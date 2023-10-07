from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from config import cfg
from utilities.hier import Encoder
from scipy.special import softmax
from utilities import dataLoader  as dl
import numpy as np
import torch
import torch.nn.functional as F


from metrics.calibration_error import calibration_error
from inference import infer

class Wrapper:
    def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, calibrater=None, is_selected=True): 
        # TODO: Move sample to app.Define sampling rate
        # TODO: Add multiple pre-processing steps
        # TODO: Method to change default value of is_selected
        # TODO: rename predict_proba

        self.__validate_input()

        self.model = model
        self.name = name
        self.tuning_space = tuning_space
        self.preprocessing_steps = preprocessing_steps
        self.preprocessing_params = preprocessing_params
        self.is_selected = is_selected
        self.calibrater_class = calibrater
        self.model_fitted = None
        self.best_params = None
        self.g_global = None
        self.roots_label = None
        self.encoder = None

    def predict_proba(self, X):
        return self.model_fitted.predict_proba(X), None

    def set_gGlobal(self, g, roots):
        self.g_global = g
        self.roots_label = roots

    def set_modelFitted(self, model_fitted):
        self.model_fitted = model_fitted

    def set_ppSteps(self, ppSteps):
        self.preprocessing_steps = ppSteps

    def set_ppParams(self, ppParams):
        self.preprocessing_params = ppParams

    def init_model(self, X, train_y_label, test_y_label):
        # Define pipeline and param_grid
        param_grid = {}
        if self.tuning_space:
            for key, value in self.tuning_space.items():
                param_grid[self.name + '__' + key] = value

        if not self.preprocessing_steps:
            pipeline = Pipeline([(self.name, self.model)])
        else:
            pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
            
            if self.preprocessing_params:
                param_grid.update(self.preprocessing_params)

        # Encode labels
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        y_train = le.fit_transform(train_y_label.to_numpy().reshape(-1, 1)).flatten()
        y_train, y_test = y_train, le.transform(test_y_label.to_numpy().reshape(-1, 1)).flatten()
        
        return pipeline, param_grid, y_train, y_test

    def __validate_input(self, model=None):
        pass

        #TODO: Validates fit and predict methods in model

    def ece(self, y_test, y_test_pred,  probas):
        ece = calibration_error(y_test, y_test_pred,  probas)
        return ece


class WrapperHier(Wrapper):
    def init_model(self, X, train_y_label, test_y_label):
            # Define pipeline and param_grid
            param_grid = {}
            if self.tuning_space:
                for key, value in self.tuning_space.items():
                    param_grid[self.name + '__' + key] = value

            if not self.preprocessing_steps:
                pipeline = Pipeline([(self.name, self.model)])
            else:
                pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
                
                if self.preprocessing_params:
                    param_grid.update(self.preprocessing_params)
            
            # Set Loss params
            # Ecode y
            self.set_gGlobal(*dl.load_full_hier(cfg.path_hier))
            en = Encoder(self.g_global, self.roots_label)

            en = en.fit(train_y_label)
            self.encoder = en
            y_train = np.array(list(map(en.node_map.get, train_y_label)))
            # y_test = np.array(list(map(partial(en.node_map.get, d=-1), test_y_label)))
            y_test = []
            for lable in test_y_label:
                 y_test.append(en.node_map.get(lable, -1))
            y_test = np.array(y_test)

            # Set input dim for NN
            try:
                num_feature = pipeline['DimensionReduction'].n_components
            except:
                num_feature = X.shape[1]

            self.model.set_params(
                 module__en=en,
                 module__dim_in=num_feature,
                 module__dim_out=len(en.G_idx.nodes()), 
                 criterion__encoder=en
            ) 

            return pipeline, param_grid, y_train, y_test

    def ece_path(self, y_true_encoded, y_test_pred,  probas):
        sum = 0
        # sum_uc = 0
        # y_true_encoded = self.encoder.transform(y_test)
        num_col = y_true_encoded.shape[1]
        num_row = y_true_encoded.shape[0]
        # y_test_pred = probas > 0.5 
        # y_test_pred = y_test_pred.astype(int)

        for col_idx in range(num_col):
            if col_idx in self.encoder.roots_idx:
                continue

            #  set proba of entry 0 as 1 - P_pos
            idxs = np.arange(num_row)
            y_true_col = y_true_encoded[:, col_idx]
            proba_col = probas[:, col_idx]
            # proba_uncalib_col = np.array(probas)[:, col_idx]
            mask = (y_true_col == 0)
            idxs_0 =idxs[mask]
            # print(idxs_0)

            proba_col[idxs_0] = 1 - proba_col[idxs_0]
            # proba_uncalib_col[idxs_0] = 1 - proba_uncalib_col[idxs_0]

            ece_col = calibration_error(y_true_col, y_test_pred[:, col_idx], proba_col)
            # ece_col_uc = calibration_error(y_true_col, np.array(y_test_pred_uncalib)[:, col_idx], proba_uncalib_col)
            
            sum += ece_col
            # sum_uc += ece_col_uc

        num_col -= len(self.encoder.roots_idx)
        ece = sum / num_col
        # ece_uc = sum_uc / num_col
        return ece
    
    def fit_calibrater(self, X, y):
        if self.calibrater_class is not None:
            self.calibrater = self.calibrater_class(self.model_fitted, encoder=self.encoder) #TODO: refactor
            self.calibrater = self.calibrater.fit(X, y)
    
    def predict_proba(self, X):
            probas_uncalib, _ = self.model_fitted.predict_proba(X)
            probas_calib = None

            if self.calibrater_class is not None:
                probas_calib = self.calibrater.predict_proba(X)
                # print(probas_calib)
            return probas_calib, probas_uncalib


    def predict(self, X, threshold=0.5):
        preds_uncalib = self.model_fitted.predict(X)
        preds_calib = None

        if self.calibrater_class is not None:
            probas_calib, _ = self.predict_proba(X)
            preds_calib = probas_calib > threshold
        # print(preds_calib)
        return preds_calib, preds_uncalib
            

class WrapperSVM(Wrapper):

        def predict_proba(self, X):
                confidence = self.model_fitted.decision_function(X)
                if confidence.ndim == 1 or confidence.shape[1] == 1:
                        confidence = np.reshape(confidence, (-1, 1))
                        confidence = np.concatenate((-1 * confidence, 1 * confidence), axis=1) # label 1 considered as positive, 
                        # print(confidence)
                        return softmax(confidence, axis=1), confidence
                return softmax(confidence, axis=1), confidence
        

class WrapperNN(Wrapper):

        def init_model(self, X, train_y_label, test_y_label):
            # Define pipeline and param_grid
            param_grid = {}
            if self.tuning_space:
                for key, value in self.tuning_space.items():
                    param_grid[self.name + '__' + key] = value

            if not self.preprocessing_steps:
                pipeline = Pipeline([(self.name, self.model)])
            else:
                pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
                
                if self.preprocessing_params:
                    param_grid.update(self.preprocessing_params)

            # Set input dim for NN
            num_class = train_y_label.nunique()
            try:
                num_feature = pipeline['DimensionReduction'].n_components
            except:
                num_feature = X.shape[1]
            self.model.set_params(module__dim_in=num_feature, module__dim_out=num_class) #TODO num_class is dependent on training set

            # Encode labels
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
            y_train = le.fit_transform(train_y_label.to_numpy().reshape(-1, 1)).flatten()
            y_train, y_test = y_train, le.transform(test_y_label.to_numpy().reshape(-1, 1)).flatten()

            return pipeline, param_grid, y_train, y_test

        def predict_proba(self, X):
            proba = self.model_fitted.predict_proba(X)
            pl_pp = Pipeline(self.model_fitted.best_estimator_.steps[:-1])
            net = self.model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
            # return proba, F.softmax(net.forward(pl_pp.transform(X)), dim=-1)
            logits = net.forward(pl_pp.transform(X)).to('cpu').data
            return proba, logits



class WrapperCHMC(WrapperHier):
        def predict_proba(self, X):
            probas_uncalib, _ = self.model_fitted.predict_proba(X)
            probas_calib = None

            if self.calibrater is not None:
                logits = self.calibrater.predict_proba(X)
                probas_calib = get_constr_out(logits, self.encoder.get_R())
                probas_calib = probas_calib.cpu().detach().numpy().astype(float)
                # print(probas_calib)
            return probas_calib, probas_uncalib


        def predict(self, X, threshold=0.5):
            preds_uncalib = self.model_fitted.predict(X)

            probas_calib, _ = self.predict_proba(X)
            preds_calib = probas_calib > threshold
            # print(preds_calib)
            return preds_calib, preds_uncalib


class WrapperCS(WrapperHier):
        # def init_model(self, X, train_y_label, test_y_label):
        #     # Define pipeline and param_grid
        #     param_grid = {}
        #     if self.tuning_space:
        #         for key, value in self.tuning_space.items():
        #             param_grid[self.name + '__' + key] = value

        #     if not self.preprocessing_steps:
        #         pipeline = Pipeline([(self.name, self.model)])
        #     else:
        #         pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
                
        #         if self.preprocessing_params:
        #             param_grid.update(self.preprocessing_params)
            
        #     # Set Loss params
        #     # Ecode y
        #     self.set_gGlobal(*dl.load_full_hier(cfg.path_hier))
        #     en = Encoder(self.g_global, self.roots_label)
        #     # y_train = en.fit_transform(train_y_label)
        #     # y_test = en.transform(test_y_label)

        #     en = en.fit(train_y_label)
        #     self.encoder = en
        #     y_train = np.array(list(map(en.node_map.get, train_y_label)))
        #     # y_test = np.array(list(map(partial(en.node_map.get, d=-1), test_y_label)))
        #     y_test = []
        #     for lable in test_y_label:
        #          y_test.append(en.node_map.get(lable, -1))
        #     y_test = np.array(y_test)


        #     # Set input dim for NN
        #     try:
        #         num_feature = pipeline['DimensionReduction'].n_components
        #     except:
        #         num_feature = X.shape[1]
            
        #     self.model.set_params(
        #          module__en=en,
        #          module__dim_in=num_feature,
        #          module__dim_out=len(en.G_idx.nodes()), 
        #          criterion__en=en
        #     ) 

        #     return pipeline, param_grid, y_train, y_test


        # def predict_proba(self, X):
        #     proba = self.model_fitted.predict_proba(X)
        #     pl_pp = Pipeline(self.model_fitted.best_estimator_.steps[:-1])
        #     net = self.model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
        #     # return proba, F.softmax(net.forward(pl_pp.transform(X)), dim=-1)
        #     return proba, net.forward(pl_pp.transform(X))

        def predict_proba(self, X):
            probas_uncalib, _ = self.model_fitted.predict_proba(X)
            probas_calib = None

            if self.calibrater is not None:
                logits = self.calibrater.predict_proba(X)
                probas_calib = F.sigmoid(logits)
                probas_calib = probas_calib.cpu().detach().numpy().astype(float)

                probas_calib = infer.infer_path(probas_calib, self.encoder)
                probas_calib = infer.run_IR(probas_calib, self.encoder)
                # print(probas_calib)
            return probas_calib, probas_uncalib


        def predict(self, X, threshold=0.5):
            preds_uncalib = self.model_fitted.predict(X)

            probas_calib, _ = self.predict_proba(X)
            preds_calib = probas_calib > threshold
            # print(preds_calib)
            return preds_calib, preds_uncalib

class WrapperLocal(WrapperHier):

        def init_model(self, X, train_y_label, test_y_label):
            # Define pipeline and param_grid
            param_grid = {}
            if self.tuning_space:
                for key, value in self.tuning_space.items():
                    param_grid[self.name + '__' + key] = value

            if not self.preprocessing_steps:
                pipeline = Pipeline([(self.name, self.model)])
            else:
                pipeline = Pipeline(self.preprocessing_steps + [(self.name, self.model)])
                
                if self.preprocessing_params:
                    param_grid.update(self.preprocessing_params)

            # Ecode y to idx in graph
            self.set_gGlobal(*dl.load_full_hier(cfg.path_hier))
            en = Encoder(self.g_global, self.roots_label)

            en = en.fit(train_y_label)
            self.encoder = en

            y_train = np.array(list(map(en.node_map.get, train_y_label)))
            y_test = []
            for lable in test_y_label:
                 y_test.append(en.node_map.get(lable, -1))
            y_test = np.array(y_test)
            
            self.model.set_encoder(en)
            return pipeline, param_grid, y_train, y_test
