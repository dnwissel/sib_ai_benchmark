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
from loss.hier import get_constr_out
from sklearn.utils.class_weight import compute_class_weight


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
        self.calibrater = calibrater
        self.model_fitted = None
        self.best_params = None
        self.g_global = None
        self.roots_label = None
        self.encoder = None
        self.path_eval = None

    def set_predictPath(self, val):
        self.path_eval = val
        
    def set_gGlobal(self, g, roots):
        self.g_global = g
        self.roots_label = roots

    def set_modelFitted(self, model_fitted):
        self.model_fitted = model_fitted

    def set_ppSteps(self, ppSteps):
        self.preprocessing_steps = ppSteps

    def set_ppParams(self, ppParams):
        self.preprocessing_params = ppParams

    def get_pipeline(self):
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
        return pipeline, param_grid

    def init_model(self, X, train_y_label, test_y_label):
        # Define pipeline and param_grid
        pipeline, param_grid = self.get_pipeline()

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
    
    def fit_calibrater(self, X, y):
        if self.calibrater is not None:
            self.calibrater.set_model(self) 
            if hasattr(self.calibrater.criterion, 'set_encoder'):
                self.calibrater.criterion.set_encoder(self.encoder)
                        # y = self.encoder.transform(y)

            # y = self.encoder.transform(y)
            self.calibrater = self.calibrater.fit(X, y)
    
    def predict_proba(self, X):
        logits = self.get_logits(X)

        if logits is None:
            probas_uncalib_all, _ = self.model_fitted.predict_proba(X)
        else:
            probas_uncalib_all = self.nonlinear(logits)
        
        probas_calib = None
        probas_uncalib = probas_uncalib_all

        if not self.path_eval:
            probas_uncalib = np.max(probas_uncalib_all, axis=-1)

        if self.calibrater is not None:
            logits = self.calibrater.get_logits(X)
            probas_calib_all = self.nonlinear(logits)
            probas_calib = probas_calib_all
            # print(probas_calib, probas_uncalib)

            if not self.path_eval:
                probas_calib = self.predict_label_proba(probas_calib_all)

        return probas_calib, probas_uncalib

    def predict(self, X, threshold=0.5):
        preds_uncalib = self.model_fitted.predict(X)
        preds_calib = None

        if self.calibrater is not None:
            # if probas_calib_all is None:
            logits = self.calibrater.get_logits(X)
            probas_calib_all = self.nonlinear(logits)
                
            if self.path_eval:
                preds_calib = probas_calib_all > threshold
            else:
                preds_calib = self.predict_label(probas_calib_all)
        # print(preds_calib)
        return preds_calib, preds_uncalib

    def predict_label(self, probas):
        preds = np.argmax(probas, axis=-1)
        return preds

    def predict_label_proba(self, probas_all):
        probas = np.max(probas_all, axis=-1)
        return probas

    def nonlinear(self, logits):
        """
        If a classifier need to be calibrated, this method must be implemented
        """
        # return F.softmax(logits, dim=-1)
        if torch.is_tensor(logits):
            logits = logits.detach().numpy()
        return softmax(logits, axis=-1)
        
    def get_logits(self):
        """
        If a classifier need to be calibrated, this method must be implemented
        """
        return None


class WrapperHier(Wrapper):
    def init_model(self, X, train_y_label, test_y_label):
        # Define pipeline and param_grid
        pipeline, param_grid = self.get_pipeline()
        
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
        # y_true_encoded = self.encoder.transform(y_test)
        num_col = y_true_encoded.shape[1]
        num_row = y_true_encoded.shape[0]

        for col_idx in range(num_col):
            if col_idx in self.encoder.roots_idx:
                continue

            #  set proba of entry 0 as 1 - P_pos
            idxs = np.arange(num_row)
            y_true_col = y_true_encoded[:, col_idx]
            proba_col = probas[:, col_idx]
            mask = (y_true_col == 0)
            idxs_0 =idxs[mask]
            # print(idxs_0)

            proba_col[idxs_0] = 1 - proba_col[idxs_0]
            ece_col = calibration_error(y_true_col, y_test_pred[:, col_idx], proba_col)
            
            sum += ece_col

        num_col -= len(self.encoder.roots_idx)
        ece = sum / num_col
        return ece

    def get_logits(self, X):
        _, logits = self.model_fitted.predict_proba(X)
        return logits
    
    def nonlinear(self, logits):
        """
        If a classifier need to be calibrated, this method must be implemented
        """
        # return F.softmax(logits, dim=-1)
        if torch.is_tensor(logits):
            logits = logits.detach().numpy()
        return F.sigmoid(logits)
        

class WrapperSVM(Wrapper):
    def get_logits(self, X):
        confidence = self.model_fitted.decision_function(X)
        if confidence.ndim == 1 or confidence.shape[1] == 1:
            confidence = np.reshape(confidence, (-1, 1))
            confidence = np.concatenate((-1 * confidence, 1 * confidence), axis=1) # label 1 considered as positive, 
        return confidence

class WrapperXGB(Wrapper):
    def get_logits(self, X):
        confidence = self.model_fitted.decision_function(X)
        return confidence


class WrapperNN(Wrapper):
    def init_model(self, X, train_y_label, test_y_label):
        # Define pipeline and param_grid
        pipeline, param_grid = self.get_pipeline()

        # Set input dim for NN
        num_class = train_y_label.nunique()
        try:
            num_feature = pipeline['DimensionReduction'].n_components
        except:
            num_feature = X.shape[1]

        # Encode labels
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        y_train = le.fit_transform(train_y_label.to_numpy().reshape(-1, 1)).flatten()
        y_test = le.transform(test_y_label.to_numpy().reshape(-1, 1)).flatten()

        class_weight = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight = torch.from_numpy(class_weight)
        class_weight = class_weight.float()
        self.model.set_params(
            module__dim_in=num_feature, 
            module__dim_out=num_class,
            criterion__weight=class_weight
        )
        return pipeline, param_grid, y_train, y_test
    
    def get_logits(self, X):
        pl_pp = Pipeline(self.model_fitted.best_estimator_.steps[:-1])
        net = self.model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
        logits = net.forward(pl_pp.transform(X)).to('cpu').data
        return logits
  

class WrapperCHMC(WrapperHier):
    def nonlinear(self, logits):
        probas_calib = get_constr_out(logits, self.encoder.get_R())
        probas_calib = probas_calib.cpu().detach().numpy().astype(float)
        return probas_calib

    def predict_label(self, probas):
        preds = infer.infer_1(probas, self.encoder)
        return preds

    #TODO: not precise
    def predict_label_proba(self, probas_all):
        preds = infer.infer_1(probas_all, self.encoder)
        probas = []
        for idx in range(probas_all.shape[0]):
            probas.append(probas_all[idx, preds[idx]])
        return np.array(probas)


class WrapperCS(WrapperHier):
    def nonlinear(self, logits):
        probas_calib = F.sigmoid(logits)
        probas_calib = probas_calib.cpu().detach().numpy().astype(float)

        probas_calib = infer.infer_path_cs(probas_calib, self.encoder)
        probas_calib = infer.run_IR(probas_calib, self.encoder)
        return probas_calib

    def predict_label(self, probas_all):
        preds, _ = infer.infer_cs(probas_all, self.encoder)
        return preds

    def predict_label_proba(self, probas_all):
        _, probas = infer.infer_cs(probas_all, self.encoder)
        return probas

class WrapperLocal(WrapperHier):
    def init_model(self, X, train_y_label, test_y_label):
        # Define pipeline and param_grid
        pipeline, param_grid = self.get_pipeline()

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

     