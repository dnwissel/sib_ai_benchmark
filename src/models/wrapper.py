from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from config import cfg
from utilities.hier import Encoder, get_lossMask, get_R
from scipy.special import softmax
from utilities import dataLoader  as dl
import numpy as np
import torch




class Wrapper:
    def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
        # TODO: Move sample to app.Define sampling rate
        # TODO: Add multiple pre-processing steps
        # TODO: Method to change default value of is_selected
        # TODO: rename predict_proba

        self.__validate_input()

        self.model = model
        self.model_fitted = None
        self.name = name
        self.tuning_space = tuning_space
        self.preprocessing_steps = preprocessing_steps
        self.preprocessing_params = preprocessing_params
        self.is_selected = is_selected
        self.best_params = None
        self.g_global = None
        self.roots_label = None

    def predict_proba(self, model_fitted, X):
        return model_fitted.predict_proba(X), None

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


class WrapperSVM(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected)

        def predict_proba(self, model_fitted, X):
                confidence = model_fitted.decision_function(X)
                if confidence.ndim == 1 or confidence.shape[1] == 1:
                        confidence = np.reshape(confidence, (-1, 1))
                        confidence = np.concatenate((-1 * confidence, 1 * confidence), axis=1) # label 1 considered as positive, 
                        # print(confidence)
                        return softmax(confidence, axis=1), confidence
                return softmax(confidence, axis=1), confidence
        

class WrapperNN(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected)

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

        def predict_proba(self, model_fitted, X):
            proba = model_fitted.predict_proba(X)
            pl_pp = Pipeline(model_fitted.best_estimator_.steps[:-1])
            net = model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
            # return proba, F.softmax(net.forward(pl_pp.transform(X)), dim=-1)
            logits = net.forward(pl_pp.transform(X)).to('cpu').data
            return proba, logits



class WrapperHier(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected)

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
            y_train = np.array(list(map(en.node_map.get, train_y_label)))
            # y_test = np.array(list(map(partial(en.node_map.get, d=-1), test_y_label)))
            y_test = []
            for lable in test_y_label:
                 y_test.append(en.node_map.get(lable, -1))
            y_test = np.array(y_test)

            # print(y_train, y_test)
            nodes = en.G_idx.nodes()
            idx_to_eval = list(set(nodes) - set(en.roots_idx))

            # Set input dim for NN
            try:
                num_feature = pipeline['DimensionReduction'].n_components
            except:
                num_feature = X.shape[1]

            self.model.set_params(
                 module__en=en,
                 module__dim_in=num_feature,
                 module__dim_out=len(en.G_idx.nodes()), 
                 criterion__en=en,
                 criterion__idx_to_eval=idx_to_eval
            ) 

            # y = y.astype(np.int64)
            return pipeline, param_grid, y_train, y_test


        def predict_proba(self, model_fitted, X):
            proba = model_fitted.predict_proba(X)
            pl_pp = Pipeline(model_fitted.best_estimator_.steps[:-1])
            net = model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
            # return proba, F.softmax(net.forward(pl_pp.transform(X)), dim=-1)
            return proba, net.forward(pl_pp.transform(X))


class WrapperHierCS(Wrapper):
        def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True): 
                super().__init__(model, name, tuning_space, preprocessing_steps, preprocessing_params, is_selected)

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
            # y_train = en.fit_transform(train_y_label)
            # y_test = en.transform(test_y_label)

            en = en.fit(train_y_label)
            y_train = np.array(list(map(en.node_map.get, train_y_label)))
            # y_test = np.array(list(map(partial(en.node_map.get, d=-1), test_y_label)))
            y_test = []
            for lable in test_y_label:
                 y_test.append(en.node_map.get(lable, -1))
            y_test = np.array(y_test)

            nodes = en.G_idx.nodes()
            idx_to_eval = list(set(nodes) - set(en.roots_idx))

            # Set input dim for NN
            try:
                num_feature = pipeline['DimensionReduction'].n_components
            except:
                num_feature = X.shape[1]
            
            self.model.set_params(
                 module__en=en,
                 module__dim_in=num_feature,
                 module__dim_out=len(en.G_idx.nodes()), 
                 criterion__en=en,
                 criterion__idx_to_eval=idx_to_eval
            ) 

            # y = y.astype(np.int64)
            return pipeline, param_grid, y_train, y_test


        def predict_proba(self, model_fitted, X):
            proba = model_fitted.predict_proba(X)
            pl_pp = Pipeline(model_fitted.best_estimator_.steps[:-1])
            net = model_fitted.best_estimator_.steps[-1][1] #TODO: make a copy
            # return proba, F.softmax(net.forward(pl_pp.transform(X)), dim=-1)
            return proba, net.forward(pl_pp.transform(X))