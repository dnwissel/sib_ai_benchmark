from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder



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