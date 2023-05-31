class Wrapper:
    
    def __init__(self, model, name, tuning_space=None, preprocessing_steps=None, preprocessing_params=None, is_selected=True, data_shape_required=False): 
        # TODO: Move sample to app.Define sampling rate
        # TODO: Add multiple pre-processing steps
        # TODO: Method to change default value of is_selected

        self.__validate_input()

        self.model = model
        self.name = name
        self.tuning_space = tuning_space
        self.preprocessing_steps = preprocessing_steps
        self.preprocessing_params = preprocessing_params
        self.is_selected = is_selected
        self.data_shape_required = data_shape_required
        self.best_params = None

    def __validate_input(self,model=None):
        pass

        #TODO: Validates fit and predict methods in model