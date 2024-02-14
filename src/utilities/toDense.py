class ToDense:
    def __init__(self):
        pass

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X):
        return X.toarray()
