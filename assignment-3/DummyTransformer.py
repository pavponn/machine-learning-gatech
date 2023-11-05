from sklearn.base import BaseEstimator, TransformerMixin


class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # This transformer does nothing during fitting
        return self

    def transform(self, X):
        # This transformer does nothing during transformation
        return X
