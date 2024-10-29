from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Clase para incorporar a la pipeline la conversion de 0 en None
class ReplaceZeroWithNone(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = pd.Series()):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].replace(0, None)

        return X
