from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid


class GridSearchValidation(BaseEstimator):
    def __init__(self, model, scorer, parameter_grid):
        self.model = model
        self.scorer = scorer
        self.parameter_grid = list(ParameterGrid(parameter_grid))

    def fit(self, X_train, y_train, X_valid, y_valid):
        best_model, best_params = None, None
        best_score = 0
        for params in self.parameter_grid:
            clf = self.model(**params)
            clf.fit(X_train, y_train)
            score_train = self.scorer(y_train, clf.predict_proba(X_train))
            score_valid = self.scorer(y_valid, clf.predict_proba(X_valid))
            print(f"Current train score equals {score_train}",
                  f"Current validation score equals {score_valid}", sep='\n')
            if score_valid > best_score:
                best_score = score_valid
                best_model = clf
                best_params = params
        self.best_params_ = best_params
        self.best_model_ = best_model

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        return self.best_model_.predict_proba(X)
