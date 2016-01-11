from step import Step
import model

import os
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.externals import joblib
from numpy import random

class FitPredict(Step):
    def __init__(self, return_estimator=False, return_feature_importances=True, return_predictions=True, prefit=False, **kwargs):
        Step.__init__(self, return_estimator=return_estimator, 
                return_feature_importances=return_feature_importances, 
                return_predictions=return_predictions, prefit=prefit, **kwargs)

    def run(self, estimator, X, y, train=None, test=None, **kwargs):
        if not prefit:
            if train is not None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = X, y

            estimator.fit(X_train, y_train)

        result = {}

        if self.return_estimator:
            result['estimator'] = estimator
        if self.return_feature_importances:
            result['feature_importances'] = model.feature_importance(estimator, X)
        if self.return_predictions:
            if test is not None:
                X_test, y_test = X[test], y[test]
            else:
                X_test, y_test = X, y
                    
            result['score'] =  pd.Series(model.y_score(estimator, X[test]),
                    index=X[test].index)

        return result

    def dump(self):
        result = self.get_result()
        if self.return_estimator:
            filename = os.path.join(self.get_dump_dirname(), 'estimator.pkl')
            joblib.dump(result['estimator'], filename)
        if self.return_feature_importances:
            filename = os.path.join(self.get_dump_dirname(), 'feature_importances.hdf')
            result['feature_importances'].to_hdf(filename, 'df')
        if self.return_feature_importances:
            filename = os.path.join(self.get_dump_dirname(), 'score.hdf')
            result['score'].to_hdf(filename, 'df')

    def load(self):
        result = {}
        if self.return_estimator:
            filename = os.path.join(self.get_dump_dirname(), 'estimator.pkl')
            result['estimator'] = joblib.load(filename)
        if self.return_feature_importances:
            filename = os.path.join(self.get_dump_dirname(), 'feature_importances.hdf')
            result['feature_importances'] = pd.read_hdf(filename, 'df')
        if self.return_feature_importances:
            filename = os.path.join(self.get_dump_dirname(), 'score.hdf')
            result['score'] = pd.read_hdf(filename, 'df')


class Fit(FitPredict):
    def __init__(**kwargs):
        FitPredict.__init__(self, return_predictions=False, prefit=False, **kwargs)

class Predict(FitPredict):
    def __init__(**kwargs):
        FitPredict.__init__(self, return_feature_importances=False,
                return_predictions=True, prefit=True, **kwargs)

class ClassificationData(Step):
    def run(self):
        X,y = datasets.make_classification(**self.__kwargs__)
        X,y = pd.DataFrame(X), pd.Series(y)

        train = np.zeros(len(X), dtype=bool)
        train[random.choice(len(X), len(X)/2)] = True
        train = pd.Series(train)
        
        return {'X': X, 'y': y, 'train': train, 'test': ~train}
