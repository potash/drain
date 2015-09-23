import datetime
from sklearn import metrics
import pandas as pd
import numpy as np
import math
from statsmodels.discrete.discrete_model import Logit

def y_score(estimator, X):
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    else:
        y = estimator.predict_proba(X)
        return y[:,1]

def auc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)

def precision(y_true,y_score, x, count=False):
    n = len(y_true)
    counts = x if count else [int(p*n) for p in x]
    
    ydf = pd.DataFrame({'y':y_true, 'risk':y_score}).sort('risk', ascending=False)
    precision = [(ydf.head(counts[i]).y.sum()/float(counts[i])) for i in range(len(counts)) ] # could be O(n) if it mattered

    return precision

def baseline(y_true):
    return y_true.sum() / float(len(y_true)) 

# for use with sklearn RFECV
def precision_scorer(estimator,X,y,p):
    y_score = estimator.predict_proba(X)[:,1]
    n = len(y)
    
    if p < 1:
        p = math.floor(p*n)
        
    ydf = pd.DataFrame({'y':y, 'risk':y_score}).sort('risk', ascending=False)
    return ydf.head(p).y.sum()/float(p) 

def summary(name, max_train_age, date, y_train, y_test, y_score):
    a = auc(y_test, y_score)
    n_test = len(y_test)
    p = precision(y_test,y_score, [.01,.02,.05,.10])
    data = {'name':name, 'date':date, 'max_train_age':max_train_age,
            'n_train':len(y_train), 'n_test':n_test, 
            'train_baseline':float(y_train.sum())/len(y_train),
            'test_baseline':float(y_test.sum())/len(y_test),
            'auc':a}
    data.update(p)
    return pd.DataFrame(data=data, index=[0])

def sk_tree(X,y, params={'max_depth':3}):
    clf = tree.DecisionTreeClassifier(**params)
    return clf.fit(X, y)

def feature_importance(estimator, X):
    if hasattr(estimator, 'coef_'):
        i = estimator.coef_[0]
    elif hasattr(estimator, 'feature_importances_'):
        i = estimator.feature_importances_
    else:
        i = [np.nan]*len(X.columns)

    return pd.DataFrame({'feature': X.columns, 'importance': i}).sort('importance')


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model = Logit(y, X)
        self.result = self.model.fit()
    
    def predict_proba(self, X):
        return self.result.predict(X)
