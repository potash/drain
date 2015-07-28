import datetime
from sklearn import metrics
import pandas as pd
import math
from statsmodels.discrete.discrete_model import Logit

from sklearn import tree
import wand.image
from sklearn.externals.six import StringIO  
import pydot

def y_score(estimator, X):
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    else:
        return estimator.predict_proba(X)[:,1]

def auc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)

def precision(y_true,y_score, proportions):
    n = len(y_true)
    counts = [int(p*n) for p in proportions]
    
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
    p = precision(y_test,y_score, proportions=[.01,.02,.05,.10])
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

def show_tree(X,y,params):
    filename ="tree.pdf"
    clf = sk_tree(X,y, params)
    export_tree(clf, filename, [c.encode('ascii') for c in X.columns])
    img = wand.image.Image(filename=filename)
    return img
    
def export_tree(clf, filename, feature_names=None):
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf(filename)

class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model = Logit(y, X)
        self.result = self.model.fit()
    
    def predict_proba(self, X):
        return self.result.predict(X)
