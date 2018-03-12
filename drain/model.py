import os
import sys
import logging
import inspect

import pandas as pd
import numpy as np

from sklearn.externals import joblib

from drain import util, metrics
from drain.step import Step, Call


class FitPredict(Step):
    """
    Step which can fit a scikit-learn estimator and make predictions.
    """
    def __init__(self, inputs,
                 return_estimator=False,
                 return_feature_importances=True,
                 return_predictions=True,
                 prefit=False,
                 predict_train=False):
        """
        Args:
            return_estimator: whether to return the fitted estimator object
            return_feature_importances: whether to return a DataFrame of feature importances
            prefit: whether the estimator input is already fitted
            predict_train: whether to make predictions on training set
        """
        Step.__init__(self, inputs=inputs, return_estimator=return_estimator,
                      return_feature_importances=return_feature_importances,
                      return_predictions=return_predictions, prefit=prefit,
                      predict_train=predict_train)

    def run(self, estimator, X, y=None, train=None, test=None, aux=None, sample_weight=None,
            feature_importances=None):
        if not self.prefit:
            if y is None:
                raise ValueError("Need outcome data y for predictions")
            if train is not None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = X, y

            y_missing = y_train.isnull()
            y_missing_count = y_missing.sum()
            if y_missing.sum() > 0:
                logging.info('Dropping %s training examples with missing outcomes'
                             % y_missing_count)
                y_train = y_train[~y_missing]
                X_train = X_train[~y_missing]

            y_train = y_train.astype(bool)

            logging.info('Fitting with %s examples, %s features' % X_train.shape)
            if 'sample_weight' in inspect.getargspec(estimator.fit).args and\
                    sample_weight is not None:
                logging.info('Using sample weight')
                sample_weight = sample_weight.loc[y_train.index]
                estimator.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                estimator.fit(X_train, y_train)

        result = {}

        if self.return_estimator:
            result['estimator'] = estimator
        if self.return_feature_importances:
            result['feature_importances'] = feature_importance(estimator, X)
        if self.return_predictions:
            if test is not None and not self.predict_train:
                X_test, y_test = X[test], y[test]
            else:
                X_test, y_test = X, y

            logging.info('Predicting %s examples' % len(X_test))
            if y_test is not None:
                y = pd.DataFrame({'test': y_test})
            else:
                y = pd.DataFrame(index=X_test.index)

            y['score'] = y_score(estimator, X_test)

            if self.predict_train:
                y['train'] = train

            if aux is not None:
                y = y.join(aux, how='left')

            result['y'] = y

        return result

    def dump(self):
        result = self.result
        if self.return_estimator:
            filename = os.path.join(self._dump_dirname, 'estimator.pkl')
            joblib.dump(result['estimator'], filename)
        if self.return_feature_importances:
            filename = os.path.join(self._dump_dirname, 'feature_importances.hdf')
            result['feature_importances'].to_hdf(filename, 'df')
        if self.return_predictions:
            filename = os.path.join(self._dump_dirname, 'y.hdf')
            result['y'].to_hdf(filename, 'df')

    def load(self):
        result = {}
        if self.return_estimator:
            filename = os.path.join(self._dump_dirname, 'estimator.pkl')
            result['estimator'] = joblib.load(filename)
        if self.return_feature_importances:
            filename = os.path.join(self._dump_dirname, 'feature_importances.hdf')
            result['feature_importances'] = pd.read_hdf(filename, 'df')
        if self.return_predictions:
            filename = os.path.join(self._dump_dirname, 'y.hdf')
            result['y'] = pd.read_hdf(filename, 'df')

        self.result = result


class Fit(FitPredict):
    def __init__(self, inputs, return_estimator=True, return_feature_importances=False):
        FitPredict.__init__(self, inputs=inputs, prefit=False,
                            return_estimator=return_estimator,
                            return_feature_importances=return_feature_importances,
                            return_predictions=False)


class Predict(FitPredict):
    def __init__(self, inputs, return_estimator=False, return_feature_importances=False):
        FitPredict.__init__(self, inputs=inputs,
                            return_feature_importances=return_feature_importances,
                            return_estimator=return_estimator,
                            return_predictions=True, prefit=True)


class PredictProduct(Step):
    def run(self, **kwargs):
        keys = list(kwargs.keys())
        ys = [kwargs[k]['y'] for k in keys]
        y = ys[0].copy()
        y.rename(columns={'score': 'score_%s' % keys[0]}, inplace=True)
        y['score_%s' % keys[1]] = ys[1].score
        y['score'] = ys[0].score * ys[1].score

        return {'y': y}


class InverseProbabilityWeights(Step):
    def run(self, y, train=None, **kwargs):
        if train is not None:
            logging.info("Using training mask")
            train = train[train].index
            intersection = y.index.intersection(train)
            if len(intersection) != len(train):
                raise ValueError("Must provide scores for every training example.")
            y = y.ix[intersection]

        return {'sample_weight': y.score**-1}


def y_score(estimator, X):
    """
    Score examples from a new matrix X
    Args:
        estimator: an sklearn estimator object
        X: design matrix with the same features that the estimator was trained on

    Returns: a vector of scores of the same length as X

    Note that estimator.predict_proba is preferred but when unavailable
    (e.g. SVM without probability calibration) decision_function is used.
    """
    try:
        y = estimator.predict_proba(X)
        return y[:, 1]
    except(AttributeError):
        return estimator.decision_function(X)


def feature_importance(estimator, X):
    if hasattr(estimator, 'coef_'):
        i = estimator.coef_[0]
    elif hasattr(estimator, 'feature_importances_'):
        i = estimator.feature_importances_
    else:
        i = [np.nan]*X.shape[1]

    features = X.columns if hasattr(X, 'columns') else range(X.shape[1])

    return pd.DataFrame({'feature': features, 'importance': i}).\
        sort_values('importance', ascending=False)


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        from statsmodels.discrete.discrete_model import Logit
        self.model = Logit(y, X)
        self.result = self.model.fit()

    def predict_proba(self, X):
        return self.result.predict(X)


def _proximity_parallel_helper(train_nodes, t, k):
    d = (train_nodes == t).sum(axis=1)
    n = d.argsort()[::-1][:k]

    return d[n], n  # distance, neighbors


def _proximity_helper(train_nodes, test_nodes, k):
    from sklearn.externals.joblib import Parallel, delayed

    results = Parallel(n_jobs=16, backend='threading')(
        delayed(_proximity_parallel_helper)(train_nodes, t, k) for t in test_nodes)
    distance, neighbors = zip(*results)
    return np.array(distance), np.array(neighbors)


def apply_forest(run):
    # store nodes in run
    run['nodes'] = pd.DataFrame(run.estimator.apply(run['data'].X), index=run['data'].X.index)


def proximity(run, ix, k):
    # look for nodes in training set proximal to the given nodes
    if 'nodes' not in run:
        apply_forest(run)
    distance, neighbors = _proximity_helper(run['nodes'][run.y.train].values,
                                            run['nodes'].loc[ix].values, k)
    neighbors = run['nodes'][run.y.train].irow(neighbors.flatten()).index
    neighbors = [neighbors[k*i:k*(i+1)] for i in range(len(ix))]
    return distance, neighbors


def y_subset(y, query=None, aux=None, subset=None, dropna=False, outcome='true',
             k=None, p=None, ascending=False, score='score', p_of='notnull'):
    """
    Subset a model "y" dataframe
    Args:
        query: operates on y, or aux if present
        subset: takes a dataframe or index thereof and subsets to that
        dropna: means drop missing outcomes
        return: top k (count) or p (proportion) if specified
        p_of: specifies what the proportion is relative to
            'notnull' means proportion is relative to labeled count
            'true' means proportion is relative to positive count
            'all' means proportion is relative to total count
    """
    if query is not None:
        if aux is None:
            y = y.query(query)
        else:
            s = aux.ix[y.index]
            if len(s) != len(y):
                logging.warning('y not a subset of aux')
            y = y.ix[s.query(query).index]

    if subset is not None:
        if hasattr(subset, 'index'):
            subset = subset.index
        y = y.ix[y.index.intersection(subset)]

    if dropna:
        y = y.dropna(subset=[outcome])

    if k is not None and p is not None:
        raise ValueError("Cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        if p_of == 'notnull':
            k = int(p*y[outcome].notnull().sum())
        elif p_of == 'true':
            k = int(p*y[outcome].sum())
        elif p_of == 'all':
            k = int(p*len(y))
        else:
            raise ValueError('Invalid value for p_of: %s' % p_of)
    else:
        k = None

    if k is not None:
        y = y.sort_values(score, ascending=ascending).head(k)

    return y


# list of arguments to y_subset() for Metric above
Y_SUBSET_ARGS = inspect.getargspec(y_subset).args


def true_score(y, outcome='true', score='score', **subset_args):
    y = y_subset(y, outcome=outcome, score=score, **subset_args)
    return util.to_float(y[outcome], y[score])


def make_metric(function):
    def metric(predict_step, **kwargs):
        y = predict_step.result['y']
        subset_args = [k for k in Y_SUBSET_ARGS if k in kwargs]
        kwargs_subset = {k: kwargs[k] for k in subset_args}
        y_true, y_score = true_score(y, **kwargs_subset)

        kwargs_metric = {k: kwargs[k] for k in kwargs if k not in Y_SUBSET_ARGS}
        r = function(y_true, y_score, **kwargs_metric)
        return r

    return metric


metric_functions = [o for o in inspect.getmembers(metrics)
                    if inspect.isfunction(o[1]) and not o[0].startswith('_')]

for name, function in metric_functions:
    function = make_metric(function)
    function.__name__ = name
    setattr(sys.modules[__name__], name, function)


def lift(predict_step, **kwargs):
    p = precision(predict_step, **kwargs)  # noqa: F821
    kwargs.pop('k', None)
    kwargs.pop('p', None)
    b = baseline(predict_step, **kwargs)  # noqa: F821
    return p/b


def lift_series(predict_step, **kwargs):
    p = precision_series(predict_step, **kwargs)  # noqa: F821

    # pass everything except k or p to baseline
    b_kwargs = {k: v for k, v in kwargs.items() if k not in ('k', 'p')}
    b = baseline(predict_step, **b_kwargs)  # noqa: F821

    return p/b


def recall(predict_step, prop=True, **kwargs):
    r = make_metric(metrics.recall)(predict_step, **kwargs)
    if prop:
        kwargs.pop('k', None)
        kwargs.pop('p', None)
        c = make_metric(metrics.recall)(predict_step, **kwargs)
        return r/c
    else:
        return r


def recall_series(predict_step, prop=True, **kwargs):
    r = make_metric(metrics.recall_series)(predict_step, **kwargs)
    if prop:
        kwargs.pop('k', None)
        kwargs.pop('p', None)
        c = make_metric(metrics.recall)(predict_step, **kwargs)
        return r/c
    else:
        return r


def overlap(self, other, **kwargs):
    y0 = self.result['y']
    y0 = y_subset(y0, **kwargs)

    y1 = other.result['y']
    y1 = y_subset(y1, **kwargs)

    return len(y0.index & y1.index)


def similarity(self, other, **kwargs):
    y0 = self.result['y']
    y0 = y_subset(y0, **kwargs)

    y1 = other.result['y']
    y1 = y_subset(y1, **kwargs)

    return np.float32(len(y0.index & y1.index)) / \
        len(y0.index | y1.index)


def rank(self, **kwargs):
    y0 = self.result['y']
    y0 = y_subset(y0, **kwargs)
    return y0.score.rank(ascending=False)


def perturb(estimator, X, bins, columns=None):
    """
    Predict on peturbations of a feature vector
    estimator: a fitted sklearn estimator
    index: the index of the example to perturb
    bins: a dictionary of column:bins arrays
    columns: list of columns if bins doesn't cover all columns
    TODO make this work when index is multiple rows
    """
    if columns is None:
        if len(bins) != X.shape[1]:
            raise ValueError("Must specify columns when not perturbing all columns")
        else:
            columns = X.columns

    n = np.concatenate(([0], np.cumsum([len(b) for b in bins])))

    X_test = np.empty((n[-1]*X.shape[0], X.shape[1]))
    r = pd.DataFrame(columns=['value', 'feature', 'index'], index=np.arange(n[-1]*X.shape[0]))
    for j, index in enumerate(X.index):
        X_test[j*n[-1]:(j+1)*n[-1], :] = X.values[j, :]
        for i, c in enumerate(columns):
            s = slice(j*n[-1] + n[i], j*n[-1] + n[i+1])
            r['value'].values[s] = bins[i]
            r['feature'].values[s] = c
            r['index'].values[s] = [index]*(n[i+1]-n[i])
            X_test[s, (X.columns == c).argmax()] = bins[i]

    y = estimator.predict_proba(X_test)[:, 1]
    r['y'] = y
    return r


def forests(**kwargs):
    steps = []
    d = dict(criterion=['entropy', 'gini'], max_features=['sqrt', 'log2'],
             n_jobs=[-1], **kwargs)
    for estimator_args in util.dict_product(d):
        steps.append(Call(
                     'sklearn.ensemble.RandomForestClassifier',
                     **estimator_args))

    return steps


def logits(**kwargs):
    steps = []
    for estimator_args in util.dict_product(dict(
            penalty=['l1', 'l2'], C=[.001, .01, .1, 1], **kwargs)):
        steps.append(Call('sklearn.linear_model.LogisticRegression',
                          **estimator_args))

    return steps


def svms(**kwargs):
    steps = []
    for estimator_args in util.dict_product(dict(
                 penalty=['l2'],
                 dual=[True, False], C=[.001, .01, .1, 1])) + \
            util.dict_product(dict(
                    penalty=['l1'], dual=[False], C=[.001, .01, .1, 1])):
        steps.append(Call('sklearn.svm.LinearSVC',
                          **estimator_args))

    return steps
