import os
from sklearn.externals import joblib
import yaml
import pandas as pd
from drain.model import model
from drain.model import util
import numpy as np
from copy import deepcopy
import matplotlib.colors
from matplotlib import cm
import matplotlib.pyplot as plt

def dict_to_df(d):
    df = pd.DataFrame(index=[0])
    for k in d:
        if isinstance(d[k], dict):
            for k2 in d[k]:
                df[k2] = [str(d[k][k2])]
        else:
            df[k] = [d[k]]
    return df

def read_model(dirname, estimator=True):
    estimator = (joblib.load(os.path.join(dirname, 'estimator.pkl'))) if estimator else None
    estimator_name = estimator.__class__.__name__ if estimator else None
    
    y = (pd.read_csv(os.path.join(dirname, 'y.csv'), index_col=0))
    train = (pd.Series.from_csv(os.path.join(dirname, 'train.csv'), index_col=0))
    test = (pd.Series.from_csv(os.path.join(dirname, 'test.csv'), index_col=0))
    params = yaml.load(open(os.path.join(dirname, 'params.yaml')))
    columns = pd.read_csv(os.path.join(dirname, 'columns.csv')).columns

    df = dict_to_df(params)
    df['estimator'] = [estimator]
    df['estimator_name'] = [estimator_name]
    df['y'] = [y]
    df['train'] = [train]
    df['test'] = [test]
    df['columns'] = [columns]
    df['params'] = [params]
    df['data'] = [util.init_object(**params['data'])]

    return df

def read_models(dirname, estimator=True):
    df = pd.concat((read_model(subdir, estimator) for subdir in get_subdirs(dirname)), ignore_index=True)
    calculate_metrics(df)

    return df

def calculate_metrics(df):
    df['auc'] = df['y'].apply(lambda y: model.auc(y['true'], y['score']))

    for p in [.01, .02, .05, .1]:
        df['precision' + str(p)] = df['y'].apply(lambda y: precision(y['true'], y['score'], p))

    df['baseline']=df.y.apply(lambda y: y.true.sum()*1.0/len(y.true))

    df['coef'] = [get_coef(row) for i,row in df.iterrows()]
    
    return df

def get_coef(row):
    if hasattr(row['estimator'], 'coef_'):
        return pd.DataFrame({'name':row['columns'], 'c':row['estimator'].coef_[0]}).sort('c')
    else:
        return pd.DataFrame()

def get_subdirs(directory):
     return [os.path.join(directory, name) for name in os.listdir(directory) 
             if os.path.isdir(os.path.join(directory, name))]

def precision(y_true, y_score, p):
    count = int(p*len(y_true))
    ydf = pd.DataFrame({'y':y_true, 'risk':y_score}).sort('risk', ascending=False)
    return ydf.head(count).y.sum()/float(count)

def insert_coef(df):
    df['coef'] = [pd.DataFrame({'name':row['columns'], 'c':row['estimator'].coef_[0]}).sort('c') 
                        for i,row in df.iterrows()]

def mask(df, key, value):
    return df[df[key] == value]

pd.DataFrame.mask = mask

def get_series(df, columns, value, indices=None, index_names=None,index='year', include_baseline=True):
    unstacked = df.set_index([index]+columns).unstack(columns)
    if indices is not None:
        indices2 = [(value,) + i for i in indices]
    else:
        indices2 = [i for i in unstacked.columns if i[0] == value]
        
    if include_baseline:
        baseline = indices2[0] + tuple()
        indices2.append(('baseline',) + indices2[0][1:])
            
    series = unstacked[indices2]
    if index_names is not None:
        series.columns=index_names + ['baseline']
    return series

# for a given example idx get the series X*\beta
def get_example_series(run, idx):
    return pd.DataFrame({'c':(run['data'].X.ix[[idx]].values[0]*run['estimator'].coef_)[0], 'name':run['columns']}).sort('c')

# http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html
# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def jenks_labels(breaks):
    return ["<= %0.1f (%s wards)" % (b, c) for b, c in zip(breaks.bins, breaks.counts)]
