import os
import sys
import yaml
from copy import deepcopy
from tempfile import NamedTemporaryFile
from pprint import pformat

from sklearn.externals import joblib
from sklearn import tree
import pandas as pd
import numpy as np

import matplotlib.colors
from matplotlib import cm
import matplotlib.pyplot as plt

from drain import model, util, metrics
from drain.model import params_dir

def read_data(row, basedir, transform=True):
    params = {'data': row['params']['data']}
    datadir = os.path.join(params_dir(basedir, params, 'data'), 'output')
    row['data'].read(datadir)

    if transform:
        row['data'].transform(**row['params']['transform'])

def read_estimator(row, basedir):
    modeldir = os.path.join(params_dir(basedir, row['params'], 'model'), 'output')
    row['estimator'] = joblib.load(os.path.join(modeldir, 'estimator.pkl'))

# pairwise returns a dataframe with intersections
def intersection(df, pairwise=False, **subset_args):
    indexes = map(lambda row: set(model.y_subset(row[1].y, **subset_args).index), df.iterrows())

    if not pairwise:
        return len(util.intersect(indexes))
    else:
        r = pd.DataFrame(index=df.index, columns=xrange(len(df)))

        for i in xrange(len(df)):
            r.values[i][i] = len(indexes[i])
            for j in xrange(i+1, len(df)):
                r.values[i][j] = len(indexes[i] & indexes[j])
        return r

def apply(df, fn, **kwargs):
    return df.apply(lambda run: fn(run=run, **kwargs), axis=1).T

def read_model(dirname, estimator=False):
    outdirname = os.path.join(dirname, 'output/')
    if not os.path.isdir(outdirname):
        return
 
    mtime = util.mtime(outdirname)

    estimator = (joblib.load(os.path.join(outdirname, 'estimator.pkl'))) if estimator else None
    
    y = pd.read_hdf(os.path.join(outdirname, 'y.hdf'), 'y')
    features = pd.read_csv(os.path.join(outdirname, 'features.csv'))
    params = yaml.load(open(os.path.join(outdirname, '../params.yaml')))

    estimator_name = params['model']['name']

    df = dict_to_df(params)
    df['timestamp'] = [mtime]
    df['dirname'] = [dirname]
    df['estimator'] = [estimator]
    df['estimator_name'] = [estimator_name]
    df['y'] = [y]
    df['features'] = [features]
    df['n_features'] = [len(features)]
    df['params'] = [params]
    df['data'] = [util.init_object(**params['data'])]

    return df

def read_models(dirname, tagname=None, estimator=False):
    if tagname is not None:
        dirname = os.path.join(dirname, 'tag', tagname)
    else:
        dirname = os.path.join(dirname, 'model')
    df = pd.concat((read_model(subdir, estimator) for subdir in get_subdirs(dirname)), ignore_index=True)

    reset_index(df, inplace=True)
    return df

# set model runs dataframe index using diff of params
def reset_index(df, inplace=False):
    diffs = dict_diff(df.params.values)

    first = True
    for c in util.union(map(set, diffs)):
        s = df[c]

        # shorten model and transform names by removing module name
        if c.endswith('name'):
            s = s.apply(lambda d: d[d.rfind('.')+1:]) 
        s = s.fillna('') # make nan empty to look nicer
        df.set_index(s.apply(lambda d: str(d)), append=(not first), inplace=True)
   
        first=False

    return df

def get_subdirs(directory):
     return [os.path.join(directory, name) for name in os.listdir(directory) 
             if os.path.isdir(os.path.join(directory, name))]

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

def show_tree(tree, feature_names,max_depth=None):
    import wand.image

    filename = NamedTemporaryFile(delete=False).name
    export_tree(tree, filename, [c.encode('ascii') for c in feature_names],max_depth)
    img = wand.image.Image(filename=filename)
    return img

def export_tree(clf, filename, feature_names=None, max_depth=None):
    from sklearn.externals.six import StringIO
    import pydot

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names, max_depth=max_depth)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)

# turn a dictionary into a dataframe row
# subdictionaries get included either multilevel or prefixed
def dict_to_df(d, multilevel=False):
    df = pd.DataFrame(index=[0])
    for k in d:
        if isinstance(d[k], dict):
            for k2 in d[k]:
                k3 = (k,k2) if multilevel else '{}_{}'.format(str(k), str(k2))
                df[k3] = [d[k][k2]]
        else:
            df[k] = [d[k]]
    return df

def dict_diff(dictionaries, multilevel=False):
    diffs = [{} for d in dictionaries]
    for top_key in ['data','model', 'transform']:
        dicts = [d[top_key] for d in dictionaries]
        keys = map(lambda d: set(d.keys()), dicts)
        intersection = reduce(lambda a,b: a&b, keys)
        
        # uncommon keys
        diff = [{k:d[k] for k in d if k not in intersection} for d in dicts]
        
        # common keys
        for key in intersection:
            if len(set(yaml.dump(d[key]) for d in dicts)) > 1:
                for d1, d2 in zip(diff, dicts):
                    d1[key] = d2[key]

        for i in xrange(len(diff)):
            if multilevel:
                diff[i] = {(top_key, k):v for k,v in diff[i].iteritems()}
            else:
                diff[i] = {'{}_{}'.format(str(top_key), str(k)):v for k,v in diff[i].iteritems()}

        if len(diff[0]) > 0: # add to total diff if non-empty
            for d1, d2 in zip(diffs, diff):
                d1.update(d2)

    return diffs
