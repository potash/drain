import os
import sys
import yaml
from collections import Hashable
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

import model, util, metrics

def to_dataframe(steps):
    args = [s.named_arguments for s in steps]
    diffs = util.diff_dicts(args, multilevel=True)

    df = pd.DataFrame(diffs)
    df.columns = [str.join('_', c) for c in df.columns]

    df['step'] = steps

    return df

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

def apply(df, fn, include_baseline=False,**kwargs):
    # make all arguments hashable so they can be column names for result
    df = df.copy()
    non_step = list(df.columns.difference({'step'}))
    df.loc[:,non_step] = df.loc[:,non_step].applymap(
        lambda d: d if isinstance(d, Hashable) 
                    else yaml.dump(d).replace('\n', ', ').rstrip(', '))

    result = df.set_index(non_step)['step'].apply(lambda step: fn(step, **kwargs)).T


    if include_baseline:
        baseline_kwargs = util.dict_subset(kwargs, 
                ['dropna', 'outcome', 'score', 'query'])
        result['baseline'] = model.baseline(df.iloc[0]['step'], 
                **baseline_kwargs)
    return result

def apply_y(df, fn, **kwargs):
    return apply(df, lambda s: fn(model.y_subset(s.get_result()['y'], **kwargs)))

def get_subdirs(directory):
     return [os.path.join(directory, name) for name in os.listdir(directory) 
             if os.path.isdir(os.path.join(directory, name))]

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
