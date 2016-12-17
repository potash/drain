import yaml

import os
import sys
from collections import Hashable
from copy import deepcopy
from tempfile import NamedTemporaryFile
from pprint import pformat

from sklearn.externals import joblib
from sklearn import tree
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.colors
from matplotlib import cm

from drain import model, util, metrics

def to_dataframe(steps):
    """
    Args: a collection of Step objects
    Returns: a DataFrame indexing the steps by their arguments
    """
    args = [s.named_arguments for s in steps]
    diffs = map(util.dict_expand, util.diff_dicts(args, multilevel=True))

    df = pd.DataFrame(diffs)

    # find unique arguments
    arg_count = Counter((c[1:] for c in df.columns))
    unique_args = {a for a in arg_count if arg_count[a] == 1}

    # prefix non-unique arguments with step name
    # otherwise use argument alone
    df.columns = [str.join('_', c[1:] if c[1:] in unique_args else c) 
            for c in df.columns]

    df['step'] = steps

    return df

def intersection(df, pairwise=False, **subset_args):
    """
    Counts the size of intersections of subsets of predicted examples.
    E.g. count the overlap between the top k of two different models
    Args:
        df: the result of to_dataframe(), Predict steps of length n_steps
        pairwise: when False, returns the mutual intersection between 
            all subsets. Otherwise returns an n_steps x n_steps matrix 
            whose i,j entry is the number of examples in the 
            intersection between the i and j step subsets.
        **subset_args: arguments to be passed to model.y_subset()
            for each predict step
    Returns: the intersection, either an integer, if pairwise is False, 
        or a DataFrame, otherwise.
    """
    indexes = map(lambda row: set(model.y_subset(row[1].step.get_result()['y'], **subset_args).index), df.iterrows())

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
    # make all arguments hashable so they can be column names for result
    df = df.copy()
    non_step = list(df.columns.difference({'step'}))
    df.loc[:,non_step] = df.loc[:,non_step].applymap(
        lambda d: d if isinstance(d, Hashable) 
                    else yaml.dump(d).replace('\n', ', ').rstrip(', '))

    result = df.set_index(non_step)['step'].apply(lambda step: fn(step, **kwargs)).T

    return result

def apply_y(df, fn, **kwargs):
    return apply(df, lambda s: fn(model.y_subset(s.get_result()['y'], **kwargs)))

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

    import matplotlib.pyplot as plt
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
