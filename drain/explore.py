from tempfile import NamedTemporaryFile
from pprint import pformat
from itertools import product, chain

from sklearn.externals import joblib
from sklearn import tree
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.colors
from matplotlib import cm

from drain import model, util, metrics, step

def expand(self, prefix=False, index=True, diff=True, existence=False):
    """
    Args: 
        prefix: whether to always use step name prefix for kwarg name.
            Default False, which uses prefixes when necessary, i.e. for 
            keywords that are shared by multiple step names.
        index: If True expand args into index. Otherwise expand into
            columns
        diff: whether to only expand keywords whose values that are 
            non-constant
        existence: whether to check for existence of a step in the tree
            instead of a diff. Only applicable when diff=True
    Returns: a DataFrame indexing the steps by their arguments
    """
    # collect kwargs resulting in a list of {name: kwargs} dicts
    dicts = [step._collect_kwargs(s) for s in self.index]
    # if any of the kwargs are themselves dicts, expand them
    #dicts = [{k: util.dict_expand(v) for k,v in s.items()} for s in dicts]

    if diff:
        if existence:
            merged_dicts = [{} for d in dicts] # the desired list of dicts

            names = util.union([set(d.keys()) for d in dicts]) # all names among these steps
            for name in names:
                ndicts = [d[name] for d in dicts if name in d.keys()] # all dicts for this name
                ndiffs = util.dict_diff(nd) # diffs for this name
                # if they were all the same
                if sum(map(len, ndiffs)) == 0: 
                    if len(ndicts) == len(self): # and every step had the name
                        continue # don't use this dict
                    else: # not every step had the name
                        exists = [name in d.keys() for d in dicts] # does the 
        else:
            merged_dicts = []
            for dd in dicts:
                merged = {}
                for name,d in dd.items():
                    merged.update({(name, k):v for k,v in d.items()})
                merged_dicts.append(merged)
            dicts = util.dict_diff(merged_dicts)
    else:
        if existence:
            raise ValueError("Cannot use existence=True when diff=False")

    # prefix_keys are the keys that will keep their prefix
    keys = list(chain(*(d.keys() for d in dicts)))
    if not prefix:
        key_count = Counter((k[1:] for k  in keys))
        prefix_keys = {a for a in key_count if key_count[a] > 1}
    else:
        prefix_keys = set(keys)

    # prefix non-unique arguments with step name
    # otherwise use argument alone
    dicts = [{str.join('_', k if k[1:] in prefix_keys else k[1:]):v 
              for k,v in d.items()} for d in dicts]

    df2 = pd.DataFrame(dicts, index=self.index)
    return df2
    columns = list(df2.columns) # remember columns for index below

    expanded = df2.join(self)

    if index:
        try:
            expanded.set_index(columns, inplace=True)
        except TypeError:
            _print_unhashable(expanded, columns)
            expanded.set_index(columns, inplace=True)

    return expanded

def dapply(self, fn, **kwargs):
    """
    apply function to each step object, passing all kwargs
    if any kwarg or function is iterable it gets grid-searched
    returns a dataframe whose columns are indexed by the non-step columns of df,
        i.e. the differential arguments of the steps
    """
    search_keys = [k for k,v in kwargs.items() if isinstance(v, list) and len(v) > 1]
    functions = util.make_list(fn)
    search = list(product(functions, util.dict_product(kwargs)))
    
    results = []
    for fn,kw in search:
        r = self.index.to_series().apply(lambda step: fn(step, **kw))
        
        name = [] if len(functions) == 1 else [fn.__name__]
        name += util.dict_subset(kw, search_keys).values()
            
        if isinstance(r, pd.DataFrame):
            columns = pd.MultiIndex.from_tuples([tuple(name + util.make_list(c)) for c in r.columns])
            r.columns = columns
        else:
            r.name = tuple(name)
        results.append(r)

    if len(results) > 1:
        result = pd.concat(results, axis=1)
        # get subset of parameters that were searched over
        column_names = [] if len(functions) == 1 else [None]
        column_names += search_keys
        column_names += [None]*(len(result.columns.names)-len(column_names))
        result.columns.names = column_names

        return StepFrame(result)
    else:
        result = results[0]
        if isinstance(result, pd.DataFrame):
            return StepFrame(result)
        else:
            result.name = functions[0].__name__
            return StepSeries(result)

class StepFrame(pd.DataFrame):
    expand = expand
    dapply = dapply

    @property
    def _constructor(self):
        return StepFrame

    @property
    def _contructor_sliced(self):
        return pd.Series

class StepSeries(pd.Series):
    expand = expand
    dapply = dapply

    @property
    def _constructor(self):
        return StepSeries

    @property
    def _contructor_expanddim(self):
        return StepFrame

def _print_unhashable(df, columns=None):
    """
    Replace unhashable values in a DataFrame with their string repr
    Args:
        df: DataFrame
        columns: columns to replace, if necessary. Default None replaces all columns.
    """
    for c in df.columns if columns is None else columns:
        if df.dtypes[c] == object:
            try:
                df[c].apply(hash)
            except TypeError:
                df[c] = df[c].dropna().apply(pformat)

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
