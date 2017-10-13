from tempfile import NamedTemporaryFile
from pprint import pformat
from itertools import product

from sklearn import tree
import pandas as pd
from collections import Counter
from six import StringIO

from drain import util, step


def explore(steps, reload=False):
    return StepFrame(index=step.load(steps, reload=reload))


def expand(self, prefix=False, index=True, diff=True, existence=True):
    """
    This function is a member of StepFrame and StepSeries. It is used to
    expand the kwargs of the steps either into the index (index=True) or
    as columns (index=False). By default (diff=True) only the kwargs which
    differ among steps are expanded.

    Note that index objects in pandas must be hashable so any unhashable
    argument values are converted to string representations (using pprint)
    when index=True.

    If "inputs" is an argument those steps' kwargs are also expanded (and
    their inputs recursively). If there are multiple steps with the same
    argument names they are prefixed by their names or if those are not set
    then by their class names. To enable prefixing for all args set
    prefix=True.

    Sometimes the difference between pipelines is that a step exists or it
    doesn't. When diff=True and existence=True, instead of expanding all
    the kwargs for that step, we expand a single column whose name is the
    step name and whose value is a boolean indicating whether the step exists
    in the given tree.

    Args:
        prefix: whether to always use step name prefix for kwarg name.
            Default False, which uses prefixes when necessary, i.e. for
            keywords that are shared by multiple step names.
        index: If True expand args into index. Otherwise expand into
            columns
        diff: whether to only expand keywords whose values that are
            non-constant
        existence: whether to check for existence of a step in the tree
            instead of a full diff. Only applicable when diff=True. See
            note above.

    Returns: a DatFrame with the arguments of the steps expanded.
    """
    # collect kwargs resulting in a list of {name: kwargs} dicts
    dicts = [step._collect_kwargs(s, drop_duplicate_names=True) for s in self.index]
    # if any of the kwargs are themselves dicts, expand them
    dicts = [{k: util.dict_expand(v) for k, v in s.items()} for s in dicts]

    if diff:
        diff_dicts = [{} for d in dicts]  # the desired list of dicts

        names = util.union([set(d.keys()) for d in dicts])  # all names among these steps
        for name in names:
            if existence:
                ndicts = [d[name] for d in dicts if name in d.keys()]  # all dicts for this name
            else:
                ndicts = [d[name] if name in d.keys() else {} for d in dicts]

            ndiffs = util.dict_diff(ndicts)  # diffs for this name

            if sum(map(len, ndiffs)) == 0:  # if they're all the same
                # but not all had the key and existence=True
                if existence and len(ndicts) < len(self):
                    for m, d in zip(diff_dicts, dicts):
                        m[name] = {tuple(): name in d.keys()}
            else:  # if there was a diff
                diff_iter = iter(ndiffs)
                for m, d in zip(diff_dicts, dicts):
                    if name in d.keys() or not existence:
                        m[name] = diff_iter.next()  # get the corresponding diff

        dicts = diff_dicts

    # restructure so name is in the key
    merged_dicts = []
    for dd in dicts:
        merged_dicts.append(util.dict_merge(*({tuple([name] + list(util.make_tuple(k))): v
                            for k, v in d.items()} for name, d in dd.items())))

    # prefix_keys are the keys that will keep their prefix
    keys = [list((k[1:] for k in d.keys())) for d in merged_dicts]
    if not prefix:
        key_count = [Counter(kk) for kk in keys]
        prefix_keys = util.union({k for k in c if c[k] > 1} for c in key_count)
    else:
        prefix_keys = util.union((set(kk) for kk in keys))

    merged_dicts = [{str.join('_', map(str, k if k[1:] in prefix_keys else k[1:])): v
                    for k, v in d.items()} for d in merged_dicts]

    expanded = pd.DataFrame(merged_dicts, index=self.index)

    if index:
        columns = list(expanded.columns)
        try:
            if len(columns) > 0:
                expanded.set_index(columns, inplace=True)
            else:
                expanded.index = [None]*len(expanded)
        except TypeError:
            _print_unhashable(expanded, columns)
            expanded.set_index(columns, inplace=True)

        df = self.__class__.__bases__[0](self, copy=True)
        df.index = expanded.index

    else:
        df = pd.concat((expanded, self), axis=1)
        # When index=False, the index is still a Step collection
        df = StepFrame(expanded)

    return df


def dapply(self, fn, pairwise=False, symmetric=True, diagonal=False, block=None, **kwargs):
    """
    Apply function to each step object in the index

    Args:
        fn: function to apply. If a list then each function is applied
        pairwise: whether to apply the function to pairs of steps
        symmetric, diagonal, block: passed to apply_pairwise when pairwise=True
        kwargs: a keyword arguments to pass to each function. Arguments
            with list value are grid searched using util.dict_product.

    Returns: a StepFrame or StepSeries
    """
    search_keys = [k for k, v in kwargs.items() if isinstance(v, list) and len(v) > 1]
    functions = util.make_list(fn)
    search = list(product(functions, util.dict_product(kwargs)))

    results = []
    for fn, kw in search:
        if not pairwise:
            r = self.index.to_series().apply(lambda step: fn(step, **kw))
        else:
            r = apply_pairwise(self, fn,
                               symmetric=symmetric, diagonal=diagonal, block=block,
                               **kw)

        name = [] if len(functions) == 1 else [fn.__name__]
        name += util.dict_subset(kw, search_keys).values()

        if isinstance(r, pd.DataFrame):
            columns = pd.MultiIndex.from_tuples(
                    [tuple(name + util.make_list(c)) for c in r.columns])
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


def apply_pairwise(self, function, symmetric=True, diagonal=False, block=None, **kwargs):
    """
    Helper function for pairwise apply.
    Args:
        steps: an ordered collection of steps
        function: function to apply, first two positional arguments are steps
        symmetric: whether function is symmetric in the two steps
        diagonal: whether to apply on the diagonal
        block: apply only when the given columns match
        kwargs: keyword arguments to pass to the function

    Returns:
        DataFrame with index and columns equal to the steps argument
    """
    steps = self.index
    r = pd.DataFrame(index=steps, columns=steps)
    for i, s1 in enumerate(steps):
        j = range(i+1 if symmetric else len(steps))
        if not diagonal:
            j.remove(i)
        other = set(steps[j])
        if block is not None:
            df = self.reset_index()
            df = df.merge(df, on=block)
            other &= set(df[df.index_x == s1].index_y)

        for s2 in other:
            r.ix[s1, s2] = function(s1, s2, **kwargs)
    return r


def _assert_step_collection(steps):
    for s in steps:
        if not isinstance(s, step.Step):
            raise ValueError("StepFrame index must consist of drain.step.Step objects")
    if len(set(steps)) != len(steps):
        raise ValueError("StepFrame steps must be unique")


class StepFrame(pd.DataFrame):
    expand = expand
    dapply = dapply

    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        _assert_step_collection(self.index.values)

    @property
    def _constructor(self):
        return StepFrame

    @property
    def _contructor_sliced(self):
        return pd.Series

    def __str__(self):
        return self.expand().__str__()

    def to_html(self, *args, **kwargs):
        return self.expand().to_html(*args, **kwargs)

    # resetting index makes it no longer a StepFrame
    def reset_index(self, *args, **kwargs):
        return pd.DataFrame(self).reset_index(*args, **kwargs)


class StepSeries(pd.Series):
    expand = expand
    dapply = dapply

    def __init__(self, *args, **kwargs):
        pd.Series.__init__(self, *args, **kwargs)
        _assert_step_collection(self.index.values)

    @property
    def _constructor(self):
        return StepSeries

    @property
    def _contructor_expanddim(self):
        return StepFrame

    def __str__(self):
        return self.expand().__str__()

    def to_html(self, *args, **kwargs):
        return self.expand().to_html(*args, **kwargs)

    def reset_index(self, *args, **kwargs):
        return pd.Series(self).reset_index(*args, **kwargs)


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
                df[c] = df[c].dropna().apply(pformat).ix[df.index]

    return df


def show_tree(tree, feature_names, max_depth=None):
    import wand.image

    filename = NamedTemporaryFile(delete=False).name
    export_tree(tree, filename, [c.encode('ascii') for c in feature_names], max_depth)
    img = wand.image.Image(filename=filename)
    return img


def export_tree(clf, filename, feature_names=None, max_depth=None):
    import pydot

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_names, max_depth=max_depth)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)
