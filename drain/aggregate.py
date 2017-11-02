"""Classes that facilitate transformation and aggregation of dataframes.

This module provides wrappers to simplify grouping and aggregation
of dataframes by index, and to consequently perform arithmetics on
the aggregated dataframes.

Examples:

    The following code would take a Pandas dataframe ``df``, and produce
    a dataframe ``res_df`` that is indexed by ``name``. The result would have
    a column ``count`` that gives the number of rows per ``name``; a column
    that gives the sum of ``score`` per ``name``, divided by the sum of
    ``arrests`` per ``name``; and finally two columns that give the sum
    of ``score`` per ``name``, and the sum of squared ``score`` per ``name``::

        aggregates = [Count(),
                      Proportion('score','arrests'),
                      Aggregate(['score', lambda x: x.score**2],'sum')
                      ]
        res_df = Aggregator(df, aggregates).aggregate('name')

    Note how we specify all definitions before performing the actual
    aggregation, and how ``Aggregator.aggregate()`` then takes an
    ``index`` to group by.

Aggregator also caches individual transformations of columns, as to
reduce redundant calculations.

Classes that endusers interface with are Aggregate, Fraction,
Count, and Proportion (all of which specify outcome columns and row-wise
aggregation functions), and Aggregator (which takes an input dataframe and an
index by which rows are being grouped).

"""

from itertools import product
from six import string_types

import pandas as pd
import numpy as np

from . import util
from .data import Column


class ColumnReduction(object):
    """Wraps and hashes a `Column` together with a function that aggregates across rows.
    """

    def __init__(self, column, agg_func):
        """Args:
            column (Column): The column that will be aggregated across rows.
            agg_func (Function): The function that will be used to aggregate rows,
                                 once the Column has been grouped by some index.
        """
        if not isinstance(column, Column):
            raise ValueError("ColumnReduction needs a Column")
        self.column = column
        self.agg_func = agg_func

        # use float32 by default for string agg functions except 'nunique',
        # e.g. 'min', 'median', 'max', etc.
        if isinstance(agg_func, string_types) and\
                agg_func != 'nunique' and\
                self.column.astype is None:
            self.column.astype = np.float32

    def __hash__(self):
        return hash((self.column, self.agg_func))

    def __eq__(self, other):
        return hash(self) == hash(other)


class ColumnFunction(object):
    """Abstract base class for functions on reduced Columns; names the outcomes.

    Having obtained Columns that have been created and aggregated along rows, with
    the same index - ColumnReductions, in other words - we might want to perform
    arithmetics on these ColumnReductions, such as element-wise division of one by the other.
    These transformations are handled by ColumnFunctions.

    Note: ColumnFunction guarantees an `apply_and_name(aggregator)`. This is the
        lowest-level function in this module that actually returns 'populated'
        DataFrames (as provided by the aggregator). Children of this class
        thus include functions on pairs of popoulated dataframes, such as
        division and addition.
    """

    def __init__(self, column_reductions, names):
        """Args:
            column_reductions (list[ColumnReduction]): List of input ColumnReductions.
            names (list[str]): A list of strings. Its length must be equal to the number
                of pd.Series' that `apply_and_name()` produces.
        """

        for cr in column_reductions:
            if not isinstance(cr, ColumnReduction):
                raise ValueError(
                        "ColumnFunction requires a list of ColumnReductions; %s is not one."
                        % repr(cr))

        self.column_reductions = column_reductions
        self.names = names

    def apply_and_name(self, aggregator):
        """Fetches the row-aggregated input columns for this ColumnFunction.

        Args:
            aggregator (Aggregator)

        Returns:
            pd.DataFrame: The dataframe has columns with names self.names
                that were created by this ColumnFunction,
                and is indexed by the index that was passed to
                aggregator.aggregate(index).
        """
        reduced_df = self._apply(aggregator)
        if len(self.names) != len(reduced_df.columns):
            raise IndexError("ColumnFunction creates more columns than it has names for.")
        reduced_df.columns = self.names
        return reduced_df

    def _apply(self, aggregator):
        """Abstract function for the actual transformation.

        Note:
            Don't call this; it gets called by `self.apply_and_name()`.

        Args:
            aggregator(Aggregator)

        """
        raise NotImplementedError

    def __div__(self, other):
        return Fraction(numerator=self, denominator=other)


class ColumnIdentity(ColumnFunction):
    """The simplest non-abstract ColumnFunction.
    """
    def __init__(self, column_reductions, names):
        ColumnFunction.__init__(self, column_reductions, names)

    def _apply(self, aggregator):
        """
        Returns:
            pd.DataFrame: A dataframe with self.column_reductions
                as columns, where the column names are self.names.
        """
        return aggregator.get_reduced(self.column_reductions)


class Aggregate(ColumnIdentity):
    """A highly convenient wrapper around ColumnReductions.

    Example::
        Aggregate(['arrests','income','age'], ['min','max','mean'])

    This would create 3x3 columns: arrests_min, arrests_max, arrests_mean, and so on.
    """

    def __init__(self, column_def, agg_func, name=None, fname=None, astype=None):
        """
        Args:
            column_def: List of (or a single) column definitions, as accepted by Column.
            agg_func: List of (or a single) row-wise aggregation function (will be passed to
                Panda's groupby.agg())
            name: List of strings of names for the column_defintions. Must be of
                same length as column_def, or be None, in which case the names
                default to the string representation of the each column definition.
            fname: List of strings of names for the aggregation functions. Must be of
                same length as agg_func or be None, in which case the names
                default to the string representation of the each aggregation function.
            astype: List of pandas dtypes, which gets passed to Column together with the
                column definitions. Must be of same length as column_def or be None,
                in which case no casting will be performed.

        Names for the resulting column reductions are of the format `columnname_functionname`.
        """

        # make list of column reductions from arguments
        # each ColumnReduction  needs a Column(definition) and an agg_func
        column_def = util.make_list(column_def)
        agg_func = util.make_list(agg_func)
        astype = util.make_list(astype)

        if len(astype) == 1:
            astype = astype*len(column_def)

        if len(astype) != len(column_def):
            raise ValueError("astype must be a datatype, or a list of datatypes")

        column_reductions = [ColumnReduction(Column(defn, at), func)
                             for (defn, at), func in product(zip(column_def, astype), agg_func)]

        # make list of names from the arguments
        # same length as the list of column reductions above!
        name = [str(c) for c in column_def] if name is None else util.make_list(name)
        if len(name) != len(column_def):
            raise ValueError("name and column_def must be same length, or name must be None.")

        fname = [str(a) for a in agg_func] if fname is None else util.make_list(fname)
        if len(fname) != len(agg_func):
            raise ValueError("fname and agg_func must be same length, or fname must be None.")

        # if there's only on agg_func, you can override the function-naming-schema,
        # by passing fname=False; then we only use the name
        if len(agg_func) == 1 and fname == [False]:
            column_names = name
        else:
            column_names = ['%s_%s' % (cn, fn) for cn, fn in product(name, fname)]

        ColumnFunction.__init__(self, column_reductions, column_names)


class Aggregator(object):
    """Binds column functions to a dataframe and allows for aggregation by a given index.
    """

    def __init__(self, df, column_functions):
        """
        Args:
            df (pd.DataFrame): A dataframe to apply column functions to, and
                which will be aggregated.
            column_functions (list[ColumnFunction]): ColumnFunctions that will
                be applied to the dataframe.
        """
        self.df = df
        self.column_functions = column_functions

        # unique column reductions from all the column functions
        self.column_reductions = set([cr for cf in column_functions
                                      for cr in cf.column_reductions])

        # unique columns across all the column reductions
        self.columns = set([c.column for c in self.column_reductions])

        # dataframe of the unique, populated columns
        # with the column objects as the dataframe's column names
        self.col_df = pd.DataFrame({col: col.apply(df) for col in self.columns})

    def get_reduced(self, column_reductions):
        """This function gets called by ColumnFunction._apply(). After a ColumnFunction
        has been passed to Aggregator's constructor, the ColumnFunction can use this function
        to request the populated, aggregated columns that correspond to its ColumnReductions.

        Args:
            column_reduction (list[ColumnReduction])

        Returns:
            pd.DataFrame: A dataframe, where the column names are ColumnReductions.
        """
        for cr in column_reductions:
            if cr not in self.column_reductions:
                raise ValueError("Column reduction %r is not known to this Aggregator!" % cr)
        return self.reduced_df[column_reductions]

    def aggregate(self, index):
        """Performs a groupby of the unique Columns by index, as constructed from self.df.

        Args:
            index (str, or pd.Index): Index or column name of self.df.

        Returns:
            pd.DataFrame: A dataframe, aggregated by index, that contains the result
                of the various ColumnFunctions, and named accordingly.
        """

        # deal with index as a string vs index as a index/MultiIndex
        if isinstance(index, string_types):
            col_df_grouped = self.col_df.groupby(self.df[index])
        else:
            self.col_df.index = pd.MultiIndex.from_arrays([self.df[i] for i in index])
            col_df_grouped = self.col_df.groupby(level=index)
            self.col_df.index = self.df.index

        # perform the actual aggregation
        self.reduced_df = pd.DataFrame({
            colred: col_df_grouped[colred.column].agg(colred.agg_func)
            for colred in self.column_reductions
            })

        # then apply the functions to produce the final dataframe
        reduced_dfs = []
        for cf in self.column_functions:
            # each apply_and_name() calls get_reduced() with the column reductions it wants
            reduced_dfs.append(cf.apply_and_name(self))

        return pd.concat(reduced_dfs, axis=1)


class Fraction(ColumnFunction):
    """Divides all pairs of column reductions from two column functions.

    Example::
        Fraction(Aggregate(['arrests','score'], 'sum'), Aggregate('score','mean'))
    This would create two columns: `arrests_sum_per_score_mean`, and
    `score_sum_per_score_mean`.
    """

    def __init__(self, numerator, denominator, name='{numerator}_per_{denominator}',
                 include_numerator=False, include_denominator=False, include_fraction=True):
        """
        Args:
            numerator (ColumnFunction)
            denominator (ColumnFunction)
            name (list[str] or str): A list of strings of length
                denominator.names*numerator.names. Output columns will be named by
                this. Defaults to `numerator.name_per_denominator.name`.
            include_numerator (bool): If the unmodified numerator will be part of the output
            include_denominator (bool): If the unmodified denominator will be part of the output
            include_fraction (bool): If the division of all pairs of columns (pd.Series) from
                the numerator and denominator will be included in the output.
        """
        self.numerator = numerator
        self.denominator = denominator
        self.include_numerator = include_numerator
        self.include_denominator = include_denominator
        self.include_fraction = include_fraction

        names = []
        if include_fraction:
            if hasattr(name, '__iter__') and len(name) == \
                    len(numerator.names)*len(denominator.names):
                names.extend(name)
            elif isinstance(name, string_types):
                names.extend([name.format(numerator=n, denominator=d)
                              for n, d in product(numerator.names, denominator.names)])
            else:
                raise ValueError('Name must either be a list of names or a format string')
        if include_numerator:
            names.extend(numerator.names)
        if include_denominator:
            names.extend(denominator.names)

        column_reductions = []
        # numerator first, then denominator
        if include_fraction:
            column_reductions += numerator.column_reductions + denominator.column_reductions
        else:
            if include_numerator:
                column_reductions += numerator.column_reductions
            if include_denominator:
                column_reductions += denominator.column_reductions

        ColumnFunction.__init__(self, column_reductions=column_reductions, names=names)

    def _apply(self, aggregator):
        """Returns a dataframe with the requested ColumnReductions.
        """

        reduced_dfs = []
        if self.include_fraction:
            n_df = self.numerator.apply_and_name(aggregator)
            d_df = self.denominator.apply_and_name(aggregator)
            reduced_dfs.extend([n_df[cn]/d_df[cd]
                                for cn, cd in product(n_df.columns, d_df.columns)])

        if self.include_numerator:
            reduced_dfs.append(self.numerator.apply_and_name(aggregator))

        if self.include_denominator:
            reduced_dfs.append(self.denominator.apply_and_name(aggregator))

        return pd.concat(reduced_dfs, axis=1)


class Count(Fraction):
    """Define various counts, sums, and proportions of Columns.

    Examples::
        Count() # count the number of rows per grouped index
        Count('Arrests') # sum column 'Arrests' by grouped index
        Count('Arrests',prop=True) # as above, but also add a column that is normalized,
            i.e. where each group's sum is divided by the size of that group
        Count('Arrests',prop=True, prop_only=True) # as above, but exclude the raw sum
        Count('Arrests', prop=lambda x: x.score**2) # create a column with the sum of `Arrests`
            per grouped index, and also create a column with the sum of `Arrests`
            per grouped index divided by the sum of `score**2` per grouped index.
        Count(['Arrests','Stops']) # create both Count('Arrests') and Count('Stops')

    By default names are set similar to this example: 'count', 'Arrests_count', 'Arrests_prop',
    'Arrests_prop_score', etc.
    """

    def __init__(self, definition=None, name=None, prop=None, prop_only=False, prop_name=None,
                 astype=None, prop_astype=None):
        """
        Args:
            definition: List of (or single) column definition, as accepted by Column. `Count`
                will create ColumnFunctions corresponding to these definitions, using `sum`
                as the row-wise aggregation function.
            name: List of (or single) string for naming above definitions. If `None`, then the
                definitions' string representation will be used.
            prop: List of (or single) column definition, as accepted by Column. Defaults to
                None, in which case `Count` only creates the columns as defined in
                `definition`. If `prop=1` or `prop=True`, then `Count` also creates a column
                in which the sums from the column definitions are divided by the length of each
                group. `prop` can also be any column definition as accepted by `Column`. In that
                case, `Count` will divide the results of `definition` by the result `prop`.
            prop_name: List of strings (or single string) for naming the column definition from
                `prop`.  Defaults to `None`, in which case the string representation of `prop`
                is used.
            astype: Pandas dtypes, or list thereof. If list, needs to be of same length as
                `definition`. `astype` will be passed to `ColumnDefinition` along with
                `definition`.
            prop_astype: Like `astype`, but for the `prop` definition.

        If no name is given and `definition` is `None` or 1, then the resulting column will
        simply be called 'count'. Otherwise, columns are named similar to 'Arrests_count',
        'Arrests_prop', 'Arrests_prop_score', etc.

        """

        if prop is None and prop_only:
            raise ValueError("Cannot calculate only the proportion when no prop")

        definition = np.float32(1) if definition is None else definition
        prop = np.float32(1) if prop is True else prop

        # if we do a vanilla count, just call it 'count'
        if definition == 1 and name is None:
            name = 'count'
            fname = False
        else:
            fname = 'count'

        denominator = Aggregate(prop, 'sum', prop_name, astype=prop_astype) if prop else None
        numerator = Aggregate(definition, 'sum', name, fname, astype=astype)
        if prop not in [None, 1]:
            fracnames = [n[:-5]+'per_'+d[:-4]
                         for n, d in product(numerator.names, denominator.names)]
        else:
            fracnames = [n[:-5]+'prop' for n in numerator.names]
        Fraction.__init__(self, numerator=numerator, denominator=denominator,
                          include_numerator=not prop_only,
                          include_denominator=False, include_fraction=prop is not None,
                          name=fracnames)


class Proportion(Count):
    """Convenience wrapper for count.

    Example::
        Proportion('Arrests','Inspections')
    Creates a column 'Arrests_prop_Inspections', which divides the sum of 'Arrests' per group
    by the sum of 'Inspections' per group.
    """
    def __init__(self, definition, denom_def=True, name=None,
                 denom_name=None, astype=None, denom_astype=None):
        Count.__init__(self, definition=definition, name=name, prop=denom_def,
                       prop_only=True, prop_name=denom_name, astype=astype,
                       prop_astype=denom_astype)


def aggregate_list(l):
    return list(np.concatenate(l.values))


def aggregate_set(l):
    return set(np.concatenate(l.values))


def aggregate_counts(l):
    lists = [list(i) for i in l.values if len(i) > 0]
    if len(lists) == 0:
        return None
    else:
        ls = np.concatenate(lists)
        return np.unique(ls, return_counts=True)


def days(date1, date2):
    """
    returns a lambda that determines the number of days between the two dates
    the dates can be strings (column names) or actual dates
    e.g. Aggregate(days('date', today), ['min','max'], 'days_since')
    TODO: should there be a Days(AggregateBase) for this? handle naming
    """
    return lambda df, date1=date1, date2=date2: _days(df, date1, date2)


def _days(df, date1, date2):
    # when df is empty the subtraction below is broken
    if len(df) == 0:
        return pd.Series(dtype=np.float64, index=df.index)
    else:
        d1 = df[date1] if isinstance(date1, string_types) else date1
        d2 = df[date2] if isinstance(date2, string_types) else date2

        return (d2 - d1) / util.day


def date_min(d):
    """
    groupby()['date_colum'].aggregate('min') returns a float?
    convert it back to a timestamp
    """
    return pd.to_datetime(d.min())


def date_max(d):
    return pd.to_datetime(d.max())
