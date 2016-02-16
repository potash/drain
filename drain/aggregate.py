from itertools import product, chain
from collections import defaultdict

import pandas as pd
import numpy as np
import sys
import dis
import StringIO
import inspect
import types

import util, data
from util import merge_dicts, hash_obj

'''
transform first (per raw column), then aggregate (per transformed column), then do column operations (which now have shared index)

allow syntax like ([lambda x: x.gpa**2, 'suspensions'], ['sum','avg']), and like
Aggregator(df, [Count(), Count('Arrest')], 'district')

make sure all transformations and aggregations are only done once
'''

def capture_print(f, *args):
    '''helper to capture stdout'''
    stdout_ = sys.stdout
    stream = StringIO.StringIO()
    sys.stdout = stream
    f(*args)
    sys.stdout = stdout_ 
    return stream.getvalue()

class FuncHash:
    '''utility class that wraps functions; makes lambdas hashable via bytecode'''
    def __init__(self, func):
        self.func = func
    def __hash__(self):
        # anonymous functions can only be cashed based on their bytecode
        if isinstance(self.func, types.LambdaType):
            return hash(capture_print(dis.dis, self.func))
        return hash(self.func)
    def __eq__(self, other):
        return hash(self) == hash(other)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def __repr__(self):
        return self.func.__repr__()

'''
class Column:
    __init__( name or lambda or function, or scalar)
    pd.Series <- .apply(df)
    __hash__: gets hashed based on input argument (so lambda function based on bytecode)
'''
class Column:

    def __init__(self, definition, astype=None):
        if hasattr(definition, '__call__'):
            self.definition = FuncHash(definition)
        else:
            self.definition = definition
        self.astype = astype

    def apply(self, df):
        if hasattr(self.definition, '__call__'):
            r =  self.definition(df)
        elif self.definition in df.columns:
            r = df[self.definition]
        elif not isinstance(self.definition, basestring):
            r = pd.Series(self.definition, index=df.index)
        else:
            raise ValueError("Invalid column definition: %s"%str(self.definition))
        return r.astype(self.astype) if self.astype else r

    def __hash__(self):
        return hash((self.definition, self.astype))
    def __eq__(self, other):
        return hash(self) == hash(other)
        
class ColumnReduction:
    '''
        __init__(column, and a reduction (i.e., a row-aggregating function); maybe also takes an astype
        __hash__: gets hashed by aggregate row reduction (or lambda bytecode) and Column
    '''
    # TODO: add astype
    def __init__(self, column, agg_func):
        if not isinstance(column, Column):
            raise ValueError("ColumnReduction needs a Column")
        self.column = column
        self.agg_func = FuncHash(agg_func) if hasattr(agg_func, '__call__') else agg_func # necessary because we sometimes pass strings for Pandas' agg()
    def __hash__(self):
        return hash((self.column, self.agg_func))
    def __eq__(self, other):
        return hash(self) == hash(other)
        
class ColumnFunction:
    '''
    __init__(list of column reductions, list of names)
    - the second list names the outputs that apply() produces
    apply_and_name() -  asks aggregator to provide the reduced columns from the list of column reductions; they need to be indexed the same
                        calls self.apply(), names the pd.Series with its list of names, return the list of named pd.Series
    apply(aggregator) - abstract, needs to be implemented by all children
    - performs some arithmetic on these pd.Series
    - returns a list of pd.Series
    '''

    def __init__(self, column_reductions, names):

        for cr in column_reductions:
            if not isinstance(cr, ColumnReduction):
                raise ValueError("ColumnFunction requires a list of ColumnReductions; %s is not one."%repr(cr))

        self.column_reductions = column_reductions
        self.names = names

    def apply_and_name(self, aggregator):
        reduced_df = self._apply(aggregator)
        if len(self.names) != len(reduced_df.columns):
            raise IndexError("The ColumnFunction creates more dataframe columns than it has names for them!")
        reduced_df.columns = self.names
        return reduced_df
    
    def _apply(self, aggregator):
        raise NotImplementedError

    def __div__(self, other):
        return Fraction(numerator=self, denominator=other)

class ColumnIdentity(ColumnFunction):
    def __init__(self, column_reductions, names):
        ColumnFunction.__init__(self, column_reductions, names)
    def _apply(self, aggregator):
        return aggregator.get_reduced(self.column_reductions)

class Aggregate(ColumnIdentity):

    # TODO: add astype
    def __init__(self, column_def, agg_func, name=None, fname=None):

        # make list of column reductions from arguments  --> each ColumnReduction  needs a Column(definition) and an agg_func
        column_def = util.make_list(column_def)
        agg_func = util.make_list(agg_func)
        column_reductions = [ColumnReduction(Column(defn), func) for defn, func in product(column_def, agg_func)]

        # make list of names from the arguments --> same length as the list of column reductions above!
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
            column_names = ['%s_%s'%(cn, fn) for cn, fn in product(name, fname)]

        ColumnFunction.__init__(self, column_reductions, column_names)

class Aggregator:
    def __init__(self, df, column_functions):
        self.df = df
        self.column_functions = column_functions

        # unique column reductions from all the column functions
        self.column_reductions = set([cr for cf in column_functions for cr in cf.column_reductions])

        # unique columns across all the column reductions
        self.columns = set([c.column for c in self.column_reductions])

        # dataframe of the unique, populated columns, with the column  objects as the dataframe's column names
        self.col_df = pd.DataFrame({col: col.apply(df) for col in self.columns})

    def get_reduced(self, column_reductions):
        for cr in column_reductions:
            if not cr in self.column_reductions:
                raise ValueError("Column reduction %r is not known to this Aggregator!"%cr)
        return self.reduced_df[column_reductions]

    def aggregate(self, index):

        # create a df that is aggregated by index, 
        # and that contains as columns all the unique 
        # column reductions (columns, aggregated by some function)
        # as requested
        if isinstance(index, basestring):
            col_df_grouped = self.col_df.groupby(self.df[index])
        else:
            self.col_df.index = pd.MultiIndex.from_arrays([self.df[i] for i in index])
            col_df_grouped = self.col_df.groupby(level=index)
            self.col_df.index = self.df.index

        self.reduced_df = pd.DataFrame({
            colred: col_df_grouped[colred.column].agg(colred.agg_func)
            for colred in self.column_reductions
            })

        reduced_dfs = []
        for cf in self.column_functions:
            # each apply_and_name() calls get_reduced() with the column reductions it wants
            reduced_dfs.append(cf.apply_and_name(self))

        return pd.concat(reduced_dfs, axis=1)

#         return pd.DataFrame({s.name: s for s in reduced_series})

class Fraction(ColumnFunction):

    ''' numerator and denominator are ColumnFunctions '''

    def __init__(self, numerator, denominator, name='{numerator}_per_{denominator}',
            include_numerator=False, include_denominator=False, include_fraction=True):
        self.numerator = numerator
        self.denominator = denominator
        self.include_numerator=include_numerator
        self.include_denominator=include_denominator
        self.include_fraction=include_fraction

        names = []
        if include_fraction:
            if hasattr(name, '__iter__') and len(name) == \
                    len(numerator.names)*len(denominator.names):
                names.extend(name)
            elif isinstance(name, basestring):
                names.extend([name.format(numerator=n, denominator=d)
                    for n,d in product(numerator.names, denominator.names)])
            else:
                raise ValueError('Name must either be a list of names of the correct length or a format string')
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
        # the incoming dataframe will have the reduced columns in order as in column_reductions above

        reduced_dfs = []
        if self.include_fraction:
            n_df = self.numerator.apply_and_name(aggregator)
            d_df = self.denominator.apply_and_name(aggregator)
            reduced_dfs.extend( [n_df[cn]/d_df[cd] for cn,cd in product(
                                            n_df.columns, d_df.columns)] )

        if self.include_numerator:
            reduced_dfs.append(self.numerator.apply_and_name(aggregator))

        if self.include_denominator:
            reduced_dfs.append(self.denominator.apply_and_name(aggregator))

        return pd.concat(reduced_dfs,axis=1)

class Count(Fraction):
    '''
    Should be callable like:
    Count('Arrests', prop=1) - then also produces columns with the proportion
    Count(['Arrests','Stops'], prop=lambda x: x.Arrests.notnull()) - calculates the count and the proportion out of the outof series;
    Count(['Arrests','Stops'], prop=lambda x: x.Arrests.notnull(), prop_only) - as above, but omits the vanilla count
    '''
    def __init__(self, definition=None, name=None, prop=None, prop_only=False, prop_name=None):

        if prop==None and prop_only==True:
            raise ValueError("Cannot calculate only the proportion when no proportion requested!")

        definition = 1 if definition is None else definition
        prop = 1 if prop==True else prop

        # if we do a vanilla count, just call it 'count'
        if definition==1 and name is None:
            name = 'count'
            fname = False
        else:
            fname = 'count'
        
        denominator = Aggregate(prop, 'sum', prop_name) if prop else None
        numerator = Aggregate(definition, 'sum', name, fname)
        if prop not in [None, 1]:
            fracnames = [n[:-5]+'prop_'+d[:-4] for n,d in zip(numerator.names, denominator.names)]
        else:
            fracnames = [n[:-5]+'prop' for n in numerator.names]
        Fraction.__init__(self, numerator=numerator, denominator=denominator, include_numerator=not prop_only,
                          include_denominator=False, include_fraction=prop is not None,
                          name=fracnames)

class Proportion(Count):
    def __init__(self, definition, denom_def, name=None, denom_name=None):
        Count.__init__(self, definition=definition, name=name, prop=denom_def, prop_only=True, prop_name=denom_name)

def _collect_columns(aggregates):
    columns = set()
    for a in aggregates:
        intersection = columns.intersection(a.columns)
        if len(intersection) > 0: raise ValueError('Repeated columns: %s' % intersection)
        columns.update(a.columns)

    if len(columns) == 0:
        raise ValueError('Aggregator needs at least one output columns')

    return columns

def aggregate_list(l):
    return list(np.concatenate(l.values))

def aggregate_set(l):
    return set(np.concatenate(l.values))

def aggregate_counts(l):
    lists = [list(i) for i in l.values if len(i) > 0]
    if len(lists) == 0:
        return None
    else:
        l = np.concatenate(lists)
        return np.unique(l, return_counts=True)

def days(date1, date2):
    """
    returns a lambda that determines the number of days between the two dates
    the dates can be strings (column names) or actual dates
    e.g. Aggregate(days('date', today), ['min','max'], 'days_since')
    TODO: should there be a Days(AggregateBase) for this? handle naming
    """
    if isinstance(date1, basestring) and isinstance(date2, basestring):
        return lambda d: (d[date2] - d[date1])/util.day
    elif isinstance(date1, basestring):
        return lambda d: (date2 - d[date1])/util.day
    elif isinstance(date2, basestring):
        return lambda d: (d[date2] - date1)/util.day
    else:
        return (date2 - date1)/util.day

def date_min(d):
    """
    groupby()['date_colum'].aggregate('min') returns a float?
    convert it back to a timestamp
    """
    return pd.to_datetime(d.min())

def date_max(d):
    return pd.to_datetime(d.max())
