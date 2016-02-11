import os
import logging
from itertools import product,chain

import pandas as pd
import numpy as np

import util, data
from util import merge_dicts, hash_obj

# the series can be:
#    - a callable (taking the DataFrame), e.g. lambda df: df.column**2
#    - one of the columns in the frame, e.g. 'column'
#    - a non-string value, e.g. 1
def get_series(series, df, astype=None):
    if hasattr(series, '__call__'):
        r = series(df)
    elif series in df.columns:
        r = df[series]
    elif not isinstance(series, basestring):
        r = pd.Series(series, index=df.index)
    else:
        raise ValueError('Invalid series: %s' % series)

    if astype is not None:
        r = r.astype(astype)
    return r
 
# AggregateSeries consist of a series and a function
class AggregateSeries(object):
    def __init__(self, series, function, astype=None):
        self.series = series
        self.function = function
        self.astype = astype
    
    def apply_series(self, df):
        return get_series(self.series, df, self.astype)
    
    def __hash__(self):
        s = hash_obj(self.series)
        f = hash_obj(self.function)
        
        return hash((s,f, self.astype))
       
class AggregateBase(object):
    def __init__(self, columns, aggregate_series):
        self.columns = columns
        self.aggregate_series = aggregate_series
      
    # default is that series and columns are one-to-one
    def apply(self, series):
        return series

    def __div__(self, other):
        f = Fraction(numerator=self, denominator=other)
        return f

class Fraction(AggregateBase):
    def __init__(self, numerator, denominator, name='{numerator}_per_{denominator}', 
            include_numerator=False, include_denominator=False, include_fraction=True):
        self.numerator = numerator
        self.denominator = denominator

        self.include_numerator=include_numerator
        self.include_denominator=include_denominator
        self.include_fraction=include_fraction

        columns = []
        if include_fraction:
            if hasattr(name, '__iter__') and len(name) == \
                    len(numerator.columns)*len(denominator.columns):
                columns.extend(name)
            elif isinstance(name, basestring):
                columns.extend([name.format(numerator=n, denominator=d)
                    for n,d in product(numerator.columns, denominator.columns)])
            else:
                raise ValueError('Name must either be a list of names of the correct length or a format string')
        if include_numerator:
            columns.extend(numerator.columns)
        if include_denominator:
            columns.extend(denominator.columns)

        aggregate_series = []
        if include_fraction:
            aggregate_series += numerator.aggregate_series + denominator.aggregate_series
        else:
            if include_numerator:
                aggregate_series += numerator.aggregate_series
            if include_denominator:
                aggregate_series += denominator.aggregate_series

        AggregateBase.__init__(self, columns=columns, aggregate_series=aggregate_series)

    def apply(self, series):
        if self.include_fraction or (self.include_numerator and self.include_denominator):
            ncolumns = self.numerator.apply(series[:len(self.numerator.aggregate_series)])
            dcolumns = self.denominator.apply(series[len(self.numerator.aggregate_series):])
        elif self.include_numerator:
            ncolumns = self.numerator.apply(series)
        elif self.include_denominator:
            dcolumns = self.denominator.apply(series)

        columns = []
        if self.include_fraction:
            columns = [n/d for n,d in product(ncolumns, dcolumns)]
        if self.include_numerator:
            columns.extend(ncolumns)
        if self.include_denominator:
            columns.extend(dcolumns)

        return columns

class Aggregate(AggregateBase):
    def __init__(self, series, f, name=None, fname=None):
        """
        series can be single or iterable
        functions can be a single function or iterable
        total aggregates are the products of series and functions
        default name is str(series), default function name is str(f)
        column names are {name}_{fname}
        unless single function and function_names=False, then column name is just name
        """
        series = util.make_list(series)
        f = util.make_list(f)

        name = series if name is None else util.make_list(name)
        fname = f if fname is None else util.make_list(fname)

        if not (len(series) == len(name)):
            raise ValueError('series and name must have same length or name must be None')
        if not (len(series) == len(name)):
            raise ValueError('f and fname must have same length or fname must be None')

        if fname == [False]:
            if len(f) > 1:
                raise ValueError('Must use function names for multiple functions')
            columns = name
        else:
            columns = ['%s_%s' % n for n in product(name, fname)]

        AggregateBase.__init__(self, columns,
                [AggregateSeries(*sf) for sf in product(series, f)])

class Count(Fraction):
    """
    Aggregate a given series (as accepted by get_series above) by summing
    By default that series is 1, resulting in a count (hence the name).
    Also useful when series is a boolean.
    If prop=True then also divide that series by a specified parent series, also summed.
    """
    def __init__(self, series=None, name=None, parent=None, parent_name=None, prop=False, prop_only=False):
        fname = 'count'
        if series is None:
            series = 1
            if name is None:
                name = 'count'
                fname = False # don't use fname when series is 'count'

        if parent is not None or prop_only:
            prop=True

        numerator = Aggregate(series, 'sum', name, fname=fname)
        for aseries in numerator.aggregate_series:
            aseries.astype = np.float32
        if not prop:
            Fraction.__init__(self, numerator=numerator, 
                    denominator=None, include_fraction=False, 
                    include_numerator=True)
        else:
            if parent is None:
                parent = 1

            denominator = Aggregate(parent, 'sum', parent_name)
            Fraction.__init__(self, numerator=numerator, 
                    denominator=denominator, 
                    include_numerator=not prop_only, 
                    # fraction names are numerator names with
                    # 'count' replaced by 'prop'
                    name=[n[:-5]+'prop' for n in numerator.columns])

class Proportion(Count):
    def __init__(self, series=None, name=None, parent=None, parent_name=None):
        Count.__init__(self, series=series, name=name, parent=parent, parent_name=parent_name, prop_only=True)

def _collect_columns(aggregates):
    columns = set()
    for a in aggregates:
        intersection = columns.intersection(a.columns)
        if len(intersection) > 0: raise ValueError('Repeated columns: %s' % intersection)
        columns.update(a.columns)

    if len(columns) == 0:
        raise ValueError('Aggregator needs at least one output columns')

    return columns

class Aggregator(object):
    def __init__(self, df, aggregates):
        self.df = df
        self.aggregates = aggregates

        self.columns = _collect_columns(aggregates)
        
        # collect all the series
        series = {} # unique series
        aggregate_series = {} # references to those series, one for each (series, function)
        series_refs = [] # series_refs[i] contains references to aggregate_series for aggregates[i]
        for a in aggregates:
            sr = []
            for aseries in a.aggregate_series:
                h = hash_obj(aseries.series)
                i = hash(aseries)
                if h not in series: # only apply each series once
                    series[h] = aseries.apply_series(df)
                    aggregate_series[h] = {i: aseries.function}
                if i not in aggregate_series[h]: # collect each aseries only once
                    aggregate_series[h][i] = aseries.function

                sr.append(i)
            series_refs.append(sr)
        
        self.adf = pd.DataFrame(series)
        self.aggregate_series = aggregate_series
        self.series_refs = series_refs
        
    # all the aggregates for a single series
    # returns a hash(aseries): series dict
    def get_series_aggregates(self, groupby, h):
        a = groupby[h].agg(self.aggregate_series[h])
        return {i:a[i] for i in a.columns}
    
    def aggregate(self, index):
        if isinstance(index, basestring):
            groupby = self.adf.groupby(self.df[index])
        else:
            # groupby does not accept a dataframe so we have to set the index then put it back
            self.adf.index = pd.MultiIndex.from_arrays([self.df[i] for i in index])
            groupby = self.adf.groupby(level=index)
            self.adf.index = self.df.index
        
        aggregated = merge_dicts(*[self.get_series_aggregates(groupby, h) for h in self.aggregate_series])

        all_series = []
        for a,r in zip(self.aggregates, self.series_refs):
            series = a.apply([aggregated[i].copy() for i in r])
            # rename series according to aggregate.columns
            for s,c in zip(series, a.columns): s.name = c
            all_series.extend(series)

        return pd.concat(all_series, axis=1)

def aggregate_list(l):
    return list(np.concatenate(l.values))

def aggregate_set(l):
    return set(np.concatenate(l.values))

def aggregate_counts(l):
    return np.unique(l.values, return_counts=True)

def concatenate_aggregate_counts(l):
    return np.unique(np.concatenate(l.values), return_counts=True)
