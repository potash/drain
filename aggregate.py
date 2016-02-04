import os
import re
import logging

import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from itertools import product,chain

import pandas as pd
import numpy as np

import util, data
from util import merge_dicts, hash_obj

# the series can be:
#    - a callable (taking the DataFrame), e.g. lambda df: df.column**2
#    - one of the columns in the frame, e.g. 'column'
#    - a non-string value, e.g. 1
def get_series(series, df):
    if hasattr(series, '__call__'):
        return series(df)
    elif series in df.columns:
        return df[series]
    elif not isinstance(series, basestring):
        return pd.Series(series, index=df.index)
    else:
        raise ValueError('Invalid series: %s' % series)
 
# AggregateSeries consist of a series and a function
class AggregateSeries(object):
    def __init__(self, series, function):
        self.series = series
        self.function = function
    
    def apply_series(self, df):
        return get_series(self.series, df)
    
    def __hash__(self):
        s = hash_obj(self.series)
        f = hash_obj(self.function)
        
        return hash((s,f))
       
class AggregateBase(object):
    def __init__(self, columns, aggregate_series):
        self.columns = columns
        self.aggregate_series = aggregate_series
      
    # default is that series and columns are one-to-one
    def apply(self, series):
        for s,name in zip(series, self.columns):
            s.name = name
        return series

    def __div__(self, other):
        f = Fraction(numerator=self, denominator=other)
        return f

class Fraction(AggregateBase):
    def __init__(self, numerator, denominator, name='{numerator}_per_{denominator}', include_numerator=False, include_denominator=False, include_fraction=True):
        self.numerator = numerator
        self.denominator = denominator

        self.include_numerator=include_numerator
        self.include_denominator=include_denominator
        self.include_fraction=include_fraction

        columns = []
        if include_fraction:
            columns.extend([name.format(numerator=n, denominator=d)
                for n,d in product(numerator.columns, denominator.columns)])
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

        for s,name in zip(columns, self.columns):
            s.name = name

        return columns

# functions can be a single function or iterable
# default name is str(series), default function name is str(function)
# column names are {name}_{function_name}
# unless single function and function_names=False, then column name is just name
# TODO: allow list of series, take product(series, functions)
class Aggregate(AggregateBase):
    def __init__(self, series, functions, name=None, function_names=None):
        if not hasattr(functions, '__iter__'):
            functions = [functions]
            if function_names is not None and function_names is not False:
                function_names = [function_names]

        if name is None: name = series

        if function_names is False:
            if len(functions) > 1:
                raise ValueError('Must use function names for multiple functions')
            columns = [name]
        else:
            if function_names is None: function_names = functions
            columns = ['%s_%s' % (name,f) for f in function_names]

        AggregateBase.__init__(self, columns,
                    [AggregateSeries(series, f) for f in functions])

# turn a given series (as accepted by get_series above) into a float for summing
# this can be up to an order of magnitude faster than summing int or float directly
def float_sum(series, name=None):
    if series is None:
        if name is None:
            name = 'count'
        return Aggregate(1.0, 'sum', name=name, function_names=False)
    elif name is None:
        name = series

    return Aggregate(lambda d: get_series(series, d).astype(np.float32), 'sum', name=name, function_names=False)

class Count(Fraction):
    def __init__(self, series=None, name=None, parent=None, parent_name=None, prop=False):
        if not prop:
            Fraction.__init__(self, numerator=float_sum(series, name), 
                    denominator=None, include_fraction=False, 
                    include_numerator=True)
        else:
            Fraction.__init__(self, numerator=float_sum(series, name), 
                    denominator=float_sum(parent, parent_name), 
                    include_numerator=True, name='{numerator}_prop')

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
        series = {} # unique
        aggregate_series = {} # references to those series (one for each (series, function)
        aggregate_functions = {} # corresponding functions
        series_refs = [] # ordered references for feeding results back to Aggregate.apply() later
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
        series = (a.apply([aggregated[i].copy() for i in r]) 
               for a,r in zip(self.aggregates, self.series_refs))
        # TODO: automatically apply aggregates.columns as names to these series here
        return pd.concat(chain(*series), axis=1)

def aggregate_list(l):
    return list(np.concatenate(l.values))

def aggregate_set(l):
    return set(np.concatenate(l.values))

def aggregate_counts(l):
    return np.unique(np.concatenate(l.values), return_counts=True)
