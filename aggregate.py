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

# with no params just counts
# Default name is column + _count
# When parent is specified count parent
# TODO: refactor this as a Fraction with include_numerator, include_denominator, include_fraction
# TODO: overload division of AggregateBase to return Fraction object
class Count(AggregateBase):
    def __init__(self, series=None, name=None, prop=False, parent=1.0, prop_only=True):
        # if parent is specified, assume we want proportion
        if parent != 1.0:
            prop = True
        self.prop = prop
        self.prop_only = prop_only

        if series is None:
            columns = [name if name is not None else 'count']
            aggregate_series = [AggregateSeries(1.0, 'sum')]
        else:
            if name is None: name = series
            columns = ['%s_count' % name]
            # converting to float32 before summing is an order of magnitude faster
            # if series is a function we need to compose it with the cast
            count_series = lambda d: get_series(series, d).astype(np.float32)
            aggregate_series = [AggregateSeries(count_series, 'sum')]
            
            if prop:
                columns.append('%s_prop' % name)
                count_series = lambda d: get_series(parent, d).astype(np.float32)
                aggregate_series.append(AggregateSeries(parent, 'sum'))

                if prop_only:
                    columns = [columns[1]]
            
        AggregateBase.__init__(self, columns, aggregate_series)
    
    def apply(self, series):
        count = series[0]
        count.name = self.columns[0]
        if self.prop:
            prop = (series[0] / series[1]).where(series[1] != 0)
            prop.name = self.columns[0] if self.prop_only else self.columns[1]

        if self.prop:
            return [prop] if self.prop_only else [count, prop]
        else:
            return [count]

# a shorthand for Count(prop_only=True)
class Proportion(Count):
    def __init__(self, series, **kwargs):
        Count.__init__(self, series=series, prop_only=True, **kwargs)

def _collect_columns(aggregates):
    columns = set()
    for a in aggregates:
        intersection = columns.intersection(a.columns)
        if len(intersection) > 0: raise ValueError('Repeated columns: %s' % intersection)
        columns.update(a.columns)
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
        return pd.concat(chain(*series), axis=1)

def aggregate_list(l):
    return list(np.concatenate(l.values))

def aggregate_set(l):
    return set(np.concatenate(l.values))

def aggregate_counts(l):
    return np.unique(np.concatenate(l.values), return_counts=True)
