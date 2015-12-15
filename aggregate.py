import os
import re
import logging

import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from itertools import product,chain

import pandas as pd
import numpy as np

from drain import util, data
from drain.util import merge_dicts, hash_obj

# AggregateSeries consist of a series and a function
# the series can be:
#    - a callable (taking the DataFrame), e.g. lambda df: df.column**2
#    - one of the columns in the frame, e.g. 'column'
#    - a non-string value, e.g. 1
class AggregateSeries(object):
    def __init__(self, series, function):
        self.series = series
        self.function = function
    
    def apply_series(self, df):
        if hasattr(self.series, '__call__'):
            return self.series(df)
        elif self.series in df.columns:
            return df[self.series]
        elif not isinstance(self.series, basestring):
            return pd.Series(self.series, index=df.index)
        else:
            raise ValueError('Invalid series: %s' % self.series)
    
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
# When parent is specified count parent and 
class Count(AggregateBase):
    def __init__(self, series=None, name=None, prop=None, parent=1):
        if series is None:
            columns = [name if name is not None else 'count']
            aggregate_series = [AggregateSeries(1, 'sum')]
        else:
            if name is None: name = series
            columns = ['%s_count' % name]
            # converting to float32 before summing is an order of magnitude faster
            aggregate_series = [AggregateSeries(lambda d: d[series].astype(np.float32), 'sum')]
            
            if prop:
                columns.append('%s_prop' % name)
                aggregate_series.append(AggregateSeries(parent, 'sum'))
            
        AggregateBase.__init__(self, columns, aggregate_series)
    
    def apply(self, series):
        series[0].name = self.columns[0]
        if len(self.columns) == 2:
            series[1] = (series[0] / series[1]).where(series[1] != 0)
            series[1].name = self.columns[1]
        return series

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

# given a series an end date and number of days, return subset in the date range
# if deta is -1 then there is no starting date
def censor(df, date_column, end_date, delta):
    df = df[ df[date_column] < end_date ]

    if delta is not None:
        start_date = end_date - delta
        df = df[ df[date_column] >= start_date ]

    return df

def aggregate_list(l):
    return list(np.concatenate(l.values))

def aggregate_set(l):
    return set(np.concatenate(l.values))

def aggregate_counts(l):
    return np.unique(np.concatenate(l.values), return_counts=True)

# spacetimes is a dict of space_name : Spacetimes 
#     e.g. ['industry_state' : Spacetime(['naics_code', 'state'], ['1y', '5y', 'all'])
# dates is a collection of dates to aggregate (all the spacetimes) to
#     e.g. [date(2012,1,1), date(2013,1,1)]
# basedir is the base directory for storing hdf files
# prefix (e.g. 'tests') is used for:
#     storing the hdf files (e.g. '{basedir}/tests/20130101.hdf')
#     feature names (e.g. tests_tract_3y_{feature_name}
# date_col is used by the default censor method
# TODO: read(left, pivot) support for multiple spatial indexes
class SpacetimeAggregator(object):
    def __init__(self, spacedeltas, dates, prefix, basedir, date_col='date'):
        self.spacedeltas = spacedeltas
        self.prefix = prefix
        
        dates = pd.to_datetime(dates) # this is a pandas DatetimeIndex
        if len(set(dates)) != len(dates):
            raise ValueError('Dates must be unique')
        self.dates = dates
        self.dirname = os.path.join(basedir, prefix)
        self.date_col = date_col
        
        self.filenames = {d: os.path.join(self.dirname, '%s.hdf' % d.strftime('%Y%m%d')) for d in dates}
        self.dtypes = {}

    def select(self, df, spacedeltas):
        include = []
        for space, deltas in spacedeltas.iteritems():
            if not self.spacedeltas[space].has_deltas(deltas):
                raise ValueError('Aggregator does not have deltas for %s: %s' % (space, deltas))
            include.extend(['%s_%s_%s_.*' % (self.prefix, space, delta) for delta in deltas])

        return data.select_features(df, exclude=['%s_.*' % self.prefix], include=include)

    # should return DataFrame of aggregations for the given date
    def aggregate(self, date, **args):
        raise NotImplementedError
        
    # should return the aggregations, pivoted and prefixed
    # if left is specified then only returns those aggregations
    # if pivot then pivot the spaces and deltas
    # if pivot and left then will iteratively left join
    # will ignore spatial index columns missing from left
    def read(self, left=None, pivot=True):
        # check to make sure left doesn't have dates that weren't aggregated
        if left is not None:
            diff = set(left.date.unique()).difference(self.dates.values)
            if len(diff) > 0:
                raise ValueError('Left frame contains dates not in aggregator: %s' % diff)

        dfs = []
        for d in self.dates:
            df = self.read_date(d, left)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True, copy=False)

        if pivot:
            if left is None:
                raise ValueError('Need left frame to pivot')
            index = left.index # preserve index for return
            left.date = pd.to_datetime(left.date) # ensure datetime64 for join

            for space, st in self.spacedeltas.iteritems():
                logging.info('Pivoting %s' % space)
                if st.spatial_index not in left.columns:
                    continue

                df_s = df[df.space == space]
                df_s = self._pivot(df_s)
              
                logging.info('Joining %s' % space)
                df_s.reset_index(inplace=True)
                df_s.rename(columns={'id':st.spatial_index}, inplace=True)

                df_s['date'] = pd.to_datetime(df_s['date']) # ensure datetime64 for join
                left = left.merge(df_s, on=[st.spatial_index, 'date'], how='left', copy=False)
 
            left.index = index
            return left
        else:
            return df

    # spacetime pivot, inplace
    def _pivot(self, df):
        df.set_index(['id', 'date', 'space', 'delta'], inplace=True)
        df = df.unstack('space').unstack('delta')
        columns = list(product(*df.columns.levels)) # list of (column, space, delta)

        # prefix columns
        df.columns = ['{0}_{1}_{2}_{3}'.format(self.prefix, space, delta, column)
            for column, space, delta in columns]

        return df

    # override this method for more complex censoring
    def censor(self, df, date, delta):
        return censor(df, self.date_col, date, delta) 

    def get_data(self, date):
        raise NotImplementedError()

    def get_aggregator(self, date):
        raise NotImplementedError()

    def aggregate(self, date):
        df = self.get_data(date)
        aggregates = self.get_aggregates(date=date, data=df)

        dfs = []
        for space, st in self.spacedeltas.iteritems():
            spatial_index = st.spatial_index
            df_s = df[df[spatial_index].notnull()] # ignore when spatial index is null

            for s, delta in st.deltas.iteritems():
                logging.info('Aggregating %s %s %s' % (date, space, s))

                df_st = self.censor(df_s, date, delta)
                aggregator = Aggregator(df_st, aggregates)
                aggregated = aggregator.aggregate(index=spatial_index)

                util.set_dtypes(aggregated, self.dtypes)
                aggregated.reset_index(inplace=True)

                aggregated.rename(columns={spatial_index:'id'}, inplace=True)
                aggregated['space'] = space
                aggregated['delta'] = s

                dfs.append(aggregated)

        return pd.concat(dfs, ignore_index=True)
    
    # read the data for the specified date
    def read_date(self, date, left=None):
        if left is not None:
            left = left[left.date == date]
            if len(left) == 0:
                return pd.DataFrame()

        logging.info('Reading date %s' % date)
        df = pd.read_hdf(self.filenames[date], key='df')

        if left is not None:
            for space, st in self.spacedeltas.iteritems():
                mask = (df['space'] == space)

                # TODO make this support spatial multi-index
                if st.spatial_index in left.columns:
                    values = left[self.spacedeltas[space].spatial_index].unique()
                    mask = mask & (~df['id'].isin(values))
                df.drop(df.index[mask], inplace=True)

        df['date'] = date
        return df
    
    # write the data for a specific date
    # cast to dtype unless it's None
    def write_date(self, date):
        logging.info('Aggregating %s' % date)
        df = self.aggregate(date)

        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

        logging.info('Writing %s %s' % (date, df.shape))
        return df.to_hdf(self.filenames[date], key='df', mode='w')

delta_chars = {
        'y':'years', 'm':'months', 'w':'weeks', 'd':'days', 'h':'hours', 
        'M':'minutes', 's':'seconds', 'u':'microseconds'
}

delta_regex = re.compile('^([0-9]+)(u|s|M|h|d|m|y)$')

def parse_delta(s):
    if s == 'all':
        return None
    else:
        l = delta_regex.findall(s)
        if len(l) == 1:
            return relativedelta(**{delta_chars[l[0][1]]:int(l[0][0])})
        else:
            raise ValueError('Invalid delta string: %s' % s)

spacetime_prefix_regex = re.compile('^(([^_]+_){3})')

# returns the {prefix}_{space}_{delta}_
def get_spacetime_prefix(column):
    return spacetime_prefix_regex.findall(column)[0][0]

class Spacedeltas(object):
    def __init__(self, spatial_index, delta_strings):
        self.spatial_index = spatial_index
        if len(set(delta_strings)) != len(delta_strings):
            raise ValueError('Duplicate deltas: %s' % delta_strings)
        self.delta_strings = delta_strings
        self.deltas = {s:parse_delta(s) for s in delta_strings}

    def has_deltas(self, delta_strings):
        return set(delta_strings).issubset(self.delta_strings)
