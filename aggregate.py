import os
import re
import logging

import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

from drain import util

def aggregate(df, columns, weight=None, index=None):
    if index is not None:
        if isinstance(index, basestring):
            df2 = df[[index]].copy()
        else:
            df2 = df[index].copy()
    else:
        df2 = pd.DataFrame(index=df.index)  
    
    # generate numerators and denominators
    agg_dict = {}
    
    for column,agg in columns.iteritems():
        if 'numerator' in agg:
            numerator = __series(df, agg['numerator'])
        else:
            numerator = pd.Series(np.ones(len(df)), index=df.index)
        if weight is not None:
            numerator *= weight
        
        df2[column+'_numerator'] = numerator
        
        # default function is sum
        if 'func' not in agg:
            agg['func'] = 'sum'
        
        agg_dict[column+'_numerator'] = agg['func']
        if 'denominator' in agg:
            denominator = __series(df, agg['denominator'])
            if weight is not None:
                denominator *= weight
            df2[column+'_denominator'] = denominator
            agg_dict[column+'_denominator'] = agg['denominator_func'] if 'denominator_func' in agg else agg['func'] 
    
    # aggregate
    df3 = df2.groupby(index).agg(agg_dict) if index is not None else df2
    
    # collect and rename
    df4 = pd.DataFrame(index=df3.index)
    for column in columns:
        if 'denominator' in columns[column]:
            df4[column] = df3[column+'_numerator']/df3[column+'_denominator']
        else:
            df4[column] = df3[column+'_numerator']
    return df4

# get a series from a dataframe
# could be a function that takes the frame as a parameter
# or a string (interpreted as a column)
# or a series or something that can be coerced into one (e.g. a scalar)
def __series(df, attr):
    if hasattr(attr, '__call__'):
        return attr(df)
    elif attr in df.columns:
        return df[attr]
    elif not isinstance(attr, basestring):
        return attr
    else:
        raise ValueError('Invalid attribute for series: {}'.format(attr))

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

# TODO write the aggregate function
# TODO pivot based on delta
# TODO prefix column
# TODO test table vs fixed format, data_column
class SpacetimeAggregator(object):
    def __init__(self, spacedeltas, dates, prefix, basedir):
        self.spacedeltas = spacedeltas
        self.prefix = prefix
        self.dates = dates
        self.dirname = os.path.join(basedir, prefix)
        
        self.filenames = {d: os.path.join(self.dirname, '%s.hdf' % d.strftime('%Y%m%d')) for d in dates}
        
    # should return DataFrame of aggregations for the given date
    def aggregate(self, date, **args):
        raise NotImplementedError

        
    # should return the aggregations, pivoted and prefixed
    # if left is specified then only returns those aggregations
    def read(self, left=None):
        dfs = []
        for d in self.dates:
            logging.info('reading date %s' % d)
            df = self.read_date(d, left)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True, copy=False)
    
    # read the data for the specified date
    def read_date(self, date, left=None):
        hdf_kwargs = {}
        if left is not None:
            left = left[left.date == date]
            if len(left) == 0:
                return pd.DataFrame()            
        df = pd.read_hdf(self.filenames[date], key='df', **hdf_kwargs)
        for index in self.spatial_indexes:
            values = left[index].unique()
            df.drop(df.index[~df[index].isin(values)], inplace=True)

        df['date'] = date
        return df
    
    # write the data for a specific date
    # cast to dtype unless it's None
    def write_date(self, date, dtype=None):
        df = self.aggregate(date)

        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        df = df.astype(np.float32) if dtype is not None else df
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

class Spacedeltas(object):
    def __init__(self, spatial_index, delta_strings):
        self.spatial_index = spatial_index
        self.deltas = {s:parse_delta(s) for s in delta_strings}
