import pandas as pd
from sklearn import preprocessing
from sklearn.utils.validation import _assert_all_finite
import re
import os
import numpy as np
import collections
from itertools import product

import random
import datetime
import model
from util import prefix_columns, join_years
import util
import warnings

class ModelData(object):
    def read(self, **args):
        raise NotImplementedError
    
    def write(self, **args):
        raise NotImplementedError
        
    def transform(self, **args):
        raise NotImplementedError

# a ModelData wrapper for sklearn.datasets.make_classification for testing
class ClassificationData(ModelData):
    def __init__(self, **args):
        self.args = args

    def read(self, dirname=None):
        if dirname is not None:
            self.df = pd.read_pickle(os.path.join(dirname, 'df.pkl'))
        else:
            from sklearn import datasets
            X,y = datasets.make_classification(**self.args)
            df = pd.DataFrame(X)
            df['y'] = y
            self.df = df

    def write(self, dirname):
        self.df.to_pickle(os.path.join(dirname, 'df.pkl'))

    def transform(self):
        self.X = self.df.drop('y', axis=1)
        self.y = self.df.y
        
        train_len = int(len(self.X)/2)
        train = pd.Series([True]*train_len + [False]*(len(self.X)-train_len))
        self.cv = (train, ~train)

# produces a query returning left at the given level in sql
def aggregation_left_sql(left, index):
    if 'aggregation_end' in left.columns:
        left = left[[index, 'aggregation_end']].dropna().drop_duplicates()
        left[index] = left[index].astype(np.int64)
        left = str.join(',', map(lambda a: "({i},'{d}')".format(i=a[0],d=a[1]), left.values))
        return 'select * from unnest(ARRAY[{left}]::aggregation_type[])'.format(left=left)
    else:
        left = left[index].dropna().drop_duplicates()
        left = left.astype(np.int64)
        left = str.join(',', map(str, left))
        return 'select * from unnest(ARRAY[{left}]) aggregation_id'.format(left=left)
    
def get_aggregate(table_name, index, engine, left):
    left_sql = aggregation_left_sql(left, index)
    join_columns = 'aggregation_id, aggregation_end' if 'aggregation_end' in left.columns else 'aggregation_id'

    sql = """
    with addresses as ({left_sql})
    select * from addresses
    join {table_name} using ({join_columns}) 
    where aggregation_level='{index}'
    """.format(left_sql=left_sql, table_name=table_name, index=index, join_columns=join_columns)
    
    return pd.read_sql(sql, engine)

# convenience function for getting level index from name
# tract -> census_tract_id, building -> building_id, etc.
def level_index(level):
    if level in ('tract', 'block'):
        level = 'census_' + level
    if level == 'community':
        level = 'community_area'
    return level + '_id'

def aggregate_prefix_column(level, column, prefix=None, delta=None):
    column_prefix = 'st_' + level + '_'
    if delta is not None:
        delta_prefix = str(delta) + 'y' if delta != -1 else 'all'
        column_prefix += delta_prefix + '_'
    if prefix is not None:
        column_prefix += prefix + '_'
    return column_prefix + column

def include_aggregations(aggregations, prefix, options):
    include = set()
    for level in aggregations:
        if level not in options:
            raise ValueError('Invalid level for {0} aggregation: {1}'.format(prefix, level))
        if isinstance(aggregations, dict):
            for delta in aggregations[level]:
                if delta not in options[level]:
                    raise ValueError('Invalid delta for {0} {1} aggregation: {2}'.format(level, prefix, delta))
                include.add(aggregate_prefix_column(level=level, delta=delta, prefix=prefix, column='') + '.*')
        else:
             include.add(aggregate_prefix_column(level=level, prefix=prefix, column=''))
    return include

def get_aggregation(table_name, levels, engine, left, prefix=None):
    temporal = 'aggregation_end' in left.columns

    for level in levels:
        index = level_index(level)
        t = get_aggregate(table_name, index, engine, left) # change to level after re-running things
        t.drop(['aggregation_level'],inplace=True,axis=1)
        if temporal:
            t.set_index(['aggregation_end', 'aggregation_id', 'aggregation_delta'], inplace=True)
            t = t.unstack()
            t.columns = [aggregate_prefix_column(level, column, prefix, delta) for column, delta in product(*t.columns.levels)]
        else:
            t.set_index('aggregation_id', inplace=True)
            t.columns = [aggregate_prefix_column(level, column, prefix) for column in t.columns]

        t.reset_index(inplace=True) # should add exclude arg to prefix_columns
        t.rename(columns={'aggregation_id':index}, inplace=True)

        index = [index, 'aggregation_end'] if temporal else index
        left = left.merge(t, on=index, how='left', copy=False)

    return left

# generate year, month, day features from specified date features
def expand_dates(df, columns=[]):
    columns=df.columns.intersection(columns)
    df2 = df.reindex(columns=set(df.columns).difference(columns))
    for column in columns:
        df2[column + '_year'] = df[column].apply(lambda x: x.year)
        df2[column + '_month'] = df[column].apply(lambda x: x.month)
        df2[column + '_day'] = df[column].apply(lambda x: x.day)
    return df2

# binarize specified categoricals
# category_classes is either a dict of (column : [class1, class2, ...]) pairs
# or a list of columns, in which case possible values are found using df[column].unique()
def binarize(df, category_classes, all_classes=False):
    if type(category_classes) is not dict:
        columns = set(category_classes).intersection(df.columns)
        category_classes = {column : df[column].unique() for column in columns}
    else:
        columns = set(category_classes.keys()).intersection(df.columns)

    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1 if not all_classes else len(classes)):
            df[category + '_' + str(classes[i]).replace( ' ', '_')] = (df[category] == classes[i])
        
    df.drop(columns, axis=1, inplace=True)                                      
    return df

def binarize_set(df, column, values=None):
    if values is None:
        values = reduce(lambda a,b: a | b, df[column])
    for value in values:
        name = values[value] if type(values) is dict else str(value)
        df[column + '_'+ name.replace(' ', '_')] = df[column].apply(lambda d: value in d)
    df.drop(column, axis=1, inplace=True)

# given a column whose entries are lists, create columns counting each element
def binarize_list(df, column, values=None):
    if values is None:
        values = set(np.concatenate(df[column].values))
    for value in values:
        name = values[value] if type(values) is dict else str(value)
        df[column + '_'+ name.replace(' ', '_')] = df[column].apply(lambda d: d.count(value))
    df.drop(column, axis=1, inplace=True)

def binarize_clusters(df, column, n_clusters, train=None):
    series = df[column]
    series = series.dropna()
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters)
    
    series = pd.DataFrame(series)
    kmeans.fit(series[train] if train is not None else series)
    
    clusters = kmeans.cluster_centers_[:,0].astype(int)
    df[column + '_cluster'] = pd.Series(kmeans.predict(series), index=series.index).apply(lambda d: clusters[d])
    
    binarize(df, {column + '_cluster': clusters}, all_classes=True) # use all_classes to handle nulls
    
    return df

# narrows df to train | test
# then narrows train and test to that
def train_test_subset(df, train, test):
    df.drop(df.index[~(train | test)], inplace=True)
    train = train.loc[df.index]
    test = test.loc[df.index]

    return df, train, test

# returns endogenous and exogenous variables
# normalization requires imputation (can't normalize null values)
# training mask is used for normalization
def Xy(df, y_column, include=None, exclude=None, train=None, category_classes={}):
    y = df[y_column]
    exclude.add(y_column)

    X = select_features(df=df, exclude=exclude, include=include)
    
    X = binarize(X, category_classes)

    #nulcols = null_columns(X, train)
    #if len(nulcols) > 0:
    #    print 'Warning: null columns ' + str(nulcols)
    
    #nonnum = non_numeric_columns(X)
    #if len(nonnum) > 0:
    #    print 'Warning: non-numeric columns ' + str(nonnum)

    X = X.astype(np.float32, copy=False)

    #inf = infinite_columns(X)
    #if len(inf) > 0:
    #    print 'Warning: columns contain infinite values' + str(inf)
    
    return (X,y)

def normalize(X, train=None):
    Xfit = X[train] if train is not None else X
    sigma = X[train].std(ddof=0)
    sigma.loc[sigma==0]=1
    mu = X[train].mean()

    X = (X - mu) / sigma
        
    return X

def impute(X, train=None):
    Xfit = X[train] if train is not None else X
    X.fillna(Xfit.mean(), inplace=True)
    return X

# select subset of strings matching a regex
# treats strings as a set!
def select_regexes(strings, regexes):
    strings = set(strings)
    select = set()
    if isinstance(strings, collections.Iterable):
        for r in regexes:
            s = set(filter(re.compile('^'  + r + '$').search, strings))
            strings -= s
            select |= s
        return select
    else:
        raise ValueError("exclude should be iterable")
        
def exclude_regexes(strings, exclude, include=None):
    e = select_regexes(strings, exclude)
    i = select_regexes(strings, include) if include is not None else set()
    return set(strings).difference(e).union(i)

def select_features(df, exclude, include=None):
    columns = exclude_regexes(strings=df.columns, exclude=exclude, include=include)
    df = df.reindex(columns = columns, copy=False)
    return df

def null_columns(df, train=None):
    if train is not None:
        df = df[train]
    nulcols = df.isnull().sum() == len(df)
    return nulcols[nulcols==True].index

def infinite_columns(df):
    columns = []
    for c in df.columns:
        try:
            _assert_all_finite(df[c])
        except:
            columns.append(c)
    return columns

def non_numeric_columns(df):
    columns = []
    for c in df.columns:
        try: 
            df[c].astype(float)
        except:
            columns.append(c)
            
    return columns

def get_correlates(df, c=.99):
    corr = df.corr().values
    for i in range(len(df.columns)):
        for j in range(i):
            if corr[i][j] >= c:
                print df.columns[i] + ', ' + df.columns[j] + ': ' + str(corr[i][j])
                
def undersample_by(y, train, p):
    a = pd.Series([random.random() < p for i in range(len(y))], index=y.index)
    return train & (y | a) 

def undersample_to(y, train, p):
    T = y[train].sum()
    F = len(y[train]) - T
    q = p*T/((1-p)*F)
    return undersample_by(y, train, q)

def censor_column(date_column, today, column = None):
    if column is None:
        column = date_column
    return "(CASE WHEN {date_column} < '{today}' THEN {column} ELSE null END)".format(
            date_column=date_column, today=today, column=column)

def nearest_neighbors_impute(df, coordinate_columns, data_columns, knr_params={}):
    from sklearn.neighbors import KNeighborsRegressor
    for column in data_columns:
        not_null = df[column].notnull()
        if (~not_null).sum() == 0:
            continue
        knr = KNeighborsRegressor(**knr_params)
        knr.fit(df.loc[not_null,coordinate_columns], df.loc[not_null,[column]])
        predicted = knr.predict(df.loc[~not_null,coordinate_columns])
        df.loc[ (~not_null),[column]] = predicted
