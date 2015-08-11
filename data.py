import pandas as pd
from sklearn import preprocessing
import re
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

def get_aggregate(table_name, level, engine, end_dates=None, deltas=None):
    sql = "select * from {table_name} where aggregation_level='{level}' ".format(table_name=table_name, level=level)

    if end_dates is not None:
        end_dates = str.join(',', map(lambda d: "'" + str(d) + "'", end_dates))
        deltas = str.join(',', map(str, deltas))
        sql = sql + " and aggregation_end in ({end_dates}) and aggregation_delta in ({deltas})".format(end_dates=end_dates, deltas=deltas)

    print sql

    t = pd.read_sql(sql, engine)
    return t

def prefix_column(level, column, prefix=None, delta=None):
    column_prefix = level[:-3] + '_'
    if prefix is not None:
        column_prefix += prefix + '_'
    if delta is not None:
        delta_prefix = str(delta) + 'y' if delta != -1 else 'all'
        column_prefix += delta_prefix + '_'
    return column_prefix + column

def get_aggregation(table_name, level_deltas, engine, end_dates=None, left=None, prefix=None):
    for level in level_deltas:
        deltas = level_deltas[level] if type(level_deltas) is dict else None
        t = get_aggregate(table_name, level, engine, end_dates, deltas)
        t.drop(['aggregation_level'],inplace=True,axis=1)
        if deltas is not None:
            t.set_index(['aggregation_end', 'aggregation_id', 'aggregation_delta'], inplace=True)
            t = t.unstack()
            t.columns = [prefix_column(level, column, prefix, delta) for column, delta in product(*t.columns.levels)]

        t.reset_index(inplace=True) # should add exclude arg to prefix_columns
        t.rename(columns={'aggregation_id':level}, inplace=True)
        if left is None:
            left = t
        else:
            index = [level, 'aggregation_end'] if deltas is not None else level
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

# binarize specified categoricals using specified category class dictionary
def binarize(df, category_classes, all_classes=False):
    #for category,classes in category_classes.iteritems():
    columns = set(category_classes.keys()).intersection(df.columns)
    
    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1 if not all_classes else len(classes)):
            df[category + '_' + str(classes[i]).replace( ' ', '_')] = (df[category] == classes[i])
        
    df.drop(columns, axis=1, inplace=True)                                      
    return df

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

# returns endogenous and exogenous variables
# normalization requires imputation (can't normalize null values)
# training mask is used for normalization
def Xy(df, y_column, include=None, exclude=None, impute=True, normalize=True, train=None, category_classes={}):
    y = df[y_column]
    exclude.add(y_column)

    X = select_features(df, include, exclude)
    
    X = binarize(X, category_classes)
    
    nulcols = null_columns(X)
    if len(nulcols) > 0:
        print 'Warning: null columns ' + str(nulcols)
    
    nonnum = non_numeric_columns(X)
    if len(nonnum) > 0:
        print 'Warning: non-numeric columns ' + str(nonnum)
    
    if impute:
        imputer = preprocessing.Imputer()
        Xfit = X[train] if train is not None else X
        imputer.fit(Xfit)
        d = imputer.transform(X)
        if normalize:
            d = preprocessing.StandardScaler().fit_transform(d)
        X = pd.DataFrame(d, index=X.index, columns = X.columns)

    
    return (X,y)

def select_features(df, include=None, exclude=None, regex=True):

    if isinstance(include, collections.Iterable):
        columns = set.union(*[set(filter(re.compile('^' + feature + '$').match, df.columns)) for feature in include])
    else: 
        columns = set(df.columns)
        
    if isinstance(exclude, collections.Iterable):
        d = set.union(*[set(filter(re.compile('^'  + feature + '$').search, df.columns)) for feature in exclude])
        columns = columns.difference(d)
    
    df = df.reindex(columns = columns)
    return df

def null_columns(df):
    nulcols = df.isnull().sum() == len(df)
    return nulcols[nulcols==True].index

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
                
def undersample_cv(d, train, p):
    a = pd.Series([random.random() < p for i in range(len(d))], index=d.index)
    return train & ((d.test_bll > 5).values | a) 

def count_unique(series):
    return series.nunique()

