import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.utils.validation import _assert_all_finite
import re
import os
import numpy as np
import collections
from itertools import product

import random
import datetime
from util import prefix_columns, join_years
import util
import warnings

def percentile(series):
    return pd.Series(stats.rankdata(series)/len(series), index=series.index)

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
# all_classes = False means the last class is skipped
# drop means drop the original column
def binarize(df, category_classes, all_classes=True, drop=True):
    if type(category_classes) is not dict:
        columns = set(category_classes).intersection(df.columns)
        category_classes = {column : df[column].unique() for column in columns}
    else:
        columns = set(category_classes.keys()).intersection(df.columns)

    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1 if not all_classes else len(classes)):
            df[category + '_' + str(classes[i]).replace( ' ', '_')] = (df[category] == classes[i])
    
    if drop:
        df.drop(columns, axis=1, inplace=True)                                  
    return df

def binarize_set(df, column, values=None):
    d = df[column].dropna() # avoid nulls
    if values is None:
        values = reduce(lambda a,b: a | b, d)
    for value in values:
        name = values[value] if type(values) is dict else str(value)
        df[column + '_'+ name.replace(' ', '_')] = d.apply(lambda c: value in c)
    df.drop(column, axis=1, inplace=True)

# given a column whose entries are lists, create columns counting each element
def count_list(df, column, values=None):
    if values is None:
        values = set(np.concatenate(df[column].values))
    for value in values:
        name = values[value] if type(values) is dict else str(value)
        df[column + '_'+ name.replace(' ', '_')] = df[column].apply(lambda d: d.count(value))
    df.drop(column, axis=1, inplace=True)

# given values, counts (as returned by np.unique(return_counts=True), find the count of a given value
def countsorted(values, counts, value):
    if len(values) == 0: # when values is empty is has float dtype and searchsorted will fail
        return 0
    i = np.searchsorted(values, value)
    if i != len(values) and values[i] == value:
        return counts[i]
    else:
        return 0

# convert (values, counts) as returned by aggregate.aggregate_counts() to dicts
# makes expand_counts much faster
def counts_to_dicts(df, column):
    d = df[column].apply(lambda c: pd.notnull(c) and len(c[0]) > 0) # index where there are counts and they aren't null
    return df.loc[d, column].apply(lambda c: {k:v for k,v in zip(*c)})

# expand a column containing value:count dictionaries
def expand_counts(df, column, values=None):
    d = counts_to_dicts(df, column)
    if len(d) > 0:
        if values is None:
            values = set(np.concatenate(d.apply(lambda c: c.keys()).values))
        for value in values:
            name = values[value] if type(values) is dict else str(value)
            df[column + '_'+ name.replace(' ', '_')] = d.apply(lambda c: c[value] if value in c else 0)
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
    sigma = Xfit.std(ddof=0)
    sigma.loc[sigma==0]=1
    mu = Xfit.mean()

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
