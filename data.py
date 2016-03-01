import random
import datetime
import re
import os
from drain import util
import warnings
import logging

from copy import deepcopy
import pandas as pd
from scipy import stats

import numpy as np
from numpy import random
from dateutil.relativedelta import relativedelta

import collections
from itertools import product

from sklearn import preprocessing, datasets
from sklearn.utils.validation import _assert_all_finite

from drain.step import Step

class ClassificationData(Step):
    def run(self):
        X,y = datasets.make_classification(**self.get_arguments())
        X,y = pd.DataFrame(X), pd.Series(y)

        train = np.zeros(len(X), dtype=bool)
        train[random.choice(len(X), len(X)/2)] = True
        train = pd.Series(train)

        return {'X': X, 'y': y, 'train': train, 'test': ~train}

class CreateEngine(Step):
    def run(self):
        return util.create_engine()

class FromSQL(Step):
    def __init__(self, query=None, to_str=None, table=None, tables=None, **kwargs):
        """
        Use tables to automatically set dependecies
        """
        if query is None:
            if table is None:
                raise ValueError("Must specify query or table")
            query = "SELECT * FROM %s" % table
            tables = [table]
        
        if tables is not None and 'SQL_DIR' in os.environ:
            self.dependencies = [os.path.join(
                    os.environ['SQL_DIR'], table.replace('.','/')) 
                        for table in tables]

        if to_str is None:
            to_str = []

        Step.__init__(self, query=query, to_str=to_str, **kwargs)

        if 'inputs' not in kwargs:
            self.inputs = [CreateEngine()]
 
    def run(self, engine):
        kwargs = self.get_arguments(query=False, to_str=False)

        df = pd.read_sql(self.query, engine, **kwargs)
        for column in self.to_str:
            if column in df.columns:
                df[column] = df[column].astype(str)

        return df

class Merge(Step):
    def run(self, *dfs):
        df = dfs[0] 
        for d in dfs[1:]:
            df = df.merge(d, **self.get_arguments(inputs=False))

        return df

# write DataFrames to an HDF store
# pass put_arguments (format, mode, data_columns, etc.) to init
# pass DataFrames by name via inputs
class ToHDF(Step):
    def __init__(self, target=True, objects_to_ascii=False, **kwargs):
        Step.__init__(self, target=True, objects_to_ascii=objects_to_ascii, **kwargs)

    def run(self, **kwargs):
        store = pd.HDFStore(os.path.join(self.get_dump_dirname(), 'result.h5'))

        for key, df in kwargs.iteritems():
            if self.objects_to_ascii:
                for c,dtype in df.dtypes.iteritems():
                    if dtype == object:
                        df[c] = df[c].str.encode("ascii", "ignore")

            logging.info('Writing %s %s' % (key, str(df.shape)))
            args = self.get_arguments().get('put_args', {}).get(key, {})

            store.put(key, df, mode='w', **deepcopy(args))

        return store

    def dump(self):
        return

    def load(self):
        self.set_result(pd.HDFStore(os.path.join(self.get_dump_dirname(), 'result.h5')))

class Shape(Step):
    def run(self, X, index=None, **kwargs):
        if index is not None:
            X = X[index]
        return {'n_rows': X.shape[0], 'n_cols': X.shape[1]}

class HoldOut(Step):
    def run(self, index, **kwargs):
        mask = np.zeros(len(index), dtype=bool)
        mask[random.choice(len(index), len(index)*self.p)] = True

        new_index = pd.Series(index.values & (~mask), index=index.index)
        holdout = pd.Series(index.values & mask, index=index.index)

        return {'index': new_index, 'holdout': holdout}

def percentile(series):
    return pd.Series(stats.rankdata(series)/len(series), index=series.index)

def prefix_columns(df, prefix, ignore=[]):
    df.columns =  [prefix + c if c not in ignore else c for c in df.columns]

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
    nulcols = df.isnull().sum() > 0
    return nulcols[nulcols].index

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
                print(df.columns[i] + ', ' + df.columns[j] + ': ' + str(corr[i][j]))
                
def undersample_by(y, train, p):
    a = pd.Series([random.random() < p for i in range(len(y))], index=y.index)
    return train & (y | a) 

def undersample_to(y, train, p):
    T = y[train].sum()
    F = len(y[train]) - T
    q = p*T/((1-p)*F)
    return undersample_by(y, train, q)

def date_censor_sql(date_column, today, column = None):
    if column is None:
        column = date_column
    return "(CASE WHEN {date_column} < '{today}' THEN {column} ELSE null END)".format(
            date_column=date_column, today=today, column=column)

# group 1 is the table name, group 2 is the query whose result is the table
extract_sql_regex = r'CREATE\s+TABLE\s+([^(\s]*)\s+AS\s*\(([^;]*)\);'
def revise_helper(query):
    match = re.search(extract_sql_regex, query, re.DOTALL | re.I)
    return match.group(1), match.group(2)

def revise_sql(query, id_column, output_table, max_date_column, min_date_column, date_column, date):
    """
    Given an expensive query that aggregates temporal data,
    Revise the results to censor before a particular date
    """
    if hasattr(id_column, '__iter__'):
        id_column = str.join(', ', id_column)
    
    sql_vars = dict(query=query, id_column=id_column, output_table=output_table, 
            max_date_column=max_date_column, min_date_column=min_date_column, 
            date_column=date_column, date=date)

    sql_vars['ids_query'] = """
    SELECT {id_column} FROM {output_table} 
    WHERE {max_date_column} >= '{date}' AND {min_date_column} < '{date}'""" .format(**sql_vars)

    sql_vars['revised_query'] = query.replace('1=1', 
            "(({id_column}) in (select * from ids_query) and {date_column} < '{date}')".format(**sql_vars))

    new_query = """
    with ids_query as ({ids_query})
    select * from ({revised_query}) t
    """.format(**sql_vars)

    return new_query

class Revise(Step):
    def __init__(self, sql_filename, id_column, max_date_column, min_date_column, 
                date_column, date, from_sql_args=None, **kwargs):
        """
        revise the query contained in sql_filename to the specified date
        id_column: the entity id column(s) linking the result of the query with its source tables
        max_date_column: the maximum date column name for an entry in the result
        min_date_column: the minimum date column name for an entry in the result
        date_column: name of the date column in the source
        date: the date to revise at
        from_sql_args: dictionary of keyword arguments to pass input FromSQL steps, 
                e.g. target=True, parse_dates
        """

        Step.__init__(self, sql_filename=sql_filename, id_column=id_column, 
                max_date_column=max_date_column, min_date_column=min_date_column, 
                date_column=date_column, date=date, 
                from_sql_args=from_sql_args, **kwargs)
        
        sql = util.read_file(sql_filename)
        table, query = revise_helper(sql)

        revised_sql = revise_sql(query=query, id_column=id_column, output_table=table,
                max_date_column=max_date_column, min_date_column=min_date_column, 
                date_column=date_column, date=date)

        if from_sql_args is None: from_sql_args = {}
        self.inputs = [FromSQL(table=table, **from_sql_args), 
                       FromSQL(revised_sql, **from_sql_args)]
        self.inputs_mapping = ['source', 'revised']

    def run(self, source, revised):
        subset = (source[self.min_date_column] < self.date) & (source[self.max_date_column] < self.date)

        return pd.concat((source[subset],revised), copy=False)

def date_select(df, date_column, date, delta):
    """
    given a series an end date and number of days, return subset in the date range
    if delta is None then there is no starting date
    """
    delta = parse_delta(delta)
    df = df.query("%s < '%s'" % (date_column, date))

    if delta is not None:
        start_date = date - delta
        df = df.query("%s >= '%s'" % (date_column, start_date))

    return df

def date_censor(df, date_columns, date):
    """
    a dictionary of date_column: [dependent_column1, ...] pairs
    censor the dependent columns when the date column is before the given end_date
    then censor the date column itself
    """
    for date_column, censor_columns in date_columns.iteritems():
        for censor_column in censor_columns:
            df[censor_column] = df[censor_column].where(df[date_column] < date)

        df[date_column] = df[date_column].where(df[date_column] < date)

    return df


delta_chars = {
        'y':'years', 'm':'months', 'w':'weeks', 'd':'days', 'h':'hours',
        'M':'minutes', 's':'seconds', 'u':'microseconds'
}

delta_regex = re.compile('^([0-9]+)(u|s|M|h|d|m|y)$')

# parse a string to a delta
# 'all' is represented by None
def parse_delta(s):
    if s == 'all':
        return None
    else:
        l = delta_regex.findall(s)
        if len(l) == 1:
            return relativedelta(**{delta_chars[l[0][1]]:int(l[0][0])})
        else:
            raise ValueError('Invalid delta string: %s' % s)

# return the index (given level) as a series with the original index 
def index_as_series(df, level=None): 
    if level is not None: 
        values = df.index.get_level_values(level) 
    else: 
        values = df.index.values 
 
    return pd.Series(values, index=df.index) 
 
# get a column or index level as series 
# if name is none return the whole index 
def get_series(df, name): 
    if name in df.columns: 
        return df[name] 
    else: 
        return index_as_series(df, name)

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
