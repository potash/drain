import re
import os
from . import util
import logging

from copy import deepcopy
import pandas as pd
from six import string_types

import numpy as np
from numpy import random
from dateutil.relativedelta import relativedelta

import collections

from sklearn import datasets
from sklearn.utils.validation import _assert_all_finite

from .step import Step, MapResults


class Column(object):
    """Defines a new or existing column that can be calculated from a dataframe.

    Column accepts as ``definition`` a string that refers to an
    existing column, or a lambda function that returns a pd.Series,
    or a constant, in which case it creates a pd.Series of that constant.

    Columns are hashed based on their definition and type.
    """

    def __init__(self, definition, astype=None):
        """Args:
            definition ({str, function, constant}): Specifies a
                new column (that is, a pd.Series) from a dataframe.
                It can be a function, an existing column name, or a
                constant (that will be replicated along rows).
            astype (Pandas dtype): A Pandas datatype to which the
                resulting pd.Series will be converted to.
        """
        self.definition = definition
        self.astype = astype

    def apply(self, df):
        """Takes a pd.DataFrame and returns the newly defined column, i.e.
        a pd.Series that has the same index as `df`.
        """
        if hasattr(self.definition, '__call__'):
            r = self.definition(df)
        elif self.definition in df.columns:
            r = df[self.definition]
        elif not isinstance(self.definition, string_types):
            r = pd.Series(self.definition, index=df.index)
        else:
            raise ValueError("Invalid column definition: %s" % str(self.definition))
        return r.astype(self.astype) if self.astype else r

    def __hash__(self):
        return hash((self.definition, self.astype))

    def __eq__(self, other):
        return hash(self) == hash(other)


class ClassificationData(Step):
    def run(self):
        X, y = datasets.make_classification(
                **self.get_arguments(inputs=False, dependencies=False))
        X, y = pd.DataFrame(X), pd.Series(y)

        train = np.zeros(len(X), dtype=bool)
        train[random.choice(len(X), int(len(X)/2))] = True
        train = pd.Series(train)

        return {'X': X, 'y': y, 'train': train, 'test': ~train}


class CreateEngine(Step):
    def run(self):
        return util.create_engine()


class CreateDatabase(Step):
    def run(self):
        return util.create_db()


class FromSQL(Step):
    def __init__(self, query=None, to_str=None, table=None,
                 tables=None, inputs=None, auto_parse_dates=True,
                 **read_sql_kwargs):
        """
        Use tables to automatically set dependecies
        """
        if query is None:
            if table is None:
                raise ValueError("Must specify query or table")
            query = "SELECT * FROM %s" % table
            tables = [table]

        if to_str is None:
            to_str = []

        if inputs is None:
            self.inputs = [CreateEngine()]

        Step.__init__(self, query=query,
                      to_str=to_str,
                      inputs=inputs,
                      auto_parse_dates=auto_parse_dates,
                      read_sql_kwargs=read_sql_kwargs)

        if tables is not None and 'SQL_DIR' in os.environ:
            self.dependencies = [os.path.join(
                    os.environ['SQL_DIR'], t.replace('.', '/'))
                        for t in tables]

    def run(self, engine):
        df = pd.read_sql(self.query, engine, **self.read_sql_kwargs)
        for column in self.to_str:
            if column in df.columns:
                df[column] = df[column].astype(str)

        if self.auto_parse_dates:
            util.parse_dates(df, errors='coerce', inplace=True)

        return df


class ToSQL(Step):
    """
    Step for util.PgSQLDatabase.to_sql()
    inputs:
        df is the DataFrame to import
        db is an instance of PgSQLDatabase
            defaults to CreateDatabase()
    TODO: once drain Steps have outputs,
        include psql/schema/name
    """
    def __init__(self, table_name, **kwargs):
        """
        Args:
            table_name: a hack because name is a special kwarg currently
                TODO: use name once refactor/init is merged
        """
        Step.__init__(self, table_name=table_name, **kwargs)

        if len(self.inputs) == 1:
            self.inputs = self.inputs + [MapResults([CreateDatabase()], 'db')]

    def run(self, df, db):
        kwargs = self.get_arguments(inputs=False)
        kwargs['name'] = kwargs.pop('table_name')
        db.to_sql(df, **kwargs)


class Merge(Step):
    def run(self, *dfs):
        df = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, **self.get_arguments(inputs=False))

        return df


class ToHDF(Step):
    """
    write DataFrames to an HDF store
    pass put_arguments (format, mode, data_columns, etc.) to init
    pass DataFrames by name via inputs
    """
    def __init__(self, objects_to_ascii=False, **kwargs):
        Step.__init__(self, objects_to_ascii=objects_to_ascii, **kwargs)
        self._target = True

    def run(self, **kwargs):
        store = pd.HDFStore(os.path.join(self._dump_dirname, 'result.h5'))

        for key, df in kwargs.items():
            if self.objects_to_ascii:
                for c, dtype in df.dtypes.items():
                    if dtype == object:
                        df[c] = df[c].str.encode("ascii", "ignore")

            logging.info('Writing %s %s' % (key, str(df.shape)))
            args = self.get_arguments().get('put_args', {}).get(key, {})

            store.put(key, df, mode='w', **deepcopy(args))

        return store

    def dump(self):
        return

    def load(self):
        self.result = pd.HDFStore(os.path.join(self._dump_dirname, 'result.h5'), mode='r')


def prefix_columns(df, prefix, ignore=[]):
    df.columns = [prefix + c if c not in ignore else c for c in df.columns]


def expand_dates(df, columns=[]):
    """
    generate year, month, day features from specified date features
    """
    columns = df.columns.intersection(columns)
    df2 = df.reindex(columns=set(df.columns).difference(columns))
    for column in columns:
        df2[column + '_year'] = df[column].apply(lambda x: x.year)
        df2[column + '_month'] = df[column].apply(lambda x: x.month)
        df2[column + '_day'] = df[column].apply(lambda x: x.day)
    return df2


def binarize(df, category_classes, all_classes=True, drop=True,
             astype=None, inplace=True, min_freq=None):
    """
    Binarize specified categoricals. Works inplace!

    Args:
        - df: the DataFrame whose columns to binarize
        - category_classes: either a dict of (column : [class1, class2, ...]) pairs
            or a collection of column names, in which case classes are
            given using df[column].unique()
        - all_classes: when False, the last class is skipped
        - drop: when True, the original categorical columns are dropped
        - astype: a type for the resulting binaries, e.g. np.float32.
            When None, use the defualt (bool).
        - inplace: whether to modify the DataFrame inplace

    Returns:
        the DataFrame with binarized columns
    """
    if type(category_classes) is not dict:
        columns = set(category_classes)
        category_classes = {column: df[column].unique() for column in columns}
    else:
        columns = category_classes.keys()

    df_new = df if inplace else df.drop(columns, axis=1)

    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1 if not all_classes else len(classes)):
            c = df[category] == classes[i]
            if not min_freq or c.sum() >= min_freq:
                if astype is not None:
                    c = c.astype(astype)
                df_new['%s_%s' % (category, str(classes[i]).replace(' ', '_'))] = c

    if drop and inplace:
        df_new.drop(columns, axis=1, inplace=True)

    return df_new


def binarize_sets(df, columns, cast=False, drop=True, min_freq=None):
    """
    Create dummies for the elements of a set-valued column. Operates in place.
    Args:
        df: data frame
        columns: either a dictionary of column: values pairs or a collection of columns.
        cast: whether or not to cast values to set
        drop: whether or not to drop the binarized columns
    TODO: make interface same as binarize(). merge the two?
    """
    for column in columns:
        d = df[column].dropna()  # avoid nulls
        if cast:
            d = d.apply(set)

        values = columns[column] if isinstance(columns, dict) else util.union(d)
        for value in values:
            name = values[value] if type(values) is dict else str(value)
            column_name = column + '_' + name.replace(' ', '_')
            series = d.apply(lambda c: value in c)
            series.fillna(0, inplace=True)
            if not min_freq or series.sum() >= min_freq:
                df[column_name] = series

    if drop:
        # list(columns) will return keys if columns was dict
        df.drop(list(columns), axis=1, inplace=True)

    return df


def counts_to_dicts(df, column):
    """
    convert (values, counts) as returned by aggregate.aggregate_counts() to dicts
    makes expand_counts much faster
    """
    # index where there are counts and they aren't null
    d = df[column].apply(lambda c: pd.notnull(c) and len(c[0]) > 0)
    return df.loc[d, column].apply(lambda c: {k: v for k, v in zip(*c)})


def expand_counts(df, column, values=None):
    """
    expand a column containing value:count dictionaries
    """
    d = counts_to_dicts(df, column)
    if len(d) > 0:
        if values is None:
            values = set(np.concatenate(d.apply(lambda c: c.keys()).values))
        for value in values:
            name = values[value] if type(values) is dict else str(value)
            df[column + '_' + name.replace(' ', '_')] =\
                d.apply(lambda c: c[value] if value in c else 0)
    df.drop(column, axis=1, inplace=True)


def binarize_clusters(df, column, n_clusters, train=None):
    series = df[column]
    series = series.dropna()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters)

    series = pd.DataFrame(series)
    kmeans.fit(series[train] if train is not None else series)

    clusters = kmeans.cluster_centers_[:, 0].astype(int)
    df[column + '_cluster'] =\
        pd.Series(kmeans.predict(series), index=series.index).apply(lambda d: clusters[d])

    # use all_classes to handle nulls
    binarize(df, {column + '_cluster': clusters}, all_classes=True)

    return df


def train_test_subset(df, train, test, drop=False):
    """
    narrows df to train | test
    then narrows train and test to that
    """
    if drop:
        df.drop(df.index[~(train | test)], inplace=True)
    else:
        df = df[(train | test)]
    train = train.loc[df.index]
    test = test.loc[df.index]

    return df, train, test


def normalize(X, train=None):
    Xfit = X[train] if train is not None else X
    sigma = Xfit.std(ddof=0)
    sigma.loc[sigma == 0] = 1
    mu = Xfit.mean()

    X = (X - mu) / sigma

    return X


class Normalize(Step):
    def run(self, X, train=None):
        return normalize(X, train=train)


def impute(X, value=None, train=None, dropna=True, inplace=True):
    """
    Performs mean imputation on a pandas dataframe.
    Args:
        train: an optional training mask with which to compute the mean
        value: instead of computing the mean, use this as the value argument to fillna
        dropna: whether to drop all null columns
        inplace: whether to perform the imputation inplace
    Returns: the imputed DataFrame
    """
    if value is None:
        Xfit = X[train] if train is not None else X
        value = Xfit.mean()
    else:
        if train is not None:
            raise ValueError("Cannot pass both train and value arguments")

    if dropna:
        null_columns = value.index[value.isnull()]
        if len(null_columns) > 0:
            logging.info('Dropping null columns: \n\t%s' % null_columns)
            if inplace:
                X.drop(null_columns, axis=1, inplace=True)
            else:
                X = X.drop(null_columns, axis=1, inplace=False)

    if inplace:
        X.fillna(value.dropna(), inplace=True)
    else:
        X = X.fillna(value.dropna(), inplace=False)

    return X


def select_regexes(strings, regexes):
    """
    select subset of strings matching a regex
    treats strings as a set
    """
    strings = set(strings)
    select = set()
    if isinstance(strings, collections.Iterable):
        for r in regexes:
            s = set(filter(re.compile('^' + r + '$').search, strings))
            strings -= s
            select |= s
        return select
    else:
        raise ValueError("exclude should be iterable")


def exclude_regexes(strings, exclude, include=None):
    e = select_regexes(strings, exclude)
    i = select_regexes(strings, include) if include is not None else set()
    return set(strings).difference(e).union(i)


def select_features(df, exclude, include=None, inplace=False):
    include = exclude_regexes(strings=df.columns, exclude=exclude, include=include)
    exclude = df.columns.difference(include)
    df2 = df.drop(exclude, axis=1, inplace=inplace)
    return df if inplace else df2


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
        except(ValueError):
            columns.append(c)
    return columns


def non_numeric_columns(df):
    columns = []
    for c in df.columns:
        try:
            df[c].astype(float)
        except(ValueError):
            columns.append(c)

    return columns


def date_censor_sql(date_column, today, column=None):
    """
    if today is None, then no censoring
    otherwise replace each column with:
        CASE WHEN {date_column} < '{today}' THEN {column} ELSE null END
    """
    if column is None:
        column = date_column

    if today is None:
        return column
    else:
        return "(CASE WHEN {date_column} < '{today}' THEN {column} ELSE null END)".format(
                date_column=date_column, today=today, column=column)


# group 1 is the table name, group 2 is the query whose result is the table
extract_sql_regex = r'CREATE\s+TABLE\s+([^(\s]*)\s+AS\s*\(([^;]*)\);'


def revise_helper(query):
    """
    given sql containing a "CREATE TABLE {table_name} AS ({query})"
    returns table_name, query
    """
    match = re.search(extract_sql_regex, query, re.DOTALL | re.I)
    return match.group(1), match.group(2)


def revise_sql(query, id_column, output_table, max_date_column,
               min_date_column, date_column, date, source_id_column=None):
    """
    Given an expensive query that aggregates temporal data,
    Revise the results to censor before a particular date
    """
    if source_id_column is None:
        source_id_column = id_column

    if hasattr(id_column, '__iter__'):
        id_column = str.join(', ', id_column)
    if hasattr(source_id_column, '__iter__'):
        source_id_column = str.join(', ', source_id_column)

    sql_vars = dict(query=query, id_column=id_column, output_table=output_table,
                    max_date_column=max_date_column, min_date_column=min_date_column,
                    date_column=date_column, date=date, source_id_column=source_id_column)

    sql_vars['ids_query'] = """
    SELECT {id_column} FROM {output_table}
    WHERE {max_date_column} >= '{date}' AND {min_date_column} < '{date}'""" .format(**sql_vars)

    sql_vars['revised_query'] = query.replace(
            '1=1',
            "(({source_id_column}) in (select * from ids_query) and {date_column} < '{date}')"
            .format(**sql_vars))

    new_query = """
    with ids_query as ({ids_query})
    select * from ({revised_query}) t
    """.format(**sql_vars)

    return new_query


class Revise(Step):
    def __init__(self, sql, id_column, max_date_column, min_date_column,
                 date_column, date, from_sql_args=None, source_id_column=None, **kwargs):
        """
        revise a query to the specified date
        sql: a path to a file or a string containing sql
        id_column: the entity id column(s) linking the query result with its source tables
        max_date_column: the maximum date column name for an entry in the result
        min_date_column: the minimum date column name for an entry in the result
        date_column: name of the date column in the source
        date: the date to revise at
        from_sql_args: dictionary of keyword arguments to pass input FromSQL steps,
                e.g. target=True, parse_dates
        """

        Step.__init__(self, sql=sql, id_column=id_column,
                      max_date_column=max_date_column, min_date_column=min_date_column,
                      date_column=date_column, date=date, source_id_column=source_id_column,
                      from_sql_args=from_sql_args, **kwargs)

        if os.path.exists(sql):
            self.dependencies = [os.path.abspath(sql)]
            sql = util.read_file(sql)

        table, query = revise_helper(sql)

        revised_sql = revise_sql(
                query=query, id_column=id_column, output_table=table,
                max_date_column=max_date_column, min_date_column=min_date_column,
                date_column=date_column, date=date, source_id_column=source_id_column)

        if from_sql_args is None:
            from_sql_args = {}
        self.inputs = [MapResults(
                    # by depending on table, revised query is given the right dependencies
                    [FromSQL(table=table, **from_sql_args),
                     FromSQL(revised_sql, tables=[table], **from_sql_args)],
                    mapping=['source', 'revised'])]

    def run(self, source, revised):
        subset = (source[self.min_date_column] < self.date) &\
                 (source[self.max_date_column] < self.date)

        # revsied might have bad dtypes because it's small
        # and pandas type inference isn't great
        # if so, convert to source dtypes
        if not revised.dtypes.equals(source.dtypes):
            revised = revised.astype(source.dtypes.to_dict())

        return pd.concat((source[subset], revised), copy=False)


def date_select(df, date_column, date, delta, max_date_column=None):
    """
    given a series an end date and number of days, return subset in the date range
    if delta is None then there is no starting date
    if max_date_column is specified then look for rows where the interval
        [date_column, max_date_column] intersects [date-delta, date+delta)
    """
    delta = parse_delta(delta)
    if delta:
        start_date = date - delta

    if not max_date_column:
        df = df.query("%s < '%s'" % (date_column, date))
        if delta:
            df = df.query("%s >= '%s'" % (date_column, start_date))
    else:
        # event not entirely after
        df = df.query("not ({min} >= '{end}' and {max} >= '{end}')".format(
                          min=date_column, max=max_date_column, end=date))
        if delta:
            # event not entirely before
            df = df.query("not ({min} < '{start}' and {max} < '{start}')".format(
                          min=date_column, max=max_date_column, start=start_date))

    return df


def date_censor(df, date_columns, date):
    """
    a dictionary of date_column: [dependent_column1, ...] pairs
    censor the dependent columns when the date column is before the given end_date
    then censor the date column itself
    """
    for date_column, censor_columns in date_columns.items():
        for censor_column in censor_columns:
            df[censor_column] = df[censor_column].where(df[date_column] < date)

        df[date_column] = df[date_column].where(df[date_column] < date)

    return df


delta_chars = {
        'y': 'years', 'm': 'months', 'w': 'weeks', 'd': 'days', 'h': 'hours',
        'M': 'minutes', 's': 'seconds', 'u': 'microseconds'
}

delta_regex = re.compile('^([0-9]+)(u|s|M|h|d|m|y)$')


def parse_delta(s):
    """
    parse a string to a delta
    'all' is represented by None
    """
    if s == 'all':
        return None
    else:
        ls = delta_regex.findall(s)
        if len(ls) == 1:
            return relativedelta(**{delta_chars[ls[0][1]]: int(ls[0][0])})
        else:
            raise ValueError('Invalid delta string: %s' % s)
