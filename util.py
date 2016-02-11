import sqlalchemy
import logging
import os
import sys
import yaml

import numpy as np
import pandas as pd

from itertools import chain, product
from datetime import datetime, timedelta, date
from sklearn import preprocessing
from scipy import stats

# useful for finding number of days in an interval: (date1 - date2) /day
day = np.timedelta64(1, 'D')

def create_engine():
    return sqlalchemy.create_engine('postgresql://{user}:{pwd}@{host}:5432/{db}'.format(
            host=os.environ['PGHOST'], db=os.environ['PGDATABASE'], user=os.environ['PGUSER'], pwd=os.environ['PGPASSWORD']))

def create_db():
    engine = create_engine()
    return PgSQLDatabase(engine)

def execute_sql(sql, engine):
    conn = engine.connect()
    trans = conn.begin()
    conn.execute(sql)
    trans.commit()

def mtime(path):
    return datetime.fromtimestamp(os.stat(path).st_mtime)

def touch(path):
    open(path, 'a').close()
    os.utime(path, None)

def intersect(sets):
    return reduce(lambda a,b: a & b, sets)

def union(sets):
    return reduce(lambda a,b: a | b, sets)

# cast numpy arrays to float32
# if there's more than one, return an array
def to_float(*args):
    floats = [np.array(a, dtype=np.float32) for a in args]
    return floats[0] if len(floats) == 1 else floats

# a fake hash that detects references to the same object
# used by Aggregator for minimzing computations
def hash_obj(obj):
    try:
        return hash((True, obj))
    except: 
        return hash((False, id(obj)))

def timestamp(year,month,day):
    """
    Convenient constructor for pandas Timestamp
    """
    return pd.Timestamp('%04d-%02d-%02d' % (year, month, day))

epoch = pd.Timestamp(0)
def date_to_days(date):
    """
    Number of days since epoch
    """
    return (date - epoch)/day

def date_ceil( month, day):
    def ceil(t):
        c = timestamp(t.year, month, day)
        return c if c >= t else timestamp(t.year+1, month, day)

    return ceil

def date_floor(month, day):
    def floor(t):
        f = timestamp(t.year, month, day)
        return f if f <= t else timestamp(t.year-1, month, day)

    return floor

def eqattr(object1, object2, attr):
    return hasattr(object1, attr) and hasattr(object2, attr) and (getattr(object1, attr) == getattr(object2, attr))

# get a class or function by name
def get_attr(name):
    i = name.rfind('.')
    cls = str(name[i+1:])
    module = str(name[:i])
    
    mod = __import__(module, fromlist=[cls])
    return getattr(mod,cls)

def init_object(name, **kwargs):
    return get_attr(name)(**kwargs)

def randtimedelta(low, high, size):
    d = np.empty(shape=size, dtype=timedelta)
    r = np.random.randint(low, high, size=size)
    for i in range(size):
        d[i] = timedelta(r[i])
    return d

def randdates(start,end, size):
    d = np.empty(shape=size, dtype=datetime)
    r = randtimedelta(0, (end-start).days, size)
    for i in range(size):
        d[i] = start + r[i]
    return d

# pandas mode is "empty if nothing has 2+ occurrences."
# this method always returns something (nan if the series is empty/nan), breaking ties arbitrarily
def mode(series):
    if series.notnull().sum() == 0:
        return np.nan
    else:
        return series.value_counts().idxmax()

# normalize a dataframes columns
# method = 'normalize': use standard score i.e. (X - \mu) / \sigma
# method = 'percentile': replace with percentile. SLOW
def normalize(df, method='standard'):
    if method == 'standard':
        return pd.DataFrame(preprocessing.scale(df), index=df.index, columns=df.columns)
    elif method == 'percentile':
        return df.rank(pct=True)

def get_collinear(df, tol=.1, verbose=False):
    q, r = np.linalg.qr(df)
    diag = r.diagonal()
    if verbose:
        for i in range(len(diag)):
            if np.abs(diag[i]) < tol:
                print(r[:,i]) # TODO print equation with column names!
    return [df.columns[i] for i in range(len(diag)) if np.abs(diag[i]) < tol]

def drop_collinear(df, tol=.1, verbose=True):
    columns = get_collinear(df, tol=tol)
    if (len(columns) > 0) and verbose:
        logging.info('Dropping collinear columns: ' + str(columns))
    df.drop(columns, axis=1, inplace=True)
    return df

def cross_join(left, right, lsuffix='_left', rsuffix='_right'):
    left.index = np.zeros(len(left))
    right.index = np.zeros(len(right))
    return left.join(right, lsuffix=lsuffix, rsuffix=rsuffix)

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def dict_subset(d, keys):
    return {k:d[k] for k in keys if k in d}


def list_expand(d, prefix=None):
    """
    Recursively expand dictionaries into lists
    e.g. list_expand({1:{2:[3,4]}, 5:[6]}) == [(1,2,3), (1,2,4), (5,6)]
    """
    if prefix is None:
        prefix = tuple()
    for k in d:
        if isinstance(d, dict):
            for i in list_expand(d[k], prefix=list(chain(prefix, (k,)))):
                yield i
        else:
            yield list(chain(prefix, make_list(k)))


def nunique(iterable):
    try:
        return len(set(iterable))
    except TypeError:
        logging.info('unhashable!')
        unique = []
        for i in iterable:
            if i not in unique:
                unique.append(i)
        return len(i)

# When multilevel is true, only look for diffs within subkeys
# TODO add tests to clarify this
def diff_dicts(dicts, multilevel=True):
    diffs = [{} for d in dicts]

    if multilevel:
        keys = dict()
        for d in union(set(d.keys()) for d in dicts):
            if d[0] in keys:
                keys[d[0]].add(d)
            else:
                keys[d[0]] = {d}

        intersection = dict()
        for k0 in keys:
            intersection.update({k: nunique(d[k] for d in dicts if k in d) for k in keys[k0]})

    else:
        keys = map(lambda d: set(d.keys()), dicts)
        intersection = {k: nunique(d[k] for d in dicts) for k in intersect(keys)}

    for diff, d in zip(diffs, dicts):
        for k in d:
            if k not in intersection or intersection[k] > 1:
                diff[k] = d[k]

    return diffs

def make_list(a):
    return [a] if not type(a) in (list, tuple) else a

# cartesian product of dict whose values are lists
# if product_keys is not None then take product on those keys only
def dict_product(d, product_keys=None):
    if product_keys is not None:
        holdout = {k:d[k] for k in d if k not in product_keys}
        d = {k:d[k] for k in d if k in product_keys}
        
    if not isinstance(d, dict):
        raise ValueError('Expected dictionary got %s' % type(d).__name__)
        
    items = d.items()
    if len(items) == 0:
        dicts =  [{}]
    else:
        keys, values = zip(*items)
        dicts = [dict_filter_none(dict(zip(keys, v))) for v in product(*values)]
    
    if product_keys is not None:
        for d in dicts:
            d.update(holdout)
            
    return dicts

# filter none values from dict
def dict_filter_none(d):
    return {k:v for k,v in d.iteritems() if v is not None}

# update a set-valued dictionary
# when key exists, union sets
def dict_update_union(d1, d2):
    for k in d2:
        if k in d1:
            d1[k].update(d2[k])
        else:
            d1[k] = d2[k]

def set_dtypes(df, dtypes):
    for column in df.columns:
        dtype = None
        if isinstance(dtypes, dict):
            if column in dtypes:
                dtype = dtypes[column]
        else:
            dtype = dtypes
        
        if dtype is not None and df[column].dtype != dtype:
            df[column] = df[column].astype(dtype)
    
def conditional_join(left, right, left_on, right_on, condition, lsuffix='_left', rsuffix='_right'):
    left_index = left[left_on].reset_index()
    right_index = right[right_on].reset_index()
    
    join_table = cross_join(left_index, right_index, lsuffix=lsuffix, rsuffix=rsuffix)
    join_table = join_table[condition(join_table)]
    
    lindex = left.index.name if left.index.name is not None else 'index'
    rindex = left.index.name if right.index.name is not None else 'index'
    if lindex == rindex:
        lindex = lindex + lsuffix
        rindex = rindex + rsuffix
    
    df = left.merge(join_table[[lindex, rindex]], left_index=True, right_on=lindex)
    df = df.merge(right, left_on=rindex, right_index=True)
    df.drop(labels=[lindex, rindex], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def join_years(left, years, period=None, column='year'):
    years = pd.DataFrame({column:years})
    if period is None:
        cond = lambda df: (df[column + '_left'] <= df[column + '_right'])
    else:
        cond = lambda df: (df[column + '_left'] <= df[column + '_right']) & (df[column +'_left'] > df[column + '_right'] - period)
        
    df = conditional_join(left, years, left_on=[column], right_on=[column], condition=cond)
    df.rename(columns={column + '_y': column}, inplace=True)
    return df

import pandas.io.sql
class PgSQLDatabase(pandas.io.sql.SQLDatabase):
    import tempfile
    # FIXME Schema is pulled from Meta object, shouldn't actually be part of signature!
    def to_sql(self, frame, name, if_exists='fail', index=True,
               index_label=None, schema=None, chunksize=None, dtype=None, pk=None, prefixes=None, raise_on_error=True):
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type.
        pk: name of column(s) to set as primary keys
        """
        table = pandas.io.sql.SQLTable(name, self, frame=frame, index=index,
                                       if_exists=if_exists, index_label=index_label,
                                       schema=schema, dtype=dtype)
        existed = table.exists()
        table.create()
        replaced = existed and if_exists=='replace'

        table_name=name
        if schema is not None:
            table_name = schema + '.' + table_name

        if pk is not None and ( (not existed) or replaced):
            if isinstance(pk, str):
                pks = pk
            else:
                pks = ", ".join(pk)
            sql = "ALTER TABLE {table_name} ADD PRIMARY KEY ({pks})".format(table_name=table_name, pks=pks)
            self.execute(sql)


        from subprocess import Popen, PIPE, STDOUT

        columns = frame.index.names + list(frame.columns) if index else frame.columns
        columns = str.join(",", map(lambda c: '"' + c + '"', columns))

        sql = "COPY {table_name} ({columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)".format(table_name=table_name, columns=columns)
        p = Popen(['psql', '-c', sql], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        frame.to_csv(p.stdin, index=index)

        psql_out = p.communicate()[0]
        logging.info(psql_out.decode()),
        
        r = p.wait()
        if raise_on_error and (r > 0):
            sys.exit(r)

        return r

    def read_table(self, name, schema=None):
        table_name=name
        if schema is not None:
            table_name = schema + '.' + table_name

        return self.read_query('select * from %s' % table_name)

    def read_sql(self, query, raise_on_error=True, **kwargs):
        from subprocess import Popen, PIPE, STDOUT

        sql = "COPY (%s) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)" % query
        p = Popen(['psql', '-c', sql], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        df = pd.read_csv(p.stdout, **kwargs)

        psql_out = p.communicate()
        logging.info(psql_out[0].decode(),)

        r = p.wait()
        if raise_on_error and (r > 0):
            sys.exit(r)
 
        return df

