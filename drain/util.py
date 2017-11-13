import sqlalchemy
import logging
import os
import sys

import numpy as np
import pandas as pd
import pandas.io.sql

from itertools import chain, product
from functools import reduce
from datetime import datetime, timedelta, date

try:
    from repoze.lru import lru_cache
except ImportError:
    from functools import lru_cache

# useful for finding number of days in an interval: (date1 - date2) /day
day = np.timedelta64(1, 'D')


def create_engine():
    return sqlalchemy.create_engine('postgresql://{user}:{pwd}@{host}:5432/{db}'.format(
            host=os.environ['PGHOST'], db=os.environ['PGDATABASE'], user=os.environ['PGUSER'],
            pwd=os.environ['PGPASSWORD']))


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


def parse_dates(df, inplace=True, *args, **kwargs):
    """
    Parse all datetime.date and datetime.datetime columns
    """
    if not inplace:
        df = df.copy()

    for c in df.columns:
        i = df[c].first_valid_index()
        if i is not None and type(df[c].ix[i]) in (date, datetime):
            df[c] = pd.to_datetime(df[c], *args, **kwargs)

    if not inplace:
        return df


def touch(path):
    open(path, 'a').close()
    os.utime(path, None)


def get_subdirs(directory):
    """
    Returns: a list of subdirectories of the given directory
    """
    return [os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def intersect(sets):
    return reduce(lambda a, b: a & b, sets)


def union(sets):
    return reduce(lambda a, b: a | b, sets, set())


def to_float(*args):
    """
    cast numpy arrays to float32
    if there's more than one, return an array
    """
    floats = [np.array(a, dtype=np.float32) for a in args]
    return floats[0] if len(floats) == 1 else floats


def timestamp(year, month, day):
    """
    Convenient constructor for pandas Timestamp
    """
    return pd.Timestamp('%04d-%02d-%02d' % (year, month, day))


epoch = np.datetime64(0, 'ns')


def date_to_days(date):
    """
    Number of days since epoch
    """
    return (date - epoch)/day


def date_ceil(month, day):
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
    return hasattr(object1, attr) and\
        hasattr(object2, attr) and\
        (getattr(object1, attr) == getattr(object2, attr))


def get_attr(name):
    """
    get a class or function by name
    """
    i = name.rfind('.')
    cls = str(name[i+1:])
    module = str(name[:i])

    mod = __import__(module, fromlist=[cls])
    return getattr(mod, cls)


def init_object(name, **kwargs):
    return get_attr(name)(**kwargs)


def randtimedelta(low, high, size):
    d = np.empty(shape=size, dtype=timedelta)
    r = np.random.randint(low, high, size=size)
    for i in range(size):
        d[i] = timedelta(r[i])
    return d


def randdates(start, end, size):
    d = np.empty(shape=size, dtype=datetime)
    r = randtimedelta(0, (end-start).days, size)
    for i in range(size):
        d[i] = start + r[i]
    return d


def mode(series):
    """
    pandas mode is "empty if nothing has 2+ occurrences."
    this method always returns something:
        nan if the series is empty/nan), breaking ties arbitrarily
    """
    if series.notnull().sum() == 0:
        return np.nan
    else:
        return series.value_counts().idxmax()


def get_collinear(df, tol=.1, verbose=False):
    q, r = np.linalg.qr(df)
    diag = r.diagonal()
    if verbose:
        for i in range(len(diag)):
            if np.abs(diag[i]) < tol:
                print(r[:, i])  # TODO print equation with column names!
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


def dict_merge(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def dict_subset(d, keys):
    return {k: d[k] for k in keys if k in d}


def drop_constant_column_levels(df):
    """
    drop the levels of a multi-level column dataframe which are constant
    operates in place
    """
    columns = df.columns
    constant_levels = [i for i, level in enumerate(columns.levels) if len(level) <= 1]
    constant_levels.reverse()

    for i in constant_levels:
        columns = columns.droplevel(i)
    df.columns = columns


def list_expand(d, prefix=None):
    """
    Recursively expand dictionaries into lists
    e.g. list_expand({1:{2:[3,4]}, 5:[6]}) == [(1,2,3), (1,2,4), (5,6)]
    """
    if prefix is None:
        prefix = tuple()
    for k in d:
        if isinstance(d, dict):
            for i in list_expand(d[k], prefix=tuple(chain(prefix, (k,)))):
                yield i
        else:
            yield tuple(chain(prefix, make_list(k)))


def dict_expand(d, prefix=None):
    """
    Recursively expand subdictionaries returning dictionary
    dict_expand({1:{2:3}, 4:5}) = {(1,2):3, 4:5}
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result.update(dict_expand(v, prefix=k))
        else:
            result[k] = v

    if prefix is not None:
        result = {make_tuple(prefix) + make_tuple(k): v
                  for k, v in result.items()}
    return result


def nunique(iterable):
    try:
        return len(set(iterable))
    except TypeError:
        # use equals to count unhashable objects
        unique = []
        for i in iterable:
            if i not in unique:
                unique.append(i)
        return len(unique)


def dict_diff(dicts):
    """
    Subset dictionaries to keys which map to multiple values
    """
    diff_keys = set()

    for k in union(set(d.keys()) for d in dicts):
        values = []
        for d in dicts:
            if k not in d:
                diff_keys.add(k)
                break
            else:
                values.append(d[k])
                if nunique(values) > 1:
                    diff_keys.add(k)
                    break

    return [dict_subset(d, diff_keys) for d in dicts]


def make_list(a):
    return [a] if not type(a) in (list, tuple) else list(a)


def make_tuple(a):
    return (a,) if not type(a) in (list, tuple) else tuple(a)


def get_collection_values(a):
    if not hasattr(a, '__iter__'):
        raise ValueError("Must pass iterable")

    return a.values() if isinstance(a, dict) else a


def dict_product(*d, **kwargs):
    """
    cartesian product of dict whose values are lists
    Args:
        d: dictionary to take product of. multiple dictionaries will first
           be merged by dict_merge
        kwargs: additional kwargs for convenience
    Returns:
        a list of dictionaries with the same keys as d and kwargs
    """
    d = dict(dict_merge(*d), **kwargs)
    holdout = {k: d[k] for k in d if not isinstance(d[k], list)}
    d = {k: d[k] for k in d if k not in holdout}

    items = d.items()
    if len(items) == 0:
        dicts = [{}]
    else:
        keys, values = zip(*items)
        dicts = [dict_filter_none(dict(zip(keys, v))) for v in product(*values)]

    for d in dicts:
        d.update(holdout)

    return dicts


def dict_filter_none(d):
    """
    filter none values from dict
    """
    return {k: v for k, v in d.items() if v is not None}


def list_filter_none(l):
    """
    filter none values from a list
    """
    return [v for v in l if v is not None]


def dict_update_union(d1, d2):
    """
    update a set-valued dictionary
    when key exists, union sets
    """
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


@lru_cache(maxsize=500)
def read_file(filename):
    with open(filename) as f:
        return f.read()


def conditional_join(left, right, left_on, right_on, condition,
                     lsuffix='_left', rsuffix='_right'):
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


class PgSQLDatabase(pandas.io.sql.SQLDatabase):
    # FIXME Schema is pulled from Meta object, shouldn't actually be part of signature!
    def to_sql(self, frame, name, if_exists='fail', index=True,
               index_label=None, schema=None, chunksize=None,
               dtype=None, pk=None, prefixes=None, raise_on_error=True):
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
        replaced = existed and if_exists == 'replace'

        table_name = name
        if schema is not None:
            table_name = schema + '.' + table_name

        if pk is not None and ((not existed) or replaced):
            if isinstance(pk, str):
                pks = pk
            else:
                pks = ", ".join(pk)
            sql = "ALTER TABLE {table_name} ADD PRIMARY KEY ({pks})".format(
                    table_name=table_name, pks=pks)
            self.execute(sql)

        from subprocess import Popen, PIPE, STDOUT

        columns = frame.index.names + list(frame.columns) if index else frame.columns
        columns = str.join(",", map(lambda c: '"' + c + '"', columns))

        sql = "COPY {table_name} ({columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)".format(
                table_name=table_name, columns=columns)
        p = Popen(['psql', '-c', sql], stdout=PIPE, stdin=PIPE, stderr=STDOUT,
                  universal_newlines=True)
        frame.to_csv(p.stdin, index=index)

        psql_out = p.communicate()[0]
        logging.info(psql_out),

        r = p.wait()
        if raise_on_error and (r > 0):
            sys.exit(r)

        return r

    def read_table(self, name, schema=None):
        table_name = name
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


def indent(s, n_spaces=2, initial=True):
    """
    Indent all new lines
    Args:
        n_spaces: number of spaces to use for indentation
        initial: whether or not to start with an indent
    """
    i = ' '*n_spaces
    t = s.replace('\n', '\n%s' % i)
    if initial:
        t = i + t
    return t


def is_instance_collection(c, cls):
    """
    Args:
        c: any object
        cls: a class or a list/tuple of classes
    Returns: True if c is a non-empty collection of objects, each of which
        is an instance of one of the specified classes.
    """
    if not (hasattr(c, '__iter__') and len(c) > 0):
        # make sure it's iterable and not empty
        return False

    cls = make_list(cls)
    for i in get_collection_values(c):
        instance = False
        for cl in cls:
            if isinstance(i, cl):
                instance = True
                break
        if not instance:
            return False

    return True
