import sqlalchemy
import os
import numpy as np
import datetime
import sys
import pandas as pd
from sklearn import preprocessing
from scipy import stats

def create_engine(**kwargs):
    return sqlalchemy.create_engine('postgresql://{user}:{pwd}@{host}:5432/{db}'.format(
            host=os.environ['PGHOST'], db=os.environ['PGDATABASE'], user=os.environ['PGUSER'], pwd=os.environ['PGPASSWORD']))

def execute_sql(sql, engine):
    conn = engine.connect()
    trans = conn.begin()
    conn.execute(sql)
    trans.commit()

def datetime64(year,month,day):
    return np.datetime64( ("%04d" % year) + '-' +  ("%02d" % month) + '-' + ("%02d" % day))

def get_class(name):
    i = name.rfind('.')
    cls = name[i+1:]
    module = name[:i]
    
    mod = __import__(module, fromlist=[cls])
    return getattr(mod,cls)

def prefix_columns(df, prefix, ignore=[]):
    df.columns =  [prefix + c if c not in ignore else c for c in df.columns]

def init_object(name, **kwargs):
    return get_class(name)(**kwargs)

def randtimedelta(low, high, size):
    d = np.empty(shape=size, dtype=datetime.timedelta)
    r = np.random.randint(low, high, size=size)
    for i in range(size):
        d[i] = datetime.timedelta(r[i])
    return d

def randdates(start,end, size):
    d = np.empty(shape=size, dtype=datetime.datetime)
    r = randtimedelta(0, (end-start).days, size)
    for i in range(size):
        d[i] = start + r[i]
    return d

def count_unique(series):
    return series.nunique()

# normalize a dataframes columns
# method = 'normalize': use standard score i.e. (X - \mu) / \sigma
# method = 'percentile': replace with percentile. SLOW
def normalize(df, method):
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
                print r[:,i] # TODO print equation with column names!
    return [df.columns[i] for i in range(len(diag)) if np.abs(diag[i]) < tol]

def drop_collinear(df, tol=.1, verbose=True):
    columns = get_collinear(df, tol=tol)
    if (len(columns) > 0) and verbose:
        print 'Dropping collinear columns: ' + str(columns)
    df.drop(columns, axis=1, inplace=True)
    return df

def cross_join(left, right, lsuffix='_left', rsuffix='_right'):
    left.index = np.zeros(len(left))
    right.index = np.zeros(len(right))
    return left.join(right, lsuffix=lsuffix, rsuffix=rsuffix)

def set_types(df, types_dict):
    for column, dtype in types_dict.iteritems():
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

import tempfile
import pandas.io.sql

class PgSQLDatabase(pandas.io.sql.SQLDatabase):
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

        table_name=name
        if schema is not None:
            table_name = schema + '.' + table_name

        if pk is not None and not existed:
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
        psql_out = p.communicate(input=frame.to_csv(index=index))[0]
        print psql_out.decode(),
        
        r = p.wait()
        if raise_on_error and (r > 0):
            sys.exit(r)

        return r
