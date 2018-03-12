from .step import Step
from .aggregate import Aggregator
from . import util, data

from itertools import chain
import pandas as pd
import logging


class AggregationBase(Step):
    """
    AggregationBase uses aggregate.Aggregator to aggregate data.
    It can include aggregations over multiple indexes and multiple data
    transformations (e.g. subsets). The combinations can be run in
    parallel and can be returned disjoint or concatenated. Finally
    the results may be pivoted and joined to other datasets.
    """
    def __init__(self, insert_args, aggregator_args, concat_args,
                 parallel=False, prefix=None, inputs=None):
        """
        Args:
            insert_args: collection of argument names to insert
                into results
            aggregator_args: collection of argument names to pass to
                get_aggregator
            concat_args: collection of argument names on which to
                concatenate results. Typically a subset (or equal
                to) aggregator_args.
            parallel: whether to distribute the aggregation over
                many inputs. uses self._parallel_kwargs to determine how
                to distribute.
            prefix: used as a prefix for feature names by join()
        """
        Step.__init__(self,
                      insert_args=insert_args,
                      concat_args=concat_args,
                      aggregator_args=aggregator_args,
                      prefix=prefix,
                      parallel=parallel,
                      inputs=inputs)

        if parallel:
            # create a new Aggregation according to parallel_kwargs
            # pass our input to those steps
            # those become the inputs to this step
            pkwargs = self.get_arguments()
            pkwargs.update(parallel=False)

            self.inputs = []
            for pk in self._parallel_kwargs:
                pkwargs.update(pk)
                a = self.__class__(**pkwargs)
                self.inputs.append(a)

        self._aggregators = {}

        """
        arguments is a list of dictionaries of argument names and values.
        it must include the special 'index' argument, whose values are
        keys to plug into the self.indexes dictionary, whose values are
        the actual index that is passed to Aggregator.aggregate()
        """
    @property
    def argument_names(self):
        return list(util.union(map(set, self.arguments)))

    def args_prefix(self, args):
        prefix = '' if self.prefix is None else self.prefix + '_'
        prefix += str.join('_', map(str, args)) + '_'
        return prefix

    # left join to the specified DataFrame
    # left should contain the index of the concatenated agg in its columns
    def join(self, left):
        fillna_value = pd.Series()
        concat_result = self.get_concat_result()

        # TODO: is it more efficient to first collect indexes from concat
        # then outer join all of the dfs
        # then left join that to left?
        for concat_args, df in concat_result.items():
            # TODO: print warning if df.index.names is not a subset of left.columns
            # and skip this df
            logging.info('Joining %s %s' % (self.prefix, str(concat_args)))
            data.prefix_columns(df, self.args_prefix(concat_args))
            if not set(df.index.names).issubset(left.columns):
                logging.info("Skipping join since aggregation index not in left: %s"
                             % df.index.names)
                continue
            left = left.merge(df, left_on=df.index.names,
                              right_index=True, how='left', copy=False)
            fillna_value = fillna_value.append(self.fillna_value(
                    df=df,
                    left=left,
                    **{k: v for k, v in zip(self.concat_args, concat_args)}))

        logging.info('Filling missing values')
        left.fillna(fillna_value, inplace=True)

        return left

    def fillna_value(self, df, left, **concat_args):
        """
        This method gives subclasses the opportunity to define how
        join() fills missing values. Return value must be compatible with
        DataFrame.fillna() value argument. Examples:
            - return 0: replace missing values with zero
            - return df.mean(): replace missing values with column mean

        This default implimentation fills counts with zero.
        TODO: identify counts more robustly instead of relying on column name

        Typically fill other fields with mean but can't do that during the join
            because that would leak information across a train/test split
        """
        value = pd.Series(
                0, index=[c for c in df.columns
                          if c.endswith('_count') and c.find('_per_') == -1])
        return value

    def select(self, df, args, inplace=False):
        """
        After joining, selects a subset of arguments
        df: the result of a call to self.join(left)
        args: a collcetion of arguments to select, as accepted by drain.util.list_expand:
            - a tuple corresponding to concat_args,
                e.g. [('District', '12h'), ('Distict', '24h')]
            - a dict to be exanded into the above,
                e.g. {'District': ['12h', '24h']}
        """
        if self.prefix is None:
            raise ValueError('Cannot do selection on an Aggregation without a prefix')

        # run list_expand and ensure all args to tuples for validation
        args = [tuple(i) for i in util.list_expand(args)]

        # check that the args passed are valid
        for a in args:
            has_arg = False
            for argument in self.arguments:
                if a == tuple(argument[k] for k in self.concat_args):
                    has_arg = True
                    break
            if not has_arg:
                raise ValueError('Invalid argument for selection: %s' % str(a))

        df = data.select_features(
                df, exclude=[self.prefix + '_.*'],
                include=map(lambda a: self.args_prefix(a) + '.*', args), inplace=inplace)

        return df

    def run(self, *args, **kwargs):
        if self.parallel:
            # use tuple to avoid mapping to positional arguments by step.merge_results()
            return tuple(chain(*args))

        if not self.parallel:
            dfs = []

            for argument in self.arguments:
                logging.info('Aggregating %s %s' % (self.prefix, argument))
                aggregator = self._get_aggregator(**argument)
                df = aggregator.aggregate(self.indexes[argument['index']])

                logging.info('Aggregated %s: %s' % (argument, df.shape))
                # insert insert_args
                for k in argument:
                    if k in self.insert_args:
                        df[k] = argument[k]
                df.set_index(self.insert_args, append=True, inplace=True)
                dfs.append(df)

            return tuple(dfs)

    def load(self):
        # overload load in order to restore result to a tuple
        Step.load(self)
        self.result = tuple(self.result)

    def get_concat_result(self):
        to_concat = {}
        dfs = self.result
        for argument, df in zip(self.arguments, dfs):
            concat_args = tuple(argument[k] for k in self.concat_args)
            if concat_args not in to_concat:
                to_concat[concat_args] = [df]
            else:
                to_concat[concat_args].append(df)
        dfs = {concat_args: pd.concat(dfs, copy=False)
               for concat_args, dfs in to_concat.items()}
        return dfs

    def _get_aggregator(self, **kwargs):
        args_tuple = (kwargs[k] for k in self.aggregator_args)
        if args_tuple in self._aggregators:
            return self._aggregators[args_tuple]
        else:
            aggregator = self.get_aggregator(
                    **util.dict_subset(kwargs, self.aggregator_args))
            self._aggregators[args_tuple] = aggregator
            return aggregator

    def get_aggregator(self, **kwargs):
        """
        Given the arguments, return an aggregator

        This method exists to allow subclasses to use Aggregator objects
        efficiently, i.e. only apply AggregateSeries once per set of
        Aggregates. If the set of Aggregates depends on some or none of
        the arguments the subclass need not recreate Aggregators
        """
        raise NotImplementedError


class AggregationJoin(Step):
    """
    first input is left and second input is aggregation
    if left step returned a dict, use MapResults to clarify e.g.:
        mapping=[{'aux': None}]
    """
    def __init__(self, inputs, **kwargs):
        Step.__init__(self, inputs=inputs, **kwargs)

    def run(self, aggregations, left):
        # aggregations = iter(self.inputs)
        # next(aggregations) # first input is left, not aggregation
        # for aggregation in aggregations:
        left_columns = list(left.columns)
        left = self.inputs[0].join(left)
        left = left.drop(left_columns, axis=1)
        return left


class SpacetimeAggregationJoin(AggregationJoin):
    """
    Specify a temporal lag between the aggregations and left
    Useful for simulating a delay in receipt of aggregation data sources
    """
    def __init__(self, inputs, lag=None, **kwargs):
        AggregationJoin.__init__(self, lag=lag, inputs=inputs, **kwargs)

    def run(self, aggregations, left):
        if self.lag is not None:
            delta = data.parse_delta(self.lag)
            for a in aggregations:
                a.reset_index(level='date', inplace=True)
                a.date = a.date.apply(lambda d: d + delta)
                a.set_index('date', append=True, inplace=True)

        return AggregationJoin.run(self, aggregations, left)


class SimpleAggregation(AggregationBase):
    """
    A simple AggreationBase subclass with a single aggregrator
    The only argument is the index
    An implementation need only define an aggregates attributes, see
    test_aggregation.SimpleCrimeAggregation for an example.
    """
    def __init__(self, inputs, indexes, prefix=None, parallel=False):
        # if indexes was not a dict but a list, make it a dict
        if not isinstance(indexes, dict):
            indexes = {index: index for index in indexes}
        self.indexes = indexes
        self.inputs = inputs

        AggregationBase.__init__(self, insert_args=[], concat_args=['index'],
                                 aggregator_args=[], parallel=parallel, prefix=prefix)

    def get_aggregator(self, **kwargs):
        return Aggregator(self.inputs[0].result, self.aggregates)

    @property
    def _parallel_kwargs(self):
        """
        Returns: a list of kwargs for each parallel input
        """
        return [{'indexes': {name: index}}
                for name, index in self.indexes.items()]

    @property
    def arguments(self):
        return [{'index': name} for name in self.indexes]


class SpacetimeAggregation(AggregationBase):
    """
    SpacetimeAggregation is an Aggregation over space and time.
    Specifically, the index is a spatial index and an additional date and delta argument select
        a subset of the data to aggregate.
    We assume that the index and deltas are independent of the date,
        so every date is aggregated to all spacedeltas
    By default the aggregator_args are date and delta (i.e. independent of aggregation index).
    To change that, pass aggregator_args=['date', 'delta', 'index'] and override get_aggregator
        to accept an index argument.
    Note that dates should be datetime.datetime, not numpy.datetime64, for yaml serialization
        and to work with dateutil.relativedelta.
    However since pandas automatically turns a datetime column in the index into datetime64
        DatetimeIndex, the left dataframe passed to join() should use datetime64!
    See test_aggregation.SpacetimeCrimeAggregation for an example.
    """
    def __init__(self, spacedeltas, dates, date_column, parallel=False, max_date_column=None,
                 censor_columns=None, aggregator_args=None, concat_args=None,
                 inputs=None, prefix=None):
        if aggregator_args is None:
            aggregator_args = ['date', 'delta']
        if concat_args is None:
            concat_args = ['index', 'delta']

        self.censor_columns = censor_columns if censor_columns is not None else {}
        self.date_column = date_column
        self.max_date_column = max_date_column
        self.dates = dates
        self.spacedeltas = spacedeltas

        """
        spacedeltas is a dict of the form {name: (index, deltas)}
            where deltas is an array of delta strings
        dates are end dates for the aggregators
        """
        AggregationBase.__init__(self,
                                 insert_args=['date'],
                                 aggregator_args=aggregator_args,
                                 concat_args=concat_args,
                                 prefix=prefix,
                                 parallel=parallel,
                                 inputs=inputs)

    @property
    def indexes(self):
        return {name: value[0] for name, value in self.spacedeltas.items()}

    @property
    def arguments(self):
        names = list(self.spacedeltas.keys())
        names.sort()

        a = []
        for date in self.dates:
            for name in names:
                for delta in self.spacedeltas[name][1]:
                    a.append({'date': date, 'delta': delta, 'index': name})

        return a

    @property
    def _parallel_kwargs(self):
        return [{'spacedeltas': self.spacedeltas, 'dates': [date]} for date in self.dates]

    def join(self, left):
        # this check doesn't work with lag!
        # TODO: fix by moving Aggregation.join() code to AggregationJoin.sun()
        # difference = set(pd.to_datetime(left.date.unique()))\
        #        .difference(pd.to_datetime(self.dates))
        # if len(difference) > 0:
        #     raise ValueError('Left contains unaggregated dates: %s' % difference)
        return AggregationBase.join(self, left)

    def get_aggregator(self, date, delta):
        df = self.get_data(date, delta)
        aggregator = Aggregator(df, self.get_aggregates(date, delta))
        return aggregator

    def get_data(self, date, delta):
        df = self.inputs[0].result
        df = data.date_select(df, self.date_column, date, delta, self.max_date_column)
        df = data.date_censor(df.copy(), self.censor_columns, date)
        return df

    def get_aggregates(self, date, delta):
        raise NotImplementedError
