from drain.step import Step
from drain.aggregate import Aggregator
from drain import util, data

from itertools import product,chain
import pandas as pd
import logging

class AggregationBase(Step):
    """
    AggregationBase uses aggregate.Aggregator to aggregate data. It can include aggregations over multiple indexes and multiple data transformations (e.g. subsets). The combinations can be run in parallel and can be returned disjointl or concatenated. Finally the results may be pivoted and joined to other datasets.
    """
    def __init__(self, insert_args, aggregator_args, concat_args, 
            parallel=False, target=False, prefix='', **kwargs):
        """
        insert_args is a collection of argument names to insert into the results
        argument names that are not in insert_args will get pivoted
        block_args will run together when parallel=True
        """

        self.insert_args = insert_args
        self.concat_args = concat_args
        self.aggregator_args = aggregator_args
        self.prefix = prefix

        Step.__init__(self, parallel=parallel, target=target and not parallel, **kwargs)

        if parallel:
            inputs = self.inputs if hasattr(self, 'inputs') else []
            self.inputs = []
            # create a new Aggregation according to parallel_kwargs
            # pass our input to those steps
            # those become the inputs to this step
            for kwargs in self.parallel_kwargs:
                a = self.__class__(parallel=False, target=target, inputs=inputs, **kwargs)
                self.inputs.append(a)

        self._aggregators = {}
    
        """
        arguments is a list of dictionaries of argument names and values.
        it must include the special 'index' argument, whose values are keys to plug into the self.indexes dictionary, whose values are the actual index
        the index is used for aggregation its index name is used to prefix the results
        """
        """
        called by __init__ when parallel=True
        to get keyword args to pass to parallelized inputs
        """
    @property
    def argument_names(self):
        return list(util.union(map(set, self.arguments)))

    def concat_args_prefix(self, argument):
        """
        given an agggregator argument, get the corresponding prefix
        """
        concat_args = tuple(argument[k] for k in self.concat_args)
        return str.join('_', map(str, concat_args))

    def join(self, left):
        # this only works if concat is true!
        index = left.index
        
        for prefix, df in self.get_concat_result().iteritems():
            data.prefix_columns(df, prefix + '_')
            left = left.merge(df, left_on=df.index.names, right_index=True, how='left')
        return left

    def run(self,*args, **kwargs):
        if self.parallel:
            return chain(*args)

        if not self.parallel:
            dfs = []

            for argument in self.arguments:
                logging.info('Aggregating %s' % argument)
                aggregator = self._get_aggregator(**argument)
                df = aggregator.aggregate(self.indexes[argument['index']])
                logging.info('Aggregated %s: %s' % (argument, df.shape))
                # insert insert_args
                for k in argument:
                    if k in self.insert_args:
                        df[k] = argument[k]
                df.set_index(self.insert_args, append=True, inplace=True)
                dfs.append(df)

            return dfs

    def get_concat_result(self):
        to_concat = {}
        dfs = self.get_result()
        for argument, df in zip(self.arguments, dfs):
            # use concat_args_prefix as the key because step.run() wants keys to be valid python variable names! maybe need to make that optional...
            concat_args_prefix = self.concat_args_prefix(argument)
            if concat_args_prefix not in to_concat:
                to_concat[concat_args_prefix] = [df]
            else:
                to_concat[concat_args_prefix].append(df)
        dfs = {concat_args_prefix:pd.concat(dfs) 
                for concat_args_prefix,dfs in to_concat.iteritems()}
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

        This method exists to allow subclasses to use Aggregator objects efficiently, i.e. only apply AggregateSeries once per set of Aggregates. If the set of Aggregates depends on some or none of the arguments the subclass need not recreate Aggregators
        """
        raise NotImplementedError

class SimpleAggregation(AggregationBase):
    """
    A simple AggreationBase subclass with a single aggregrator
    The only argument is the index
    An implementation need only define an aggregates attributes, see test_aggregation.SimpleCrimeAggregation for an example.
    """
    def __init__(self, indexes, **kwargs):

        AggregationBase.__init__(self, indexes=indexes, insert_args=[], concat_args=['index'], aggregator_args=[], **kwargs)

        # if indexes was not a dict but a list, make it a dict
        if not isinstance(indexes, dict):
            self.indexes = {index:index for index in indexes}

    def get_aggregator(self, **kwargs):
        return Aggregator(self.inputs[0].get_result(), self.aggregates)

    @property
    def parallel_kwargs(self):
        return [{'indexes': [index]} for index in self.indexes]

    @property
    def arguments(self):
        return [{'index':name} for name in self.indexes]

class SpacetimeAggregation(AggregationBase):
    """
    SpacetimeAggregation is an Aggregation over space and time.
    Specifically, the index is a spatial index and an additional date and delta argument select a subset of the data to aggregate.
    We assume that the index and deltas are independent of the date, so every date is aggregated to all spacedeltas
    By default the aggregator_args are date and delta (i.e. independent of aggregation index).
    To change that, pass aggregator_args=['date', 'delta', 'index'] and override get_aggregator to accept an index argument.
    Note that dates should be datetime.datetime, not numpy.datetime64, for yaml serialization and to work with dateutil.relativedelta.
    However since pandas automatically turns a datetime column in the index into datetime64 DatetimeIndex, the left dataframe passed to join() should use datetime64!
    See test_aggregation.SpacetimeCrimeAggregation for an example.
    """
    def __init__(self, spacedeltas, dates, date_column,
            censor_columns=None, aggregator_args=None, concat_args=None, **kwargs):
        if aggregator_args is None: aggregator_args = ['date', 'delta']
        if concat_args is None: concat_args = ['index', 'delta']

        self.censor_columns = censor_columns if censor_columns is not None else {}
        self.date_column = date_column

        """
        spacedeltas is a dict of the form {name: (index, deltas)} where deltas is an array of delta strings
        dates are end dates for the aggregators
        """
        AggregationBase.__init__(self,
                spacedeltas=spacedeltas, dates=dates, 
                insert_args=['date'], aggregator_args=aggregator_args,
                concat_args=concat_args, **kwargs)

    @property
    def indexes(self):
        return {name:value[0] for name,value in self.spacedeltas.iteritems()}

    @property
    def arguments(self):
        a = []
        for date in self.dates:
            for name,spacedeltas in self.spacedeltas.iteritems():
                for delta in spacedeltas[1]:
                    a.append({'date':date, 'delta': delta, 'index':name})

        return a

    @property
    def parallel_kwargs(self):
        return [{'spacedeltas':self.spacedeltas, 'dates':[date]} for date in self.dates]

    def get_aggregator(self, date, delta):
        df = self.get_data(date, delta)
        aggregator = Aggregator(df, self.get_aggregates(date, delta))
        return aggregator

    def get_data(self, date, delta):
        df = self.inputs[0].get_result()
        df = data.date_select(df, self.date_column, date, delta)
        df = data.date_censor(df, self.censor_columns, date)
        return df

    def get_aggregates(self, date, delta):
        raise NotImplementedError
