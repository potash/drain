# drain

## Introduction

Drain integrates Python machine learning tasks with 
[drake](https://github.com/Factual/drake), resulting in a robust and efficient 
machine learning pipeline. Drain additionally provides a library of methods 
for both the processing data going into the pipeline and exploration of models 
coming out of the pipeline.

`drake` is a useful data science pipeline tool because it allows the user to 
define dependencies between files (usually the inputs and outputs of data processing
steps) in a simple-to-read bash file. 
Once the dependency graph (a DAG) is defined, `drake` can take care of running 
all the steps necessary to produce a desired step's output. `drake` also looks out 
for changes to files (by inspecting their timestamp); if a file gets updated,
`drake` can re-run only those steps that are children of said file.

While highly useful, our data science pipelines frequently require features 
that `drake` does not provide. Thus, while `drain` relies on `drake` 
behind the scenes, it offers a different and extended interface:

1. In `drain`, a step does not need to produce a file. This way, you can
 define a DAG of steps, and decide later which parts of it should be cached,
 and which steps should always re-run. This allows the user to think of steps 
 as natural units of work (instead of as file-producing units of work). For 
 example, you might have a step that performs a costly imputation (and which 
 you thus would like to cache), and another step that drops some columns from 
 your dataframe (and which you do not want to cache, as that would be a waste 
 of disk space). In `drain`, these two steps follow the same template, and 
 caching is enabled by passing a flag to `drain`.
 
2. `drain` steps are written in Python.

3. `drain` takes care of filepaths behind the scenes. The user does not define
 dependencies between files, but rather dependencies between steps. This is 
 extremely useful when your pipeline has to handle a large and variable 
 amount of files, e.g. when you iterate over many different models 
 or feature subsets.

4. Finally, `drain` keeps track not only of file timestamps, 
 but also of each step's parameters (it's 'signature'). If you change some 
 setting in one of your data processing steps, then `drain` will run all
 those follow-up steps that are affected by this change.

## Inputs Mapping

### Arithmetic

This is a toy example, in which each `Step` produces a number.

1. We define a simple `Step` that wraps a numeric value:
	```
	class Scalar(Step):
		def __init__(self, value, **kwargs):
			
			# note how we do not need to say self.value=value; the parent constructor does that for us
			Step.__init__(self, value=value, **kwargs)

		def run(self):
			return self.value
	```

2.	
	``` > s = Scalar(value=5)
 	```

	Note that the result of a step's `run()` method is accessible via `get_result()`.

3. Steps can use the results of others steps, called `inputs`. For example we can define an `Add` step which adds the values of its inputs:
	```
	class Add(Step):
		def __init__(self, inputs):
			Step.__init__(self, inputs=inputs)

		def run(self, *values)
			return sum((i.get_result() for i in self.inputs))
	```

	In order to avoid calling `get_result()`, drain does so-called inputs mapping which is explained in the corresponding section below. In its most basic form, inputs mapping allows us to rewrite `Add.run` as follows:

	```
	def run(self, *values):
		return sum(values)
	```

	```
	a = Add(inputs = [Scalar(value=v) for v in range(1,10)])
	```
	
## How does `drain` work?

`drain` is a pretty lightweight wrapper around `drake`; its core functionality 
are only a few hundred lines of code.


## Steps

A workflow consists of steps, each of which are inherit from the drain.step.Step class.  Each step must implement the `run()` method, whose return value is the `result` of the step. A step should be a deterministic function from its constructor arguments to its result.

### Serialization

Because a step is only a function of its arguments, serialization and hashing is easy. We use YAML for serialization, and hash the YAML for hashing. Thus all arguments to a step's constructor should be YAML serializable.

### Arguments

There are two kinds of arguments to a step: input steps, and everything else.

## Inputs

The step attribute `inputs` should be a list of input step objects. Steps appearing in other arguments will not be run correctly. Inputs can be specified in two ways.

### Passed inputs

Inputs can be passed through the constructor argument keyword `inputs`. Note that the `Step.__init__` superconstructor automatically assigns all keywords to object attributes.

### Declared inputs

Inputs can be declared within a step's constructor by setting the `inputs` attribute.

## Inputs mapping

The `inputs_mapping` argument to a step allows for convenience and flexibility in passing that step's inputs' results to the step's `run()` method.

### Default behavior

By default, results are passed as positional arguments. So a step with `inputs=[a, b]` will have `run` called as
```
run(a.get_result(), b.get_result())
```

When a step produces multiple items as the result of run() it is often useful to name them and return them as a dictionary. Dictionary results are merged (with later inputs overriding earlier ones?) and passed to `run` as keyword arguments. So if inputs `a` and `b` had dictionary results with keys `a_0, a_1` and `b_0, b_1`, respectively, then `run` will be called as

```
run(a_0=a.get_result()['a_0'], a_1=a.get_result()['a_1'],
    b_0=a.get_result()['b_0'], b_1=b.get_result()['b_1'])
```

### Custom behavior
This mapping of input results to run arguments can be customized when constructing steps. For example if the results of `a` and `b` are objects then specifying
```
inputs_mapping = ['a', 'b']
```
will result in the call
```
run(a=a.get_result(), b=b.get_result()
```
If `a` and `b` return dicts then the mapping can be used to change their keywords or exclude the values:
```
inputs_mapping = [{'a_0':'alpha_0', 'a_1': None}, {'b_1':'beta_1'}]
```
will result in the call
```
run(alpha_0=a.get_result()['a_0'],
    b_0=a.get_result()['b_0'], beta_1=b.get_result()['beta_1'])
```
where:
- `a_0` and `b_1` have been renamed to `alpha_0` and `alpha_1`, respectively
- `a_1` has been excluded, and
- `b_0` has been preserved.

To ignore the inputs mapping simply define
```
def run(self, *args, **kwargs):
    results = [i.get_result() for i in self.inputs]
```

## Execution

Given a collection of steps, drain executes them by generating a temporary Drakefile for them and then calling `drake`.

## Exploration

## metrics

## Future improvements
option to store in db instead of files
