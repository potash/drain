=====
Usage
=====


Arithmetic Example
------------------

This is a toy example, in which each ``Step`` produces a number.

1. We define a simple ``Step`` that wraps a numeric value:

   ::

       class Scalar(Step):
           def __init__(self, value, **kwargs):

               # note how we do not need to say self.value=value; the parent constructor does that for us
               Step.__init__(self, value=value, **kwargs)

           def run(self):
               return self.value

2. ::

       > s = Scalar(value=5)

   Note that the result of a step's ``run()`` method is accessible via
   ``get_result()``.

3. Steps can use the results of others steps, called ``inputs``. For
   example we can define an ``Add`` step which adds the values of its
   inputs:

   ::

       class Add(Step):
           def __init__(self, inputs):
               Step.__init__(self, inputs=inputs)

           def run(self, *values)
               return sum((i.get_result() for i in self.inputs))

   In order to avoid calling ``get_result()``, drain does so-called
   inputs mapping which is explained in the corresponding section below.
   In its most basic form, inputs mapping allows us to rewrite
   ``Add.run`` as follows:

   ::

       def run(self, *values):
           return sum(values)

   ::

       a = Add(inputs = [Scalar(value=v) for v in range(1,10)])

How does ``drain`` work?
------------------------

``drain`` is a pretty lightweight wrapper around ``drake``; its core
functionality is only a few hundred lines of code.

Steps
-----

A workflow consists of steps, each of which is inherited from the
drain.step.Step class. Each step must implement the ``run()`` method,
whose return value is the ``result`` of the step. A step should be a
deterministic function from its constructor arguments to its result.

Because a step is only a function of its arguments, serialization and
hashing is easy. We use YAML for serialization, and hash the YAML for
hashing. Thus all arguments to a step's constructor should be YAML
serializable.

Design decisions
~~~~~~~~~~~~~~~~



-  ``Step``'s constructor accepts any keyword argument, but does **not**
   accept positional arguments.
-  A ``Step`` can decide to only accept certain keyword arguments by
   defining a custom ``__init__()``.
-  Reserved keyword arguments are ``name``, ``target``, ``inputs``,
   ``inputs_mapping``, and ``resources``. These are handled specifically
   by ``Step.__new__()``.
-  When passing keyword arguments to a ``Step`` constructor, then all
   the arguments (except ``name`` and ``target``) become part of the
   signature (i.e., they will be part of this ``Step``'s serialization).
   Any instance of a ``Step`` automatically has an attribute ``_kwargs``
   holding these arguments.
-  When a ``Step`` does not override ``__init__()`` (i.e., when it uses
   the default ``Step.__init__()``), then all the keyword arguments that
   are being passed become attributes of the new instance. This is a
   mere convenience functionality. It can be overriden simply by
   overriding ``__init__()``, and it does not affect serialization.

Each ``Step`` has several reserved keyword arguments, namely ``target``,
``name,``\ inputs\_mapping\ ``,``\ resources\ ``, and``\ inputs\`.

``name`` and ``target``
~~~~~~~~~~~~~~~~~~~~~~~

``name`` defaults to None and ``target`` to ``False``. ``name`` is a
string and allows you to name your current ``Step``; this is useful
later, when handling the step graph. ``target`` decides if the
``Step``'s output should be cached on disk or not. These two arguments
are *not* serialized.

``inputs``
~~~~~~~~~~

The step attribute ``inputs`` should be a list of input step objects.
Steps appearing in other arguments will not be run correctly. Note that
the ``Step.__init__`` superconstructor automatically assigns all
keywords to object attributes.

Inputs can also be declared within a step's constructor by setting the
``inputs`` attribute.

``inputs_mapping``
~~~~~~~~~~~~~~~~~~

The ``inputs_mapping`` argument to a step allows for convenience and
flexibility in passing that step's inputs' results to the step's
``run()`` method.

Default behavior
^^^^^^^^^^^^^^^^

By default, results are passed as positional arguments. So a step with
``inputs=[a, b]`` will have ``run`` called as

::

    run(a.get_result(), b.get_result())

When a step produces multiple items as the result of run() it is often
useful to name them and return them as a dictionary. Dictionary results
are merged (with later inputs overriding earlier ones?) and passed to
``run`` as keyword arguments. So if inputs ``a`` and ``b`` had
dictionary results with keys ``a_0, a_1`` and ``b_0, b_1``,
respectively, then ``run`` will be called as

::

    run(a_0=a.get_result()['a_0'], a_1=a.get_result()['a_1'],
        b_0=a.get_result()['b_0'], b_1=b.get_result()['b_1'])

Custom behavior
^^^^^^^^^^^^^^^

This mapping of input results to run arguments can be customized when
constructing steps. For example if the results of ``a`` and ``b`` are
objects then specifying

::

    inputs_mapping = ['a', 'b']

will result in the call

::

    run(a=a.get_result(), b=b.get_result()

If ``a`` and ``b`` return dicts then the mapping can be used to change
their keywords or exclude the values:

::

    inputs_mapping = [{'a_0':'alpha_0', 'a_1': None}, {'b_1':'beta_1'}]

will result in the call

::

    run(alpha_0=a.get_result()['a_0'],
        b_0=a.get_result()['b_0'], beta_1=b.get_result()['beta_1'])

where: - ``a_0`` and ``b_1`` have been renamed to ``alpha_0`` and
``alpha_1``, respectively - ``a_1`` has been excluded, and - ``b_0`` has
been preserved.

To ignore the inputs mapping simply define

::

    def run(self, *args, **kwargs):
        results = [i.get_result() for i in self.inputs]
