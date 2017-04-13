drain
=====

|image0| |image1| |image2| |image3| |image4|

Drain is a lightweight framework for writing reproducible data science workflows in Python. The core features are:

 - Turn a Python workflow ([DAG]()) into steps that can be run by a tool like `make`.
 - Transparently pass the results of one step as the input to another, handling any caching that the user requests using efficient tools like [HDF]() and [joblib]().
 - Enable easy parallel execution of workflows.
 - Execute only those steps that are determined to be necessary based on timestamps (both source code and data) and dependencies, virtually guaranteeing *reproducibility* of results and efficient development.

It is designed around these principles:

 - Simplicity: drain is very lightweight and easy to use. The core is just a few hundred lines of code. The steps you write in drain get executed with minimal overhead, making drain workflows easy to debug and manage.

 - Reusability: Drain leverages mature tools [drake]() to execute workflows, giving us easy and robust parallelism for free. Drain provides a library of steps for data science workflows including feature generation and selection, model fitting and comparison.

 - Generality: Virtually any workflow can be realized in drain. The core was written with extensibility in mind so new storage backends and job schedulers, for example, will be easy to incorporate.


.. |image0| image:: https://img.shields.io/pypi/v/drain.svg
   :target: https://pypi.python.org/pypi/drain
.. |image1| image:: https://api.travis-ci.org/dssg/drain.svg
   :target: https://travis-ci.org/dssg/drain
.. |image2| image:: https://readthedocs.org/projects/drain/badge/?version=latest
   :target: https://drain.readthedocs.io/en/latest/?badge=latest
.. |image3| image:: https://pyup.io/repos/github/potash/drain/shield.svg
   :target: https://pyup.io/repos/github/dssg/drain/
.. |image4| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
