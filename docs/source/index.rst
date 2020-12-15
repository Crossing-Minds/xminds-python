xminds
======

Crossing Minds Data Science Python Library and API client.


Contents
--------

.. autosummary::
   :toctree: generated
   :recursive:

   xminds.api
   xminds.ds
   xminds.lib


Source code
-----------

See our `GitHub repository <https://github.com/Crossing-Minds/xminds-python>`_ for the source code.


About Crossing Minds
--------------------
We are a start-up based in San Francisco & with offices in Paris. We specialize in recommender systems and are currently building a universal recommendation API, allowing anyone to harness the power of recommendations.

Along the way, we developed some useful and efficient tools for general Machine Learning and optimisation, for recommender systems specifically, and some tools for data wrangling. We believe these tools should be for anyone to use, so please feel free to use this library for any of your Machine Learning and data manipulation needs.

Please visit `our website <https://crossingminds.com/>`_ if you want to learn more!

Getting started
---------------

Requires Python >= 3.6 to function properly.

Installing the package
~~~~~~~~~~~~~~~~~~~~~~

Using pip:

.. code:: sh

   pip install xminds

Not supported on Conda yet.

Using the library
~~~~~~~~~~~~~~~~~

The library currently requires using exact paths, so to use xminds.lib.arrays.to_structured the recommended way is currently:

.. code:: python

   from xminds.lib.arrays import to_structured
   arr = to_structured([
      ('a', numpy.arange(5)),
      ('b', numpy.ones(5))
   ])

Our next release will allow the following pattern:

.. code:: python

   import xminds  # not supported in current release
   arr = xminds.to_structured([  # not supported in current release
      ('a', numpy.arange(5)),
      ('b', numpy.ones(5))
   ])

Documentation
-------------

This website is the documentation for the Python library and the Python client of the API.

For our universal recommendation API’s documentation, see: `API
Documentation`_.

Contributing
------------

Any contributions are *greatly appreciated*!

Reporting issues
~~~~~~~~~~~~~~~~

Please open an issue on the GitHub repository, `here`_. Such issues are
extremely valuable to us.

Making changes
~~~~~~~~~~~~~~

If you like what we do and would like to improve it, feel free to
contribute.

1. Clone the repository:

.. code:: sh

     git clone https://github.com/Crossing-Minds/xminds-python.git

2. Install prerequisites:

.. code:: sh

   pip install -r requirements.txt

3. Create a branch and make additions / changes there
4. Open a Pull Request on GitHub from your branch to master


Release notes
-------------

So far we have released array tools (numpy array, numpy structured array).

We plan on publishing some of our Recommender system tools and utils as well as some Gaussian Processes optimisation tools and Linear Algebra tools, so stay tuned!


License
-------

Our tools are licensed under the MIT License. See `License`_ for more
detail.

Contacting us
-------------

For code-related issues, please open issues on the GitHub repository.

To request new features or functions, you may also open an issue on the
GitHub repository.

You can also write to us at contact [at] crossingminds.com for business
/ hiring.

Keeping in touch
~~~~~~~~~~~~~~~~

Follow us on `Twitter`_, `LinkedIn`_. We also organize meetups (remote
at the moment, in person once it’s safe to do so again), so follow us on
`meetup`_!

Hiring
~~~~~~

We are always looking for great talent! You can check out our
`LinkedIn`_ and `AngeList`_ pages for openings, or contact us directly
at contact [at] crossingminds.com for spontaneous candidatures.


.. _License: License

.. _our website: https://crossingminds.com/
.. _readthedocs: https://xminds.readthedocs.io/en/latest/
.. _API Documentation: https://docs.api.crossingminds.com/
.. _Twitter: https://twitter.com/crossing_minds
.. _LinkedIn: https://www.linkedin.com/company/crossing-minds/
.. _meetup: https://www.meetup.com/Events-at-Crossing-Minds/
.. _AngeList: https://angel.co/company/crossing-minds
.. _here: https://github.com/Crossing-Minds/xminds-python/issues

.. |Crossing Minds| image:: https://static.crossingminds.com/img/logo.png
   :target: https://crossingminds.com
