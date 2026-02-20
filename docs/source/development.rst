Development
===========

Setup
-----

.. code-block:: bash

   git clone https://github.com/sonic182/llm-async.git
   cd llm-async
   poetry install

Running Tests
-------------

.. code-block:: bash

   poetry run pytest -p no:sugar

Running a single test:

.. code-block:: bash

   poetry run pytest -k "test_name"

Linting and Formatting
-----------------------

.. code-block:: bash

   poetry run ruff check llm_async tests
   poetry run ruff format llm_async tests

Building
--------

.. code-block:: bash

   poetry build

Building the Docs
-----------------

.. code-block:: bash

   poetry run sphinx-build -b html docs/source docs/build/html

Roadmap
-------

- Support for additional providers (e.g., Grok, Anthropic direct API)
- More advanced tool features
- Response caching and retry mechanisms
