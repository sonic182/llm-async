xAI Responses Endpoint
======================

``OpenAIResponsesProvider`` can be used with xAI's Responses-compatible endpoint.

Reference: `xAI text generation docs <https://docs.x.ai/developers/model-capabilities/text/generate-text>`_

- Endpoint: ``https://api.x.ai/v1/responses``
- Provider ``base_url``: ``https://api.x.ai/v1`` (``/responses`` is appended automatically)
- API key: use an xAI-issued key (for example ``XAI_API_KEY``), not an OpenAI key

.. code-block:: python

   from llm_async.providers import OpenAIResponsesProvider

   provider = OpenAIResponsesProvider(
       api_key="YOUR_XAI_API_KEY",
       base_url="https://api.x.ai/v1",
   )

See ``examples/xai_responses.py`` for a runnable example that loads
``X_AI_API_KEY`` from ``.env``.
