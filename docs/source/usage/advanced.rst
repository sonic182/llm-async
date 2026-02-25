Advanced Usage
==============

Custom Base URL
---------------

Override the default API endpoint for any provider:

.. code-block:: python

   from llm_async import OpenAIProvider

   provider = OpenAIProvider(
       api_key="your-api-key",
       base_url="https://custom-openai-endpoint.com/v1"
   )

HTTP/2
------

Enable HTTP/2 on the underlying ``aiosonic.HTTPClient``:

.. code-block:: python

   from llm_async import OpenAIProvider

   provider = OpenAIProvider(
       api_key="your-openai-api-key",
       http2=True,
   )

Combine with ``client_kwargs`` for additional HTTP client options:

.. code-block:: python

   provider = OpenAIProvider(
       api_key="your-openai-api-key",
       http2=True,
       client_kwargs={"timeout": 30, "verify": True},
   )

HTTP/2 is supported by all providers that inherit from ``BaseProvider``:
``OpenAIProvider``, ``OpenRouterProvider``, ``ClaudeProvider``, ``GoogleProvider``,
and ``OpenAIResponsesProvider``.

See also ``examples/openai_responses_repl_http2_prompt_cache.py`` for an interactive
OpenAI Responses REPL that combines HTTP/2 with tool calling, prompt caching, and
``previous_response_id`` conversation chaining.

Direct API Requests
-------------------

Make raw requests to any provider API endpoint using the ``request()`` method.
Supports all HTTP verbs: GET, POST, PUT, DELETE, PATCH.

.. code-block:: python

   import asyncio
   import os
   from llm_async import OpenAIProvider

   async def main():
       provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

       response = await provider.request("GET", "/models")
       print(f"Available models: {len(response.get('data', []))}")

       response = await provider.request("POST", "/endpoint", json_data={"key": "value"})

       response = await provider.request("GET", "/endpoint", custom_header="value")

   asyncio.run(main())

See ``examples/provider_request.py`` for a complete example.
