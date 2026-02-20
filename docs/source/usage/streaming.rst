Streaming
=========

All providers support streaming via ``stream=True``. The ``acomplete`` call returns an async iterator
that yields :class:`~llm_async.models.StreamChunk` objects as tokens arrive.

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider

   async def main():
       provider = OpenAIProvider(api_key="your-openai-api-key")
       async for chunk in await provider.acomplete(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Give me a recipe for tortilla española."}],
           stream=True,
       ):
           print(chunk.delta, end="", flush=True)

   asyncio.run(main())

The same pattern works for all providers: ``ClaudeProvider``, ``GoogleProvider``, ``OpenRouterProvider``,
and ``OpenAIResponsesProvider``.

Example streaming output across providers::

   --- OpenAI streaming response ---
   1. Peel and slice potatoes.
   2. Par-cook potatoes briefly.
   3. Whisk eggs with salt and pepper.
   4. Sauté onions until translucent (optional).
   5. Combine potatoes and eggs in a pan and cook until set.
   6. Fold and serve.
   --- Claude streaming response ---
   1. Prepare potatoes by peeling and slicing.
   2. Fry or boil until tender.
   3. Beat eggs and season.
   4. Mix potatoes with eggs and cook gently.
   5. Serve warm.

See ``examples/stream_all_providers.py`` for a runnable multi-provider streaming demo.
