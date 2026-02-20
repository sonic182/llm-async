Quickstart
==========

Basic completion with OpenAI:

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider
   from llm_async.models.message import Message

   async def main():
       provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
       response = await provider.acomplete(
           model="gpt-4o-mini",
           messages=[
               Message("system", "You are a helpful assistant."),
               Message("user", "Give me 3 ideas for a CLI tool."),
           ],
       )
       print(response.main_response.content)

   asyncio.run(main())

:class:`~llm_async.models.message.Message` is the preferred way to build messages.
Plain dicts are also accepted.

Streaming tokens as they arrive:

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider
   from llm_async.models.message import Message

   async def main():
       provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
       async for chunk in await provider.acomplete(
           model="gpt-4o-mini",
           messages=[Message("user", "Give me 3 ideas for a CLI tool.")],
           stream=True,
       ):
           print(chunk.delta, end="", flush=True)

   asyncio.run(main())
