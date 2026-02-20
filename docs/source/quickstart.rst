Quickstart
==========

Basic completion with OpenAI:

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider

   async def main():
       provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
       response = await provider.acomplete(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Give me 3 ideas for a CLI tool."}],
       )
       print(response.main_response.content)

   asyncio.run(main())

Streaming tokens as they arrive:

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider

   async def main():
       provider = OpenAIProvider(api_key="YOUR_OPENAI_API_KEY")
       async for chunk in await provider.acomplete(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Give me 3 ideas for a CLI tool."}],
           stream=True,
       ):
           print(chunk.delta, end="", flush=True)

   asyncio.run(main())
