Basic Chat Completion
=====================

Use :class:`~llm_async.models.message.Message` to build messages — it provides role
validation and type safety. Plain dicts are also accepted by all providers.

OpenAI
------

.. code-block:: python

   import asyncio
   from llm_async import OpenAIProvider
   from llm_async.models.message import Message

   async def main():
       provider = OpenAIProvider(api_key="your-openai-api-key")
       response = await provider.acomplete(
           model="gpt-4o-mini",
           messages=[
               Message("system", "You are a helpful assistant."),
               Message("user", "Hello, how are you?"),
           ]
       )
       print(response.main_response.content)

   asyncio.run(main())

OpenRouter
----------

.. code-block:: python

   import asyncio
   import os
   from llm_async import OpenRouterProvider
   from llm_async.models.message import Message

   async def main():
       provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
       response = await provider.acomplete(
           model="openrouter/auto",
           messages=[
               Message("system", "You are a helpful assistant."),
               Message("user", "Hello, how are you?"),
           ],
           http_referer="https://github.com/your-username/your-app",
           x_title="My AI App"
       )
       print(response.main_response.content)

   asyncio.run(main())

Google Gemini
-------------

.. code-block:: python

   import asyncio
   from llm_async.providers.google import GoogleProvider
   from llm_async.models.message import Message

   async def main():
       provider = GoogleProvider(api_key="your-google-gemini-api-key")
       response = await provider.acomplete(
           model="gemini-2.5-flash",
           messages=[
               Message("user", "Hello, how are you?"),
           ]
       )
       print(response.main_response.content)

   asyncio.run(main())
