Structured Outputs
==================

Enforce JSON schema validation on model responses for consistent, type-safe outputs.

Supported providers: **OpenAI**, **Google Gemini**, **OpenRouter**.
Claude does not support structured outputs.

.. code-block:: python

   import asyncio
   import json
   from llm_async import OpenAIProvider
   from llm_async.models.response_schema import ResponseSchema
   from llm_async.providers.google import GoogleProvider

   response_schema = ResponseSchema(
       schema={
           "type": "object",
           "properties": {
               "answer": {"type": "string"},
               "confidence": {"type": "number"}
           },
           "required": ["answer", "confidence"],
           "additionalProperties": False,
       }
   )

   async def main():
       openai_provider = OpenAIProvider(api_key="your-openai-key")
       response = await openai_provider.acomplete(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "What is the capital of France?"}],
           response_schema=response_schema
       )
       result = json.loads(response.main_response.content)
       print(f"OpenAI: {result}")

       google_provider = GoogleProvider(api_key="your-google-key")
       response = await google_provider.acomplete(
           model="gemini-2.5-flash",
           messages=[{"role": "user", "content": "What is the capital of France?"}],
           response_schema=response_schema
       )
       result = json.loads(response.main_response.content)
       print(f"Gemini: {result}")

   asyncio.run(main())
