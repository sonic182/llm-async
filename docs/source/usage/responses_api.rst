OpenAI Responses API with Prompt Caching
=========================================

``OpenAIResponsesProvider`` uses OpenAI's Responses API for stateless conversation management.
Use ``previous_response_id`` to continue a conversation without resending full history,
and ``prompt_cache_key`` to route requests to the same machine for consistent cache hits.

.. code-block:: python

   import asyncio
   import uuid
   from llm_async.models import Tool
   from llm_async.models.message import Message
   from llm_async.providers import OpenAIResponsesProvider

   calculator_tool = Tool(
       name="calculator",
       description="Perform basic arithmetic operations",
       parameters={
           "type": "object",
           "properties": {
               "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
               "a": {"type": "number"},
               "b": {"type": "number"}
           },
           "required": ["operation", "a", "b"]
       }
   )

   def calculator(operation: str, a: float, b: float) -> float:
       if operation == "add":
           return a + b
       elif operation == "subtract":
           return a - b
       elif operation == "multiply":
           return a * b
       elif operation == "divide":
           return a / b
       return 0

   async def main():
       provider = OpenAIResponsesProvider(api_key="your-openai-api-key")
       session_id = uuid.uuid4().hex

       response = await provider.acomplete(
           model="gpt-4.1",
           messages=[Message("user", "What is 15 + 27? Use the calculator tool.")],
           tools=[calculator_tool],
           tool_choice="required",
           prompt_cache_key=session_id,
       )

       tool_call = response.main_response.tool_calls[0]
       tool_result = await provider.execute_tool(tool_call, {"calculator": calculator})

       final_response = await provider.acomplete(
           model="gpt-4.1",
           messages=[tool_result],
           tools=[calculator_tool],
           previous_response_id=response.original["id"],
           prompt_cache_key=session_id,
       )
       print(final_response.main_response.content)

   asyncio.run(main())

Key benefits:

- **No history overhead**: reference previous turns via ``previous_response_id`` instead of resending messages.
- **Prompt caching**: ``prompt_cache_key`` routes requests to the same machine for cache hits.
- **Reduced costs**: cached prefixes consume 90% fewer tokens.
- **Lower latency**: cached prefixes are processed faster.

How it works:

1. First request establishes a response context and caches the prompt prefix (≥1024 tokens).
2. Subsequent requests reference the first response via ``previous_response_id``.
3. Using the same ``prompt_cache_key`` routes requests to the same machine.
4. Only new content (tool outputs, user messages) needs to be sent.
5. Cached prefixes remain active for 5–10 minutes of inactivity (up to 1 hour off-peak).

See ``examples/openai_responses_tool_call_with_previous_id.py`` for a complete working example.
