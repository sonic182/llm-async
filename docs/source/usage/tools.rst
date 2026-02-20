Tool Usage
==========

Tools let you expose Python functions to the LLM. Define a :class:`~llm_async.models.Tool`,
pass it to ``acomplete``, then execute the returned tool call and send the result back.

.. code-block:: python

   import asyncio
   import os
   from llm_async.models import Tool
   from llm_async.providers import OpenAIProvider

   calculator_tool = Tool(
       name="calculator",
       description="Perform basic arithmetic operations",
       parameters={
           "type": "object",
           "properties": {
               "operation": {
                   "type": "string",
                   "enum": ["add", "subtract", "multiply", "divide"]
               },
               "a": {"type": "number"},
               "b": {"type": "number"}
           },
           "required": ["operation", "a", "b"]
       },
       input_schema={
           "type": "object",
           "properties": {
               "operation": {
                   "type": "string",
                   "enum": ["add", "subtract", "multiply", "divide"]
               },
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
       provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
       tools_map = {"calculator": calculator}
       messages = [{"role": "user", "content": "What is 15 + 27?"}]

       response = await provider.acomplete(
           model="gpt-4o-mini",
           messages=messages,
           tools=[calculator_tool]
       )

       tool_call = response.main_response.tool_calls[0]
       tool_result = await provider.execute_tool(tool_call, tools_map)

       messages_with_tool = messages + [response.main_response.original] + [tool_result]
       final_response = await provider.acomplete(
           model="gpt-4o-mini",
           messages=messages_with_tool
       )
       print(final_response.main_response.content)

   asyncio.run(main())

See ``examples/tool_call_all_providers.py`` for a cross-provider tool-call demo.
