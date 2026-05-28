Agent and Tool Decorator
========================

Use :class:`~llm_async.agent.Agent` with the :func:`~llm_async.tool.tool` decorator
when you want llm-async to handle the tool-call loop for you. The agent sends your
tools to the provider, executes requested tool calls, appends tool results to the
conversation, and keeps calling the model until it returns a final response or reaches
``max_iter``.

.. code-block:: python

   import asyncio
   import os

   from llm_async import Agent, OpenAIProvider, tool


   @tool
   def calculator(operation: str, a: float, b: float) -> float:
       """Perform basic arithmetic operations."""
       if operation == "add":
           return a + b
       if operation == "subtract":
           return a - b
       if operation == "multiply":
           return a * b
       if operation == "divide":
           return a / b
       raise ValueError(f"Unsupported operation: {operation}")


   async def main() -> None:
       agent = Agent(
           provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")),
           model="gpt-4o-mini",
           tools=[calculator],
       )

       response = await agent.acomplete(
           messages=[
               {
                   "role": "user",
                   "content": "What is 15 + 27? Use the calculator.",
               }
           ]
       )
       print(response.main_response.content)


   asyncio.run(main())

Decorated Tools
---------------

The :func:`~llm_async.tool.tool` decorator wraps a regular Python function and attaches
a generated :class:`~llm_async.models.Tool` definition to ``function.tool``.

The generated schema uses the function name as the tool name, the function docstring as
the tool description, and supported type annotations for parameters:

.. code-block:: python

   from llm_async import tool


   @tool
   def word_count(text: str) -> int:
       """Count the number of words in a text string."""
       return len(text.split())


   assert word_count.tool.name == "word_count"
   assert word_count.tool.parameters["properties"]["text"] == {"type": "string"}

Supported parameter annotations include ``str``, ``int``, ``float``, ``bool``,
``list``, ``list[T]``, ``dict``, and optional unions such as ``str | None``. Parameters
without supported annotations are omitted from the generated schema. Parameters with
defaults, and optional parameters, are not marked as required.

Agent Options
-------------

``Agent`` accepts these options:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Description
   * - ``provider``
     - Provider instance used for ``acomplete`` and tool execution.
   * - ``model``
     - Model name passed to the provider on every completion request.
   * - ``tools``
     - List of functions decorated with :func:`~llm_async.tool.tool`.
   * - ``parallel_tools``
     - Execute multiple tool calls concurrently when ``True``. Defaults to ``True``.
   * - ``max_iter``
     - Maximum model/tool-call iterations before returning the last response. Defaults to ``10``.

Additional keyword arguments passed to :meth:`~llm_async.agent.Agent.acomplete` are
forwarded to the provider, so provider-specific options such as ``temperature`` or
``tool_choice`` can still be used.

See ``examples/agent_tool_decorator.py`` for a multi-provider example.
