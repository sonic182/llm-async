llm-async — Async multi-provider LLM client for Python
=======================================================

High-performance, async-first LLM client for OpenAI, Claude, Google Gemini, and OpenRouter.
Built on top of `aiosonic <https://github.com/sonic182/aiosonic>`_ for fast, low-latency HTTP
and true asyncio streaming across providers.

Features
--------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 20 15

   * - Feature
     - OpenAI
     - Claude
     - Google Gemini
     - OpenRouter
   * - Chat Completions
     - ✅
     - ✅
     - ✅
     - ✅
   * - Tool Calling
     - ✅
     - ✅
     - ✅
     - ✅
   * - Streaming
     - ✅
     - ✅
     - ✅
     - ✅
   * - Structured Outputs
     - ✅
     - ❌
     - ✅
     - ✅

- **Async-first**: Built with asyncio for high-performance, non-blocking operations.
- **Unified interface**: Same message/tool/streaming patterns across all providers.
- **Tool Calling**: Unified tool definitions with execution helpers.
- **Structured Outputs**: JSON schema validation on responses (OpenAI, Google, OpenRouter).
- **Extensible**: Add new providers by inheriting from ``BaseProvider``.
- **Tested**: Comprehensive test suite with high coverage.

Performance
~~~~~~~~~~~

- Built on `aiosonic <https://github.com/sonic182/aiosonic>`_ for fast, low-overhead async HTTP.
- True asyncio end-to-end: concurrent requests across providers with minimal overhead.
- Designed for fast tool-call round-trips and low-latency streaming.

Why llm-async?
--------------

- Async-first performance (aiosonic-based) vs. sync or heavier HTTP stacks.
- Unified provider interface: same message/tool/streaming patterns across OpenAI, Claude, Gemini, OpenRouter.
- Structured outputs (OpenAI, Google, OpenRouter) with JSON schema validation.
- Tool-call round-trip helpers for consistent multi-turn execution.
- Minimal surface area: easy to extend with new providers via ``BaseProvider``.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   providers/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   development
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
