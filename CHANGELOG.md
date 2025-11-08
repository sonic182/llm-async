# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- **OpenAIResponsesProvider**: New provider for OpenAI's Responses API with advanced features:
  - Stateless multi-turn conversations using `previous_response_id` to reference prior responses without resending full message history
  - Support for OpenAI's `prompt_cache_key` parameter for session-based prompt caching
  - Extended `ToolCall` model with helper methods: `to_responses_api_message()`, `from_responses_api_function_call()`, `function_call_output()`
  - Custom message normalization for Responses API format (uses `type` field instead of `role` for tool calls and outputs)
- Examples demonstrating stateless conversations:
  - `examples/openai_responses_tool_call.py`: Full message history approach
  - `examples/openai_responses_tool_call_with_previous_id.py`: Stateless multi-turn conversation (weather tool example)
- Documentation in README.md explaining Responses API advantages and stateless conversation patterns
- **Generic request method**: Added `async request(method, path, json_data=None, **kwargs)` to `BaseProvider`:
  - Allows direct API calls to any endpoint (GET, POST, PUT, DELETE, PATCH)
  - Works across all providers (OpenAI, Claude, Google, OpenRouter, OpenAIResponses)
  - Example: `await provider.request("GET", "/models")` to list available models
  - See `examples/provider_request.py` for usage example

## [0.3.0] - 2025-11-08
### Added
- Introduced unified `Message` model with normalization helpers and provider-agnostic serialization
- Expanded sample scripts to demonstrate multi-provider tool calling (`examples/tool_call_all_providers.py`) and streaming (`examples/stream_all_providers.py`) with shared prompts and dotenv-loaded credentials
### Removed
- Deprecated `MainResponse` helper in favor of the new `Message` abstraction

## [0.2.0] - 2025-11-07
### Changed
- Removed automatic execution of tools - tool calls are now returned to the caller for explicit handling
- Removed Pub/Sub backend - tools no longer require event subscription system
- Simplified tool call workflow for better control and transparency
### Added
- Clear and simple function tool call example demonstrating explicit tool handling

## [0.1.1] - 2025-10-31
### Added
- Makefile for easier launch commands with poetry
### Changed
- Updated package metadata in `pyproject.toml`:
  - Added `readme` field pointing to README.md
  - Added `license` field (MIT)
  - Added `repository` URL
  - Updated author information with correct name and email

## [0.1.0] - 2025-10-31
### Added
- Initial public release.
- Async-first providers: OpenAI, Claude (Anthropic), Google Gemini, OpenRouter.
- Chat completion API with asyncio-native `acomplete` method and streaming support.
- Tool calling support with unified `Tool` model and examples for cross-provider execution.
- Structured outputs support (OpenAI, Google, OpenRouter) with JSON schema enforcement.
- Pub/Sub local backend for tool execution events and example usage.
- Tests covering providers, streaming, tool calls, and response schema handling.

### Changed
- N/A (first release)

### Fixed
- N/A (first release)

### Notes
- This changelog entry was bootstrapped from `README.md` and existing tests.
- For developer commands, testing and build instructions see `README.md`.

[Unreleased]: https://github.com/sonic182/llm-async/compare/0.3.0...HEAD
[0.3.0]: https://github.com/sonic182/llm-async/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/sonic182/llm-async/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/sonic182/llm-async/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/sonic182/llm-async/releases/tag/0.1.0
