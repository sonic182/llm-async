# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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

[Unreleased]: https://github.com/sonic182/llm_async/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sonic182/llm_async/releases/tag/v0.1.0
