# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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

[Unreleased]: https://github.com/sonic182/llm_async/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sonic182/llm_async/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/sonic182/llm_async/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sonic182/llm_async/releases/tag/v0.1.0
