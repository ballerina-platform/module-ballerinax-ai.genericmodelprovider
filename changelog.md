# Change Log

This file documents all significant changes made to the Ballerina ai.genericmodelprovider package across releases.

## [Un-released]

### Changed
- Display the provider as `Generic Model Provider` instead of `Generic Model Provider (OpenAI-compatible)`

### Added
- [Add a generic OpenAI-compatible model provider](https://github.com/wso2-enterprise/integration-product-management/issues/1025)
  - Works with any service exposing the OpenAI Chat Completions API by configuring the service URL, API key and a free-form model name.
  - Adapts automatically to common differences between servers (the max tokens field name, forced tool choice support, structured output support), remembering the fallback per provider instance.
  - Optional `CompatibilityConfig` to override the request shape: max tokens field, tool choice mode, structured output mode, system role, auth header and scheme, extra headers, extra body fields, dropped fields, and reasoning content preservation.
