## Overview

The generic model provider is an **OpenAI-compatible** model provider. It works with any LLM service that exposes an
OpenAI-compatible Chat Completions API (`/chat/completions`).

Configure the service URL, API key and model name to use providers such as Groq, Together AI, Fireworks AI, xAI,
Z.ai (GLM), OpenRouter, Mistral, vLLM, LM Studio and Ollama, without waiting for a dedicated connector.

> **Note:** Only OpenAI-compatible APIs are supported. Services with a different API format (for example, the native
> Anthropic Messages API or the Gemini API) require their dedicated model provider.

The provider implements `ai:ModelProvider`, so it can be used with `ai:Agent` and `generate` of the `ballerina/ai`
module.

### How it adapts to different servers

OpenAI-compatible servers differ in small ways. Instead of requiring provider-specific settings, the provider:

- Sends only the fields you set. It never sends fields that some servers reject (such as `messages[].name`, `n`,
  `user` or `logprobs`) and omits `tools` when there are none.
- Sends `max_tokens`, and switches to `max_completion_tokens` if the server rejects `max_tokens`.
- Forces the internal result tool in `generate`, and falls back to `tool_choice: "auto"` with an instruction if the
  server does not support forced tool choice. If the model does not support tools at all, `generate` uses JSON mode
  instead. If the model replies with text instead of a tool call, the JSON in the text is parsed.
- Reads reasoning output from either `reasoning_content` or `reasoning`, removes inline `<think>` blocks from
  responses, and accepts tool call arguments as a JSON string or an object.

A fallback is used only when the server's error message names the rejected feature, and it is remembered for the
lifetime of the provider instance only after the fallback request succeeds, so it costs at most one extra request.
Remembered fallbacks are logged at the debug level because they are expected. To avoid the extra request, set the
matching [compatibility](#compatibility-configuration) field (for example, `maxTokensField: MAX_COMPLETION_TOKENS` for
OpenAI reasoning models such as `gpt-5`).

## Quickstart

### Step 1: Import the module

```ballerina
import ballerina/ai;
import ballerinax/ai.genericmodelprovider;
```

### Step 2: Initialize the model provider

```ballerina
configurable string apiKey = ?;

final ai:ModelProvider model = check new genericmodelprovider:ModelProvider(
    serviceUrl = "https://api.groq.com/openai/v1",
    modelType = "llama-3.3-70b-versatile",
    apiKey = apiKey
);
```

For a local server that does not require authentication, omit `apiKey`:

```ballerina
final ai:ModelProvider localModel = check new genericmodelprovider:ModelProvider(
    serviceUrl = "http://localhost:1234/v1",
    modelType = "qwen2.5-7b-instruct"
);
```

### Step 3: Invoke chat completion

```ballerina
ai:ChatMessage[] chatMessages = [{role: "user", content: "hi"}];
ai:ChatAssistantMessage response = check model->chat(chatMessages, tools = []);

chatMessages.push(response);
```

### Step 4: Generate structured output

```ballerina
type Summary record {|
    string title;
    string[] keyPoints;
|};

string text = "Ballerina is an open-source programming language for writing network-aware integrations.";
Summary summary = check model->generate(`Summarize the following text: ${text}`);
```

### Step 5: Use with an agent

```ballerina
final ai:Agent agent = check new (
    systemPrompt = {role: "Assistant", instructions: "You are a helpful assistant."},
    model = model,
    tools = []
);
```

### Authentication

The API key is sent as `Authorization: Bearer <apiKey>`. Omit `apiKey` for local servers that do not require
authentication. Azure OpenAI is not covered by this provider; use the dedicated `ballerinax/ai.azure` provider.

## Common service URLs

| Service | `serviceUrl` | API key |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | Required |
| Together AI | `https://api.together.ai/v1` | Required |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | Required |
| xAI | `https://api.x.ai/v1` | Required |
| Z.ai (GLM) | `https://api.z.ai/api/paas/v4` | Required |
| OpenRouter | `https://openrouter.ai/api/v1` | Required |
| Mistral | `https://api.mistral.ai/v1` | Required |
| vLLM | `http://localhost:8000/v1` | Only if the server is started with an API key |
| LM Studio | `http://localhost:1234/v1` | Not required by default |
| Ollama | `http://localhost:11434/v1` | Not required |

> **Self-hosted servers (Ollama, vLLM, LM Studio):** set `compatibility = {structuredOutputMode: genericmodelprovider:JSON_SCHEMA}`.
> These servers enforce the JSON schema while generating, which makes `generate` reliable even with small models.
> Ollama accepts but ignores a forced `tool_choice`, so the default tool-call strategy depends on the model choosing
> to call the tool with correct arguments, which small models often fail to do.

Use the `serviceUrl` up to, but excluding, `/chat/completions`. The model name is passed to the service as-is (e.g.
`meta-llama/Llama-3.3-70B-Instruct-Turbo` for Together AI).

## Compatibility configuration

All settings are optional. Use them only when a server needs a request shape that the provider cannot detect
automatically, or to skip the first failed request when you already know what the server supports.

| Field | Default | Description |
|---|---|---|
| `maxTokensField` | Auto-detected | `MAX_TOKENS` or `MAX_COMPLETION_TOKENS` |
| `toolChoiceMode` | Auto-detected | How `generate` requests the result tool: `FORCED`, `AUTO`, or `NONE` (omit `tool_choice`) |
| `structuredOutputMode` | `TOOL_CALL` | How `generate` obtains structured output: `TOOL_CALL`, `JSON_SCHEMA` (`response_format`), or `JSON_OBJECT` (JSON mode with the schema in the prompt). With `JSON_SCHEMA`, a server that does not support `json_schema` causes a `JSON_OBJECT` retry on every call, so prefer `JSON_OBJECT` for such servers |
| `systemRole` | `SYSTEM` | Role used for system messages: `SYSTEM` or `DEVELOPER` |
| `extraHeaders` | `{}` | Additional headers sent with every request |
| `extraBody` | `{}` | Additional top-level request fields (e.g. provider-specific parameters) |
| `dropParams` | `[]` | Top-level request fields that must never be sent |
| `preserveReasoningContent` | `false` | Send reasoning content back with assistant tool-call messages (required by some thinking models such as DeepSeek in thinking mode) |

Example: passing a provider-specific parameter (Z.ai thinking mode).

```ballerina
final ai:ModelProvider glmModel = check new genericmodelprovider:ModelProvider(
    serviceUrl = "https://api.z.ai/api/paas/v4",
    modelType = "glm-4.6",
    apiKey = zaiApiKey,
    compatibility = {
        extraBody: {thinking: {"type": "disabled"}}
    }
);
```
