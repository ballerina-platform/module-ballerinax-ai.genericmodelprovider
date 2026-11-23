// Copyright (c) 2026 WSO2 LLC (http://www.wso2.com).
//
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/http;

# Configurations for controlling the behaviours when communicating with a remote HTTP endpoint.
@display {label: "Connection Configuration"}
public type ConnectionConfig record {|

    # The HTTP version understood by the client. HTTP/1.1 is the default because many self-hosted
    # OpenAI-compatible servers do not support HTTP/2
    @display {label: "HTTP Version"}
    http:HttpVersion httpVersion = http:HTTP_1_1;

    # Configurations related to HTTP/1.x protocol
    @display {label: "HTTP1 Settings"}
    http:ClientHttp1Settings http1Settings?;

    # Configurations related to HTTP/2 protocol
    @display {label: "HTTP2 Settings"}
    http:ClientHttp2Settings http2Settings?;

    # The maximum time to wait (in seconds) for a response before closing the connection
    @display {label: "Timeout"}
    decimal timeout = 60;

    # The choice of setting `forwarded`/`x-forwarded` header
    @display {label: "Forwarded"}
    string forwarded = "disable";

    # Configurations associated with request pooling
    @display {label: "Pool Configuration"}
    http:PoolConfiguration poolConfig?;

    # HTTP caching related configurations
    @display {label: "Cache Configuration"}
    http:CacheConfig cache?;

    # Specifies the way of handling compression (`accept-encoding`) header
    @display {label: "Compression"}
    http:Compression compression = http:COMPRESSION_AUTO;

    # Configurations associated with the behaviour of the Circuit Breaker
    @display {label: "Circuit Breaker Configuration"}
    http:CircuitBreakerConfig circuitBreaker?;

    # Configurations associated with retrying
    @display {label: "Retry Configuration"}
    http:RetryConfig retryConfig?;

    # Configurations associated with inbound response size limits
    @display {label: "Response Limit Configuration"}
    http:ResponseLimitConfigs responseLimits?;

    # SSL/TLS-related options
    @display {label: "Secure Socket Configuration"}
    http:ClientSecureSocket secureSocket?;

    # Proxy server related options
    @display {label: "Proxy Configuration"}
    http:ProxyConfig proxy?;

    # Enables the inbound payload validation functionality which provided by the constraint package. Enabled by default
    @display {label: "Payload Validation"}
    boolean validation = true;
|};

# Name of the request field used to send the maximum number of tokens to generate.
public enum MaxTokensField {
    # The `max_tokens` field, accepted by most OpenAI-compatible servers
    MAX_TOKENS = "max_tokens",
    # The `max_completion_tokens` field, required by newer OpenAI reasoning models
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
}

# How the `generate` method asks the model to call the internal result tool.
public enum ToolChoiceMode {
    # Force the result tool via `tool_choice` (most reliable where supported)
    FORCED = "forced",
    # Send `tool_choice: "auto"` and instruct the model to call the result tool
    AUTO = "auto",
    # Omit `tool_choice` and instruct the model to call the result tool
    NONE = "none"
}

# Strategy used by the `generate` method to obtain structured output.
public enum StructuredOutputMode {
    # Expose the expected type as a tool and read the tool call arguments
    TOOL_CALL = "tool_call",
    # Use `response_format` with `type: json_schema`
    JSON_SCHEMA = "json_schema",
    # Use `response_format` with `type: json_object` and describe the schema in the prompt
    JSON_OBJECT = "json_object"
}

# Role used to send system instructions.
public enum SystemRole {
    # The standard `system` role
    SYSTEM = "system",
    # The `developer` role used by some newer OpenAI-style APIs
    DEVELOPER = "developer"
}

# Optional settings to adjust request and response handling for a specific OpenAI-compatible server.
# All settings are optional. By default, the provider adapts automatically: it starts with the most widely
# supported request shape and falls back when the server rejects an optional feature.
@display {label: "Compatibility Configuration"}
public type CompatibilityConfig record {|
    # Field used to send the maximum token count. If not set, `max_tokens` is used first and
    # `max_completion_tokens` is used if the server rejects it
    @display {label: "Max Tokens Field"}
    MaxTokensField? maxTokensField = ();

    # How the result tool is requested in `generate`. If not set, the tool call is forced first and
    # the provider falls back to `AUTO` if the server rejects a forced tool choice
    @display {label: "Tool Choice Mode"}
    ToolChoiceMode? toolChoiceMode = ();

    # Strategy used by `generate` to obtain structured output
    @display {label: "Structured Output Mode"}
    StructuredOutputMode structuredOutputMode = TOOL_CALL;

    # Role used for system messages
    @display {label: "System Role"}
    SystemRole systemRole = SYSTEM;

    # Additional HTTP headers sent with every request
    @display {label: "Extra Headers"}
    map<string> extraHeaders = {};

    # Additional top-level fields merged into every request body (e.g. provider-specific parameters).
    # These override the fields set by the provider
    @display {label: "Extra Request Body Fields"}
    map<json> extraBody = {};

    # Top-level request body fields that must never be sent (e.g. parameters rejected by the server)
    @display {label: "Fields to Drop"}
    string[] dropParams = [];

    # Whether to send the model's reasoning content back on assistant tool-call messages in subsequent turns.
    # Required by some thinking models (e.g. DeepSeek thinking mode) and rejected by others
    @display {label: "Preserve Reasoning Content"}
    boolean preserveReasoningContent = false;
|};

// ---------------------------------------------------------------------------------------------------------------------
// Internal wire-level records. These are intentionally open and lenient so that responses from any
// OpenAI-compatible server can be consumed.
// ---------------------------------------------------------------------------------------------------------------------

type ToolCallFunction record {
    string name;
    # Arguments as a JSON-encoded string (standard) or a JSON object (some servers)
    json arguments?;
};

type ResponseToolCall record {
    string? id?;
    string 'type?;
    ToolCallFunction 'function?;
};

type ResponseMessage record {
    string? role?;
    json content?;
    string? reasoning_content?;
    string? reasoning?;
    ResponseToolCall[]? tool_calls?;
    # Deprecated single function call field
    ToolCallFunction? function_call?;
};

type ChatChoice record {
    int index?;
    ResponseMessage message?;
    string? finish_reason?;
};

type Usage record {
    int? prompt_tokens?;
    int? completion_tokens?;
    int? total_tokens?;
};

type ChatCompletionResponse record {
    string? id?;
    string model?;
    ChatChoice[]? choices?;
    Usage? usage?;
};
