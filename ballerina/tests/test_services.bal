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

// Mock OpenAI-compatible server. The behaviour is selected by the `style` path segment:
//
// - `standard`     : a normal OpenAI-compatible server
// - `zai`          : rejects any `tool_choice` other than "auto", returns `reasoning_content` and returns tool call
//                    arguments as JSON objects
// - `ollama`       : ignores `tool_choice` and answers `generate` with JSON inside ```json fences
// - `oseries`      : rejects `max_tokens` (accepts `max_completion_tokens`)
// - `local`        : rejects requests that carry an `Authorization` header
// - `nojsonschema` : rejects `response_format.type == "json_schema"` as unsupported, accepts `json_object`
// - `badschema`    : rejects a `json_schema` response format as an invalid schema (not as unsupported)
// - `notools`      : rejects any request carrying `tools` (Ollama-style "does not support tools"),
//                    accepts `json_object`
// - `bare`         : rejects any request carrying `tools` and any request carrying `response_format`
// - `toolfailed`   : rejects any request carrying `tools` with Groq's `tool_use_failed` error (a failed tool call
//                    produced by the model, not a lack of tool support), accepts requests without `tools`
// - `maxtokens`    : a standard server with prompt-triggered max tokens rejections (see `mockMaxTokensRejection`)
//
// Error and edge-case behaviour is selected by trigger strings in the last user message.

const MOCK_GET_RESULTS_TOOL = "getResults";

const MOCK_SERVER_ERROR_MESSAGE = "The model is currently overloaded. Please try again later.";
const MOCK_ZAI_TOOL_CHOICE_ERROR = "tool_choice only supports auto";
const MOCK_OSERIES_MAX_TOKENS_ERROR = "Unsupported parameter: 'max_tokens' is not supported with this model. " +
        "Use 'max_completion_tokens' instead.";
const MOCK_JSON_SCHEMA_ERROR = "Invalid parameter: 'response_format' of type 'json_schema' is not supported " +
        "with this model.";
const MOCK_INVALID_SCHEMA_ERROR = "Invalid schema for response_format 'json_schema': In context=(), " +
        "'additionalProperties' is required to be supplied and to be false.";
const MOCK_SCHEMA_KEYWORD_ERROR = "Invalid schema: patternProperties is not supported";
const MOCK_BARE_KEYWORD_ERROR = "patternProperties is not supported";
const MOCK_CLOSE_TAG_STRING = "a</think>c";
const MOCK_THINK_TAG_STRING = "a<think>b</think>c";
const MOCK_NO_TOOLS_ERROR = "registry.ollama.ai/library/x does not support tools";
const MOCK_TOOL_USE_FAILED_ERROR = "Failed to call a function. Please adjust your prompt. See 'failed_generation' " +
        "for more details.";
const MOCK_NO_RESPONSE_FORMAT_ERROR = "Unrecognized request argument supplied: response_format";
const MOCK_CONTEXT_LENGTH_ERROR = "Request too large: context length exceeded (8192 tokens).";
const MOCK_MAX_COMPLETION_TOKENS_ERROR = "Unsupported parameter: 'max_completion_tokens' is not supported " +
        "with this model.";
const MOCK_THINK_REASONING = "The user wants the weather, so I will call the weather tool.";
const MOCK_REASONING_TEXT = "The user asked about the weather, so I should call the weather tool.";
const MOCK_TEXT_RESPONSE_PREFIX = "Mock response to: ";
const MOCK_IMAGE_DESCRIPTION = "A bright orange sunset over the ocean.";
const MOCK_STRING_RESULT = "Hello from the mock model";
const MOCK_NILABLE_STRING_RESULT = "Why did the developer go broke? Because he used up all his cache.";

// ---------------------------------------------------------------------------------------------------------------------
// Request recorder
// ---------------------------------------------------------------------------------------------------------------------

isolated map<json> mockLastBodies = {};
isolated map<map<string>> mockLastHeaders = {};
isolated map<int> mockRequestCounts = {};
isolated map<json[]> mockRequestHistory = {};

# Records a request received by the mock server.
#
# + key - The recorder key (the mock server style)
# + body - The request body
# + headers - The request headers, with lower-cased names
isolated function mockRecordRequest(string key, json body, map<string> headers) {
    lock {
        mockLastBodies[key] = body.clone();
    }
    lock {
        mockLastHeaders[key] = headers.clone();
    }
    lock {
        mockRequestCounts[key] = (mockRequestCounts[key] ?: 0) + 1;
    }
    lock {
        json[] history = mockRequestHistory[key] ?: [];
        history.push(body.clone());
        mockRequestHistory[key] = history;
    }
}

# Returns the bodies of the chat completion requests received for a style after the given request count.
#
# + style - The mock server style
# + fromCount - The request count before the requests of interest were sent
# + return - The request bodies, in the order they were received
isolated function getRequestBodiesSince(string style, int fromCount) returns map<json>[] {
    lock {
        json[] history = mockRequestHistory[style] ?: [];
        map<json>[] bodies = [];
        foreach int i in fromCount ..< history.length() {
            json body = history[i];
            bodies.push(body is map<json> ? body.clone() : {});
        }
        return bodies.clone();
    }
}

# Returns the body of the last chat completion request received for a style.
#
# + style - The mock server style
# + return - The last request body, or an empty map
isolated function getLastRequestBody(string style) returns map<json> {
    lock {
        json body = mockLastBodies[style];
        return body is map<json> ? body.clone() : {};
    }
}

# Returns the headers of the last chat completion request received for a style.
#
# + style - The mock server style
# + return - The last request headers (lower-cased names), or an empty map
isolated function getLastRequestHeaders(string style) returns map<string> {
    lock {
        return (mockLastHeaders[style] ?: {}).clone();
    }
}

# Returns the number of chat completion requests received for a style.
#
# + style - The mock server style
# + return - The request count
isolated function getRequestCount(string style) returns int {
    lock {
        return mockRequestCounts[style] ?: 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------------------------------------------------

service /llm on new http:Listener(8080) {

    resource function post [string style]/chat/completions(http:Request req) returns http:Response {
        map<string> headers = mockExtractHeaders(req);
        json|http:ClientError payload = req.getJsonPayload();
        if payload !is map<json> {
            mockRecordRequest(style, (), headers);
            return mockErrorResponse(400, "Request body must be a JSON object");
        }
        mockRecordRequest(style, payload, headers);
        return mockHandleChatCompletion(style, payload, headers);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------------------------------------------------

type MockToolCall record {|
    string id;
    string name;
    json arguments;
|};

isolated function mockHandleChatCompletion(string style, map<json> body, map<string> headers) returns http:Response {
    http:Response? authError = mockCheckAuth(style, headers);
    if authError is http:Response {
        return authError;
    }
    if style == "oseries" && body.hasKey("max_tokens") {
        return mockErrorResponse(400, MOCK_OSERIES_MAX_TOKENS_ERROR);
    }
    if style == "zai" && body.hasKey("tool_choice") && body["tool_choice"] != "auto" {
        return mockErrorResponse(400, MOCK_ZAI_TOOL_CHOICE_ERROR);
    }
    json responseFormat = body["response_format"];
    string? responseFormatType = mockGetResponseFormatType(responseFormat);
    if style == "nojsonschema" && responseFormatType == "json_schema" {
        return mockErrorResponse(400, MOCK_JSON_SCHEMA_ERROR);
    }
    if style == "badschema" && responseFormatType == "json_schema" {
        return mockJsonResponse(400, {
            "error": {
                "message": MOCK_INVALID_SCHEMA_ERROR,
                "type": "invalid_request_error",
                "param": "response_format",
                "code": ()
            }
        });
    }
    if style == "keywordschema" && responseFormatType == "json_schema" {
        // A schema-keyword rejection without an "Invalid schema" prefix.
        return mockJsonResponse(400, {
            "error": {"param": "response_format", "message": MOCK_BARE_KEYWORD_ERROR}
        });
    }
    if style == "patternschema" && responseFormatType == "json_schema" {
        return mockJsonResponse(400, {
            "error": {
                "message": MOCK_SCHEMA_KEYWORD_ERROR,
                "type": "invalid_request_error",
                "param": "response_format"
            }
        });
    }
    if (style == "notools" || style == "bare") && body.hasKey("tools") {
        return mockErrorResponse(400, MOCK_NO_TOOLS_ERROR);
    }
    if style == "toolfailed" && body.hasKey("tools") {
        return mockJsonResponse(400, {
            "error": {
                "message": MOCK_TOOL_USE_FAILED_ERROR,
                "type": "invalid_request_error",
                "code": "tool_use_failed",
                "failed_generation": "<function=getResults>{\"value\": 2}</function>"
            }
        });
    }
    if style == "bare" && body.hasKey("response_format") {
        return mockErrorResponse(400, MOCK_NO_RESPONSE_FORMAT_ERROR);
    }

    string model = body["model"] is string ? <string>body["model"] : "mock-model";
    string prompt = mockGetLastUserMessageText(body["messages"]);

    if style == "maxtokens" {
        http:Response? rejection = mockMaxTokensRejection(prompt, body);
        if rejection is http:Response {
            return rejection;
        }
    }
    if prompt.includes("TRIGGER_SERVER_ERROR") {
        return mockErrorResponse(500, MOCK_SERVER_ERROR_MESSAGE);
    }
    if prompt.includes("TRIGGER_CONTEXT_LENGTH") {
        return mockJsonResponse(400, {
            "error": {"message": MOCK_CONTEXT_LENGTH_ERROR, "type": "invalid_request_error", "code": "context_length"}
        });
    }
    if prompt.includes("TRIGGER_ECHO_AUTH") {
        // Simulates a gateway that echoes the received credential in its error response.
        return mockErrorResponse(401, string `Invalid credential: ${headers["authorization"] ?: ""}`);
    }
    int? triggeredStatus = mockGetTriggeredStatus(prompt);
    if triggeredStatus is int {
        return mockErrorResponse(triggeredStatus, string `Simulated HTTP ${triggeredStatus} failure from the server`);
    }
    if prompt.includes("TRIGGER_HTML_RESPONSE") {
        http:Response response = new;
        response.statusCode = 200;
        response.setTextPayload("<html><body><h1>Welcome to the gateway</h1></body></html>", "text/html");
        return response;
    }
    if prompt.includes("TRIGGER_NULL_FIELDS") {
        // Some servers send explicit nulls for optional fields.
        return mockJsonResponse(200, {
            "id": (),
            "object": "chat.completion",
            "model": (),
            "choices": [
                {
                    "index": (),
                    "message": {
                        "role": "assistant",
                        "content": (),
                        "tool_calls": [
                            {"id": "call_null_1", "type": (), "function": {"name": "getWeather", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        });
    }
    if prompt.includes("TRIGGER_EMPTY_CHOICES") {
        return mockJsonResponse(200, {
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "model": model,
            "choices": []
        });
    }

    map<json>[] tools = mockGetTools(body["tools"]);
    string[] toolNames = from map<json> tool in tools
        select mockGetToolName(tool);
    if toolNames.indexOf(MOCK_GET_RESULTS_TOOL) is int || responseFormatType is string
            || prompt.includes("GEN_") || prompt.includes("Please rate this blog") {
        return mockHandleGenerate(style, model, prompt, tools, responseFormat, responseFormatType);
    }

    if toolNames.length() > 0 {
        string toolName = toolNames[0];
        http:Response? special = mockSpecialToolCallResponse(style, model, prompt, toolName);
        if special is http:Response {
            return special;
        }
        if prompt.includes("TRIGGER_MALFORMED_ARGS") {
            return mockToolCallResponse(style, model,
                    [{id: "call_bad_1", name: toolName, arguments: "{this is not json"}], ());
        }
        if prompt.includes("TRIGGER_PARALLEL_TOOL_CALLS") {
            return mockToolCallResponse(style, model, [
                        {id: "call_parallel_1", name: toolName, arguments: {"city": "Colombo"}},
                        {id: "call_parallel_2", name: toolName, arguments: {"city": "Kandy"}}
                    ], ());
        }
        if prompt.includes("TRIGGER_TOOL_CALL") {
            string? reasoning = style == "zai" || prompt.includes("TRIGGER_REASONING") ? MOCK_REASONING_TEXT : ();
            return mockToolCallResponse(style, model,
                    [{id: "call_weather_1", name: toolName, arguments: {"city": "Colombo"}}], reasoning);
        }
    }
    if prompt.includes("TRIGGER_THINK_TEXT") {
        return mockTextResponse(style, model, "<think>Let me consider the question.</think>Final");
    }
    return mockTextResponse(style, model, MOCK_TEXT_RESPONSE_PREFIX + prompt);
}

# Returns the tool call edge-case response selected by a trigger in the prompt, if any.
#
# + style - The mock server style
# + model - The model name
# + prompt - The prompt text
# + toolName - The name of the first tool in the request
# + return - The response, or `()` if no edge-case trigger is present
isolated function mockSpecialToolCallResponse(string style, string model, string prompt, string toolName)
        returns http:Response? {
    json validArguments = "{\"city\":\"Colombo\"}";
    json[] toolCalls = [];
    if prompt.includes("TRIGGER_EMPTY_ID") {
        toolCalls = [mockRawToolCall("", toolName, validArguments)];
    } else if prompt.includes("TRIGGER_NULL_ID") {
        toolCalls = [mockRawToolCall((), toolName, validArguments)];
    } else if prompt.includes("TRIGGER_ARGS_NULL_STRING") {
        toolCalls = [mockRawToolCall("call_args_1", toolName, "null")];
    } else if prompt.includes("TRIGGER_ARGS_EMPTY_STRING") {
        toolCalls = [mockRawToolCall("call_args_1", toolName, "")];
    } else if prompt.includes("TRIGGER_ARGS_JSON_NULL") {
        toolCalls = [mockRawToolCall("call_args_1", toolName, ())];
    } else if prompt.includes("TRIGGER_ARGS_ARRAY") {
        toolCalls = [mockRawToolCall("call_args_1", toolName, "[1]")];
    } else if prompt.includes("TRIGGER_FUNCTION_CALL_ONLY") {
        map<json> message = {
            "role": "assistant",
            "content": (),
            "function_call": {"name": toolName, "arguments": validArguments}
        };
        return mockJsonResponse(200, mockCompletion(style, model, message, "function_call"));
    } else if prompt.includes("TRIGGER_THINK_TOOL_CALL") {
        map<json> message = {
            "role": "assistant",
            "content": "<think>" + MOCK_THINK_REASONING + "</think>\n",
            "tool_calls": [mockRawToolCall("call_think_1", toolName, validArguments)]
        };
        return mockJsonResponse(200, mockCompletion(style, model, message, "tool_calls"));
    } else {
        return ();
    }
    map<json> message = {"role": "assistant", "content": (), "tool_calls": toolCalls};
    return mockJsonResponse(200, mockCompletion(style, model, message, "tool_calls"));
}

isolated function mockRawToolCall(string? id, string name, json arguments) returns json => {
    "id": id,
    "type": "function",
    "function": {"name": name, "arguments": arguments}
};

# Returns the max tokens rejection selected by a trigger in the prompt of the `maxtokens` style, if any.
#
# + prompt - The prompt text
# + body - The request body
# + return - The error response, or `()` if the request is accepted
isolated function mockMaxTokensRejection(string prompt, map<json> body) returns http:Response? {
    if prompt.includes("TRIGGER_MT_UNRELATED") {
        return mockErrorResponse(400, MOCK_CONTEXT_LENGTH_ERROR);
    }
    if prompt.includes("TRIGGER_MT_VALUE") && body.hasKey("max_tokens") {
        return mockJsonResponse(400, {
            "error": {"message": "max_tokens must be <= 4096", "type": "invalid_request_error", "param": "max_tokens"}
        });
    }
    if prompt.includes("TRIGGER_MT_FASTAPI") {
        // A pydantic validation error that echoes the whole request (including `max_tokens`) in `input`.
        return mockJsonResponse(422, {
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "messages", 0, "content"],
                    "msg": "Input should be a valid string",
                    "input": body.clone()
                }
            ]
        });
    }
    if prompt.includes("TRIGGER_MT_BOTH") {
        if body.hasKey("max_tokens") {
            return mockErrorResponse(400, MOCK_OSERIES_MAX_TOKENS_ERROR);
        }
        if body.hasKey("max_completion_tokens") {
            return mockErrorResponse(400, MOCK_MAX_COMPLETION_TOKENS_ERROR);
        }
    }
    return ();
}

# Returns the HTTP status requested by a `TRIGGER_STATUS_<code>` trigger in the prompt, if any.
#
# + prompt - The prompt text
# + return - The status code, or `()`
isolated function mockGetTriggeredStatus(string prompt) returns int? {
    foreach int status in [401, 403, 429, 503] {
        if prompt.includes(string `TRIGGER_STATUS_${status}`) {
            return status;
        }
    }
    return ();
}

isolated function mockHandleGenerate(string style, string model, string prompt, map<json>[] tools,
        json responseFormat, string? responseFormatType) returns http:Response {
    json value = mockGetGenerateValue(prompt);
    map<json>? schema = mockGetResultSchema(tools, responseFormat);
    json payload = mockWrapGenerateValue(value, schema);
    string payloadText = prompt.includes("GEN_MALFORMED") ? "this is not json"
        : prompt.includes("GEN_BARE_NULL") ? "null" : payload.toJsonString();

    if style == "ollama" {
        // Reasoning models may emit draft JSON in a leading think block before the answer.
        string reasoning = prompt.includes("GEN_THINK_PREFIX") ? "<think>Draft: {\"result\": 1}</think>\n" : "";
        return mockTextResponse(style, model,
                reasoning + "Sure! Here is the result:\n\n```json\n" + payloadText + "\n```\n");
    }
    boolean hasResultsTool = tools.some(tool => mockGetToolName(tool) == MOCK_GET_RESULTS_TOOL);
    if responseFormatType is string || !hasResultsTool {
        // `json_object`/`json_schema` response formats, or a prompt-only JSON request without tools.
        return mockTextResponse(style, model, payloadText);
    }
    json arguments = prompt.includes("GEN_MALFORMED") ? payloadText : payload;
    string? reasoning = style == "zai" ? "Computing the requested value." : ();
    return mockToolCallResponse(style, model,
            [{id: "call_results_1", name: MOCK_GET_RESULTS_TOOL, arguments}], reasoning);
}

# Returns the value the mock model generates for a `generate` prompt.
#
# + prompt - The prompt text
# + return - The value
isolated function mockGetGenerateValue(string prompt) returns json {
    if prompt.includes("GEN_MISMATCH") {
        return "definitely not a number";
    }
    if prompt.includes("GEN_INT") {
        return 2;
    }
    if prompt.includes("GEN_CLOSE_TAG_STRING") {
        return MOCK_CLOSE_TAG_STRING;
    }
    if prompt.includes("GEN_THINK_STRING") {
        return MOCK_THINK_TAG_STRING;
    }
    if prompt.includes("GEN_STRING") {
        return MOCK_STRING_RESULT;
    }
    if prompt.includes("GEN_BOOL") {
        return true;
    }
    if prompt.includes("GEN_REVIEWS_LIST") {
        return [reviewRecord, {"rating": 4, "comment": "Covers the basics of gardening."}];
    }
    if prompt.includes("GEN_REVIEW") || prompt.includes("Please rate this blog") {
        return reviewRecord;
    }
    if prompt.includes("GEN_NILABLE_NULL") {
        return ();
    }
    if prompt.includes("GEN_NILABLE_VALUE") {
        return MOCK_NILABLE_STRING_RESULT;
    }
    if prompt.includes("GEN_IMAGE") {
        return MOCK_IMAGE_DESCRIPTION;
    }
    return "Unrecognized prompt";
}

# Wraps a generated value the way the requested schema expects. Non-object types are expected to be
# wrapped in a `result` property; when no schema is available (`json_object`), non-object values are wrapped.
#
# + value - The generated value
# + schema - The requested JSON schema, if any
# + return - The value to send back
isolated function mockWrapGenerateValue(json value, map<json>? schema) returns json {
    if schema is map<json> {
        json properties = schema["properties"];
        if properties is map<json> && properties.length() == 1 && properties.hasKey("result") {
            return {"result": value};
        }
        return value;
    }
    return value is map<json> ? value : {"result": value};
}

isolated function mockGetResultSchema(map<json>[] tools, json responseFormat) returns map<json>? {
    foreach map<json> tool in tools {
        if mockGetToolName(tool) == MOCK_GET_RESULTS_TOOL {
            json fn = tool["function"];
            if fn is map<json> && fn["parameters"] is map<json> {
                return <map<json>>fn["parameters"];
            }
        }
    }
    if responseFormat is map<json> {
        json jsonSchema = responseFormat["json_schema"];
        if jsonSchema is map<json> && jsonSchema["schema"] is map<json> {
            return <map<json>>jsonSchema["schema"];
        }
    }
    return ();
}

isolated function mockGetResponseFormatType(json responseFormat) returns string? {
    if responseFormat is map<json> && responseFormat["type"] is string {
        return <string>responseFormat["type"];
    }
    return ();
}

isolated function mockGetTools(json tools) returns map<json>[] {
    if tools !is json[] {
        return [];
    }
    map<json>[] result = [];
    foreach json tool in tools {
        if tool is map<json> {
            result.push(tool);
        }
    }
    return result;
}

isolated function mockGetToolName(map<json> tool) returns string {
    json fn = tool["function"];
    if fn is map<json> && fn["name"] is string {
        return <string>fn["name"];
    }
    return "";
}

# Returns the text of the last user message (string content or the text parts of an array content).
#
# + messages - The request messages
# + return - The text, or an empty string
isolated function mockGetLastUserMessageText(json messages) returns string {
    if messages !is json[] {
        return "";
    }
    int i = messages.length() - 1;
    while i >= 0 {
        json message = messages[i];
        if message is map<json> && message["role"] == "user" {
            return mockGetContentText(message["content"]);
        }
        i -= 1;
    }
    return "";
}

isolated function mockGetContentText(json content) returns string {
    if content is string {
        return content;
    }
    if content is json[] {
        string text = "";
        foreach json part in content {
            if part is map<json> && part["text"] is string {
                text += <string>part["text"];
            }
        }
        return text;
    }
    return "";
}

isolated function mockToolCallResponse(string style, string model, MockToolCall[] calls, string? reasoning)
        returns http:Response {
    json[] toolCalls = [];
    foreach MockToolCall call in calls {
        json arguments = call.arguments;
        // Z.ai-style servers return the arguments as a JSON object instead of a JSON-encoded string.
        json wireArguments = style == "zai" || arguments is string ? arguments : arguments.toJsonString();
        toolCalls.push({
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": wireArguments}
        });
    }
    map<json> message = {"role": "assistant", "content": (), "tool_calls": toolCalls};
    if reasoning is string {
        message["reasoning_content"] = reasoning;
    }
    return mockJsonResponse(200, mockCompletion(style, model, message, "tool_calls"));
}

isolated function mockTextResponse(string style, string model, string content) returns http:Response {
    map<json> message = {"role": "assistant", "content": content};
    if style == "zai" {
        message["reasoning_content"] = "Let me think about how to answer this.";
    }
    return mockJsonResponse(200, mockCompletion(style, model, message, "stop"));
}

isolated function mockCompletion(string style, string model, map<json> message, string finishReason)
        returns map<json> {
    map<json> response = {
        "id": "chatcmpl-mock-" + style,
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finishReason}]
    };
    // Some servers (e.g. Ollama, Z.ai) are lenient about usage; omit it to verify lenient parsing.
    if style != "ollama" && style != "zai" {
        response["usage"] = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20};
    }
    return response;
}

// ---------------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------------

isolated function mockCheckAuth(string style, map<string> headers) returns http:Response? {
    if style == "local" && headers.hasKey("authorization") {
        return mockErrorResponse(400, "Unexpected Authorization header: the local server does not use authentication");
    }
    return ();
}

isolated function mockExtractHeaders(http:Request req) returns map<string> {
    map<string> headers = {};
    foreach string name in req.getHeaderNames() {
        string|http:HeaderNotFoundError value = req.getHeader(name);
        if value is string {
            headers[name.toLowerAscii()] = value;
        }
    }
    return headers;
}

isolated function mockJsonResponse(int statusCode, json payload) returns http:Response {
    http:Response response = new;
    response.statusCode = statusCode;
    response.setJsonPayload(payload);
    return response;
}

isolated function mockErrorResponse(int statusCode, string message) returns http:Response =>
    mockJsonResponse(statusCode, {"error": {"message": message}});
