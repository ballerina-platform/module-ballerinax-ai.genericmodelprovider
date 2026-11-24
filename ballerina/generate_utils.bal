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

import ballerina/ai;
import ballerina/ai.observe;
import ballerina/http;
import ballerina/log;

# The JSON schema sent to the model for the expected type.
type ResponseSchema record {|
    # The schema. Always an object schema
    map<json> schema;
    # Whether the original schema was an object schema. When `false`, the original schema is wrapped
    # under the `result` property and the value must be unwrapped when parsing
    boolean isOriginallyJsonObject = true;
|};

# Everything needed to send a `generate` request, shared by all structured output strategies.
type GenerateRequestContext record {|
    # The HTTP client
    http:Client llmClient;
    # The model name
    string modelType;
    # The temperature, if configured
    decimal? temperature;
    # The maximum number of tokens, if configured
    int? maxTokens;
    # Compatibility settings
    readonly & CompatibilityConfig compatibility;
    # Request headers
    map<string> & readonly requestHeaders;
    # Learned fallbacks for the provider instance
    AdaptiveState adaptiveState;
    # The user message content (a string or an array of content parts)
    json userContent;
    # The expected response schema
    ResponseSchema responseSchema;
|};

const GENERATE_RESULT_FIELD = "result";
const GET_RESULTS_TOOL_NAME = "getResults";
const GENERATE_JSON_CONVERSION_ERROR = "FromJsonStringError";
const GENERATE_CONVERSION_ERROR = "ConversionError";
const GENERATE_PARSE_ERROR_MESSAGE = "Error occurred while attempting to parse the response from the " +
    "LLM as the expected type. Retrying and/or validating the prompt could fix the response.";
const GENERATE_NO_RELEVANT_RESPONSE = "No relevant response from the LLM";
const GET_RESULTS_TOOL_INSTRUCTION = "You must call the `getResults` tool exactly once, passing your final answer " +
    "as the tool arguments. The arguments must conform to the tool's parameter schema. " +
    "Do not respond with plain text.";

# The outcome of a single `generate` request strategy, together with what was sent in the request that produced it
# (used to record the span attributes of the attempt that is finally used).
type GenerateAttempt record {|
    # The response of the final request, or its error
    ChatCompletionResponse|ai:Error response;
    # The messages sent in the final request
    json[] messages;
    # The tools sent in the final request, if any
    json[]? tools = ();
|};

# Maximum number of chat completion requests sent by the fallback chain of a single `generate` call
# (excluding the max tokens field retry done by `sendChatCompletion`).
const MAX_GENERATE_REQUESTS = 3;

# Generates a value of the expected type for the given prompt. Called from the native `Generator` class;
# the parameter order must be kept in sync with it.
#
# + llmClient - The HTTP client
# + modelType - The model name
# + temperature - The temperature, if configured
# + maxTokens - The maximum number of tokens, if configured
# + compatibility - Compatibility settings
# + requestHeaders - Request headers
# + adaptiveState - Learned fallbacks for the provider instance
# + prompt - The prompt
# + expectedResponseTypedesc - The expected type
# + return - The generated value, or an error
isolated function generateLlmResponse(http:Client llmClient, string modelType, decimal? temperature, int? maxTokens,
        readonly & CompatibilityConfig compatibility, map<string> & readonly requestHeaders,
        AdaptiveState adaptiveState, ai:Prompt prompt, typedesc<json> expectedResponseTypedesc)
        returns anydata|ai:Error {
    observe:GenerateContentSpan span = observe:createGenerateContentSpan(modelType);
    span.addProvider(PROVIDER_NAME);
    if temperature is decimal {
        span.addTemperature(temperature);
    }

    json userContent;
    ResponseSchema responseSchema;
    do {
        userContent = check buildUserContent(prompt);
        responseSchema = check getExpectedResponseSchema(expectedResponseTypedesc);
    } on fail ai:Error err {
        span.close(err);
        return err;
    }

    GenerateRequestContext ctx = {
        llmClient,
        modelType,
        temperature,
        maxTokens,
        compatibility,
        requestHeaders,
        adaptiveState,
        userContent,
        responseSchema
    };

    GenerateAttempt attempt = sendStructuredOutputRequest(ctx);
    span.addInputMessages(attempt.messages);
    json[]? tools = attempt.tools;
    if tools is json[] {
        span.addTools(tools);
    }
    ChatCompletionResponse|ai:Error response = attempt.response;
    if response is ai:Error {
        span.close(response);
        return response;
    }

    string? responseId = response?.id;
    if responseId is string {
        span.addResponseId(responseId);
    }
    int? inputTokens = response?.usage?.prompt_tokens;
    if inputTokens is int {
        span.addInputTokenCount(inputTokens);
    }
    int? outputTokens = response?.usage?.completion_tokens;
    if outputTokens is int {
        span.addOutputTokenCount(outputTokens);
    }

    ChatChoice|ai:Error choice = getFirstChoice(response);
    if choice is ai:Error {
        span.close(choice);
        return choice;
    }
    string? finishReason = choice?.finish_reason;
    if finishReason is string {
        span.addFinishReason(finishReason);
    }

    json|ai:Error output = extractGenerateOutput(choice, responseSchema);
    if output is ai:Error {
        span.close(output);
        return output;
    }

    anydata|ai:Error result = convertGenerateOutput(output, expectedResponseTypedesc, responseSchema);
    if result is ai:Error {
        span.close(result);
        return result;
    }

    span.addOutputMessages(result.toJson());
    span.addOutputType(observe:JSON);
    span.close();
    return result;
}

# Sends the `generate` request using the configured structured output strategy, applying the runtime fallbacks.
# Sends at most `MAX_GENERATE_REQUESTS` requests.
#
# + ctx - The request context
# + return - The final attempt
isolated function sendStructuredOutputRequest(GenerateRequestContext ctx) returns GenerateAttempt {
    match ctx.compatibility.structuredOutputMode {
        JSON_SCHEMA => {
            return sendJsonSchemaRequest(ctx);
        }
        JSON_OBJECT => {
            return sendJsonObjectRequest(ctx, MAX_GENERATE_REQUESTS);
        }
    }
    return sendToolCallRequest(ctx);
}

# Sends the `generate` request using the `getResults` tool.
# - If the tool choice mode is not configured and the server rejects a forced tool choice, retries once in `AUTO`
#   mode (remembered only if that succeeds).
# - If a request in `AUTO` or `NONE` mode is rejected because of the tools, falls back to the JSON object strategy
#   (remembered only if that succeeds; the remembered fallback is used only when the tool choice mode is not
#   configured).
#
# + ctx - The request context
# + return - The final attempt
isolated function sendToolCallRequest(GenerateRequestContext ctx) returns GenerateAttempt {
    ToolChoiceMode? configuredMode = ctx.compatibility.toolChoiceMode;
    if configuredMode is () && ctx.adaptiveState.isToolsRejected() {
        return sendJsonObjectRequest(ctx, MAX_GENERATE_REQUESTS);
    }
    ToolChoiceMode mode = configuredMode ?: (ctx.adaptiveState.isForcedToolChoiceRejected() ? AUTO : FORCED);

    int requestsSent = 1;
    GenerateAttempt attempt = sendToolCallRequestWithMode(ctx, mode);
    ChatCompletionResponse|ai:Error response = attempt.response;
    if response is ai:Error && mode == FORCED && configuredMode is ()
            && (isFieldRejected(response, "tool_choice") || isFieldRejected(response, "tool")) {
        log:printDebug(string `The server rejected a forced tool choice for model '${ctx.modelType}'; ` +
                "retrying with 'tool_choice: auto'. Set 'compatibility.toolChoiceMode' to skip this retry",
                statusCode = getStatusCode(response));
        mode = AUTO;
        requestsSent += 1;
        attempt = sendToolCallRequestWithMode(ctx, AUTO);
        response = attempt.response;
        if isUsableResponse(response) {
            // Remember the fallback only when it worked, so unrelated errors do not change the mode.
            ctx.adaptiveState.markForcedToolChoiceRejected();
        }
    }

    if response is ai:Error && mode != FORCED && isFieldRejected(response, "tool") {
        log:printDebug(string `The server rejected tool calling for model '${ctx.modelType}'; ` +
                "falling back to the JSON object strategy. Set 'compatibility.structuredOutputMode' to " +
                "'JSON_OBJECT' to skip this retry", statusCode = getStatusCode(response));
        GenerateAttempt fallback = sendJsonObjectRequest(ctx, MAX_GENERATE_REQUESTS - requestsSent);
        if isUsableResponse(fallback.response) {
            ctx.adaptiveState.markToolsRejected();
        }
        return fallback;
    }
    return attempt;
}

# Sends the `generate` request using the `getResults` tool with the given tool choice mode.
#
# + ctx - The request context
# + mode - The tool choice mode
# + return - The attempt
isolated function sendToolCallRequestWithMode(GenerateRequestContext ctx, ToolChoiceMode mode)
        returns GenerateAttempt {
    json[] tools = getGetResultsTool(ctx.responseSchema.schema);
    map<json> fields = {tools};
    if mode == FORCED {
        fields["tool_choice"] = {'type: "function", 'function: {name: GET_RESULTS_TOOL_NAME}};
    } else if mode == AUTO {
        fields["tool_choice"] = "auto";
    }
    // The instruction is sent in every mode, including FORCED: some servers (e.g. Ollama) accept but ignore
    // `tool_choice`, and without the instruction the model may answer in text or misuse the tool arguments.
    json[] messages = [
        buildGenerateSystemMessage(ctx, GET_RESULTS_TOOL_INSTRUCTION),
        buildGenerateUserMessage(ctx)
    ];
    GenerateAttempt attempt = sendGenerateChatCompletion(ctx, messages, fields);
    attempt.tools = tools;
    return attempt;
}

# Sends the `generate` request with a `json_schema` response format. Falls back to the JSON object strategy for
# this call only if the server states that the field is unsupported (not when it rejects a particular schema).
#
# + ctx - The request context
# + return - The final attempt
isolated function sendJsonSchemaRequest(GenerateRequestContext ctx) returns GenerateAttempt {
    map<json> fields = {
        response_format: {
            'type: "json_schema",
            json_schema: {
                name: GET_RESULTS_TOOL_NAME,
                schema: ctx.responseSchema.schema
            }
        }
    };
    GenerateAttempt attempt = sendGenerateChatCompletion(ctx, [buildGenerateUserMessage(ctx)], fields);
    ChatCompletionResponse|ai:Error response = attempt.response;
    if response is ai:Error && !isSchemaRejected(response) && (isUnsupportedFieldError(response, "response_format")
            || isUnsupportedFieldError(response, "json_schema")) {
        log:printWarn(string `The server does not support 'response_format' of type 'json_schema' for model '${
                ctx.modelType}'; retrying with 'json_object'`, statusCode = getStatusCode(response));
        return sendJsonObjectRequest(ctx, MAX_GENERATE_REQUESTS - 1);
    }
    return attempt;
}

# Sends the `generate` request with a `json_object` response format and describes the schema in a system message.
# If the server rejects `response_format`, retries once without it and remembers that only if the retry succeeds.
# `response_format` is not sent once that is remembered, or when only one request remains in the budget (a last-resort
# attempt should not be rejectable for an optional field).
#
# + ctx - The request context
# + maxRequests - Maximum number of requests this strategy may send (at least 1)
# + return - The final attempt
isolated function sendJsonObjectRequest(GenerateRequestContext ctx, int maxRequests) returns GenerateAttempt {
    string instruction = string `Respond only with a JSON value (no prose, no code fences) that conforms to this JSON schema: ${
            ctx.responseSchema.schema.toJsonString()}`;
    json[] messages = [buildGenerateSystemMessage(ctx, instruction), buildGenerateUserMessage(ctx)];
    if maxRequests <= 1 || ctx.adaptiveState.isJsonObjectRejected() {
        return sendGenerateChatCompletion(ctx, messages, {});
    }

    GenerateAttempt attempt = sendGenerateChatCompletion(ctx, messages, {response_format: {'type: "json_object"}});
    ChatCompletionResponse|ai:Error response = attempt.response;
    if response is ai:Error && isFieldRejected(response, "response_format") {
        log:printDebug(string `The server rejected 'response_format' of type 'json_object' for model '${
                ctx.modelType}'; retrying without 'response_format'`, statusCode = getStatusCode(response));
        GenerateAttempt retryAttempt = sendGenerateChatCompletion(ctx, messages, {});
        if isUsableResponse(retryAttempt.response) {
            ctx.adaptiveState.markJsonObjectRejected();
        }
        return retryAttempt;
    }
    return attempt;
}

# Checks whether a response is usable, i.e. it is a chat completion with at least one choice. A fallback is
# remembered only when it produces a usable response.
#
# + response - The response, or its error
# + return - `true` if the response has at least one choice
isolated function isUsableResponse(ChatCompletionResponse|ai:Error response) returns boolean =>
    response is ChatCompletionResponse && getFirstChoice(response) is ChatChoice;

# Checks whether the server rejected the contents of the JSON schema (e.g. a schema keyword it does not support),
# rather than the `json_schema` response format itself. Such errors must not fall back to a mode that does not
# enforce the schema.
#
# + err - The error
# + return - `true` if the error identifies a rejected schema
isolated function isSchemaRejected(error err) returns boolean {
    if !isRequestRejected(err) {
        return false;
    }
    string text = getProviderErrorText(err);
    foreach string marker in ["invalid schema", "invalid json schema", "schema is invalid", "schema contains",
            "schema keyword", "unsupported keyword"] {
        if text.includes(marker) {
            return true;
        }
    }
    return false;
}

# Builds the request body and sends it via `sendChatCompletion`.
#
# + ctx - The request context
# + messages - The request messages
# + fields - Additional strategy-specific request fields
# + return - The attempt
isolated function sendGenerateChatCompletion(GenerateRequestContext ctx, json[] messages, map<json> fields)
        returns GenerateAttempt {
    map<json> body = {
        model: ctx.modelType,
        messages
    };
    decimal? temperature = ctx.temperature;
    if temperature is decimal {
        body["temperature"] = temperature;
    }
    foreach [string, json] [key, value] in fields.entries() {
        body[key] = value;
    }
    ChatCompletionResponse|ai:Error response = sendChatCompletion(ctx.llmClient, ctx.requestHeaders, body,
            ctx.maxTokens, ctx.compatibility, ctx.adaptiveState);
    return {response, messages};
}

# Builds the user message of a `generate` request.
#
# + ctx - The request context
# + return - The user message
isolated function buildGenerateUserMessage(GenerateRequestContext ctx) returns json => {
    role: ai:USER,
    content: ctx.userContent
};

# Builds a system message using the configured system role.
#
# + ctx - The request context
# + content - The system instruction
# + return - The system message
isolated function buildGenerateSystemMessage(GenerateRequestContext ctx, string content) returns json => {
    role: ctx.compatibility.systemRole,
    content
};

# Wraps a non-object schema under a `result` property so it can be used as tool parameters or a JSON object.
# Root-level definitions (`$defs`, `definitions`) stay on the wrapper, so root-relative `$ref`s remain valid.
#
# + schema - The original schema
# + return - The (possibly wrapped) response schema
isolated function generateJsonObjectSchema(map<json> schema) returns ResponseSchema {
    string[] supportedMetaDataFields = ["$schema", "$id", "$anchor", "$comment", "title", "description", "$defs",
        "definitions"];

    if schema["type"] == "object" {
        return {schema};
    }

    map<json> updatedSchema = map from var [key, value] in schema.entries()
        where supportedMetaDataFields.indexOf(key) is int
        select [key, value];

    updatedSchema["type"] = "object";
    map<json> content = map from var [key, value] in schema.entries()
        where supportedMetaDataFields.indexOf(key) !is int
        select [key, value];

    updatedSchema["properties"] = {[GENERATE_RESULT_FIELD]: content};
    updatedSchema["required"] = [GENERATE_RESULT_FIELD];

    return {schema: updatedSchema, isOriginallyJsonObject: false};
}

# Returns the response schema for the expected type.
#
# + expectedResponseTypedesc - The expected type
# + return - The response schema, or an error
isolated function getExpectedResponseSchema(typedesc<json> expectedResponseTypedesc) returns ResponseSchema|ai:Error =>
    generateJsonObjectSchema(check generateJsonSchemaForTypedescAsJson(expectedResponseTypedesc));

# Returns the `getResults` tool definition.
#
# + parameters - The tool parameters schema
# + return - The tools array containing the `getResults` tool
isolated function getGetResultsTool(map<json> parameters) returns json[] => [
    {
        'type: "function",
        'function: {
            name: GET_RESULTS_TOOL_NAME,
            description: "Call this tool with the final answer to the user prompt. The arguments are the answer itself, " +
                    "structured according to the parameter schema.",
            parameters
        }
    }
];

# Extracts the JSON output from a choice: the `getResults` tool call arguments (or the first tool call's), or
# otherwise JSON extracted from the text content.
#
# + choice - The first choice of the response
# + responseSchema - The expected response schema
# + return - The JSON output, or an `ai:LlmInvalidGenerationError` if there is no usable output
isolated function extractGenerateOutput(ChatChoice choice, ResponseSchema responseSchema) returns json|ai:Error {
    ResponseMessage? message = choice?.message;
    if message is () {
        return error ai:LlmInvalidGenerationError(string `${GENERATE_NO_RELEVANT_RESPONSE}: the response has no message`);
    }

    ToolCallFunction? toolCall = selectResultToolCall(message);
    if toolCall is ToolCallFunction {
        json arguments = toolCall?.arguments;
        if arguments is string {
            ParsedJson? parsed = extractJsonFromText(arguments);
            if parsed is ParsedJson {
                return parsed.value;
            }
        } else if arguments !is () {
            return arguments;
        }
    }

    string? text = getResponseText(message);
    if text is string && text.trim() != "" {
        ParsedJson? parsed = extractJsonFromText(text);
        json parsedValue = parsed?.value;
        // When a string is expected, a plain text answer is usable as is. This also covers text that happens to
        // parse as a non-string scalar or `null` (e.g. `42`, `true`), which must stay a string.
        if isWrappedStringSchema(responseSchema) && (parsed is () || parsedValue is ()|boolean|int|float|decimal) {
            string cleaned = stripThinkBlocks(text).trim();
            return extractCodeFenceContent(cleaned) ?: cleaned;
        }
        if parsed is ParsedJson {
            // A parsed JSON `null` is returned as is, so that it can be converted to a nilable expected type.
            return parsed.value;
        }
    }

    string reason = toolCall is ToolCallFunction ? "the tool call arguments are not valid JSON"
        : "the response contains neither a tool call nor JSON content";
    return error ai:LlmInvalidGenerationError(string `${GENERATE_NO_RELEVANT_RESPONSE}: ${reason}`);
}

# Selects the tool call carrying the result: the one named `getResults`, otherwise the first tool call, otherwise
# the deprecated `function_call`.
#
# + message - The response message
# + return - The selected tool call function, or `()`
isolated function selectResultToolCall(ResponseMessage message) returns ToolCallFunction? {
    ResponseToolCall[]? toolCalls = message?.tool_calls;
    if toolCalls is ResponseToolCall[] && toolCalls.length() > 0 {
        foreach ResponseToolCall toolCall in toolCalls {
            ToolCallFunction? func = toolCall?.'function;
            if func is ToolCallFunction && func.name == GET_RESULTS_TOOL_NAME {
                return func;
            }
        }
        foreach ResponseToolCall toolCall in toolCalls {
            ToolCallFunction? func = toolCall?.'function;
            if func is ToolCallFunction {
                return func;
            }
        }
    }
    return message?.function_call;
}

# Checks whether the schema was wrapped and the wrapped value is a plain string.
#
# + responseSchema - The response schema
# + return - `true` if the expected value is a wrapped string
isolated function isWrappedStringSchema(ResponseSchema responseSchema) returns boolean {
    if responseSchema.isOriginallyJsonObject {
        return false;
    }
    json properties = responseSchema.schema["properties"];
    if properties !is map<json> {
        return false;
    }
    json inner = properties[GENERATE_RESULT_FIELD];
    return inner is map<json> && inner["type"] == "string";
}

# A JSON value successfully parsed from text. Wrapping the value distinguishes a parsed JSON `null` from a
# parse failure.
type ParsedJson record {|
    # The parsed value
    json value;
|};

# Extracts a JSON value from model text output. The intact text is parsed first, so that `<think>` tags inside JSON
# string values are preserved. Otherwise, removes `<think>` blocks and markdown code fences, then tries the whole
# text, followed by the first `{...}` or `[...]` span.
#
# + text - The text output
# + return - The extracted JSON value, or `()` if none could be found
isolated function extractJsonFromText(string text) returns ParsedJson? {
    string trimmed = text.trim();
    if trimmed == "" {
        return ();
    }
    json|error intact = trimmed.fromJsonString();
    if intact is json {
        return {value: intact};
    }
    string cleaned = stripThinkBlocks(trimmed).trim();
    if cleaned == "" {
        return ();
    }
    string[] candidates = [cleaned];
    string? fenced = extractCodeFenceContent(cleaned);
    if fenced is string {
        candidates.unshift(fenced);
    }
    foreach string candidate in candidates {
        json|error parsed = candidate.fromJsonString();
        if parsed is json {
            return {value: parsed};
        }
    }
    foreach string candidate in candidates {
        json? parsed = extractJsonSpan(candidate);
        if parsed !is () {
            return {value: parsed};
        }
    }
    return ();
}

# Removes `<think>...</think>` blocks from text. Text before a dangling `</think>` is removed as well.
#
# + text - The text
# + return - The text without reasoning blocks
isolated function stripThinkBlocks(string text) returns string {
    string result = text;
    while true {
        int? startIndex = result.indexOf("<think>");
        if startIndex is () {
            break;
        }
        int? endIndex = result.indexOf("</think>", startIndex);
        if endIndex is () {
            break;
        }
        result = result.substring(0, startIndex) + result.substring(endIndex + "</think>".length());
    }
    int? danglingEnd = result.lastIndexOf("</think>");
    if danglingEnd is int {
        result = result.substring(danglingEnd + "</think>".length());
    }
    return result;
}

# Returns the content of the first markdown code fence in the text, if any.
#
# + text - The text
# + return - The fenced content, or `()` if there is no code fence
isolated function extractCodeFenceContent(string text) returns string? {
    int? fenceStart = text.indexOf("```");
    if fenceStart is () {
        return ();
    }
    int contentStart = fenceStart + 3;
    int? lineEnd = text.indexOf("\n", contentStart);
    if lineEnd is int {
        string infoString = text.substring(contentStart, lineEnd).trim();
        // Skip a language tag such as `json`, but keep content written on the same line as the fence.
        if infoString == "" || !(infoString.startsWith("{") || infoString.startsWith("[")) {
            contentStart = lineEnd + 1;
        }
    }
    int? fenceEnd = text.indexOf("```", contentStart);
    string content = fenceEnd is int ? text.substring(contentStart, fenceEnd) : text.substring(contentStart);
    content = content.trim();
    return content == "" ? () : content;
}

# Parses the first `{...}` or `[...]` span in the text (from the first opening bracket to the last matching
# closing bracket).
#
# + text - The text
# + return - The parsed JSON value, or `()`
isolated function extractJsonSpan(string text) returns json? {
    int? objectStart = text.indexOf("{");
    int? arrayStart = text.indexOf("[");
    [string, string][] bracketPairs = objectStart is int && (arrayStart is () || objectStart < arrayStart)
        ? [["{", "}"], ["[", "]"]] : [["[", "]"], ["{", "}"]];
    foreach [string, string] [open, close] in bracketPairs {
        int? startIndex = text.indexOf(open);
        int? endIndex = text.lastIndexOf(close);
        if startIndex is int && endIndex is int && endIndex > startIndex {
            json|error parsed = text.substring(startIndex, endIndex + 1).fromJsonString();
            if parsed is json {
                return parsed;
            }
        }
    }
    return ();
}

# Converts the extracted output to the expected type, unwrapping the `result` property if the schema was wrapped.
#
# + output - The extracted JSON output
# + expectedResponseTypedesc - The expected type
# + responseSchema - The response schema
# + return - The value of the expected type, or an `ai:LlmInvalidGenerationError`
isolated function convertGenerateOutput(json output, typedesc<json> expectedResponseTypedesc,
        ResponseSchema responseSchema) returns anydata|ai:Error {
    anydata|error result = parseResponseAsType(output, expectedResponseTypedesc,
            responseSchema.isOriginallyJsonObject);
    if result is error {
        return error ai:LlmInvalidGenerationError(string `Invalid value returned from the LLM Client, expected: '${
                expectedResponseTypedesc.toBalString()}', found '${output.toJsonString()}'`, result);
    }
    anydata|error typedResult = result.ensureType(expectedResponseTypedesc);
    if typedResult is error {
        return error ai:LlmInvalidGenerationError(string `Invalid value returned from the LLM Client, expected: '${
                expectedResponseTypedesc.toBalString()}', found '${(typeof result).toBalString()}'`, typedResult);
    }
    return typedResult;
}

# Converts a JSON value to the expected type. When the schema was wrapped, the value under `result` is used
# (or the value itself if the model did not wrap it).
#
# + resp - The JSON value
# + expectedResponseTypedesc - The expected type
# + isOriginallyJsonObject - Whether the schema was not wrapped
# + return - The converted value, or an error
isolated function parseResponseAsType(json resp, typedesc<json> expectedResponseTypedesc,
        boolean isOriginallyJsonObject) returns anydata|error {
    json value = resp;
    if !isOriginallyJsonObject && resp is map<json> && resp.hasKey(GENERATE_RESULT_FIELD) {
        value = resp[GENERATE_RESULT_FIELD];
    }
    anydata|error result = trap value.fromJsonWithType(expectedResponseTypedesc);
    if result is error {
        return handleParseResponseError(result);
    }
    return result;
}

# Maps JSON conversion errors to a user-friendly error.
#
# + chatResponseError - The conversion error
# + return - The mapped error
isolated function handleParseResponseError(error chatResponseError) returns error {
    string msg = chatResponseError.message();
    if msg.includes(GENERATE_JSON_CONVERSION_ERROR) || msg.includes(GENERATE_CONVERSION_ERROR) {
        return error(GENERATE_PARSE_ERROR_MESSAGE, chatResponseError);
    }
    return chatResponseError;
}
