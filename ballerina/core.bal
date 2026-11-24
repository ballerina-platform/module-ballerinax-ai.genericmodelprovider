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
import ballerina/http;
import ballerina/log;

const CHAT_COMPLETIONS_PATH = "/chat/completions";
const PROVIDER_NAME = "openai-compatible";
const MAX_REASONING_ENTRIES = 1000;
const MAX_LOGGED_BODY_LENGTH = 1000;
const REDACTED_CREDENTIAL = "***";

# Remembers request-shape fallbacks learned at runtime for a single provider instance, so that a
# rejected optional feature is retried only once per provider instance.
isolated class AdaptiveState {
    private MaxTokensField? learnedMaxTokensField = ();
    private boolean forcedToolChoiceRejected = false;
    private boolean toolsRejected = false;
    private boolean jsonObjectRejected = false;

    isolated function getMaxTokensField(readonly & CompatibilityConfig compatibility) returns MaxTokensField {
        MaxTokensField? configured = compatibility.maxTokensField;
        if configured is MaxTokensField {
            return configured;
        }
        lock {
            return self.learnedMaxTokensField ?: MAX_TOKENS;
        }
    }

    isolated function setMaxTokensField(MaxTokensField 'field) {
        lock {
            self.learnedMaxTokensField = 'field;
        }
    }

    isolated function isForcedToolChoiceRejected() returns boolean {
        lock {
            return self.forcedToolChoiceRejected;
        }
    }

    isolated function markForcedToolChoiceRejected() {
        lock {
            self.forcedToolChoiceRejected = true;
        }
    }

    isolated function isToolsRejected() returns boolean {
        lock {
            return self.toolsRejected;
        }
    }

    isolated function markToolsRejected() {
        lock {
            self.toolsRejected = true;
        }
    }

    isolated function isJsonObjectRejected() returns boolean {
        lock {
            return self.jsonObjectRejected;
        }
    }

    isolated function markJsonObjectRejected() {
        lock {
            self.jsonObjectRejected = true;
        }
    }
}

# Stores reasoning content returned with tool calls, keyed by `getReasoningKey`, so it can be sent back on the
# corresponding assistant message in later turns (only used when `preserveReasoningContent` is enabled).
# The store is bounded; the oldest entries are evicted once `MAX_REASONING_ENTRIES` is reached.
# If two responses produce the same key with different reasoning (identical tool calls whose originating response
# cannot be told apart), the key is marked ambiguous and no reasoning is returned for it, rather than reasoning that
# may belong to another response.
isolated class ReasoningStore {
    # Reasoning by key; `()` marks an ambiguous key
    private final map<string?> reasoningByToolCall = {};

    isolated function put(string key, string reasoning) {
        lock {
            if self.reasoningByToolCall.hasKey(key) {
                if self.reasoningByToolCall[key] != reasoning {
                    self.reasoningByToolCall[key] = ();
                }
                return;
            }
            if self.reasoningByToolCall.length() >= MAX_REASONING_ENTRIES {
                // Maps preserve insertion order, so the first key is the oldest entry.
                string oldest = self.reasoningByToolCall.keys()[0];
                _ = self.reasoningByToolCall.remove(oldest);
            }
            self.reasoningByToolCall[key] = reasoning;
        }
    }

    isolated function get(string key) returns string? {
        lock {
            return self.reasoningByToolCall[key];
        }
    }
}

# Builds the immutable header map sent with every request.
#
# + apiKey - API key, if any. It is sent as `Authorization: Bearer <apiKey>`. No auth header is sent when this is
#            `()` or empty
# + compatibility - Compatibility settings providing the extra headers
# + return - The headers to send with every request
isolated function buildRequestHeaders(string? apiKey, readonly & CompatibilityConfig compatibility)
        returns map<string> & readonly {
    map<string> headers = {...compatibility.extraHeaders};
    if apiKey is string && apiKey.trim() != "" {
        headers["Authorization"] = string `Bearer ${apiKey}`;
    }
    return headers.cloneReadOnly();
}

# Creates the HTTP client for the given service URL and connection configuration.
#
# + serviceUrl - Base URL of the OpenAI-compatible API (e.g. `https://api.groq.com/openai/v1`)
# + connectionConfig - Connection configuration
# + return - The HTTP client or an error
isolated function createHttpClient(string serviceUrl, ConnectionConfig connectionConfig) returns http:Client|ai:Error {
    string url = serviceUrl.trim();
    while url.endsWith("/") {
        url = url.substring(0, url.length() - 1);
    }
    if url == "" {
        return error ai:Error("The service URL of the OpenAI-compatible API must not be empty");
    }
    http:ClientConfiguration clientConfig = {
        httpVersion: connectionConfig.httpVersion,
        timeout: connectionConfig.timeout,
        forwarded: connectionConfig.forwarded,
        poolConfig: connectionConfig.poolConfig,
        compression: connectionConfig.compression,
        circuitBreaker: connectionConfig.circuitBreaker,
        retryConfig: connectionConfig.retryConfig
    };
    http:ClientHttp1Settings? http1Settings = connectionConfig.http1Settings;
    if http1Settings is http:ClientHttp1Settings {
        clientConfig.http1Settings = http1Settings;
    }
    http:ClientHttp2Settings? http2Settings = connectionConfig.http2Settings;
    if http2Settings is http:ClientHttp2Settings {
        clientConfig.http2Settings = http2Settings;
    }
    http:CacheConfig? cache = connectionConfig.cache;
    if cache is http:CacheConfig {
        clientConfig.cache = cache;
    }
    http:ResponseLimitConfigs? responseLimits = connectionConfig.responseLimits;
    if responseLimits is http:ResponseLimitConfigs {
        clientConfig.responseLimits = responseLimits;
    }
    http:ClientSecureSocket? secureSocket = connectionConfig.secureSocket;
    if secureSocket is http:ClientSecureSocket {
        clientConfig.secureSocket = secureSocket;
    }
    http:ProxyConfig? proxy = connectionConfig.proxy;
    if proxy is http:ProxyConfig {
        clientConfig.proxy = proxy;
    }
    http:Client|error httpClient = new (url, clientConfig);
    if httpClient is error {
        return error ai:Error("Failed to initialize the OpenAI-compatible model provider client", httpClient);
    }
    return httpClient;
}

# Applies `extraBody` and `dropParams` to a request body.
#
# + body - The request body built by the provider
# + compatibility - Compatibility settings
# + return - The final request body
isolated function finalizeRequestBody(map<json> body, readonly & CompatibilityConfig compatibility) returns map<json> {
    map<json> finalBody = {...body};
    foreach [string, json] [key, value] in compatibility.extraBody.entries() {
        finalBody[key] = value;
    }
    foreach string param in compatibility.dropParams {
        if finalBody.hasKey(param) {
            _ = finalBody.remove(param);
        }
    }
    return finalBody;
}

# Sends a JSON POST request and returns the JSON response. HTTP error responses and transport failures are mapped to
# `ai:LlmConnectionError`, whose detail carries `statusCode` and `body` (when available) to allow fallback decisions.
# A successful response that is not valid JSON is mapped to `ai:LlmInvalidResponseError`.
#
# + llmClient - The HTTP client
# + path - Request path relative to the service URL
# + headers - Request headers
# + body - Request body
# + return - The JSON response, or an error
isolated function postJson(http:Client llmClient, string path, map<string> & readonly headers, map<json> body)
        returns json|ai:LlmConnectionError|ai:LlmInvalidResponseError {
    http:Response|http:ClientError response = llmClient->post(path, body, headers);
    if response is http:ClientError {
        return error ai:LlmConnectionError(string `Error while connecting to the model: ${response.message()}`,
                response);
    }

    int statusCode = response.statusCode;
    if statusCode >= 300 {
        string|http:ClientError textPayload = response.getTextPayload();
        // Some gateways echo the request credentials in error responses; never return or log them.
        string responseBody = redactCredentials((textPayload is string) ? textPayload : "", headers);
        string message = string `Error while connecting to the model: HTTP ${statusCode}`;
        string? providerMessage = extractProviderErrorMessage(responseBody);
        if providerMessage is string {
            message = string `${message}: ${providerMessage}`;
        }
        log:printDebug("OpenAI-compatible endpoint returned an error response", statusCode = statusCode,
                body = truncate(responseBody, MAX_LOGGED_BODY_LENGTH));
        return error ai:LlmConnectionError(message, statusCode = statusCode, body = responseBody);
    }

    json|http:ClientError payload = response.getJsonPayload();
    if payload is http:ClientError {
        return error ai:LlmInvalidResponseError(
                "Invalid response received from the model: the response is not valid JSON", payload);
    }
    return payload;
}

# Replaces the credentials sent in the request headers (the `Authorization` header and any header whose name suggests
# a secret, such as an API key or token) with a placeholder.
#
# + text - The text to redact
# + headers - The request headers
# + return - The redacted text
isolated function redactCredentials(string text, map<string> & readonly headers) returns string {
    string redacted = text;
    foreach [string, string] [name, value] in headers.entries() {
        if !isCredentialHeader(name) {
            continue;
        }
        string secret = value.trim();
        int? schemeEnd = secret.indexOf(" ");
        if schemeEnd is int {
            string scheme = secret.substring(0, schemeEnd).toLowerAscii();
            if scheme == "bearer" || scheme == "basic" {
                secret = secret.substring(schemeEnd + 1).trim();
            }
        }
        // Credentials of any length are redacted: leaking a short key is worse than masking unrelated text.
        if secret != "" {
            redacted = replaceAllOccurrences(redacted, secret, REDACTED_CREDENTIAL);
        }
    }
    return redacted;
}

# Checks whether a header name suggests that the header carries a credential.
#
# + name - The header name
# + return - `true` if the header likely carries a credential
isolated function isCredentialHeader(string name) returns boolean {
    string lowerName = name.toLowerAscii();
    foreach string marker in ["auth", "key", "token", "secret", "cookie", "password"] {
        if lowerName.includes(marker) {
            return true;
        }
    }
    return false;
}

# Replaces every occurrence of a literal substring.
#
# + text - The text
# + target - The substring to replace (must not be empty)
# + replacement - The replacement
# + return - The text with every occurrence replaced
isolated function replaceAllOccurrences(string text, string target, string replacement) returns string {
    string result = "";
    int startIndex = 0;
    int? matchIndex = text.indexOf(target);
    while matchIndex is int {
        result += text.substring(startIndex, matchIndex) + replacement;
        startIndex = matchIndex + target.length();
        matchIndex = text.indexOf(target, startIndex);
    }
    return result + text.substring(startIndex);
}

# Extracts a human-readable message from a provider error body (`{"error": {"message": ...}}`,
# `{"error": "..."}`, `{"message": ...}` or a plain string).
#
# + body - The error response body
# + return - The extracted message, if any
isolated function extractProviderErrorMessage(anydata body) returns string? {
    json|error payload = (body is string) ? body.fromJsonString() : body.toJson();
    if payload is map<json> {
        json err = payload["error"];
        if err is map<json> && err["message"] is string {
            return <string>err["message"];
        }
        if err is string {
            return err;
        }
        if payload["message"] is string {
            return <string>payload["message"];
        }
        if payload["detail"] is string {
            return <string>payload["detail"];
        }
    }
    if body is string && body.trim() != "" {
        return body.length() > 500 ? body.substring(0, 500) : body;
    }
    return ();
}

# Returns the HTTP status code carried by an error from `postJson`, if any.
#
# + err - The error
# + return - The HTTP status code, or `()`
isolated function getStatusCode(error err) returns int? {
    var statusCode = err.detail()["statusCode"];
    return (statusCode is int) ? statusCode : ();
}

# Returns the response body carried by an error from `postJson`, if any.
#
# + err - The error
# + return - The response body, or an empty string
isolated function getErrorBody(error err) returns string {
    var body = err.detail()["body"];
    return (body is string) ? body : "";
}

# Checks whether an error is a client-side request rejection (HTTP 400 or 422).
#
# + err - The error
# + return - `true` if the server rejected the request
isolated function isRequestRejected(error err) returns boolean {
    int? statusCode = getStatusCode(err);
    return statusCode == 400 || statusCode == 422;
}

# Returns the provider's error text (message, param and code from the error body) in lower case. Falls back to the
# raw body only when it is not JSON. The raw JSON body is not used because some servers echo the whole request in
# validation errors, which would make every field appear to be rejected.
#
# + err - The error
# + return - The provider error text in lower case
isolated function getProviderErrorText(error err) returns string =>
    string:'join(" ", ...getProviderErrorDiagnostics(err));

# Returns the individual diagnostics of the provider's error body, each in lower case. The error object (its message,
# param, code and type) is one diagnostic, and each FastAPI/pydantic validation error is a separate diagnostic.
# Falls back to the raw body only when it is not JSON.
#
# + err - The error
# + includeParam - Whether the `param` of the error object is part of its diagnostic
# + return - The diagnostics in lower case
isolated function getProviderErrorDiagnostics(error err, boolean includeParam = true) returns string[] {
    string body = getErrorBody(err);
    json|error payload = body.fromJsonString();
    if payload is error {
        return [truncate(body, MAX_LOGGED_BODY_LENGTH).toLowerAscii()];
    }
    string[] diagnostics = [];
    if payload is map<json> {
        json errorField = payload["error"];
        map<json> errorSource = (errorField is map<json>) ? errorField : payload;
        string[] parts = [];
        if errorField is string {
            parts.push(errorField);
        }
        foreach string key in ["message", "param", "code", "type"] {
            json value = errorSource[key];
            if value is string|int && (includeParam || key != "param") {
                parts.push(value.toString());
            }
        }
        if parts.length() > 0 {
            diagnostics.push(string:'join(" ", ...parts));
        }
        json detail = payload["detail"];
        if detail is string {
            diagnostics.push(detail);
        } else if detail is json[] {
            // FastAPI/pydantic validation errors: use only `loc` and `msg`, not the echoed `input`.
            foreach json item in detail {
                if item is map<json> {
                    diagnostics.push(string `${item["loc"].toString()} ${item["msg"].toString()}`);
                }
            }
        }
    }
    return from string diagnostic in diagnostics
        select diagnostic.toLowerAscii();
}

# Checks whether an error is a request rejection whose provider error text mentions the given term.
#
# + err - The error
# + term - The term to look for (e.g. a request field name such as `max_tokens`)
# + return - `true` if the server rejected the request and its error text mentions the term
isolated function isFieldRejected(error err, string term) returns boolean =>
    isRequestRejected(err) && getProviderErrorText(err).includes(term.toLowerAscii());

# Checks whether an error is a request rejection stating that the given field is unsupported or unknown
# (as opposed to having an invalid value). The field and the rejection reason must appear in the same diagnostic, so
# that an unrelated diagnostic (e.g. another field that is not permitted) does not count as rejecting the field.
#
# + err - The error
# + fieldName - The request field name
# + matchParam - Whether a field named only in the error's `param` counts. When `false`, the field must be named in
#                the error message itself, so that an error about the field's contents (which reports the field as
#                its `param`) does not count as rejecting the field
# + return - `true` if the server rejected the field as unsupported
isolated function isUnsupportedFieldError(error err, string fieldName, boolean matchParam = true) returns boolean {
    if !isRequestRejected(err) {
        return false;
    }
    string lowerFieldName = fieldName.toLowerAscii();
    foreach string diagnostic in getProviderErrorDiagnostics(err, matchParam) {
        if !diagnostic.includes(lowerFieldName) {
            continue;
        }
        foreach string marker in ["unsupported", "not supported", "unknown", "unrecognized", "not allowed",
                "not permitted", "extra", "instead", "invalid parameter", "unexpected"] {
            if diagnostic.includes(marker) {
                return true;
            }
        }
    }
    return false;
}

# Checks whether a response contains at least one choice, i.e. whether it is usable as a chat completion.
#
# + response - The JSON response
# + return - `true` if the response has a non-empty `choices` array
isolated function hasChoices(json response) returns boolean {
    if response !is map<json> {
        return false;
    }
    json choices = response["choices"];
    return choices is json[] && choices.length() > 0;
}

# Truncates a string to the given length.
#
# + value - The string
# + maxLength - The maximum length
# + return - The truncated string
isolated function truncate(string value, int maxLength) returns string =>
    value.length() > maxLength ? value.substring(0, maxLength) : value;

# Sends a chat completion request. Applies the max tokens field (falling back between `max_tokens` and
# `max_completion_tokens` once if the server rejects the field), `extraBody` and `dropParams`, and parses the
# response leniently.
#
# + llmClient - The HTTP client
# + headers - Request headers
# + body - Request body without the max tokens field
# + maxTokens - Maximum tokens to generate. The field is omitted when `()`
# + compatibility - Compatibility settings
# + adaptiveState - Learned fallbacks for this provider instance
# + return - The parsed response, or an error
isolated function sendChatCompletion(http:Client llmClient, map<string> & readonly headers, map<json> body,
        int? maxTokens, readonly & CompatibilityConfig compatibility, AdaptiveState adaptiveState)
        returns ChatCompletionResponse|ai:Error {
    MaxTokensField maxTokensField = adaptiveState.getMaxTokensField(compatibility);
    json|ai:Error response = postJson(llmClient, CHAT_COMPLETIONS_PATH, headers,
            buildChatBody(body, maxTokens, maxTokensField, compatibility));

    if response is ai:LlmConnectionError && maxTokens is int && compatibility.maxTokensField is ()
            && isUnsupportedFieldError(response, maxTokensField) {
        MaxTokensField alternative = maxTokensField == MAX_TOKENS ? MAX_COMPLETION_TOKENS : MAX_TOKENS;
        // Expected adaptation (e.g. OpenAI reasoning models only accept 'max_completion_tokens'); it is remembered,
        // so it costs one extra request per provider instance. Logged at debug level for that reason.
        log:printDebug(string `The server rejected '${maxTokensField}'; retrying with '${alternative}'. Set ` +
                string `'compatibility.maxTokensField' to skip this retry`);
        response = postJson(llmClient, CHAT_COMPLETIONS_PATH, headers,
                buildChatBody(body, maxTokens, alternative, compatibility));
        if response is json && hasChoices(response) {
            // Remember the alternative only after a usable completion, so an empty retry does not change the field.
            adaptiveState.setMaxTokensField(alternative);
        }
    }

    if response is ai:Error {
        return response;
    }
    ChatCompletionResponse|error parsed = response.cloneWithType();
    if parsed is error {
        return error ai:LlmInvalidResponseError("Invalid response received from the model", parsed);
    }
    return parsed;
}

# Builds the final chat completion request body.
#
# + body - The request body without the max tokens field
# + maxTokens - Maximum tokens to generate. The field is omitted when `()`
# + maxTokensField - The field used to send the maximum tokens
# + compatibility - Compatibility settings
# + return - The final request body
isolated function buildChatBody(map<json> body, int? maxTokens, MaxTokensField maxTokensField,
        readonly & CompatibilityConfig compatibility) returns map<json> {
    map<json> requestBody = {...body};
    if maxTokens is int {
        requestBody[maxTokensField] = maxTokens;
    }
    return finalizeRequestBody(requestBody, compatibility);
}

# Returns the first choice of a chat completion response.
#
# + response - The chat completion response
# + return - The first choice, or an `ai:LlmInvalidResponseError` if there are none
isolated function getFirstChoice(ChatCompletionResponse response) returns ChatChoice|ai:LlmInvalidResponseError {
    ChatChoice[]? choices = response?.choices;
    if choices is () || choices.length() == 0 {
        return error ai:LlmInvalidResponseError("Empty response from the model");
    }
    return choices[0];
}

# Returns the text content of a response message. Handles string content and arrays of text parts.
#
# + message - The response message
# + return - The text content, or `()` if there is none
isolated function getResponseText(ResponseMessage message) returns string? {
    json content = message?.content;
    if content is string {
        return content;
    }
    if content is json[] {
        string[] parts = [];
        foreach json part in content {
            if part is string {
                parts.push(part);
            } else if part is map<json> && part["text"] is string {
                parts.push(<string>part["text"]);
            }
        }
        return parts.length() > 0 ? string:'join("", ...parts) : ();
    }
    return ();
}

# Returns the reasoning content of a response message from `reasoning_content` or `reasoning`.
#
# + message - The response message
# + return - The reasoning content, or `()`
isolated function getReasoningContent(ResponseMessage message) returns string? =>
    message?.reasoning_content ?: message?.reasoning;
