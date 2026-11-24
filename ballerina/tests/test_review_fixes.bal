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
import ballerina/test;

const STYLE_PATTERN_SCHEMA = "patternschema";

// ---------------------------------------------------------------------------------------------------------------------
// Credential redaction
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["errors"]}
function testEchoedCredentialIsRedactedFromError() {
    ai:ChatAssistantMessage|ai:Error result = secretKeyProvider->chat(
            {role: ai:USER, content: "TRIGGER_ECHO_AUTH hello"});
    assertConnectionErrorWithoutSecret(result, 401);
    if result is ai:Error {
        test:assertTrue(result.message().includes(REDACTED_CREDENTIAL),
                string `The credential must be replaced with a placeholder, found: ${result.message()}`);
        test:assertFalse(getErrorBody(result).includes(TEST_SECRET_API_KEY),
                "The API key must not appear in the error body");
    }
}

@test:Config {groups: ["errors"]}
function testRedactCredentialsCoversCredentialHeaders() {
    map<string> & readonly headers = {
        "Authorization": "Bearer sk-bearer-secret-1",
        "x-api-key": "api-key-secret-2",
        "X-Request-Source": "integration-tests"
    };
    string redacted = redactCredentials(
            "a sk-bearer-secret-1 b api-key-secret-2 c integration-tests sk-bearer-secret-1", headers);
    test:assertEquals(redacted, "a *** b *** c integration-tests ***");
}

// ---------------------------------------------------------------------------------------------------------------------
// Unsupported-field detection
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["adaptive"]}
function testUnsupportedFieldRequiresFieldAndReasonInSameDiagnostic() {
    json body = {
        "detail": [
            {"loc": ["body", "max_tokens"], "msg": "Input should be less than or equal to 4096"},
            {"loc": ["body", "foo"], "msg": "Extra inputs are not permitted"}
        ]
    };
    ai:LlmConnectionError err = error ai:LlmConnectionError("rejected", statusCode = 422, body = body.toJsonString());
    test:assertFalse(isUnsupportedFieldError(err, "max_tokens"),
            "A value error on 'max_tokens' combined with an unrelated diagnostic must not count as unsupported");

    json unsupported = {"detail": [{"loc": ["body", "max_tokens"], "msg": "Extra inputs are not permitted"}]};
    ai:LlmConnectionError unsupportedErr = error ai:LlmConnectionError("rejected", statusCode = 422,
            body = unsupported.toJsonString());
    test:assertTrue(isUnsupportedFieldError(unsupportedErr, "max_tokens"));

    json openAiStyle = {"error": {"message": "Unrecognized request argument supplied", "param": "max_tokens"}};
    ai:LlmConnectionError openAiErr = error ai:LlmConnectionError("rejected", statusCode = 400,
            body = openAiStyle.toJsonString());
    test:assertTrue(isUnsupportedFieldError(openAiErr, "max_tokens"),
            "The message and param of one error object form a single diagnostic");
}

@test:Config {groups: ["adaptive"]}
function testJsonSchemaKeywordRejectionDoesNotFallBack() returns error? {
    ModelProvider schemaProvider = check new (mockUrl(STYLE_PATTERN_SCHEMA), TEST_MODEL, TEST_API_KEY,
            compatibility = {structuredOutputMode: JSON_SCHEMA});
    int initialCount = getRequestCount(STYLE_PATTERN_SCHEMA);
    Review|error result = schemaProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertTrue(result.message().includes(MOCK_SCHEMA_KEYWORD_ERROR),
            string `The schema error must propagate, found: ${result.message()}`);
    test:assertEquals(getRequestCount(STYLE_PATTERN_SCHEMA), initialCount + 1,
            "A rejected schema keyword must not trigger the json_object fallback");
}

// ---------------------------------------------------------------------------------------------------------------------
// Fallbacks are remembered only after a usable response
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["adaptive"]}
function testMaxTokensFallbackIsNotRememberedAfterEmptyRetry() returns error? {
    ModelProvider oProvider = check new (mockUrl(OSERIES), "o4-mini", TEST_API_KEY, maxTokens = 32);
    ai:ChatAssistantMessage|ai:Error first = oProvider->chat({role: ai:USER, content: "TRIGGER_EMPTY_CHOICES hi"});
    test:assertTrue(first is ai:LlmInvalidResponseError, string `Expected an empty response error: ${describe(first)}`);

    int initialCount = getRequestCount(OSERIES);
    _ = check oProvider->chat({role: ai:USER, content: "Next question"});
    map<json>[] bodies = getRequestBodiesSince(OSERIES, initialCount);
    test:assertEquals(bodies.length(), 2, "The next call must start again with 'max_tokens'");
    test:assertTrue(bodies[0].hasKey("max_tokens"));
}

@test:Config {groups: ["adaptive"]}
function testAutoToolChoiceFallbackIsNotRememberedAfterEmptyRetry() returns error? {
    ModelProvider zaiProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY);
    int|error first = zaiProvider->generate(`GEN_INT TRIGGER_EMPTY_CHOICES What is 1 + 1?`);
    test:assertTrue(first is error, "An empty retry must fail");

    int initialCount = getRequestCount(ZAI);
    int second = check zaiProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(second, 2);
    map<json>[] bodies = getRequestBodiesSince(ZAI, initialCount);
    test:assertEquals(bodies.length(), 2, "The next call must try a forced tool choice again");
    test:assertEquals(bodies[0]["tool_choice"], FORCED_RESULTS_TOOL_CHOICE);
}

// ---------------------------------------------------------------------------------------------------------------------
// Generate output parsing
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["generate"]}
function testGeneratePreservesThinkTagsInsideJsonStrings() returns error? {
    string result = check provider->generate(`GEN_THINK_STRING Say something`);
    test:assertEquals(result, MOCK_THINK_TAG_STRING);
}

@test:Config {groups: ["generate"]}
function testGenerateBareNullTextForNilableType() returns error? {
    ModelProvider ollamaProvider = check new (mockUrl(OLLAMA), "llama3.2");
    string? result = check ollamaProvider->generate(`GEN_BARE_NULL Tell me nothing`);
    test:assertEquals(result, ());
}

@test:Config {groups: ["generate"]}
function testWrappedSchemaKeepsRootDefinitions() {
    map<json> schema = {
        "type": "array",
        "items": {"$ref": "#/$defs/Entry"},
        "$defs": {"Entry": {"type": "object", "properties": {"name": {"type": "string"}}}}
    };
    ResponseSchema wrapped = generateJsonObjectSchema(schema);
    test:assertFalse(wrapped.isOriginallyJsonObject);
    test:assertEquals(wrapped.schema["$defs"], schema["$defs"]);
    test:assertEquals(wrapped.schema["properties"], {"result": {"type": "array", "items": {"$ref": "#/$defs/Entry"}}});
}

// ---------------------------------------------------------------------------------------------------------------------
// Tool call IDs and reasoning
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["chat"]}
function testFallbackToolCallIdsSkipExplicitIds() returns error? {
    ai:ChatMessage[] messages = [
        {role: ai:USER, content: "Weather?"},
        {
            role: ai:ASSISTANT,
            toolCalls: [
                {name: "getWeather", id: "call_getWeather_1", arguments: {"city": "Colombo"}},
                {name: "getWeather", arguments: {"city": "Kandy"}}
            ]
        },
        {role: "function", name: "getWeather", id: "call_getWeather_1", content: "31C"},
        {role: "function", name: "getWeather", content: "27C"}
    ];
    json[] request = check buildChatRequestMessages(messages, SYSTEM, ());
    map<json> assistant = check asMap(request[1]);
    map<json>[] toolCalls = getToolCalls(assistant);
    test:assertEquals(toolCalls[0]["id"], "call_getWeather_1");
    test:assertEquals(toolCalls[1]["id"], "call_getWeather_2",
            "A fallback ID must not reuse an explicit ID in the transcript");
    test:assertEquals((check asMap(request[3]))["tool_call_id"], "call_getWeather_2");
}

@test:Config {groups: ["chat"]}
function testReasoningIsPreservedForToolCallWithoutId() returns error? {
    ReasoningStore store = new;
    ai:ChatAssistantMessage assistant = check convertToAssistantMessage(check toChoice({
        "message": {
            "role": "assistant",
            "reasoning_content": MOCK_REASONING_TEXT,
            "tool_calls": [{"id": (), "type": "function", "function": {"name": "getWeather", "arguments": "{}"}}]
        }
    }), store);
    json[] request = check buildChatRequestMessages([{role: ai:USER, content: "Weather?"}, assistant], SYSTEM, store);
    test:assertEquals((check asMap(request[1]))["reasoning_content"], MOCK_REASONING_TEXT);
}

@test:Config {groups: ["chat"]}
function testReusedToolCallIdKeepsReasoningOfItsOwnResponse() returns error? {
    ReasoningStore store = new;
    ai:ChatAssistantMessage first = check convertToAssistantMessage(check toChoice(
            toolCallChoice("call_0", "{\"city\":\"Colombo\"}", "Reasoning of conversation A")), store);
    ai:ChatAssistantMessage second = check convertToAssistantMessage(check toChoice(
            toolCallChoice("call_0", "{\"city\":\"Kandy\"}", "Reasoning of conversation B")), store);

    json[] requestA = check buildChatRequestMessages([{role: ai:USER, content: "A"}, first], SYSTEM, store);
    json[] requestB = check buildChatRequestMessages([{role: ai:USER, content: "B"}, second], SYSTEM, store);
    test:assertEquals((check asMap(requestA[1]))["reasoning_content"], "Reasoning of conversation A");
    test:assertEquals((check asMap(requestB[1]))["reasoning_content"], "Reasoning of conversation B");
}

function toolCallChoice(string id, string arguments, string reasoning) returns json => {
    "message": {
        "role": "assistant",
        "reasoning_content": reasoning,
        "tool_calls": [{"id": id, "type": "function", "function": {"name": "getWeather", "arguments": arguments}}]
    }
};

function toChoice(json choice) returns ChatChoice|error => choice.cloneWithType();

function asMap(json value) returns map<json>|error => value.ensureType();
