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
const STYLE_KEYWORD_SCHEMA = "keywordschema";

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

@test:Config {groups: ["errors"]}
function testShortCredentialIsRedacted() returns error? {
    test:assertEquals(redactCredentials("bad key: k1", {"Authorization": "Bearer k1"}), "bad key: ***");

    ModelProvider shortKeyProvider = check new (mockUrl(STANDARD), TEST_MODEL, "sk-7");
    ai:ChatAssistantMessage|ai:Error result = shortKeyProvider->chat(
            {role: ai:USER, content: "TRIGGER_ECHO_AUTH hello"});
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertFalse(result.toString().includes("sk-7"), "A short API key must not appear in the error");
    test:assertFalse(getErrorBody(result).includes("sk-7"), "A short API key must not appear in the error body");
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
function testJsonSchemaBareKeywordRejectionDoesNotFallBack() returns error? {
    ModelProvider schemaProvider = check new (mockUrl(STYLE_KEYWORD_SCHEMA), TEST_MODEL, TEST_API_KEY,
            compatibility = {structuredOutputMode: JSON_SCHEMA});
    int initialCount = getRequestCount(STYLE_KEYWORD_SCHEMA);
    Review|error result = schemaProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertTrue(result.message().includes(MOCK_BARE_KEYWORD_ERROR),
            string `The schema error must propagate, found: ${result.message()}`);
    test:assertEquals(getRequestCount(STYLE_KEYWORD_SCHEMA), initialCount + 1,
            "An error that names response_format only as its param must not trigger the json_object fallback");
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
function testGeneratePreservesThinkTagsInsideFencedJson() returns error? {
    ModelProvider ollamaProvider = check new (mockUrl(OLLAMA), "llama3.2");
    string result = check ollamaProvider->generate(`GEN_THINK_STRING Say something`);
    test:assertEquals(result, MOCK_THINK_TAG_STRING);
}

@test:Config {groups: ["generate"]}
function testGenerateIgnoresDraftJsonInLeadingThinkBlock() returns error? {
    ModelProvider ollamaProvider = check new (mockUrl(OLLAMA), "llama3.2");
    int result = check ollamaProvider->generate(`GEN_INT GEN_THINK_PREFIX What is 1 + 1?`);
    test:assertEquals(result, 2);
}

@test:Config {groups: ["generate"]}
function testExtractJsonFromTextFallsBackToRemovingAllThinkBlocks() {
    ParsedJson? parsed = extractJsonFromText("Answer: <think>maybe {broken</think> {\"a\": 1}");
    test:assertEquals(parsed, {value: {"a": 1}});
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

@test:Config {groups: ["chat"]}
function testIdenticalToolCallsWithDifferentReasoningOmitReasoning() returns error? {
    foreach string? id in ["call_0", ()] {
        ReasoningStore store = new;
        ai:ChatAssistantMessage first = check convertToAssistantMessage(check toChoice(
                toolCallChoiceWithId(id, "{\"city\":\"Colombo\"}", "Reasoning of conversation A")), store);
        ai:ChatAssistantMessage second = check convertToAssistantMessage(check toChoice(
                toolCallChoiceWithId(id, "{\"city\":\"Colombo\"}", "Reasoning of conversation B")), store);

        json[] requestA = check buildChatRequestMessages([{role: ai:USER, content: "A"}, first], SYSTEM, store);
        json[] requestB = check buildChatRequestMessages([{role: ai:USER, content: "B"}, second], SYSTEM, store);
        test:assertFalse((check asMap(requestA[1])).hasKey("reasoning_content"),
                "Reasoning must be omitted when its response cannot be identified");
        test:assertFalse((check asMap(requestB[1])).hasKey("reasoning_content"),
                "Reasoning must be omitted when its response cannot be identified");
    }
}

@test:Config {groups: ["chat"]}
function testIdenticalToolCallsWithSameReasoningKeepReasoning() returns error? {
    ReasoningStore store = new;
    ai:ChatAssistantMessage first = check convertToAssistantMessage(check toChoice(
            toolCallChoiceWithId((), "{}", MOCK_REASONING_TEXT)), store);
    _ = check convertToAssistantMessage(check toChoice(toolCallChoiceWithId((), "{}", MOCK_REASONING_TEXT)), store);
    json[] request = check buildChatRequestMessages([{role: ai:USER, content: "A"}, first], SYSTEM, store);
    test:assertEquals((check asMap(request[1]))["reasoning_content"], MOCK_REASONING_TEXT);
}

function toolCallChoiceWithId(string? id, string arguments, string reasoning) returns json => {
    "message": {
        "role": "assistant",
        "reasoning_content": reasoning,
        "tool_calls": [{"id": id, "type": "function", "function": {"name": "getWeather", "arguments": arguments}}]
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Second round of follow-ups
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["errors"]}
function testJsonEscapedCredentialIsRedacted() returns error? {
    map<string> & readonly headers = {"x-api-key": "abc\"def12"};
    test:assertEquals(redactCredentials("{\"error\": \"bad key abc\\\"def12\"}", headers), "{\"error\": \"bad key ***\"}");

    string quotedKey = "k\"y";
    ModelProvider quotedKeyProvider = check new (mockUrl(STANDARD), TEST_MODEL, quotedKey);
    ai:ChatAssistantMessage|ai:Error result = quotedKeyProvider->chat(
            {role: ai:USER, content: "TRIGGER_ECHO_AUTH hello"});
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertFalse(result.message().includes(quotedKey), "The decoded key must not appear in the message");
    test:assertFalse(getErrorBody(result).includes("k\\\"y"), "The escaped key must not appear in the body");
    test:assertTrue(result.message().includes(REDACTED_CREDENTIAL));
}

@test:Config {groups: ["errors"]}
function testShortCredentialIsRedactedOnlyWhereItStandsAlone() {
    test:assertEquals(redactCredentials("key a rejected; 'max_tokens' is not supported",
            {"Authorization": "Bearer a"}), "key *** rejected; 'max_tokens' is not supported");
}

@test:Config {groups: ["adaptive"]}
function testMaxTokensFallbackWorksWithShortApiKey() returns error? {
    ModelProvider oProvider = check new (mockUrl(OSERIES), "o4-mini", "a", maxTokens = 16);
    int initialCount = getRequestCount(OSERIES);
    ai:ChatAssistantMessage response = check oProvider->chat({role: ai:USER, content: "Short key"});
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Short key");
    test:assertEquals(getRequestCount(OSERIES), initialCount + 2,
            "Redacting a short key must not prevent the 'max_completion_tokens' retry");
}

@test:Config {groups: ["adaptive"]}
function testJsonSchemaFormatUnsupportedDetection() {
    [json, boolean][] cases = [
        [{"error": {"message": "For response_format, patternProperties is not supported"}}, false],
        [{"error": {"param": "response_format", "message": "patternProperties is not supported"}}, false],
        [{"error": {"message": "Invalid schema for response_format 'x': 'patternProperties' is not supported"}}, false],
        [{"error": {"message": MOCK_JSON_SCHEMA_ERROR}}, true],
        [{"error": {"message": "Unrecognized request argument supplied: response_format"}}, true],
        [{"error": {"message": "response_format is not supported by this model"}}, true],
        [{"detail": [{"loc": ["body", "response_format"], "msg": "Extra inputs are not permitted"}]}, true]
    ];
    foreach [json, boolean] [body, expected] in cases {
        ai:LlmConnectionError err = error ai:LlmConnectionError("rejected", statusCode = 400,
                body = body.toJsonString());
        test:assertEquals(isJsonSchemaFormatUnsupported(err), expected, body.toJsonString());
    }
}

@test:Config {groups: ["chat"]}
function testToolCallWithoutReasoningMakesIdenticalCallAmbiguous() returns error? {
    foreach boolean reasoningFirst in [true, false] {
        ReasoningStore store = new;
        json withReasoning = toolCallChoiceWithId((), "{}", MOCK_REASONING_TEXT);
        json withoutReasoning = {
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": (), "type": "function", "function": {"name": "getWeather", "arguments": "{}"}}]
            }
        };
        ai:ChatAssistantMessage first = check convertToAssistantMessage(
                check toChoice(reasoningFirst ? withReasoning : withoutReasoning), store);
        ai:ChatAssistantMessage second = check convertToAssistantMessage(
                check toChoice(reasoningFirst ? withoutReasoning : withReasoning), store);
        foreach ai:ChatAssistantMessage assistant in [first, second] {
            json[] request = check buildChatRequestMessages([{role: ai:USER, content: "Q"}, assistant], SYSTEM, store);
            test:assertFalse((check asMap(request[1])).hasKey("reasoning_content"),
                    "Reasoning must be omitted when an identical call arrived without reasoning");
        }
    }
}

@test:Config {groups: ["chat"]}
function testEvictedReasoningKeyIsAmbiguousWhenReused() {
    ReasoningStore store = new;
    store.put("reused-key", "Reasoning of conversation A");
    foreach int i in 0 ..< MAX_REASONING_ENTRIES {
        store.put(string `other-key-${i}`, "Other reasoning");
    }
    test:assertEquals(store.get("reused-key"), (), "The oldest key must have been evicted");
    store.put("reused-key", "Reasoning of conversation B");
    test:assertEquals(store.get("reused-key"), (),
            "A key evicted earlier must not return reasoning that may belong to another conversation");
}

@test:Config {groups: ["generate"]}
function testGeneratePreservesClosingThinkTagInsideFencedJson() returns error? {
    ModelProvider ollamaProvider = check new (mockUrl(OLLAMA), "llama3.2");
    string result = check ollamaProvider->generate(`GEN_CLOSE_TAG_STRING Say something`);
    test:assertEquals(result, MOCK_CLOSE_TAG_STRING);
}
