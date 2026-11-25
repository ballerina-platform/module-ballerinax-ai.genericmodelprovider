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

@test:Config {groups: ["adaptive"]}
function testMaxTokensFallsBackToMaxCompletionTokensAndIsRemembered() returns error? {
    ModelProvider oProvider = check new (mockUrl(OSERIES), "o4-mini", TEST_API_KEY, maxTokens = 128);

    int initialCount = getRequestCount(OSERIES);
    ai:ChatAssistantMessage first = check oProvider->chat({role: ai:USER, content: "First question"});
    test:assertEquals(first.content, MOCK_TEXT_RESPONSE_PREFIX + "First question");
    test:assertEquals(getRequestCount(OSERIES), initialCount + 2,
            "The first call must be retried once with 'max_completion_tokens'");
    map<json> body = getLastRequestBody(OSERIES);
    test:assertEquals(body["max_completion_tokens"], 128);
    test:assertFalse(body.hasKey("max_tokens"));

    ai:ChatAssistantMessage second = check oProvider->chat({role: ai:USER, content: "Second question"});
    test:assertEquals(second.content, MOCK_TEXT_RESPONSE_PREFIX + "Second question");
    test:assertEquals(getRequestCount(OSERIES), initialCount + 3,
            "The learned 'max_completion_tokens' field must be used directly on the next call");
    body = getLastRequestBody(OSERIES);
    test:assertEquals(body["max_completion_tokens"], 128);
    test:assertFalse(body.hasKey("max_tokens"));
}

@test:Config {groups: ["adaptive"]}
function testConfiguredMaxTokensFieldIsUsedDirectly() returns error? {
    ModelProvider oProvider = check new (mockUrl(OSERIES), "o4-mini", TEST_API_KEY, maxTokens = 64,
            compatibility = {maxTokensField: MAX_COMPLETION_TOKENS});
    int initialCount = getRequestCount(OSERIES);
    _ = check oProvider->chat({role: ai:USER, content: "Configured field"});
    test:assertEquals(getRequestCount(OSERIES), initialCount + 1);
    test:assertEquals(getLastRequestBody(OSERIES)["max_completion_tokens"], 64);
}

@test:Config {groups: ["adaptive"]}
function testZaiGenerateFallsBackToAutoToolChoiceAndRemembers() returns error? {
    ModelProvider zaiProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY);

    int initialCount = getRequestCount(ZAI);
    int first = check zaiProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(first, 2);
    test:assertEquals(getRequestCount(ZAI), initialCount + 2,
            "A rejected forced tool choice must be retried once with 'auto'");
    test:assertEquals(getLastRequestBody(ZAI)["tool_choice"], "auto");

    Review second = check zaiProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    test:assertEquals(second, reviewRecord);
    test:assertEquals(getRequestCount(ZAI), initialCount + 3,
            "The learned 'auto' tool choice must be used directly on the next call");
    test:assertEquals(getLastRequestBody(ZAI)["tool_choice"], "auto");
}

@test:Config {groups: ["adaptive"]}
function testZaiGenerateWithConfiguredAutoToolChoice() returns error? {
    ModelProvider zaiProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY,
            compatibility = {toolChoiceMode: AUTO});
    int initialCount = getRequestCount(ZAI);
    boolean result = check zaiProvider->generate(`GEN_BOOL Is the sky blue?`);
    test:assertTrue(result);
    test:assertEquals(getRequestCount(ZAI), initialCount + 1);
    test:assertEquals(getLastRequestBody(ZAI)["tool_choice"], "auto");
}

@test:Config {groups: ["adaptive"]}
function testOllamaGenerateParsesFencedJsonText() returns error? {
    ModelProvider ollamaProvider = check new (mockUrl(OLLAMA), "llama3.2");
    Review review = check ollamaProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    test:assertEquals(review, reviewRecord);

    int count = check ollamaProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(count, 2);
}

@test:Config {groups: ["adaptive"]}
function testJsonSchemaFallsBackToJsonObjectOnEveryCall() returns error? {
    ModelProvider schemaProvider = check new (mockUrl(NO_JSON_SCHEMA), TEST_MODEL, TEST_API_KEY,
            compatibility = {structuredOutputMode: JSON_SCHEMA});

    // JSON_SCHEMA mode remembers nothing: every call tries json_schema first and falls back to json_object.
    foreach string title in [blog2.title, blog1.title] {
        int initialCount = getRequestCount(NO_JSON_SCHEMA);
        Review review = check schemaProvider->generate(`GEN_REVIEW Please review this blog: ${title}`);
        test:assertEquals(review, reviewRecord);
        map<json>[] bodies = getRequestBodiesSince(NO_JSON_SCHEMA, initialCount);
        test:assertEquals(bodies.length(), 2, "Each call must try json_schema and then json_object");
        test:assertEquals(getResponseFormatType(bodies[0]), "json_schema");
        test:assertEquals(bodies[1]["response_format"], {"type": "json_object"});
        test:assertFalse(bodies[1].hasKey("tools"), "No tools are expected in JSON_OBJECT mode");
    }
}

@test:Config {groups: ["adaptive"]}
function testJsonSchemaInvalidSchemaDoesNotFallBack() returns error? {
    ModelProvider schemaProvider = check new (mockUrl(STYLE_BAD_SCHEMA), TEST_MODEL, TEST_API_KEY,
            compatibility = {structuredOutputMode: JSON_SCHEMA});
    int initialCount = getRequestCount(STYLE_BAD_SCHEMA);
    Review|error result = schemaProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertTrue(result.message().includes("Invalid schema for response_format"),
            string `The provider error must propagate, found: ${result.message()}`);
    test:assertEquals(getRequestCount(STYLE_BAD_SCHEMA), initialCount + 1,
            "An invalid-schema rejection must not trigger the json_object fallback");
}

@test:Config {groups: ["adaptive"]}
function testUnrelatedErrorInForcedModeIsNotRetried() returns error? {
    ModelProvider forcedProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY);
    int initialCount = getRequestCount(STANDARD);
    int|error result = forcedProvider->generate(`GEN_INT TRIGGER_CONTEXT_LENGTH What is 1 + 1?`);
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertEquals(result.detail()["statusCode"], 400);
    test:assertTrue(result.message().includes("context length exceeded"));
    test:assertEquals(getRequestCount(STANDARD), initialCount + 1, "An unrelated 400 must not be retried");

    // The next call still forces the result tool.
    int next = check forcedProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(next, 2);
    test:assertEquals(getRequestCount(STANDARD), initialCount + 2);
    test:assertEquals(getLastRequestBody(STANDARD)["tool_choice"], FORCED_RESULTS_TOOL_CHOICE);
}

@test:Config {groups: ["adaptive"]}
function testToolsRejectedFallsBackToJsonObjectAndRemembers() returns error? {
    ModelProvider noToolsProvider = check new (mockUrl(STYLE_NO_TOOLS), "llama2", TEST_API_KEY);

    int initialCount = getRequestCount(STYLE_NO_TOOLS);
    Review first = check noToolsProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    test:assertEquals(first, reviewRecord);
    map<json>[] bodies = getRequestBodiesSince(STYLE_NO_TOOLS, initialCount);
    test:assertTrue(bodies.length() <= 3, string `At most 3 requests are allowed per call, found ${bodies.length()}`);
    map<json> lastBody = bodies[bodies.length() - 1];
    test:assertFalse(lastBody.hasKey("tools"), "The successful request must not carry tools");

    initialCount = getRequestCount(STYLE_NO_TOOLS);
    int second = check noToolsProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(second, 2);
    bodies = getRequestBodiesSince(STYLE_NO_TOOLS, initialCount);
    test:assertEquals(bodies.length(), 1, "The tools-rejected fallback must be remembered");
    test:assertFalse(bodies[0].hasKey("tools"));
    test:assertFalse(bodies[0].hasKey("tool_choice"));
    test:assertEquals(bodies[0]["response_format"], {"type": "json_object"});
}

@test:Config {groups: ["adaptive"]}
function testToolsAndResponseFormatRejected() returns error? {
    ModelProvider bareProvider = check new (mockUrl(STYLE_BARE), "tiny-model", TEST_API_KEY);

    int initialCount = getRequestCount(STYLE_BARE);
    Review first = check bareProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    test:assertEquals(first, reviewRecord);
    test:assertTrue(getRequestCount(STYLE_BARE) - initialCount <= 3, "At most 3 requests are allowed per call");

    // Whatever the second call needs, the json_object rejection must be remembered for the calls after it.
    int second = check bareProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(second, 2);

    initialCount = getRequestCount(STYLE_BARE);
    boolean third = check bareProvider->generate(`GEN_BOOL Is the sky blue?`);
    test:assertTrue(third);
    map<json>[] bodies = getRequestBodiesSince(STYLE_BARE, initialCount);
    test:assertEquals(bodies.length(), 1, "The json_object rejection must be remembered");
    test:assertFalse(bodies[0].hasKey("tools"));
    test:assertFalse(bodies[0].hasKey("response_format"));
}

@test:Config {groups: ["adaptive"]}
function testJsonObjectModeRemembersResponseFormatRejection() returns error? {
    ModelProvider bareProvider = check new (mockUrl(STYLE_BARE), "tiny-model", TEST_API_KEY,
            compatibility = {structuredOutputMode: JSON_OBJECT});

    int initialCount = getRequestCount(STYLE_BARE);
    Review first = check bareProvider->generate(`GEN_REVIEW Please review this blog: ${blog2.title}`);
    test:assertEquals(first, reviewRecord);
    map<json>[] bodies = getRequestBodiesSince(STYLE_BARE, initialCount);
    test:assertEquals(bodies.length(), 2);
    test:assertEquals(bodies[0]["response_format"], {"type": "json_object"});
    test:assertFalse(bodies[1].hasKey("response_format"));

    initialCount = getRequestCount(STYLE_BARE);
    Review second = check bareProvider->generate(`GEN_REVIEW Please review this blog: ${blog1.title}`);
    test:assertEquals(second, reviewRecord);
    test:assertEquals(getRequestCount(STYLE_BARE), initialCount + 1);
    test:assertFalse(getLastRequestBody(STYLE_BARE).hasKey("response_format"));
}

@test:Config {groups: ["adaptive"]}
function testMaxTokensUnrelatedErrorIsNotRetried() {
    assertMaxTokensNotRetried("TRIGGER_MT_UNRELATED", 400);
}

@test:Config {groups: ["adaptive"]}
function testMaxTokensValueErrorIsNotRetried() {
    assertMaxTokensNotRetried("TRIGGER_MT_VALUE", 400);
}

@test:Config {groups: ["adaptive"]}
function testMaxTokensFastApiEchoedInputIsNotRetried() {
    assertMaxTokensNotRetried("TRIGGER_MT_FASTAPI", 422);
}

@test:Config {groups: ["adaptive"]}
function testMaxTokensBothFieldsRejected() returns error? {
    ModelProvider mtProvider = check new (mockUrl(STYLE_MAX_TOKENS), TEST_MODEL, TEST_API_KEY, maxTokens = 100);
    int initialCount = getRequestCount(STYLE_MAX_TOKENS);
    ai:ChatAssistantMessage|ai:Error result = mtProvider->chat({role: ai:USER, content: "TRIGGER_MT_BOTH hi"});
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    map<json>[] bodies = getRequestBodiesSince(STYLE_MAX_TOKENS, initialCount);
    test:assertEquals(bodies.length(), 2, "Only one max tokens retry is allowed");
    test:assertEquals(bodies[0]["max_tokens"], 100);
    test:assertEquals(bodies[1]["max_completion_tokens"], 100);

    // Nothing is remembered: the next call starts with `max_tokens` again.
    initialCount = getRequestCount(STYLE_MAX_TOKENS);
    _ = check mtProvider->chat({role: ai:USER, content: "A normal question"});
    bodies = getRequestBodiesSince(STYLE_MAX_TOKENS, initialCount);
    test:assertEquals(bodies.length(), 1);
    test:assertEquals(bodies[0]["max_tokens"], 100);
    test:assertFalse(bodies[0].hasKey("max_completion_tokens"));
}

@test:Config {groups: ["adaptive"]}
function testConfiguredAutoToolChoiceMode() returns error? {
    ModelProvider autoProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY,
            compatibility = {toolChoiceMode: AUTO});
    int initialCount = getRequestCount(STANDARD);
    int result = check autoProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(result, 2);
    map<json>[] bodies = getRequestBodiesSince(STANDARD, initialCount);
    test:assertEquals(bodies.length(), 1);
    test:assertEquals(bodies[0]["tool_choice"], "auto");
    assertResultToolInstructionSent(bodies[0]);
}

@test:Config {groups: ["adaptive"]}
function testConfiguredNoneToolChoiceMode() returns error? {
    ModelProvider noneProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY,
            compatibility = {toolChoiceMode: NONE});
    int initialCount = getRequestCount(STANDARD);
    int result = check noneProvider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(result, 2);
    map<json>[] bodies = getRequestBodiesSince(STANDARD, initialCount);
    test:assertEquals(bodies.length(), 1);
    test:assertFalse(bodies[0].hasKey("tool_choice"), "'tool_choice' must be omitted in NONE mode");
    test:assertTrue(bodies[0]["tools"] is json[], "The result tool must still be sent in NONE mode");
    assertResultToolInstructionSent(bodies[0]);
}

@test:Config {groups: ["adaptive"]}
function testConfiguredForcedToolChoiceDoesNotFallBack() returns error? {
    ModelProvider forcedProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY,
            compatibility = {toolChoiceMode: FORCED});
    foreach int attempt in 0 ..< 2 {
        int initialCount = getRequestCount(ZAI);
        int|error result = forcedProvider->generate(`GEN_INT What is 1 + 1?`);
        if result !is ai:LlmConnectionError {
            test:assertFail(string `Expected ai:LlmConnectionError on attempt ${attempt}, found: ${describe(result)}`);
        }
        test:assertTrue(result.message().includes(MOCK_ZAI_TOOL_CHOICE_ERROR));
        test:assertEquals(getRequestCount(ZAI), initialCount + 1, "A configured FORCED mode must not fall back");
        test:assertEquals(getLastRequestBody(ZAI)["tool_choice"], FORCED_RESULTS_TOOL_CHOICE);
    }
}

const STYLE_BAD_SCHEMA = "badschema";
const STYLE_NO_TOOLS = "notools";
const STYLE_BARE = "bare";
const STYLE_MAX_TOKENS = "maxtokens";

final readonly & map<json> FORCED_RESULTS_TOOL_CHOICE = {"type": "function", "function": {"name": "getResults"}};

function assertMaxTokensNotRetried(string trigger, int expectedStatus) {
    ModelProvider|ai:Error mtProvider = new (mockUrl(STYLE_MAX_TOKENS), TEST_MODEL, TEST_API_KEY, maxTokens = 100);
    if mtProvider is ai:Error {
        test:assertFail(mtProvider.message());
    }
    int initialCount = getRequestCount(STYLE_MAX_TOKENS);
    ai:ChatAssistantMessage|ai:Error result = mtProvider->chat({role: ai:USER, content: trigger + " hello"});
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError for ${trigger}, found: ${describe(result)}`);
    }
    test:assertEquals(result.detail()["statusCode"], expectedStatus);
    test:assertEquals(getRequestCount(STYLE_MAX_TOKENS), initialCount + 1,
            string `${trigger}: the max tokens field must not be switched`);
    test:assertEquals(getLastRequestBody(STYLE_MAX_TOKENS)["max_tokens"], 100);
}

function assertResultToolInstructionSent(map<json> body) {
    json[] messages = getMessages(body);
    string instructions = "";
    foreach json message in messages {
        if message is map<json> && message["role"] != "user" && message["content"] is string {
            instructions += <string>message["content"];
        }
    }
    test:assertTrue(instructions.includes("getResults"),
            string `An instruction to call the result tool is expected, found messages: ${messages.toJsonString()}`);
}

function getResponseFormatType(map<json> body) returns json {
    json responseFormat = body["response_format"];
    return responseFormat is map<json> ? responseFormat["type"] : ();
}

// ---------------------------------------------------------------------------------------------------------------------
// Unsupported-field detection and remembered fallbacks
// ---------------------------------------------------------------------------------------------------------------------

const STYLE_PATTERN_SCHEMA = "patternschema";
const STYLE_KEYWORD_SCHEMA = "keywordschema";

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

// ---------------------------------------------------------------------------------------------------------------------
// Tool calling and JSON mode fallback detection
// ---------------------------------------------------------------------------------------------------------------------

const STYLE_TOOL_FAILED = "toolfailed";

@test:Config {groups: ["adaptive"]}
function testToolCallingUnsupportedDetection() {
    [json, boolean][] cases = [
        // The server does not support the forced tool choice or tool calling.
        [{"error": {"message": MOCK_ZAI_TOOL_CHOICE_ERROR}}, true],
        [{"error": {"message": MOCK_NO_TOOLS_ERROR}}, true],
        [{"error": {"message": "Unsupported parameter: 'tools' is not supported with this model.", "param": "tools"}},
            true],
        [{"error": {"message": "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"}},
            true],
        [{"detail": [{"loc": ["body", "tool_choice"], "msg": "Extra inputs are not permitted"}]}, true],
        // A failed tool call produced by the model (Groq).
        [{"error": {"message": MOCK_TOOL_USE_FAILED_ERROR, "type": "invalid_request_error", "code": "tool_use_failed"}},
            false],
        // A rejected tool definition (OpenAI).
        [
            {
                "error": {
                    "message": "Invalid schema for function 'getResults': In context=(), 'additionalProperties' is " +
                        "required to be supplied and to be false.",
                    "type": "invalid_request_error",
                    "param": "tools[0].function.parameters",
                    "code": "invalid_function_parameters"
                }
            },
            false
        ],
        [{"error": {"message": "Invalid 'tools[0].function.name': string does not match pattern.", "param": "tools[0].function.name"}},
            false],
        // Unrelated errors that mention tools.
        [{"error": {"message": "The tool call arguments are too long"}}, false],
        [{"error": {"message": MOCK_CONTEXT_LENGTH_ERROR}}, false]
    ];
    foreach [json, boolean] [body, expected] in cases {
        ai:LlmConnectionError err = error ai:LlmConnectionError("rejected", statusCode = 400,
                body = body.toJsonString());
        test:assertEquals(isToolCallingUnsupported(err), expected, body.toJsonString());
    }
    ai:LlmConnectionError serverError = error ai:LlmConnectionError("failed", statusCode = 500,
            body = {"error": {"message": MOCK_NO_TOOLS_ERROR}}.toJsonString());
    test:assertFalse(isToolCallingUnsupported(serverError), "Only request rejections (400/422) count");
}

@test:Config {groups: ["adaptive"]}
function testToolUseFailedDoesNotFallBack() returns error? {
    ModelProvider groqProvider = check new (mockUrl(STYLE_TOOL_FAILED), "llama-3.3-70b-versatile", TEST_API_KEY);
    foreach int attempt in 0 ..< 2 {
        int initialCount = getRequestCount(STYLE_TOOL_FAILED);
        int|error result = groqProvider->generate(`GEN_INT What is 1 + 1?`);
        if result !is ai:LlmConnectionError {
            test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
        }
        test:assertTrue(result.message().includes(MOCK_TOOL_USE_FAILED_ERROR), result.message());
        test:assertEquals(getRequestCount(STYLE_TOOL_FAILED), initialCount + 1,
                "A failed tool call must not trigger the tool choice or JSON mode fallback");
        map<json> body = getLastRequestBody(STYLE_TOOL_FAILED);
        test:assertTrue(body.hasKey("tools"), string `Attempt ${attempt}: tool calling must not be disabled`);
        test:assertEquals(body["tool_choice"], FORCED_RESULTS_TOOL_CHOICE,
                string `Attempt ${attempt}: the forced tool choice must not be disabled`);
    }
}

@test:Config {groups: ["adaptive"]}
function testJsonObjectFallbackRequiresUnsupportedResponseFormat() {
    [json, boolean][] cases = [
        [{"error": {"message": MOCK_NO_RESPONSE_FORMAT_ERROR}}, true],
        [{"error": {"message": "Invalid parameter: 'response_format' of type 'json_object' is not supported with this model."}},
            true],
        [{"detail": [{"loc": ["body", "response_format"], "msg": "Extra inputs are not permitted"}]}, true],
        // The server supports `json_object` but rejected the request for another reason.
        [
            {
                "error": {
                    "message": "'messages' must contain the word 'json' in some form, to use 'response_format' of " +
                        "type 'json_object'.",
                    "param": "messages"
                }
            },
            false
        ],
        [{"error": {"message": "Invalid value for 'response_format.type'.", "param": "response_format"}}, false]
    ];
    foreach [json, boolean] [body, expected] in cases {
        ai:LlmConnectionError err = error ai:LlmConnectionError("rejected", statusCode = 400,
                body = body.toJsonString());
        test:assertEquals(isUnsupportedFieldError(err, "response_format"), expected, body.toJsonString());
    }
}
