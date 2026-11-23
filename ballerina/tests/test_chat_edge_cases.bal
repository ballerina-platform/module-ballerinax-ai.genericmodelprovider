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

// ---------------------------------------------------------------------------------------------------------------------
// Tool call IDs and arguments
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["chat"]}
function testChatBlankToolCallIdIsTreatedAsMissing() returns error? {
    check assertMissingToolCallIdRoundTrip("TRIGGER_EMPTY_ID");
}

@test:Config {groups: ["chat"]}
function testChatNullToolCallIdIsTreatedAsMissing() returns error? {
    check assertMissingToolCallIdRoundTrip("TRIGGER_NULL_ID");
}

@test:Config {groups: ["chat"]}
function testChatNullStringToolArgumentsBecomeEmptyObject() returns error? {
    test:assertEquals(check getSingleToolCallArguments("TRIGGER_ARGS_NULL_STRING"), {});
}

@test:Config {groups: ["chat"]}
function testChatEmptyStringToolArgumentsBecomeEmptyObject() returns error? {
    test:assertEquals(check getSingleToolCallArguments("TRIGGER_ARGS_EMPTY_STRING"), {});
}

@test:Config {groups: ["chat"]}
function testChatJsonNullToolArgumentsBecomeEmptyObject() returns error? {
    test:assertEquals(check getSingleToolCallArguments("TRIGGER_ARGS_JSON_NULL"), {});
}

@test:Config {groups: ["chat"]}
function testChatNonObjectToolArgumentsReturnLlmError() {
    ai:ChatAssistantMessage|ai:Error result = provider->chat(
            {role: ai:USER, content: "TRIGGER_ARGS_ARRAY weather?"}, weatherTools);
    if result !is ai:LlmError {
        test:assertFail(string `Expected ai:LlmError, found: ${describe(result)}`);
    }
    test:assertEquals(result.message(), MALFORMED_ARGUMENTS_ERROR);
}

@test:Config {groups: ["chat"]}
function testChatDeprecatedFunctionCallResponse() returns error? {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: "TRIGGER_FUNCTION_CALL_ONLY weather?"}, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected the deprecated function_call to be returned as a tool call");
    }
    test:assertEquals(toolCalls.length(), 1);
    test:assertEquals(toolCalls[0].name, "getWeather");
    test:assertEquals(toolCalls[0].arguments, {"city": "Colombo"});
    test:assertEquals(response.content, ());
}

@test:Config {groups: ["chat"]}
function testChatPairsMixedToolCallIds() returns error? {
    ai:ChatMessage[] messages = [
        <ai:ChatUserMessage>{role: ai:USER, content: "Weather and time in three cities"},
        <ai:ChatAssistantMessage>{
            role: ai:ASSISTANT,
            toolCalls: [
                {name: "getWeather", arguments: {"city": "Colombo"}},
                {name: "getWeather", arguments: {"city": "Kandy"}, id: "call_explicit_kandy"},
                {name: "getTime", arguments: {"city": "Kandy"}},
                {name: "getWeather", arguments: {"city": "Galle"}, id: "  "}
            ]
        },
        // The result with an explicit ID comes first; the others carry no ID (or a blank one).
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", id: "call_explicit_kandy", content: "24C"},
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", content: "31C"},
        <ai:ChatFunctionMessage>{role: "function", name: "getTime", id: "", content: "10:30"},
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", content: "29C"},
        <ai:ChatUserMessage>{role: ai:USER, content: "Summarize"}
    ];
    _ = check provider->chat(messages);

    map<json> body = getLastRequestBody(STANDARD);
    map<json>[] toolCalls = getToolCalls(getMessage(body, 1));
    string[] callIds = from map<json> toolCall in toolCalls
        select toolCall["id"] is string ? <string>toolCall["id"] : "";
    test:assertEquals(callIds[1], "call_explicit_kandy");
    test:assertTrue(callIds[0].startsWith("call_getWeather_"), string `Unexpected fallback id: ${callIds[0]}`);
    test:assertTrue(callIds[2].startsWith("call_getTime_"), string `Unexpected fallback id: ${callIds[2]}`);
    test:assertTrue(callIds[3].startsWith("call_getWeather_"), string `A blank id must get a fallback: ${callIds[3]}`);
    test:assertNotEquals(callIds[0], callIds[3]);

    // Results: explicit Kandy -> its own id; then the first unanswered getWeather (Colombo); then getTime;
    // then the remaining getWeather (Galle).
    test:assertEquals(getMessage(body, 2)["tool_call_id"], "call_explicit_kandy");
    test:assertEquals(getMessage(body, 3)["tool_call_id"], callIds[0]);
    test:assertEquals(getMessage(body, 4)["tool_call_id"], callIds[2]);
    test:assertEquals(getMessage(body, 5)["tool_call_id"], callIds[3]);
}

// ---------------------------------------------------------------------------------------------------------------------
// Think blocks
// ---------------------------------------------------------------------------------------------------------------------

@test:Config {groups: ["chat"]}
function testChatStripsThinkBlocks() returns error? {
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "TRIGGER_THINK_TEXT hi"});
    test:assertEquals(response.content, "Final");
}

@test:Config {groups: ["chat"]}
function testChatThinkOnlyContentWithToolCallsIsNil() returns error? {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: "TRIGGER_THINK_TOOL_CALL weather?"}, weatherTools);
    test:assertEquals(response.content, ());
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    test:assertEquals(toolCalls[0].id, "call_think_1");
}

@test:Config {groups: ["chat"]}
function testChatPreservesThinkBlockAsReasoningContent() returns error? {
    ModelProvider reasoningProvider = check new (mockUrl(STANDARD), "qwen3", TEST_API_KEY,
            compatibility = {preserveReasoningContent: true});
    ai:ChatUserMessage question = {role: ai:USER, content: "TRIGGER_THINK_TOOL_CALL weather?"};
    ai:ChatAssistantMessage first = check reasoningProvider->chat(question, weatherTools);
    ai:FunctionCall[]? toolCalls = first.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    ai:ChatMessage[] followUp = [
        question,
        first,
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", id: toolCalls[0].id, content: "31C"}
    ];
    _ = check reasoningProvider->chat(followUp, weatherTools);
    map<json> assistantMessage = getMessage(getLastRequestBody(STANDARD), 1);
    test:assertEquals(assistantMessage["reasoning_content"], MOCK_THINK_REASONING);
}

// ---------------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------------

# Asserts that a tool call returned with a missing ID has no `id`, and that sending it back uses a paired
# `call_<name>_<n>` fallback ID.
#
# + trigger - The mock trigger producing the tool call
# + return - An error, if any
function assertMissingToolCallIdRoundTrip(string trigger) returns error? {
    ai:ChatUserMessage question = {role: ai:USER, content: trigger + " weather?"};
    ai:ChatAssistantMessage response = check provider->chat(question, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    test:assertFalse(toolCalls[0].hasKey("id"), string `A blank/null id must be absent, found: ${toolCalls[0].toString()}`);
    test:assertEquals(toolCalls[0].arguments, {"city": "Colombo"});

    ai:ChatMessage[] followUp = [
        question,
        response,
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", content: "31C"},
        <ai:ChatUserMessage>{role: ai:USER, content: "Thanks"}
    ];
    _ = check provider->chat(followUp, weatherTools);
    map<json> body = getLastRequestBody(STANDARD);
    string callId = check getToolCalls(getMessage(body, 1))[0]["id"].ensureType();
    test:assertTrue(callId.startsWith("call_getWeather_"), string `Unexpected fallback id: ${callId}`);
    test:assertEquals(getMessage(body, 2)["tool_call_id"], callId);
}

# Returns the arguments of the single tool call produced by a trigger.
#
# + trigger - The mock trigger
# + return - The arguments, or an error
function getSingleToolCallArguments(string trigger) returns map<json>?|error {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: trigger + " weather?"}, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () || toolCalls.length() != 1 {
        test:assertFail("Expected exactly one tool call");
    }
    return toolCalls[0].arguments;
}
