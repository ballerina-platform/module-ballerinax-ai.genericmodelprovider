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

@test:Config {groups: ["chat"]}
function testChatWithSingleUserMessage() returns error? {
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "Hello there"});
    test:assertEquals(response.role, ai:ASSISTANT);
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Hello there");
    test:assertEquals(response.toolCalls, ());

    test:assertEquals(getLastRequestBody(STANDARD), {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Hello there"}]
    });
}

@test:Config {groups: ["chat"]}
function testChatWithFullAgentChain() returns error? {
    ai:ChatMessage[] messages = [
        <ai:ChatSystemMessage>{role: ai:SYSTEM, content: "You are a helpful weather assistant."},
        <ai:ChatUserMessage>{role: ai:USER, content: "What is the weather in Colombo?", name: "alice"},
        <ai:ChatAssistantMessage>{
            role: ai:ASSISTANT,
            toolCalls: [{id: "call_abc123", name: "getWeather", arguments: {"city": "Colombo"}}]
        },
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", id: "call_abc123", content: "31C and sunny"},
        <ai:ChatUserMessage>{role: ai:USER, content: "Thanks! Should I carry an umbrella?"}
    ];
    ai:ChatAssistantMessage response = check provider->chat(messages, weatherTools);
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Thanks! Should I carry an umbrella?");

    map<json> body = getLastRequestBody(STANDARD);
    json[] requestMessages = check decodeToolCallArguments(getMessages(body));
    test:assertEquals(requestMessages, [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What is the weather in Colombo?"},
        {
            "role": "assistant",
            "content": (),
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "getWeather", "arguments": {"city": "Colombo"}}
                }
            ]
        },
        {"role": "tool", "tool_call_id": "call_abc123", "content": "31C and sunny"},
        {"role": "user", "content": "Thanks! Should I carry an umbrella?"}
    ]);

    // `content: null` must be explicitly present on the assistant tool-call message.
    map<json> assistantMessage = getMessage(body, 2);
    test:assertTrue(assistantMessage.hasKey("content"), "Assistant tool-call message must carry 'content: null'");
    test:assertEquals(assistantMessage["content"], ());
    // Tool-call arguments must be sent as JSON strings.
    test:assertTrue(getToolCalls(assistantMessage)[0]["function"] is map<json>);
    map<json> fn = check getToolCalls(assistantMessage)[0]["function"].cloneWithType();
    test:assertTrue(fn["arguments"] is string, "Tool-call arguments must be sent as a JSON string");

    // `name` is never sent on any message.
    foreach json message in getMessages(body) {
        map<json> m = check message.cloneWithType();
        test:assertFalse(m.hasKey("name"), string `'name' must not be sent: ${m.toJsonString()}`);
    }
    test:assertEquals(body["tools"], expectedWeatherToolsJson);
}

@test:Config {groups: ["chat"]}
function testChatGeneratesPairedFallbackToolCallIds() returns error? {
    ai:ChatMessage[] messages = [
        <ai:ChatUserMessage>{role: ai:USER, content: "Compare the weather in Colombo and Kandy"},
        <ai:ChatAssistantMessage>{
            role: ai:ASSISTANT,
            toolCalls: [
                {name: "getWeather", arguments: {"city": "Colombo"}},
                {name: "getWeather", arguments: {"city": "Kandy"}},
                {name: "getTime", arguments: {"city": "Kandy"}}
            ]
        },
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", content: "31C"},
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", content: "24C"},
        <ai:ChatFunctionMessage>{role: "function", name: "getTime", content: "10:30"},
        <ai:ChatUserMessage>{role: ai:USER, content: "Which one is warmer?"}
    ];
    _ = check provider->chat(messages);

    map<json> body = getLastRequestBody(STANDARD);
    map<json>[] toolCalls = getToolCalls(getMessage(body, 1));
    test:assertEquals(toolCalls.length(), 3);
    string[] callIds = [];
    foreach map<json> toolCall in toolCalls {
        json id = toolCall["id"];
        test:assertTrue(id is string, "A fallback tool call id must be generated");
        callIds.push(<string>id);
    }
    test:assertTrue(callIds[0].startsWith("call_getWeather_"), string `Unexpected fallback id: ${callIds[0]}`);
    test:assertTrue(callIds[1].startsWith("call_getWeather_"), string `Unexpected fallback id: ${callIds[1]}`);
    test:assertTrue(callIds[2].startsWith("call_getTime_"), string `Unexpected fallback id: ${callIds[2]}`);
    test:assertNotEquals(callIds[0], callIds[1], "Fallback ids for repeated calls must be unique");

    // The nth result for a name must pair with the nth call for that name.
    foreach int i in 0 ..< 3 {
        map<json> toolMessage = getMessage(body, i + 2);
        test:assertEquals(toolMessage["role"], "tool");
        test:assertEquals(toolMessage["tool_call_id"], callIds[i]);
    }
}

@test:Config {groups: ["chat"]}
function testChatOmitsToolsWhenEmpty() returns error? {
    _ = check provider->chat({role: ai:USER, content: "No tools here"}, []);
    map<json> body = getLastRequestBody(STANDARD);
    test:assertFalse(body.hasKey("tools"), "'tools' must be omitted when there are no tools");
    test:assertFalse(body.hasKey("tool_choice"), "'tool_choice' must not be sent by chat");
}

@test:Config {groups: ["chat"]}
function testChatToolDefinitionsHaveNoStrict() returns error? {
    _ = check provider->chat({role: ai:USER, content: "Just say hi"}, weatherTools);
    map<json> body = getLastRequestBody(STANDARD);
    test:assertEquals(body["tools"], expectedWeatherToolsJson);
    json[] tools = check body["tools"].cloneWithType();
    foreach json tool in tools {
        map<json> toolMap = check tool.cloneWithType();
        map<json> fn = check toolMap["function"].cloneWithType();
        test:assertFalse(toolMap.hasKey("strict") || fn.hasKey("strict"), "'strict' must not be sent");
    }
    foreach string forbidden in ["n", "user", "logprobs", "parallel_tool_calls", "tool_choice"] {
        test:assertFalse(body.hasKey(forbidden), string `'${forbidden}' must not be sent`);
    }
}

@test:Config {groups: ["chat"]}
function testChatOmitsMaxTokensAndTemperatureWhenNotSet() returns error? {
    _ = check provider->chat({role: ai:USER, content: "Defaults only"});
    map<json> body = getLastRequestBody(STANDARD);
    foreach string 'field in ["temperature", "max_tokens", "max_completion_tokens", "stop"] {
        test:assertFalse(body.hasKey('field), string `'${'field}' must be omitted when not set`);
    }
}

@test:Config {groups: ["chat"]}
function testChatSendsMaxTokensAndTemperatureWhenSet() returns error? {
    ModelProvider configured = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY, maxTokens = 256,
            temperature = 0.2);
    _ = check configured->chat({role: ai:USER, content: "With limits"});
    map<json> body = getLastRequestBody(STANDARD);
    test:assertEquals(body["max_tokens"], 256);
    test:assertEquals(body["temperature"].toString(), "0.2");
    test:assertFalse(body.hasKey("max_completion_tokens"));
}

@test:Config {groups: ["chat"]}
function testChatSendsStopWhenSet() returns error? {
    _ = check provider->chat({role: ai:USER, content: "Count to ten"}, [], "END");
    test:assertEquals(getLastRequestBody(STANDARD)["stop"], "END");
}

@test:Config {groups: ["chat"]}
function testChatReturnsTextWhenToolsAreSent() returns error? {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: "Tell me a fun fact about clouds"}, weatherTools);
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Tell me a fun fact about clouds");
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    test:assertTrue(toolCalls is () || toolCalls.length() == 0, "No tool calls expected for a text answer");
}

@test:Config {groups: ["chat"]}
function testChatReturnsToolCall() returns error? {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: "TRIGGER_TOOL_CALL What is the weather in Colombo?"}, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    test:assertEquals(toolCalls, [{id: "call_weather_1", name: "getWeather", arguments: {"city": "Colombo"}}]);
}

@test:Config {groups: ["chat"]}
function testChatReturnsParallelToolCalls() returns error? {
    ai:ChatAssistantMessage response = check provider->chat(
            {role: ai:USER, content: "TRIGGER_PARALLEL_TOOL_CALLS Weather in Colombo and Kandy?"}, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected parallel tool calls");
    }
    test:assertEquals(toolCalls, [
        {id: "call_parallel_1", name: "getWeather", arguments: {"city": "Colombo"}},
        {id: "call_parallel_2", name: "getWeather", arguments: {"city": "Kandy"}}
    ]);
}

@test:Config {groups: ["chat"]}
function testChatAcceptsObjectToolArguments() returns error? {
    ModelProvider zaiProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY);
    ai:ChatAssistantMessage response = check zaiProvider->chat(
            {role: ai:USER, content: "TRIGGER_TOOL_CALL What is the weather in Colombo?"}, weatherTools);
    ai:FunctionCall[]? toolCalls = response.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    test:assertEquals(toolCalls[0].name, "getWeather");
    test:assertEquals(toolCalls[0].arguments, {"city": "Colombo"});
    test:assertEquals(toolCalls[0].id, "call_weather_1");
}

@test:Config {groups: ["chat"]}
function testChatTextWithReasoningContentReturnsContent() returns error? {
    ModelProvider zaiProvider = check new (mockUrl(ZAI), "glm-4.6", TEST_API_KEY);
    ai:ChatAssistantMessage response = check zaiProvider->chat({role: ai:USER, content: "Hi GLM"});
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Hi GLM");
}

@test:Config {groups: ["chat"]}
function testChatWithDeveloperSystemRole() returns error? {
    ModelProvider developerProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY,
            compatibility = {systemRole: DEVELOPER});
    _ = check developerProvider->chat([
        <ai:ChatSystemMessage>{role: ai:SYSTEM, content: "Be concise."},
        <ai:ChatUserMessage>{role: ai:USER, content: "Hello"}
    ]);
    test:assertEquals(getMessages(getLastRequestBody(STANDARD)), [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Hello"}
    ]);
}

@test:Config {groups: ["chat"]}
function testChatAppliesExtraBodyDropParamsAndExtraHeaders() returns error? {
    ModelProvider customProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY, temperature = 0.5,
            compatibility = {
                extraBody: {"top_k": 40, "chat_template_kwargs": {"enable_thinking": false}},
                dropParams: ["temperature"],
                extraHeaders: {"X-Custom-Header": "custom-value", "HTTP-Referer": "https://example.com"}
            });
    _ = check customProvider->chat({role: ai:USER, content: "Custom request"});

    map<json> body = getLastRequestBody(STANDARD);
    test:assertEquals(body["top_k"], 40);
    test:assertEquals(body["chat_template_kwargs"], {"enable_thinking": false});
    test:assertFalse(body.hasKey("temperature"), "'temperature' must be dropped");

    map<string> headers = getLastRequestHeaders(STANDARD);
    test:assertEquals(headers["x-custom-header"], "custom-value");
    test:assertEquals(headers["http-referer"], "https://example.com");
    test:assertEquals(headers["authorization"], "Bearer " + TEST_API_KEY);
}

@test:Config {groups: ["chat"]}
function testChatSendsBearerAuthByDefault() returns error? {
    _ = check provider->chat({role: ai:USER, content: "Auth check"});
    map<string> headers = getLastRequestHeaders(STANDARD);
    test:assertEquals(headers["authorization"], "Bearer " + TEST_API_KEY);
}

@test:Config {groups: ["chat"]}
function testChatSendsNoAuthWithoutApiKey() returns error? {
    ModelProvider localProvider = check new (mockUrl(LOCAL), "llama3.2");
    ai:ChatAssistantMessage response = check localProvider->chat({role: ai:USER, content: "Local hello"});
    test:assertEquals(response.content, MOCK_TEXT_RESPONSE_PREFIX + "Local hello");
    test:assertFalse(getLastRequestHeaders(LOCAL).hasKey("authorization"));
}


@test:Config {groups: ["chat"]}
function testChatPreservesReasoningContent() returns error? {
    ModelProvider reasoningProvider = check new (mockUrl(STANDARD), "deepseek-reasoner", TEST_API_KEY,
            compatibility = {preserveReasoningContent: true});
    ai:ChatUserMessage question = {role: ai:USER, content: "TRIGGER_TOOL_CALL TRIGGER_REASONING Weather in Colombo?"};
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
    test:assertEquals(assistantMessage["role"], "assistant");
    test:assertEquals(assistantMessage["reasoning_content"], MOCK_REASONING_TEXT);
}

@test:Config {groups: ["chat"]}
function testChatDoesNotSendReasoningContentByDefault() returns error? {
    ModelProvider defaultProvider = check new (mockUrl(STANDARD), "deepseek-reasoner", TEST_API_KEY);
    ai:ChatUserMessage question = {role: ai:USER, content: "TRIGGER_TOOL_CALL TRIGGER_REASONING Weather in Colombo?"};
    ai:ChatAssistantMessage first = check defaultProvider->chat(question, weatherTools);
    ai:FunctionCall[]? toolCalls = first.toolCalls;
    if toolCalls is () {
        test:assertFail("Expected a tool call");
    }
    ai:ChatMessage[] followUp = [
        question,
        first,
        <ai:ChatFunctionMessage>{role: "function", name: "getWeather", id: toolCalls[0].id, content: "31C"}
    ];
    _ = check defaultProvider->chat(followUp, weatherTools);

    map<json> assistantMessage = getMessage(getLastRequestBody(STANDARD), 1);
    test:assertFalse(assistantMessage.hasKey("reasoning_content"),
            "'reasoning_content' must not be sent unless preserveReasoningContent is enabled");
}
