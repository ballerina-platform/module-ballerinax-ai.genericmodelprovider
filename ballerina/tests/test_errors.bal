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

@test:Config {groups: ["errors"]}
function testChatServerErrorReturnsConnectionError() {
    ai:ChatAssistantMessage|ai:Error result = provider->chat({role: ai:USER, content: "TRIGGER_SERVER_ERROR now"});
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertTrue(result.message().includes(MOCK_SERVER_ERROR_MESSAGE),
            string `The error message must include the provider message, found: ${result.message()}`);
    test:assertEquals(result.detail()["statusCode"], 500);
}

@test:Config {groups: ["errors"]}
function testChatEmptyChoicesReturnsInvalidResponseError() {
    ai:ChatAssistantMessage|ai:Error result = provider->chat({role: ai:USER, content: "TRIGGER_EMPTY_CHOICES now"});
    if result !is ai:LlmInvalidResponseError {
        test:assertFail(string `Expected ai:LlmInvalidResponseError, found: ${describe(result)}`);
    }
}

@test:Config {groups: ["errors"]}
function testChatMalformedToolArgumentsReturnsLlmError() {
    ai:ChatAssistantMessage|ai:Error result = provider->chat(
            {role: ai:USER, content: "TRIGGER_MALFORMED_ARGS What is the weather?"}, weatherTools);
    if result !is ai:LlmError {
        test:assertFail(string `Expected ai:LlmError, found: ${describe(result)}`);
    }
    test:assertEquals(result.message(), MALFORMED_ARGUMENTS_ERROR);
}

@test:Config {groups: ["errors"]}
function testGenerateServerErrorReturnsConnectionError() {
    int|error result = provider->generate(`GEN_INT TRIGGER_SERVER_ERROR What is 1 + 1?`);
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError, found: ${describe(result)}`);
    }
    test:assertTrue(result.message().includes(MOCK_SERVER_ERROR_MESSAGE),
            string `The error message must include the provider message, found: ${result.message()}`);
}

@test:Config {groups: ["errors"]}
function testGenerateEmptyChoicesReturnsLlmError() {
    int|error result = provider->generate(`GEN_INT TRIGGER_EMPTY_CHOICES What is 1 + 1?`);
    test:assertTrue(result is ai:LlmError, string `Expected ai:LlmError, found: ${describe(result)}`);
}

@test:Config {groups: ["errors"]}
function testGenerateMalformedToolArgumentsReturnsLlmError() {
    int|error result = provider->generate(`GEN_INT GEN_MALFORMED What is 1 + 1?`);
    test:assertTrue(result is ai:LlmError, string `Expected ai:LlmError, found: ${describe(result)}`);
}

@test:Config {groups: ["errors"]}
function testModelProviderInitWithEmptyModelName() {
    ModelProvider|ai:Error result = new (mockUrl(STANDARD), "  ", TEST_API_KEY);
    test:assertTrue(result is ai:Error, "An empty model name must be rejected");
}

@test:Config {groups: ["errors"]}
function testModelProviderInitWithEmptyServiceUrl() {
    ModelProvider|ai:Error result = new ("", TEST_MODEL, TEST_API_KEY);
    test:assertTrue(result is ai:Error, "An empty service URL must be rejected");
}

@test:Config {groups: ["errors"]}
function testModelProviderInitAcceptsTrailingSlash() returns error? {
    ModelProvider _ = check new (mockUrl(STANDARD) + "/", TEST_MODEL, TEST_API_KEY);
}
