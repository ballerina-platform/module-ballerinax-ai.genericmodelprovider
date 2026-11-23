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

const TEST_SECRET_API_KEY = "sk-test-SECRET-123";

final ModelProvider secretKeyProvider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_SECRET_API_KEY);

@test:Config {groups: ["errors"], dataProvider: httpErrorStatuses}
function testChatHttpErrorStatusReturnsConnectionError(int status) {
    ai:ChatAssistantMessage|ai:Error result = secretKeyProvider->chat(
            {role: ai:USER, content: string `TRIGGER_STATUS_${status} hello`});
    assertConnectionErrorWithoutSecret(result, status);
}

@test:Config {groups: ["errors"], dataProvider: httpErrorStatuses}
function testGenerateHttpErrorStatusReturnsConnectionError(int status) {
    int|error result = secretKeyProvider->generate(`GEN_INT TRIGGER_STATUS_${status} What is 1 + 1?`);
    assertConnectionErrorWithoutSecret(result, status);
}

@test:Config {groups: ["errors"]}
function testChatNonJsonResponseReturnsInvalidResponseError() {
    ai:ChatAssistantMessage|ai:Error result = provider->chat({role: ai:USER, content: "TRIGGER_HTML_RESPONSE hi"});
    test:assertTrue(result is ai:LlmInvalidResponseError,
            string `Expected ai:LlmInvalidResponseError, found: ${describe(result)}`);
}

@test:Config {groups: ["errors"]}
function testGenerateNonJsonResponseReturnsInvalidResponseError() {
    int|error result = provider->generate(`GEN_INT TRIGGER_HTML_RESPONSE What is 1 + 1?`);
    test:assertTrue(result is ai:LlmInvalidResponseError,
            string `Expected ai:LlmInvalidResponseError, found: ${describe(result)}`);
}

function httpErrorStatuses() returns int[][] => [[401], [429], [503]];

function assertConnectionErrorWithoutSecret(any|error result, int expectedStatus) {
    if result !is ai:LlmConnectionError {
        test:assertFail(string `Expected ai:LlmConnectionError for HTTP ${expectedStatus}, found: ${describe(result)}`);
    }
    test:assertEquals(result.detail()["statusCode"], expectedStatus);
    test:assertFalse(result.message().includes(TEST_SECRET_API_KEY), "The API key must not appear in the message");
    test:assertFalse(result.toString().includes(TEST_SECRET_API_KEY), "The API key must not appear in the error");
}
