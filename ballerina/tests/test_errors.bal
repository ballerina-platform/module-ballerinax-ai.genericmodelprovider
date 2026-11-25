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
