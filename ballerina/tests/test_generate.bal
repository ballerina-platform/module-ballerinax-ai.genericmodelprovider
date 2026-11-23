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

type ReviewList Review[];

@test:Config {groups: ["generate"]}
function testGenerateInt() returns error? {
    int result = check provider->generate(`GEN_INT What is 1 + 1?`);
    test:assertEquals(result, 2);

    // By default the result tool is forced.
    map<json> body = getLastRequestBody(STANDARD);
    test:assertEquals(body["tool_choice"], {"type": "function", "function": {"name": MOCK_GET_RESULTS_TOOL}});
    // The tool instruction is sent even when the tool is forced, for servers that ignore `tool_choice`.
    map<json> systemMessage = getMessage(body, 0);
    test:assertEquals(systemMessage["role"], "system");
    test:assertTrue((check systemMessage["content"].ensureType(string)).includes(MOCK_GET_RESULTS_TOOL));
    json[] tools = check body["tools"].cloneWithType();
    test:assertEquals(tools.length(), 1);
    map<json> tool = check tools[0].cloneWithType();
    test:assertEquals(tool["type"], "function");
    map<json> fn = check tool["function"].cloneWithType();
    test:assertEquals(fn["name"], MOCK_GET_RESULTS_TOOL);
    test:assertTrue(fn["parameters"] is map<json>);
    test:assertFalse(tool.hasKey("strict") || fn.hasKey("strict"), "'strict' must not be sent");
    test:assertFalse(body.hasKey("temperature"));
    test:assertFalse(body.hasKey("max_tokens"));
    test:assertEquals(getLastRequestHeaders(STANDARD)["authorization"], "Bearer " + TEST_API_KEY);
}

@test:Config {groups: ["generate"]}
function testGenerateString() returns error? {
    string result = check provider->generate(`GEN_STRING Say hello`);
    test:assertEquals(result, MOCK_STRING_RESULT);
}

@test:Config {groups: ["generate"]}
function testGenerateBoolean() returns error? {
    boolean result = check provider->generate(`GEN_BOOL Is water wet?`);
    test:assertTrue(result);
}

@test:Config {groups: ["generate"]}
function testGenerateRecord() returns error? {
    Review result = check provider->generate(`GEN_REVIEW Please review this blog.
        Title: ${blog2.title}
        Content: ${blog2.content}`);
    test:assertEquals(result, reviewRecord);

    // Text-only prompts are sent as a plain string with the interpolations inlined.
    string content = check getUserMessage(getLastRequestBody(STANDARD))["content"].ensureType();
    test:assertTrue(content.includes(blog2.title));
}

@test:Config {groups: ["generate"]}
function testGenerateRecordArray() returns error? {
    ReviewList result = check provider->generate(`GEN_REVIEWS_LIST Review these blogs:
        ${blog1.title} and ${blog2.title}`);
    test:assertEquals(result, [reviewRecord, {rating: 4, comment: "Covers the basics of gardening."}]);
}

@test:Config {groups: ["generate"]}
function testGenerateNilableWithValue() returns error? {
    string? result = check provider->generate(`GEN_NILABLE_VALUE Tell me a joke`);
    test:assertEquals(result, MOCK_NILABLE_STRING_RESULT);
}

@test:Config {groups: ["generate"]}
function testGenerateNilableWithNull() returns error? {
    int? result = check provider->generate(`GEN_NILABLE_NULL How many unicorns are in this room?`);
    test:assertEquals(result, ());
}

@test:Config {groups: ["generate"]}
function testGenerateWithImageBytes() returns error? {
    ai:ImageDocument image = {content: sampleImageBytes};
    string result = check provider->generate(`GEN_IMAGE Describe this image: ${image}`);
    test:assertEquals(result, MOCK_IMAGE_DESCRIPTION);

    json content = getUserMessage(getLastRequestBody(STANDARD))["content"];
    map<json> imagePart = check getImagePart(content);
    map<json> imageUrl = check imagePart["image_url"].cloneWithType();
    string url = check imageUrl["url"].ensureType();
    test:assertTrue(url.startsWith("data:image/"), string `Expected a data URL, found: ${url}`);
    test:assertTrue(url.endsWith(sampleImageBytes.toBase64()), "The data URL must carry the base64-encoded bytes");
}

@test:Config {groups: ["generate"]}
function testGenerateWithImageUrl() returns error? {
    ai:ImageDocument image = {content: sampleImageUrl};
    string result = check provider->generate(`GEN_IMAGE Describe this image: ${image}`);
    test:assertEquals(result, MOCK_IMAGE_DESCRIPTION);

    json content = getUserMessage(getLastRequestBody(STANDARD))["content"];
    map<json> imagePart = check getImagePart(content);
    test:assertEquals(imagePart["image_url"], {"url": sampleImageUrl});
}

@test:Config {groups: ["generate"]}
function testGenerateWithUnsupportedFileDocument() {
    ai:FileDocument file = {content: {fileId: "file-abc123"}};
    string|error result = provider->generate(`GEN_STRING Summarize this file: ${file}`);
    test:assertTrue(result is ai:Error, "An unsupported document must result in an ai:Error");
}

@test:Config {groups: ["generate"]}
function testGenerateTypeMismatch() {
    int|error result = provider->generate(`GEN_MISMATCH What is 1 + 1?`);
    if result !is ai:LlmInvalidGenerationError {
        test:assertFail(string `Expected ai:LlmInvalidGenerationError, found: ${describe(result)}`);
    }
}

# Returns the single `image_url` content part of a user message content.
#
# + content - The user message content
# + return - The image part, or an error
function getImagePart(json content) returns map<json>|error {
    if content !is json[] {
        test:assertFail(string `Expected the user content to be an array of parts, found: ${content.toJsonString()}`);
    }
    map<json>[] imageParts = [];
    boolean hasText = false;
    foreach json part in content {
        map<json> partMap = check part.cloneWithType();
        if partMap["type"] == "image_url" {
            imageParts.push(partMap);
        } else if partMap["type"] == "text" {
            hasText = true;
        }
    }
    test:assertTrue(hasText, "Expected a text part alongside the image part");
    test:assertEquals(imageParts.length(), 1, "Expected exactly one image_url part");
    return imageParts[0];
}
