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

const TEST_API_KEY = "not-a-real-api-key";
const TEST_MODEL = "mock-chat-model";
const MOCK_BASE_URL = "http://localhost:8080/llm/";

const STANDARD = "standard";
const ZAI = "zai";
const OLLAMA = "ollama";
const OSERIES = "oseries";
const LOCAL = "local";
const NO_JSON_SCHEMA = "nojsonschema";

const MALFORMED_ARGUMENTS_ERROR = "Invalid or malformed arguments received in function call response.";

# Returns the mock service URL for a style.
#
# + style - The mock server style
# + return - The service URL
function mockUrl(string style) returns string => MOCK_BASE_URL + style;

// Shared provider against the standard mock server (no temperature or max tokens configured).
final ModelProvider provider = check new (mockUrl(STANDARD), TEST_MODEL, TEST_API_KEY);

type Blog record {
    string title;
    string content;
};

type Review record {|
    int rating;
    string comment;
|};

const blog1 = {
    title: "Tips for Growing a Beautiful Garden",
    content: string `Spring is the perfect time to start your garden.
        Begin by preparing your soil with organic compost and ensure proper drainage.`
};

const blog2 = {
    title: "Essential Tips for Sports Performance",
    content: string `Success in sports requires dedicated preparation and training.
        Begin by establishing a proper warm-up routine and maintaining good form.`
};

const reviewRecord = {
    rating: 8,
    comment: "Talks about essential aspects of sports performance including warm-up, form, equipment, and nutrition."
};

final ai:ChatCompletionFunctions[] weatherTools = [
    {
        name: "getWeather",
        description: "Get the current weather for a city",
        parameters: {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Name of the city"}},
            "required": ["city"]
        }
    }
];

final json[] expectedWeatherToolsJson = [
    {
        "type": "function",
        "function": {
            "name": "getWeather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "Name of the city"}},
                "required": ["city"]
            }
        }
    }
];

final byte[] sampleImageBytes = [137, 80, 78, 71, 13, 10, 26, 10];
const sampleImageUrl = "https://example.com/images/sunset.png";

# Returns the `messages` array of a recorded request body.
#
# + body - The request body
# + return - The messages
function getMessages(map<json> body) returns json[] {
    json messages = body["messages"];
    if messages is json[] {
        return messages;
    }
    test:assertFail("Request body does not contain a 'messages' array");
}

# Returns the message at the given index of a recorded request body as a map.
#
# + body - The request body
# + index - The message index
# + return - The message
function getMessage(map<json> body, int index) returns map<json> {
    json[] messages = getMessages(body);
    test:assertTrue(messages.length() > index, string `Expected at least ${index + 1} messages`);
    json message = messages[index];
    if message is map<json> {
        return message;
    }
    test:assertFail(string `Message ${index} is not a JSON object`);
}

# Returns the `tool_calls` of an assistant message.
#
# + message - The assistant message
# + return - The tool calls as maps
function getToolCalls(map<json> message) returns map<json>[] {
    json toolCalls = message["tool_calls"];
    if toolCalls !is json[] {
        test:assertFail("Assistant message does not contain a 'tool_calls' array");
    }
    map<json>[] result = [];
    foreach json toolCall in toolCalls {
        if toolCall !is map<json> {
            test:assertFail("Tool call is not a JSON object");
        }
        result.push(toolCall);
    }
    return result;
}

# Asserts that every tool call argument in the messages is a JSON-encoded string and replaces it with the decoded
# value so that the messages can be compared structurally.
#
# + messages - The request messages
# + return - The messages with decoded tool call arguments
function decodeToolCallArguments(json[] messages) returns json[]|error {
    json[] result = [];
    foreach json message in messages {
        if message !is map<json> || message["tool_calls"] !is json[] {
            result.push(message);
            continue;
        }
        map<json> decodedMessage = message.clone();
        json[] decodedCalls = [];
        foreach json toolCall in <json[]>message["tool_calls"] {
            map<json> call = check toolCall.cloneWithType();
            map<json> fn = check call["function"].cloneWithType();
            json arguments = fn["arguments"];
            if arguments !is string {
                test:assertFail(
                        string `Tool call arguments must be sent as a JSON string, found: ${arguments.toJsonString()}`);
            }
            fn["arguments"] = check arguments.fromJsonString();
            call["function"] = fn;
            decodedCalls.push(call);
        }
        decodedMessage["tool_calls"] = decodedCalls;
        result.push(decodedMessage);
    }
    return result;
}

# Returns a string representation of a value or an error, for assertion messages.
#
# + value - The value or error
# + return - The string representation
function describe(any|error value) returns string => value is error ? value.toString() : value.toString();

# Returns the (last) user message of a recorded request body.
#
# + body - The recorded request body
# + return - The user message
function getUserMessage(map<json> body) returns map<json> {
    json[] messages = getMessages(body);
    foreach int i in int:range(messages.length() - 1, -1, -1) {
        json message = messages[i];
        if message is map<json> && message["role"] == "user" {
            return message;
        }
    }
    test:assertFail("Request body does not contain a user message");
}
