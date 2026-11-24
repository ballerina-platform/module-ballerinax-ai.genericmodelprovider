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
import ballerina/crypto;

const MALFORMED_TOOL_CALL_ARGUMENTS_ERROR = "Invalid or malformed arguments received in function call response.";
const THINK_START_TAG = "<think>";
const THINK_END_TAG = "</think>";

# Builds the `messages` array of a Chat Completions request, preserving the message order.
#
# + messages - The chat messages, or a single user message
# + systemRole - The role used to send system messages
# + reasoningStore - Store of reasoning content to send back on assistant tool-call messages, if enabled
# + return - The request messages, or an `ai:Error` if a message cannot be converted
isolated function buildChatRequestMessages(ai:ChatMessage[]|ai:ChatUserMessage messages, SystemRole systemRole,
        ReasoningStore? reasoningStore) returns json[]|ai:Error {
    ai:ChatMessage[] chatMessages = (messages is ai:ChatUserMessage) ? [messages] : messages;
    ToolCallIdTracker idTracker = {reservedIds: collectExplicitToolCallIds(chatMessages)};
    json[] requestMessages = [];
    foreach ai:ChatMessage message in chatMessages {
        if message is ai:ChatUserMessage {
            json userContent = check buildUserContent(message.content);
            requestMessages.push({role: ai:USER, content: userContent});
        } else if message is ai:ChatSystemMessage {
            string systemContent = check buildTextContent(message.content);
            requestMessages.push({role: systemRole, content: systemContent});
        } else if message is ai:ChatAssistantMessage {
            requestMessages.push(buildAssistantRequestMessage(message, idTracker, reasoningStore));
        } else if message is ai:ChatFunctionMessage {
            requestMessages.push({
                role: "tool",
                tool_call_id: resolveToolResultId(message.name, message?.id, idTracker),
                content: message.content ?: ""
            });
        }
    }
    return requestMessages;
}

# Tracks tool call IDs while building request messages, so that tool calls and tool results without IDs of their
# own are given matching deterministic fallback IDs (`call_<name>_<n>`).
type ToolCallIdTracker record {|
    # Per-tool-name counters used to generate fallback IDs for tool calls
    map<int> callCounts = {};
    # Per-tool-name counters used to generate fallback IDs for tool results that match no unanswered call
    map<int> resultCounts = {};
    # Per-tool-name IDs of tool calls that have not been answered yet, in call order
    map<string[]> unansweredCalls = {};
    # IDs that fallback IDs must not reuse: the explicit IDs in the transcript and the fallback IDs already generated
    map<()> reservedIds = {};
|};

# Collects the explicit tool call IDs of all assistant and tool result messages, so that generated fallback IDs never
# collide with them.
#
# + messages - The chat messages
# + return - The explicit IDs
isolated function collectExplicitToolCallIds(ai:ChatMessage[] messages) returns map<()> {
    map<()> ids = {};
    foreach ai:ChatMessage message in messages {
        if message is ai:ChatAssistantMessage {
            foreach ai:FunctionCall toolCall in message.toolCalls ?: [] {
                string? id = normalizeToolCallId(toolCall?.id);
                if id is string {
                    ids[id] = ();
                }
            }
        } else if message is ai:ChatFunctionMessage {
            string? id = normalizeToolCallId(message?.id);
            if id is string {
                ids[id] = ();
            }
        }
    }
    return ids;
}

# Builds an assistant request message, including its tool calls and, when available, its reasoning content.
#
# + message - The assistant message
# + idTracker - Tracker used to assign fallback tool call IDs
# + reasoningStore - Store of reasoning content to send back, if enabled
# + return - The request message
isolated function buildAssistantRequestMessage(ai:ChatAssistantMessage message, ToolCallIdTracker idTracker,
        ReasoningStore? reasoningStore) returns map<json> {
    string? content = message.content;
    map<json> requestMessage = {role: ai:ASSISTANT, content};
    ai:FunctionCall[] toolCalls = message.toolCalls ?: [];
    if toolCalls.length() == 0 {
        if content is () {
            requestMessage["content"] = "";
        }
        return requestMessage;
    }

    json[] requestToolCalls = [];
    foreach ai:FunctionCall toolCall in toolCalls {
        string id = normalizeToolCallId(toolCall?.id)
            ?: nextToolCallId(toolCall.name, idTracker.callCounts, idTracker.reservedIds);
        string[]? unanswered = idTracker.unansweredCalls[toolCall.name];
        if unanswered is string[] {
            unanswered.push(id);
        } else {
            idTracker.unansweredCalls[toolCall.name] = [id];
        }
        requestToolCalls.push({
            id,
            'type: "function",
            'function: {
                name: toolCall.name,
                arguments: (toolCall.arguments ?: {}).toJsonString()
            }
        });
    }
    requestMessage["tool_calls"] = requestToolCalls;

    if reasoningStore is ReasoningStore {
        // Agents may split a response with parallel tool calls into one assistant message per call, so the reasoning
        // is looked up by the first tool call of this message.
        string? reasoning = reasoningStore.get(getReasoningKey(toolCalls[0]));
        if reasoning is string {
            requestMessage["reasoning_content"] = reasoning;
        }
    }
    return requestMessage;
}

# Resolves the `tool_call_id` of a tool result. A result with its own ID keeps it (and marks that call as answered);
# a result without an ID is paired with the oldest unanswered call to the same tool, or else given a fallback ID.
#
# + name - The tool name
# + id - The ID carried by the tool result message, if any
# + idTracker - Tracker of tool call IDs
# + return - The tool call ID to send
isolated function resolveToolResultId(string name, string? id, ToolCallIdTracker idTracker) returns string {
    string? resultId = normalizeToolCallId(id);
    string[]? unanswered = idTracker.unansweredCalls[name];
    if resultId is string {
        if unanswered is string[] {
            int? index = unanswered.indexOf(resultId);
            if index is int {
                _ = unanswered.remove(index);
            }
        }
        return resultId;
    }
    if unanswered is string[] && unanswered.length() > 0 {
        return unanswered.shift();
    }
    return nextToolCallId(name, idTracker.resultCounts, idTracker.reservedIds);
}

# Returns the tool call ID, treating a blank ID as missing.
#
# + id - The tool call ID
# + return - The ID, or `()` if it is missing or blank
isolated function normalizeToolCallId(string? id) returns string? => id is string && id.trim() != "" ? id : ();

# Generates a deterministic fallback tool call ID (`call_<name>_<n>`) for the nth occurrence of a tool name,
# skipping IDs that are already reserved. The generated ID is reserved as well.
#
# + name - The tool name
# + counts - Per-tool-name occurrence counters, updated in place
# + reservedIds - IDs that must not be generated, updated in place
# + return - The generated ID
isolated function nextToolCallId(string name, map<int> counts, map<()> reservedIds) returns string {
    int occurrence = counts[name] ?: 0;
    string id;
    while true {
        occurrence += 1;
        id = string `call_${name}_${occurrence}`;
        if !reservedIds.hasKey(id) {
            break;
        }
    }
    counts[name] = occurrence;
    reservedIds[id] = ();
    return id;
}

# Returns the key under which the reasoning of a tool call is stored: a hash of the tool call's ID (when the server
# returned one), name and arguments. Keying by the whole tool call, rather than by the ID alone, keeps a server that
# reuses tool call IDs from attaching one response's reasoning to another response's tool call, and also allows
# reasoning to be replayed for tool calls without an ID.
#
# + toolCall - The tool call
# + return - The reasoning key
isolated function getReasoningKey(ai:FunctionCall toolCall) returns string {
    json identity = [normalizeToolCallId(toolCall?.id), toolCall.name, toolCall.arguments ?: {}];
    return crypto:hashSha256(identity.toJsonString().toBytes()).toBase16();
}

# Builds the `tools` array of a Chat Completions request.
#
# + tools - The tool definitions
# + return - The request tools
isolated function buildChatRequestTools(ai:ChatCompletionFunctions[] tools) returns json[] =>
    from ai:ChatCompletionFunctions tool in tools
    select {
        'type: "function",
        'function: {
            name: tool.name,
            description: tool.description,
            parameters: tool?.parameters ?: {"type": "object", "properties": {}}
        }
    };

# Converts a chat completion choice into an assistant message.
#
# + choice - The chat completion choice
# + reasoningStore - Store in which reasoning content is saved against the returned tool calls, if enabled
# + return - The assistant message, or an `ai:Error` if the response is invalid
isolated function convertToAssistantMessage(ChatChoice choice, ReasoningStore? reasoningStore)
        returns ai:ChatAssistantMessage|ai:Error {
    ResponseMessage? message = choice?.message;
    if message is () {
        return error ai:LlmInvalidResponseError("Invalid response received from the model: the choice has no message");
    }

    ai:FunctionCall[] functionCalls = [];
    ResponseToolCall[]? toolCalls = message?.tool_calls;
    if toolCalls is ResponseToolCall[] && toolCalls.length() > 0 {
        foreach ResponseToolCall toolCall in toolCalls {
            ToolCallFunction? toolCallFunction = toolCall?.'function;
            if toolCallFunction is () {
                return error ai:LlmError(MALFORMED_TOOL_CALL_ARGUMENTS_ERROR,
                    error("The tool call does not contain a function"));
            }
            ai:FunctionCall functionCall = {
                name: toolCallFunction.name,
                arguments: check parseToolCallArguments(toolCallFunction?.arguments)
            };
            string? id = normalizeToolCallId(toolCall?.id);
            if id is string {
                functionCall.id = id;
            }
            functionCalls.push(functionCall);
        }
    } else {
        // Fall back to the deprecated `function_call` field.
        ToolCallFunction? functionCall = message?.function_call;
        if functionCall is ToolCallFunction {
            functionCalls.push({
                name: functionCall.name,
                arguments: check parseToolCallArguments(functionCall?.arguments)
            });
        }
    }

    string? content = getResponseText(message);
    string? thinkReasoning = ();
    if content is string && content.includes(THINK_END_TAG) {
        thinkReasoning = extractThinkReasoning(content);
        content = stripThinkBlocks(content).trim();
    }
    if content is string && content.trim() == "" && functionCalls.length() > 0 {
        content = ();
    }

    if reasoningStore is ReasoningStore {
        string? reasoning = getReasoningContent(message) ?: thinkReasoning;
        if reasoning is string {
            foreach ai:FunctionCall functionCall in functionCalls {
                reasoningStore.put(getReasoningKey(functionCall), reasoning);
            }
        }
    }

    return {
        role: ai:ASSISTANT,
        content,
        toolCalls: functionCalls.length() > 0 ? functionCalls : ()
    };
}

# Extracts the reasoning text inside `<think>...</think>` blocks, including the text before a dangling `</think>`.
# Removes the same text that `stripThinkBlocks` removes.
#
# + text - The response text
# + return - The reasoning text, or `()` if there is none
isolated function extractThinkReasoning(string text) returns string? {
    string[] pieces = [];
    string rest = text;
    while true {
        int? startIndex = rest.indexOf(THINK_START_TAG);
        if startIndex is () {
            break;
        }
        int? endIndex = rest.indexOf(THINK_END_TAG, startIndex);
        if endIndex is () {
            break;
        }
        pieces.push(rest.substring(startIndex + THINK_START_TAG.length(), endIndex).trim());
        rest = rest.substring(0, startIndex) + rest.substring(endIndex + THINK_END_TAG.length());
    }
    int? danglingEnd = rest.lastIndexOf(THINK_END_TAG);
    if danglingEnd is int {
        pieces.unshift(rest.substring(0, danglingEnd).trim());
    }
    string reasoning = string:'join("\n", ...pieces.filter(piece => piece != "")).trim();
    return reasoning == "" ? () : reasoning;
}

# Parses tool call arguments, which may be a JSON-encoded string, a JSON object, or absent. Missing, empty and
# `null` arguments are treated as an empty object.
#
# + arguments - The arguments as received from the server
# + return - The arguments as a JSON object, or an `ai:LlmError` if they are malformed
isolated function parseToolCallArguments(json arguments) returns map<json>|ai:LlmError {
    json parsed = arguments;
    if arguments is string {
        if arguments.trim() == "" {
            return {};
        }
        json|error result = arguments.fromJsonString();
        if result is error {
            return error ai:LlmError(MALFORMED_TOOL_CALL_ARGUMENTS_ERROR, result);
        }
        parsed = result;
    }
    if parsed is () {
        return {};
    }
    if parsed is map<json> {
        return parsed;
    }
    return error ai:LlmError(MALFORMED_TOOL_CALL_ARGUMENTS_ERROR,
        error(string `Expected the arguments to be a JSON object, but found: ${parsed.toJsonString()}`));
}
