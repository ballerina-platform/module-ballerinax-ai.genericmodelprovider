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
import ballerina/ai.observe;
import ballerina/http;
import ballerina/jballerina.java;
import ballerina/log;

# A model provider for any LLM service that exposes an OpenAI-compatible Chat Completions API
# (e.g. Groq, Together AI, Fireworks AI, xAI, Z.ai/GLM, OpenRouter, vLLM, LM Studio, Ollama).
#
# The provider uses the most widely supported request shape by default and adapts automatically when the
# server rejects an optional feature. Use `compatibility` to fine-tune the behaviour for a specific server.
@display {label: "Generic Model Provider (OpenAI-compatible)"}
public isolated distinct client class ModelProvider {
    *ai:ModelProvider;

    // NOTE: The field names below are read by the native `Generator` class. Keep them in sync.
    private final http:Client llmClient;
    private final string modelType;
    private final decimal? temperature;
    private final int? maxTokens;
    private final readonly & CompatibilityConfig compatibility;
    private final map<string> & readonly requestHeaders;
    private final AdaptiveState adaptiveState = new;
    private final ReasoningStore reasoningStore = new;

    # Initializes the OpenAI-compatible model provider.
    #
    # + serviceUrl - Base URL of the OpenAI-compatible API, up to but excluding `/chat/completions`
    #                (e.g. `https://api.groq.com/openai/v1`, `http://localhost:8000/v1`)
    # + modelType - Name of the model as expected by the service (e.g. `llama-3.3-70b-versatile`, `glm-4.6`)
    # + apiKey - API key of the service. Optional for local servers that do not require authentication
    # + maxTokens - Upper limit for the number of tokens in the response. Not sent if not specified
    # + temperature - Temperature for controlling randomness in the model's output. Not sent if not specified
    # + compatibility - Optional settings to adjust request and response handling for a specific server
    # + connectionConfig - Additional HTTP connection configuration
    # + return - `()` on successful initialization; otherwise, an `ai:Error`
    public isolated function init(@display {label: "Service URL"} string serviceUrl,
            @display {label: "Model Name"} string modelType,
            @display {label: "API Key"} string? apiKey = (),
            @display {label: "Maximum Tokens"} int? maxTokens = (),
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Compatibility Configuration"} CompatibilityConfig compatibility = {},
            @display {label: "Connection Configuration"} *ConnectionConfig connectionConfig) returns ai:Error? {
        if modelType.trim() == "" {
            return error ai:Error("The model name must not be empty");
        }
        readonly & CompatibilityConfig compatibilityConfig = compatibility.cloneReadOnly();
        self.llmClient = check createHttpClient(serviceUrl, connectionConfig);
        self.modelType = modelType;
        self.maxTokens = maxTokens;
        self.temperature = temperature;
        self.compatibility = compatibilityConfig;
        self.requestHeaders = buildRequestHeaders(apiKey, compatibilityConfig);
    }

    # Sends a chat request to the model with the given messages and tools.
    #
    # + messages - List of chat messages or a single user message
    # + tools - Tool definitions to be used for the tool call
    # + stop - Stop sequence to stop the completion
    # + return - Function to be called, chat response or an error in-case of failures
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages, ai:ChatCompletionFunctions[] tools = [],
            string? stop = ()) returns ai:ChatAssistantMessage|ai:Error {
        observe:ChatSpan span = observe:createChatSpan(self.modelType);
        span.addProvider(PROVIDER_NAME);
        if stop is string {
            span.addStopSequence(stop);
        }
        decimal? temperature = self.temperature;
        if temperature is decimal {
            span.addTemperature(temperature);
        }

        ReasoningStore? reasoningStore = self.compatibility.preserveReasoningContent ? self.reasoningStore : ();
        json[]|ai:Error requestMessages = buildChatRequestMessages(messages, self.compatibility.systemRole,
                reasoningStore);
        if requestMessages is ai:Error {
            span.close(requestMessages);
            return requestMessages;
        }
        span.addInputMessages(requestMessages);

        map<json> body = {
            model: self.modelType,
            messages: requestMessages
        };
        if temperature is decimal {
            body["temperature"] = temperature;
        }
        if stop is string {
            body["stop"] = stop;
        }
        if tools.length() > 0 {
            json[] requestTools = buildChatRequestTools(tools);
            body["tools"] = requestTools;
            span.addTools(tools);
        }

        ChatCompletionResponse|ai:Error response = sendChatCompletion(self.llmClient, self.requestHeaders, body,
                self.maxTokens, self.compatibility, self.adaptiveState);
        if response is ai:Error {
            log:printError(string `Error response received from the OpenAI-compatible Chat Completions API: ${
                    response.message()}`);
            span.close(response);
            return response;
        }

        ChatChoice|ai:Error choice = getFirstChoice(response);
        if choice is ai:Error {
            span.close(choice);
            return choice;
        }

        string? responseId = response?.id;
        if responseId is string {
            span.addResponseId(responseId);
        }
        int? inputTokens = response?.usage?.prompt_tokens;
        if inputTokens is int {
            span.addInputTokenCount(inputTokens);
        }
        int? outputTokens = response?.usage?.completion_tokens;
        if outputTokens is int {
            span.addOutputTokenCount(outputTokens);
        }
        string? finishReason = choice?.finish_reason;
        if finishReason is string {
            span.addFinishReason(finishReason);
        }

        ai:ChatAssistantMessage|ai:Error message = convertToAssistantMessage(choice, reasoningStore);
        if message is ai:Error {
            span.close(message);
            return message;
        }
        span.addOutputMessages(message);
        span.addOutputType(observe:TEXT);
        span.close();
        return message;
    }

    # Sends a chat request to the model and generates a value that belongs to the type
    # corresponding to the type descriptor argument.
    #
    # + prompt - The prompt to use in the chat messages
    # + td - Type descriptor specifying the expected return type format
    # + return - Generates a value that belongs to the type, or an error if generation fails
    isolated remote function generate(ai:Prompt prompt, @display {label: "Expected type"} typedesc<anydata> td = <>)
            returns td|ai:Error = @java:Method {
        'class: "io.ballerina.lib.ai.genericmodelprovider.Generator"
    } external;
}
