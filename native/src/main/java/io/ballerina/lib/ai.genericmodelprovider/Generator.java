/*
 * Copyright (c) 2026, WSO2 LLC. (https://www.wso2.com).
 *
 * WSO2 LLC. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package io.ballerina.lib.ai.genericmodelprovider;

import io.ballerina.runtime.api.Environment;
import io.ballerina.runtime.api.Module;
import io.ballerina.runtime.api.utils.StringUtils;
import io.ballerina.runtime.api.values.BObject;
import io.ballerina.runtime.api.values.BTypedesc;

/**
 * Native entry point for the dependently-typed {@code generate} method of the generic (OpenAI-compatible) model
 * provider. It reads the provider state and calls back the Ballerina function {@code generateLlmResponse}.
 *
 * <p>The field names read here are a contract with {@code ballerina/model-provider.bal}.</p>
 *
 * @since 1.0.0
 */
public class Generator {
    private static final Module MODULE = new Module("ballerinax", "ai.genericmodelprovider", "1");

    public static Object generate(Environment env, BObject modelProvider,
                                  BObject prompt, BTypedesc expectedResponseTypedesc) {
        return env.getRuntime().callFunction(
                MODULE, "generateLlmResponse", null,
                modelProvider.get(StringUtils.fromString("llmClient")),
                modelProvider.get(StringUtils.fromString("modelType")),
                modelProvider.get(StringUtils.fromString("temperature")),
                modelProvider.get(StringUtils.fromString("maxTokens")),
                modelProvider.get(StringUtils.fromString("compatibility")),
                modelProvider.get(StringUtils.fromString("requestHeaders")),
                modelProvider.get(StringUtils.fromString("adaptiveState")),
                prompt, expectedResponseTypedesc);
    }
}
