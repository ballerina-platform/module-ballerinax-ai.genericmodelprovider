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
import ballerina/constraint;

const UNSUPPORTED_DOCUMENT_ERROR = "Only text, image and audio documents are supported.";
const TEXT_ONLY_DOCUMENT_ERROR = "Only text documents are supported in system messages.";

# Returns a plain string when the content is text-only, otherwise an array of OpenAI content parts.
#
# + content - The message content
# + return - The text content, the content parts, or an `ai:Error` if a document cannot be converted
isolated function buildUserContent(string|ai:Prompt content) returns string|json[]|ai:Error {
    if content is string {
        return content;
    }
    map<json>[] parts = check flattenPrompt(content, false);
    if parts.every(part => part["type"] == "text") {
        return joinTextParts(parts);
    }
    return parts;
}

# Flattens content to text (system messages). Non-text documents -> ai:Error.
#
# + content - The message content
# + return - The text content, or an `ai:Error` if the content contains a non-text document
isolated function buildTextContent(string|ai:Prompt content) returns string|ai:Error {
    if content is string {
        return content;
    }
    // Mirrors the `ai` module's string flattening of prompts, which trims the result.
    return joinTextParts(check flattenPrompt(content, true)).trim();
}

# Converts a prompt into OpenAI content parts by interleaving its strings with its insertions.
# Adjacent text is merged into a single text part and empty text is dropped.
#
# + prompt - The prompt
# + textOnly - Whether only text is allowed; any non-text document then results in an error
# + return - The content parts, or an `ai:Error`
isolated function flattenPrompt(ai:Prompt prompt, boolean textOnly) returns map<json>[]|ai:Error {
    string[] & readonly strings = prompt.strings;
    anydata[] insertions = prompt.insertions;
    map<json>[] parts = [];
    string text = strings.length() > 0 ? strings[0] : "";

    foreach int i in 0 ..< insertions.length() {
        anydata insertion = insertions[i];
        if insertion is ai:Document|ai:Chunk {
            text = check appendDocument(insertion, text, parts, textOnly);
        } else if insertion is (ai:Document|ai:Chunk)[] {
            foreach ai:Document|ai:Chunk doc in insertion {
                text = check appendDocument(doc, text, parts, textOnly);
            }
        } else {
            text += insertion.toString();
        }
        text += strings[i + 1];
    }
    appendTextPart(text, parts);
    return parts;
}

# Adds a document to the content being built. Text is appended to the pending text, while images and audio
# flush the pending text into a text part and are added as their own content parts.
#
# + doc - The document or chunk
# + pendingText - The text accumulated since the last content part
# + parts - The content parts built so far
# + textOnly - Whether only text documents are allowed
# + return - The pending text after adding the document, or an `ai:Error`
isolated function appendDocument(ai:Document|ai:Chunk doc, string pendingText, map<json>[] parts, boolean textOnly)
        returns string|ai:Error {
    if doc is ai:TextDocument|ai:TextChunk {
        return pendingText + doc.content;
    }
    if textOnly {
        return error ai:Error(TEXT_ONLY_DOCUMENT_ERROR);
    }
    map<json> part;
    if doc is ai:ImageDocument {
        part = check buildImagePart(doc);
    } else if doc is ai:AudioDocument {
        part = check buildAudioPart(doc);
    } else {
        return error ai:Error(UNSUPPORTED_DOCUMENT_ERROR);
    }
    appendTextPart(pendingText, parts);
    parts.push(part);
    return "";
}

# Adds a text content part, unless the text is empty.
#
# + text - The text
# + parts - The content parts to add to
isolated function appendTextPart(string text, map<json>[] parts) {
    if text.length() > 0 {
        parts.push({'type: "text", text});
    }
}

# Concatenates the text of all text content parts.
#
# + parts - The content parts
# + return - The concatenated text
isolated function joinTextParts(map<json>[] parts) returns string {
    string text = "";
    foreach map<json> part in parts {
        json partText = part["text"];
        if partText is string {
            text += partText;
        }
    }
    return text;
}

# Builds an `image_url` content part for an image document.
#
# + doc - The image document
# + return - The content part, or an `ai:Error` if the URL is invalid
isolated function buildImagePart(ai:ImageDocument doc) returns map<json>|ai:Error {
    ai:Url|byte[] content = doc.content;
    string url;
    if content is ai:Url {
        ai:Url|constraint:Error validationRes = constraint:validate(content);
        if validationRes is error {
            return error ai:Error(validationRes.message(), validationRes.cause());
        }
        url = content;
    } else {
        url = string `data:${doc.metadata?.mimeType ?: "image/*"};base64,${content.toBase64()}`;
    }
    return {'type: "image_url", image_url: {url}};
}

# Builds an `input_audio` content part for an audio document. Only binary audio content is supported.
#
# + doc - The audio document
# + return - The content part, or an `ai:Error`
isolated function buildAudioPart(ai:AudioDocument doc) returns map<json>|ai:Error {
    ai:Url|byte[] content = doc.content;
    if content is ai:Url {
        return error ai:Error("URL-based audio content is not supported. Provide the audio content as bytes.");
    }
    string format = check resolveAudioFormat(doc.metadata);
    return {'type: "input_audio", input_audio: {data: content.toBase64(), format}};
}

# Resolves the audio format from the `format` metadata field, or else from the MIME type
# (e.g. `audio/mpeg` -> `mp3`, `audio/wav` -> `wav`).
#
# + metadata - The document metadata
# + return - The audio format, or an `ai:Error` if it cannot be determined
isolated function resolveAudioFormat(ai:Metadata? metadata) returns string|ai:Error {
    json format = metadata is ai:Metadata ? metadata["format"] : ();
    if format is string && format.trim() != "" {
        return format.trim().toLowerAscii();
    }
    string? mimeType = metadata?.mimeType;
    if mimeType is string {
        string normalized = mimeType.trim().toLowerAscii();
        int? paramIndex = normalized.indexOf(";");
        if paramIndex is int {
            normalized = normalized.substring(0, paramIndex).trim();
        }
        match normalized {
            "audio/mpeg"|"audio/mp3"|"audio/mpeg3"|"audio/x-mpeg-3" => {
                return "mp3";
            }
            "audio/wav"|"audio/x-wav"|"audio/wave"|"audio/vnd.wave" => {
                return "wav";
            }
        }
        if normalized.startsWith("audio/") && normalized.length() > "audio/".length() {
            return normalized.substring("audio/".length());
        }
    }
    return error ai:Error("Unable to determine the audio format. Specify it in the 'format' field of the metadata "
        + "(e.g. 'mp3' or 'wav') or set a MIME type such as 'audio/mpeg' or 'audio/wav'.");
}
