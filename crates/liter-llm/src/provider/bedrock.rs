use std::borrow::Cow;

#[cfg(feature = "bedrock")]
use crate::error::LiterLlmError;
use crate::error::Result;
use crate::provider::{Provider, StreamFormat};
use crate::types::ChatCompletionChunk;

/// Default AWS region for Bedrock when none is specified.
const DEFAULT_REGION: &str = "us-east-1";

/// Map reasoning effort levels to budget_tokens for Claude-on-Bedrock extended thinking.
fn reasoning_effort_to_budget_tokens(effort: &str) -> u64 {
    match effort {
        "low" => 1024,
        "medium" => 4096,
        "high" => 16384,
        _ => 4096,
    }
}

/// Extract a document format from a MIME type string.
///
/// E.g. `"application/pdf"` → `"pdf"`, `"text/csv"` → `"csv"`.
fn format_from_media_type(media_type: &str) -> &str {
    media_type.split('/').nth(1).unwrap_or("pdf")
}

/// Convert OpenAI-format message content (plain string or content-part array) to
/// Bedrock Converse content blocks (`text` / `image` / `document`).
///
/// Shared by both user messages and tool results: Converse's `toolResult.content`
/// accepts the same block shapes as a user turn's `content`, so this is reused
/// rather than duplicated for the `"tool"` role. ~keep
fn convert_content_to_bedrock_blocks(content: Option<&serde_json::Value>) -> Vec<serde_json::Value> {
    use serde_json::json;

    if let Some(text) = content.and_then(|c| c.as_str()) {
        vec![json!({"text": text})]
    } else if let Some(array) = content.and_then(|c| c.as_array()) {
        array
            .iter()
            .filter_map(|part| {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" => {
                        let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        Some(json!({"text": text}))
                    }
                    "image_url" => {
                        let url = part.pointer("/image_url/url").and_then(|u| u.as_str()).unwrap_or("");
                        if let Some(data_part) = url.strip_prefix("data:") {
                            let mut iter = data_part.splitn(2, ';');
                            let media_type = iter.next().unwrap_or("image/jpeg");
                            let b64 = iter.next().and_then(|s| s.strip_prefix("base64,")).unwrap_or("");
                            Some(json!({
                                "image": {
                                    "format": media_type.split('/').nth(1).unwrap_or("jpeg"),
                                    "source": {"bytes": b64}
                                }
                            }))
                        } else {
                            Some(json!({"text": url}))
                        }
                    }
                    "document" => {
                        let data = part.pointer("/document/data").and_then(|d| d.as_str()).unwrap_or("");
                        let media_type = part
                            .pointer("/document/media_type")
                            .and_then(|m| m.as_str())
                            .unwrap_or("application/pdf");
                        let format = format_from_media_type(media_type);
                        Some(json!({
                            "document": {
                                "name": "doc",
                                "format": format,
                                "source": {"bytes": data}
                            }
                        }))
                    }
                    _ => None,
                }
            })
            .collect()
    } else {
        vec![json!({"text": ""})]
    }
}

/// Determine the DNS suffix for a given AWS region.
///
/// - Standard/GovCloud regions: `amazonaws.com`
/// - European Sovereign Cloud (EUSC, `eusc-*`): `amazonaws.eu`
/// - China (`cn-*`): `amazonaws.com.cn`
fn dns_suffix_for_region(region: &str) -> &'static str {
    if region.starts_with("eusc-") {
        "amazonaws.eu"
    } else if region.starts_with("cn-") {
        "amazonaws.com.cn"
    } else {
        "amazonaws.com"
    }
}

/// Percent-encode a model ID for use in a URL path segment.
///
/// Bedrock model IDs can contain colons and slashes that must be encoded.
fn percent_encode_model(model: &str) -> String {
    let mut encoded = String::with_capacity(model.len());
    for byte in model.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            other => {
                encoded.push('%');
                let hi = char::from_digit(u32::from(other >> 4), 16).unwrap_or('0');
                let lo = char::from_digit(u32::from(other & 0xf), 16).unwrap_or('0');
                encoded.push(hi.to_ascii_uppercase());
                encoded.push(lo.to_ascii_uppercase());
            }
        }
    }
    encoded
}

/// AWS Bedrock provider.
///
/// Differences from the OpenAI-compatible baseline:
/// - Routes `bedrock/` prefixed model names to the Bedrock runtime endpoint.
/// - The model prefix is stripped before the model ID is sent in the request.
/// - Requests carry a credential when one is configured; with none, and a
///   `base_url` override, the provider is usable against a mock server.
///
/// # Authentication
///
/// Either a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, sent as
/// `Authorization: Bearer <token>` and needing no signing (so it works with the
/// `bedrock` feature off), or a SigV4 access-key pair from explicit config (see
/// [`BedrockProvider::with_credentials`]) or the environment
/// (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`), which
/// needs the feature. Precedence: explicit config, then the token, then
/// environment credentials.
///
/// # Region resolution
///
/// The region is resolved in priority order:
/// 1. Explicit value passed to [`BedrockProvider::new`] or [`BedrockProvider::from_config`].
/// 2. `AWS_DEFAULT_REGION` environment variable.
/// 3. `AWS_REGION` environment variable.
/// 4. Hard-coded default: `us-east-1`.
///
/// # Configuration
///
/// ```rust,ignore
/// let config = ClientConfigBuilder::new("unused-for-sigv4")
///     .build();
/// let client = DefaultClient::new(config, Some("bedrock/anthropic.claude-3-sonnet-20240229-v1:0"))?;
/// ```
pub struct BedrockProvider {
    region: String,
    /// Cached base URL: `https://bedrock-runtime.{region}.{dns_suffix}`.
    base_url: String,
    /// Cached cross-region prefix from `BEDROCK_CROSS_REGION` env var at
    /// construction time (e.g. `Some("us.")`) so we avoid reading the
    /// environment on every request.
    cross_region_prefix: Option<String>,
    /// Explicit AWS access key ID, overriding `AWS_ACCESS_KEY_ID` when set.
    access_key_id: Option<String>,
    /// Explicit AWS secret access key, overriding `AWS_SECRET_ACCESS_KEY` when set.
    secret_access_key: Option<String>,
    /// Explicit AWS session token, overriding `AWS_SESSION_TOKEN` when set.
    session_token: Option<String>,
}

impl BedrockProvider {
    /// Construct with the given AWS region.
    ///
    /// The base URL is derived from the region's DNS suffix. To override it
    /// entirely, set `BEDROCK_BASE_URL` in the environment.
    #[must_use]
    pub fn new(region: impl Into<String>) -> Self {
        let region = region.into();
        let custom_base_url = std::env::var("BEDROCK_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.trim_end_matches('/').to_string());
        let base_url = custom_base_url.clone().unwrap_or_else(|| {
            let dns_suffix = dns_suffix_for_region(&region);
            format!("https://bedrock-runtime.{region}.{dns_suffix}")
        });
        let cross_region_prefix = if custom_base_url.is_some() {
            None
        } else {
            std::env::var("BEDROCK_CROSS_REGION")
                .ok()
                .filter(|v| !v.is_empty())
                .map(|v| format!("{v}."))
        };
        Self {
            region,
            base_url,
            cross_region_prefix,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
        }
    }

    /// Construct using region from the environment, falling back to `us-east-1`.
    ///
    /// Reads `AWS_DEFAULT_REGION` then `AWS_REGION`.
    #[must_use]
    pub fn from_env() -> Self {
        let region = std::env::var("AWS_DEFAULT_REGION")
            .or_else(|_| std::env::var("AWS_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_owned());
        Self::new(region)
    }

    /// Construct from explicit, optional config values, falling back to the
    /// environment for anything left unset.
    ///
    /// Region resolution order: `region` -> `AWS_DEFAULT_REGION` ->
    /// `AWS_REGION` -> `us-east-1`. Credentials and the cross-region prefix
    /// fall back to their respective environment variables at request time
    /// (see [`BedrockProvider::with_credentials`] and
    /// [`BedrockProvider::with_cross_region_prefix`]).
    #[must_use]
    pub fn from_config(
        region: Option<String>,
        cross_region_prefix: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        let region = region
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
            .or_else(|| std::env::var("AWS_REGION").ok())
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        Self::new(region)
            .with_cross_region_prefix(cross_region_prefix)
            .with_credentials(access_key_id, secret_access_key, session_token)
    }

    /// Override the cross-region inference profile prefix (e.g. `"us"`).
    ///
    /// When `None`, the prefix cached from `BEDROCK_CROSS_REGION` at
    /// construction time (if any) is left untouched.
    #[must_use]
    pub fn with_cross_region_prefix(mut self, prefix: Option<String>) -> Self {
        if let Some(prefix) = prefix {
            let prefix = if prefix.ends_with('.') {
                prefix
            } else {
                format!("{prefix}.")
            };
            self.cross_region_prefix = Some(prefix);
        }
        self
    }

    /// Set explicit AWS credentials for SigV4 signing, overriding the
    /// `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
    /// environment variables when present.
    #[must_use]
    pub fn with_credentials(
        mut self,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        self.access_key_id = access_key_id;
        self.secret_access_key = secret_access_key;
        self.session_token = session_token;
        self
    }

    /// Return the AWS region this provider is configured for.
    #[must_use]
    #[allow(dead_code)]
    pub fn region(&self) -> &str {
        &self.region
    }

    /// The Bedrock API key from `AWS_BEARER_TOKEN_BEDROCK`, when it applies.
    ///
    /// Resolved per request, so a rotated key needs no new client — the same reason
    /// the SigV4 path resolves `AWS_ACCESS_KEY_ID` at signing time. A complete pair from
    /// [`BedrockProvider::with_credentials`] wins: an in-code credential must not be
    /// redirected by an ambient variable. Against *environment* credentials the token
    /// wins, matching the AWS SDKs. ~keep
    fn bearer_token(&self) -> Option<String> {
        // ~keep Empty is not set: `LlmConfig` fills the absent half of a partial pair with
        // `String::new()`, which would otherwise drop the token and sign with no secret.
        let has_explicit_pair = self.access_key_id.as_deref().is_some_and(|key| !key.is_empty())
            && self.secret_access_key.as_deref().is_some_and(|key| !key.is_empty());
        if has_explicit_pair {
            return None;
        }
        std::env::var("AWS_BEARER_TOKEN_BEDROCK")
            .ok()
            .filter(|token| !token.is_empty())
    }
}

impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    /// Base URL for the Bedrock runtime service.
    ///
    /// When a `base_url` override is set in [`ClientConfig`] (as in tests),
    /// the override takes precedence and this value is never used.
    fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Bedrock uses SigV4 signing rather than a static authorization header.
    ///
    /// Returns `None` so the HTTP layer skips adding an `Authorization` header.
    /// Authentication headers are injected by [`BedrockProvider::signing_headers`]:
    /// a Bearer token when one is set, otherwise SigV4 when the `bedrock` feature
    /// is enabled.
    fn auth_header<'a>(&'a self, _api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        None
    }

    fn matches_model(&self, model: &str) -> bool {
        model.starts_with("bedrock/")
    }

    fn strip_model_prefix<'m>(&self, model: &'m str) -> &'m str {
        model.strip_prefix("bedrock/").unwrap_or(model)
    }

    /// Validate that the provider is usable in the current environment.
    ///
    /// When the `bedrock` feature is enabled, checks that a usable credential is
    /// available: either a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, or both
    /// an AWS access key and secret key (explicit config, or `AWS_ACCESS_KEY_ID`
    /// and `AWS_SECRET_ACCESS_KEY` in the environment). With neither, no request
    /// can be authenticated, so this returns an error instead of letting the
    /// request continue toward an unsigned or malformed send.
    ///
    /// Called once at client construction and again on every request from
    /// [`BedrockProvider::transform_request`], since credentials can become
    /// unavailable between the two.
    ///
    /// When the `bedrock` feature is disabled (e.g. in tests with `base_url`
    /// override), validation is skipped so callers can connect to a mock server
    /// without real AWS credentials.
    fn validate(&self) -> Result<()> {
        #[cfg(feature = "bedrock")]
        {
            // ~keep A Bedrock API key authenticates on its own; the SigV4 pair is not required.
            if self.bearer_token().is_some() {
                return Ok(());
            }

            let has_access_key = self.access_key_id.is_some() || std::env::var("AWS_ACCESS_KEY_ID").is_ok();
            let has_secret_key = self.secret_access_key.is_some() || std::env::var("AWS_SECRET_ACCESS_KEY").is_ok();
            // ~keep Both keys are required: sigv4_sign has no instance-profile/SSO fallback,
            // so a request missing either one can never actually be signed.
            if !has_access_key || !has_secret_key {
                return Err(LiterLlmError::Authentication {
                    message: "AWS Bedrock requires credentials. \
                              Set a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, or supply an \
                              access-key pair explicitly via config or as AWS_ACCESS_KEY_ID and \
                              AWS_SECRET_ACCESS_KEY (and optionally AWS_SESSION_TOKEN) in the \
                              environment."
                        .into(),
                    status: 401,
                });
            }
        }
        Ok(())
    }

    /// Bedrock uses AWS EventStream binary framing, not SSE.
    fn stream_format(&self) -> StreamFormat {
        StreamFormat::AwsEventStream
    }

    /// Build the full URL for a Bedrock Converse API request.
    ///
    /// Chat completions map to `/model/{encoded_model}/converse`.
    /// Embeddings map to `/model/{encoded_model}/invoke`.
    /// All other paths are passed through unchanged.
    ///
    /// When the `BEDROCK_CROSS_REGION` environment variable is set, the
    /// cross-region inference profile prefix is prepended to the model ID.
    /// For example, with `BEDROCK_CROSS_REGION=us`, model
    /// `anthropic.claude-3-sonnet-20240229-v1:0` becomes
    /// `us.anthropic.claude-3-sonnet-20240229-v1:0`.
    fn build_url(&self, endpoint_path: &str, model: &str) -> String {
        let base = self.base_url();
        let effective_model = self.apply_cross_region_prefix(model);
        let encoded_model = percent_encode_model(&effective_model);
        if endpoint_path.contains("chat/completions") {
            format!("{base}/model/{encoded_model}/converse")
        } else if endpoint_path.contains("embeddings") {
            format!("{base}/model/{encoded_model}/invoke")
        } else {
            format!("{base}{endpoint_path}")
        }
    }

    /// Build the streaming URL: `/model/{id}/converse-stream`.
    fn build_stream_url(&self, endpoint_path: &str, model: &str) -> String {
        let base = self.base_url();
        let effective_model = self.apply_cross_region_prefix(model);
        let encoded_model = percent_encode_model(&effective_model);
        if endpoint_path.contains("chat/completions") {
            format!("{base}/model/{encoded_model}/converse-stream")
        } else {
            self.build_url(endpoint_path, model)
        }
    }

    /// Convert an OpenAI-style chat request to Bedrock Converse API format.
    ///
    /// Key differences from the OpenAI format:
    /// - System messages are extracted to a top-level `system` array.
    /// - Messages use `content` arrays with typed blocks (`text`, `toolUse`, `toolResult`).
    /// - Generation parameters live in `inferenceConfig`.
    /// - Tools are described in `toolConfig.tools[].toolSpec`.
    /// - `temperature` and `top_p` outside Bedrock's documented `[0.0, 1.0]` range are
    ///   rejected with a `BadRequest` error rather than forwarded and left for Bedrock to reject.
    ///
    /// Embedding requests (`input` present, no `messages`) are routed to
    /// [`transform_bedrock_embed_request`] instead: Bedrock has no Converse-style
    /// unified embeddings API, so this dispatch mirrors the same
    /// `input`/`messages` discriminator used by [`super::vertex`].
    fn transform_request(&self, body: &mut serde_json::Value) -> Result<()> {
        use serde_json::json;

        // ~keep Re-checked on every request (not just at client construction): credentials
        // can become unavailable between construction and send. `signing_headers` now also
        // hard-errors on a signing failure (see #42), but this precheck fails fast with a
        // clearer, missing-credentials-specific message before any signing work happens.
        self.validate()?;

        // ~keep Embedding bodies have `input` and no `messages`. Without this branch they
        // ~keep fell through to the Converse transform below, which unconditionally removes
        // ~keep `messages` (absent here) via `unwrap_or_default()` and rebuilds the body as
        // ~keep `{"messages": []}`, silently discarding the entire `input`.
        if body.get("input").is_some() && body.get("messages").is_none() {
            return transform_bedrock_embed_request(body);
        }

        // ~keep The Bedrock Converse API's InferenceConfiguration documents both `temperature`
        // ~keep and `topP` as 0-1, narrower than this crate's OpenAI-shaped 0.0-2.0 doc for
        // ~keep `temperature`, regardless of the underlying foundation model (Anthropic, Titan,
        // ~keep Llama, ...) since Converse enforces this at the gateway, not the model. See
        // ~keep https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html.
        // ~keep Reject out-of-range values locally rather than forwarding them for a 400.
        super::validate_sampling_param_range(body, "temperature", "Bedrock", 0.0, 1.0)?;
        super::validate_sampling_param_range(body, "top_p", "Bedrock", 0.0, 1.0)?;

        let messages = body
            .as_object_mut()
            .and_then(|o| o.remove("messages"))
            .and_then(|v| match v {
                serde_json::Value::Array(arr) => Some(arr),
                _ => None,
            })
            .unwrap_or_default();

        let mut system_parts = vec![];
        let mut converse_messages = vec![];

        for msg in &messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let content = msg.get("content");

            match role {
                "system" | "developer" => {
                    if let Some(text) = content.and_then(|c| c.as_str()) {
                        system_parts.push(json!({"text": text}));
                    } else if let Some(array) = content.and_then(|c| c.as_array()) {
                        for part in array {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                system_parts.push(json!({"text": text}));
                            }
                        }
                    }
                }
                "user" => {
                    let parts = convert_content_to_bedrock_blocks(content);
                    converse_messages.push(json!({"role": "user", "content": parts}));
                }
                "assistant" => {
                    let mut parts = vec![];
                    if let Some(text) = content.and_then(|c| c.as_str())
                        && !text.is_empty()
                    {
                        parts.push(json!({"text": text}));
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tool_calls {
                            let input = match tc.pointer("/function/arguments").and_then(|a| a.as_str()) {
                                Some(args_str) => serde_json::from_str(args_str).unwrap_or_else(|error| {
                                    tracing::warn!(
                                        %error,
                                        "Bedrock tool_calls[].function.arguments was not valid JSON; using an \
                                         empty object"
                                    );
                                    json!({})
                                }),
                                None => json!({}),
                            };
                            parts.push(json!({
                                "toolUse": {
                                    "toolUseId": tc.get("id"),
                                    "name": tc.pointer("/function/name"),
                                    "input": input
                                }
                            }));
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({"text": ""}));
                    }
                    converse_messages.push(json!({"role": "assistant", "content": parts}));
                }
                "tool" => {
                    let tool_call_id = msg.get("tool_call_id").and_then(|t| t.as_str()).unwrap_or("");
                    // toolResult.content accepts the same text/image/document block shapes as a
                    // user turn's content, so a `ToolMessage::content` of `UserContent::Parts`
                    // reaches Bedrock natively via the shared block conversion. ~keep
                    let result_content = convert_content_to_bedrock_blocks(content);
                    let is_error = msg.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                    let status = if is_error { "error" } else { "success" };
                    converse_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tool_call_id,
                                "content": result_content,
                                "status": status
                            }
                        }]
                    }));
                }
                _ => {}
            }
        }

        let mut inference_config = json!({});
        if let Some(max_tokens) = body.get("max_tokens").or_else(|| body.get("max_completion_tokens")) {
            inference_config["maxTokens"] = max_tokens.clone();
        }
        if let Some(temp) = body.get("temperature") {
            inference_config["temperature"] = temp.clone();
        }
        if let Some(top_p) = body.get("top_p") {
            inference_config["topP"] = top_p.clone();
        }
        if let Some(stop) = body.get("stop") {
            let sequences = if let Some(s) = stop.as_str() {
                vec![json!(s)]
            } else {
                stop.as_array().cloned().unwrap_or_default()
            };
            inference_config["stopSequences"] = json!(sequences);
        }

        let tool_config = body.get("tools").and_then(|tools| {
            tools.as_array().map(|arr| {
                let bedrock_tools: Vec<serde_json::Value> = arr
                    .iter()
                    .map(|t| {
                        let parameters = t
                            .pointer("/function/parameters")
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object"}));
                        json!({
                            "toolSpec": {
                                "name": t.pointer("/function/name"),
                                "description": t.pointer("/function/description"),
                                "inputSchema": {"json": parameters}
                            }
                        })
                    })
                    .collect();
                json!({"tools": bedrock_tools})
            })
        });

        let mut additional_model_fields: Option<serde_json::Value> = None;
        if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str()) {
            let budget_tokens = reasoning_effort_to_budget_tokens(effort);
            additional_model_fields = Some(json!({
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
            }));
        }

        if let Some(response_format) = body.get("response_format") {
            let rf_type = response_format.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match rf_type {
                "json_schema" => {
                    let schema = response_format.get("json_schema").and_then(|js| js.get("schema"));
                    let schema_str = schema
                        .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
                        .unwrap_or_default();
                    let instruction = if schema_str.is_empty() {
                        "You MUST respond with valid JSON only. No other text.".to_owned()
                    } else {
                        format!(
                            "You MUST respond with valid JSON only that conforms to this schema:\n```json\n{schema_str}\n```\nNo other text outside the JSON."
                        )
                    };
                    system_parts.push(json!({"text": instruction}));
                }
                "json_object" => {
                    system_parts.push(json!({"text": "You MUST respond with valid JSON only. No other text."}));
                }
                _ => {}
            }
        }

        let guardrail_config = body.get("extra_body").and_then(|eb| eb.get("guardrailConfig")).cloned();

        let mut new_body = json!({
            "messages": converse_messages,
        });
        if !system_parts.is_empty() {
            new_body["system"] = json!(system_parts);
        }
        if let Some(obj) = inference_config.as_object()
            && !obj.is_empty()
        {
            new_body["inferenceConfig"] = inference_config;
        }
        if let Some(tc) = tool_config {
            new_body["toolConfig"] = tc;
        }
        if let Some(amf) = additional_model_fields {
            new_body["additionalModelRequestFields"] = amf;
        }
        if let Some(gc) = guardrail_config {
            new_body["guardrailConfig"] = gc;
        }

        // ~keep Bedrock's Converse API has real equivalents for these two OpenAI fields:
        // ~keep `serviceTier: {type}` accepts "priority"|"default"|"flex"|"reserved", so every
        // ~keep OpenAI value except "auto" maps directly; "auto" has no Bedrock counterpart and
        // ~keep omitting `serviceTier` already gives provider-default behaviour, which is the
        // ~keep same effect. `requestMetadata` is a string-to-string map used for CloudTrail/
        // ~keep CloudWatch filtering, the same shape as our `metadata` tag map.
        if let Some(tier) = body.get("service_tier").and_then(|v| v.as_str())
            && tier != "auto"
        {
            new_body["serviceTier"] = json!({"type": tier});
        }
        if let Some(metadata) = body.get("metadata").and_then(|v| v.as_object())
            && !metadata.is_empty()
        {
            new_body["requestMetadata"] = json!(metadata);
        }

        // ~keep `logprobs`/`top_logprobs`, `audio` and `web_search_options` have no Bedrock
        // ~keep Converse API equivalent (InferenceConfiguration has no logprobs field; Claude on
        // ~keep Bedrock has no audio-output or built-in web-search-tool configuration). They are
        // ~keep already dropped by the wholesale rebuild above; warn so a caller who asked for
        // ~keep any of them can tell the request silently ignored that part of the ask.
        let logprobs_requested = body.get("logprobs").and_then(|v| v.as_bool()).unwrap_or(false);
        if logprobs_requested || body.get("top_logprobs").is_some() {
            tracing::warn!(
                "chat request set logprobs/top_logprobs, which Bedrock's Converse API does not \
                 support; the fields were dropped and the response will not include log \
                 probabilities"
            );
        }
        if body.get("audio").is_some() {
            tracing::warn!(
                "chat request set `audio`, which was dropped: Bedrock's Converse API has no \
                 audio output support"
            );
        }
        if body.get("web_search_options").is_some() {
            tracing::warn!(
                "chat request set `web_search_options`, which was dropped: Bedrock's Converse \
                 API has no built-in web-search tool configuration equivalent"
            );
        }

        *body = new_body;
        Ok(())
    }

    /// Normalize a Bedrock Converse API response to OpenAI chat completion format.
    ///
    /// Bedrock wraps the assistant's message in `output.message.content[]` blocks.
    /// Stop reasons use Bedrock terminology (`end_turn`, `tool_use`, etc.) and are
    /// mapped to the OpenAI `finish_reason` set.
    ///
    /// **Known limitation:** The `model` field in the normalized response is
    /// always `""`.  Bedrock does not include the model name in its response
    /// body — the model is only present in the request URL path.  Threading
    /// the model through would require a signature change to `transform_response`.
    ///
    /// Embedding responses (Titan's `{"embedding": [...]}` or Cohere's
    /// `{"embeddings": [[...], ...]}`) are routed to
    /// [`transform_bedrock_embed_response`] instead. Unlike the request side,
    /// this dispatch cannot use the model ID — `transform_response` has no
    /// model parameter (see the limitation above) and an `InvokeModel`
    /// response body carries no model field either — so it sniffs the
    /// response shape instead, same as [`super::vertex::transform_gemini_response`].
    fn transform_response(&self, body: &mut serde_json::Value) -> Result<()> {
        use serde_json::json;

        if body.get("embedding").is_some() || body.get("embeddings").is_some() {
            return transform_bedrock_embed_response(body);
        }

        let stop_reason = body.get("stopReason").and_then(|s| s.as_str()).unwrap_or("end_turn");
        let usage = body.get("usage").cloned();

        let content_blocks = body
            .pointer("/output/message/content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        let text: String = content_blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("");

        let tool_calls: Vec<serde_json::Value> = content_blocks
            .iter()
            .filter_map(|b| {
                b.get("toolUse").map(|tu| {
                    let arguments = serde_json::to_string(tu.get("input").unwrap_or(&json!({}))).unwrap_or_default();
                    json!({
                        "id": tu.get("toolUseId"),
                        "type": "function",
                        "function": {
                            "name": tu.get("name"),
                            "arguments": arguments
                        }
                    })
                })
            })
            .collect();

        let finish_reason = match stop_reason {
            "end_turn" => "stop",
            "tool_use" => "tool_calls",
            "max_tokens" => "length",
            "stop_sequence" => "stop",
            "content_filtered" | "guardrail_intervened" => "content_filter",
            _ => "stop",
        };

        let input_tokens = usage
            .as_ref()
            .and_then(|u| u.get("inputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .as_ref()
            .and_then(|u| u.get("outputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let response_id = body
            .get("requestId")
            .or_else(|| body.get("conversationId"))
            .cloned()
            .unwrap_or_else(|| json!("bedrock-resp"));

        let content_value: serde_json::Value = if text.is_empty() { json!(null) } else { json!(text) };

        let mut message = json!({"role": "assistant", "content": content_value});
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }

        *body = json!({
            "id": response_id,
            "object": "chat.completion",
            "created": super::unix_timestamp_secs(),
            "model": "",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        });

        Ok(())
    }

    /// Compute the authentication headers for the request.
    ///
    /// With a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, returns one
    /// `Authorization: Bearer <token>` header and skips signing.
    ///
    /// Otherwise, when the `bedrock` feature is enabled, derives the `Authorization`,
    /// `x-amz-date`, and (when a session token is present) `x-amz-security-token`
    /// headers from the current request parameters and AWS credentials.
    ///
    /// When the `bedrock` feature is disabled, returns an empty vector so
    /// requests work against override base-URLs (e.g. mock servers in tests).
    ///
    /// # Errors
    ///
    /// `Provider::transform_request` (called earlier in the send path, see
    /// [`BedrockProvider::transform_request`]) re-validates that credentials are
    /// present on every request and hard-errors before any network I/O, which
    /// closes the realistic failure mode (missing or revoked credentials). The
    /// only way `sigv4_sign` can still fail here despite that precheck is an
    /// internal SigV4 library error (malformed signing params). Fix for #42:
    /// that failure is now propagated as a hard error instead of silently
    /// falling back to an unsigned request. ~keep
    fn signing_headers(&self, method: &str, url: &str, body: &[u8]) -> Result<Vec<(String, String)>> {
        // ~keep Outside the `bedrock` feature gate deliberately: a Bearer header needs no
        // signing machinery, so a default build authenticates without pulling in aws-sigv4.
        if let Some(token) = self.bearer_token() {
            return Ok(vec![("authorization".to_owned(), format!("Bearer {token}"))]);
        }

        #[cfg(feature = "bedrock")]
        {
            sigv4_sign(
                method,
                url,
                body,
                &self.region,
                self.access_key_id.as_deref(),
                self.secret_access_key.as_deref(),
                self.session_token.as_deref(),
            )
            .map_err(|error| {
                tracing::error!(
                    error = %error,
                    %method,
                    "Bedrock SigV4 signing failed after credential precheck passed"
                );
                error
            })
        }

        #[cfg(not(feature = "bedrock"))]
        {
            let _ = (method, url, body);
            Ok(vec![])
        }
    }
}

/// `input_type` sent for Bedrock Cohere embed models.
///
/// Cohere's v3 embed models require this field, but OpenAI's `EmbeddingRequest` has
/// no equivalent concept to map from, so a default must be chosen. `search_document`
/// is the indexing-side value — correct for the dominant use of an embeddings
/// endpoint. A caller embedding *queries* wants `search_query` and will get subtly
/// mismatched vectors; that needs a provider-options passthrough to fix properly,
/// which this constant deliberately does not fake. ~keep
const COHERE_EMBED_DEFAULT_INPUT_TYPE: &str = "search_document";

/// Convert an OpenAI-style embedding request to a Bedrock model-family `InvokeModel` body.
///
/// Bedrock has no unified embeddings API — each model family defines its own wire
/// shape, and there is nothing else in `transform_request`'s signature to key off
/// of, so this dispatches on `body["model"]`. That field is populated by
/// `Client::prepare_request` (`crates/liter-llm/src/client/mod.rs`), which inserts
/// the already-prefix-stripped model ID into the body immediately before calling
/// `transform_request` — so it is reliably present here.
///
/// Supported families:
/// - Amazon Titan (`amazon.titan-embed-*`): `{"inputText": "..."}`.
/// - Cohere Embed (`cohere.embed-*`): `{"texts": [...], "input_type": "search_document"}`.
///
/// Any other model prefix is rejected with a clear error rather than guessed at.
///
/// ~keep Titan's `invoke` endpoint accepts exactly one string per call (batch Titan
/// ~keep embedding is a separate async `CreateModelInvocationJob` API, not this
/// ~keep synchronous path). A batched OpenAI `input` is reduced to its first element;
/// ~keep the rest are dropped with a warning rather than silently discarded.
fn transform_bedrock_embed_request(body: &mut serde_json::Value) -> Result<()> {
    use crate::error::LiterLlmError;
    use serde_json::json;

    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_owned();

    let input = body.get("input").cloned().unwrap_or_default();
    let texts: Vec<String> = match &input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) if arr.iter().all(serde_json::Value::is_string) => {
            arr.iter().filter_map(|v| v.as_str().map(ToOwned::to_owned)).collect()
        }
        _ => {
            return Err(LiterLlmError::BadRequest {
                message: "Bedrock embedding adapters support text input only; use a multimodal-compatible custom provider for image embeddings".into(),
                status: 400,
            });
        }
    };

    if model.starts_with("amazon.titan-embed") {
        // ~keep Reject rather than truncate. OpenAI's contract is that `data[]` parallels
        // ~keep `input[]`, so returning one vector for an N-element batch hands the caller
        // ~keep silently misaligned embeddings — the kind of defect that corrupts a vector
        // ~keep store and only surfaces much later as bad retrieval.
        if texts.len() > 1 {
            return Err(LiterLlmError::BadRequest {
                message: format!(
                    "Bedrock Titan embedding model '{model}' accepts a single input per call, but \
                     {} were supplied; issue one request per input (batch Titan embedding is a \
                     separate asynchronous job API)",
                    texts.len()
                ),
                status: 400,
            });
        }
        let text = texts.first().cloned().unwrap_or_default();
        let mut new_body = json!({"inputText": text});
        if let Some(dimensions) = body.get("dimensions") {
            new_body["dimensions"] = dimensions.clone();
        }
        *body = new_body;
        return Ok(());
    }

    if model.starts_with("cohere.embed") {
        *body = json!({
            "texts": texts,
            "input_type": COHERE_EMBED_DEFAULT_INPUT_TYPE
        });
        return Ok(());
    }

    Err(LiterLlmError::BadRequest {
        message: format!(
            "unsupported Bedrock embedding model '{model}': liter-llm currently supports \
             amazon.titan-embed-* and cohere.embed-* embedding models"
        ),
        status: 400,
    })
}

/// Normalize a Bedrock `InvokeModel` embedding response to OpenAI's embeddings list format.
///
/// Dispatched by response shape rather than model ID: `transform_response` has no
/// model parameter, and an `InvokeModel` response body carries no model field
/// either (same limitation noted on [`BedrockProvider::transform_response`]).
///
/// - Titan (`{"embedding": [...], "inputTextTokenCount": N}`) -> single embedding,
///   with `inputTextTokenCount` threaded through as `prompt_tokens`.
/// - Cohere (`{"embeddings": [[...], ...]}`) -> one embedding per input text.
///   ~keep Bedrock's Cohere embed response carries no token-usage field, so
///   ~keep usage is reported as zero rather than guessed at.
fn transform_bedrock_embed_response(body: &mut serde_json::Value) -> Result<()> {
    use serde_json::json;

    if let Some(embedding) = body.get("embedding").cloned() {
        let prompt_tokens = body.get("inputTextTokenCount").and_then(|v| v.as_u64()).unwrap_or(0);
        *body = json!({
            "object": "list",
            "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
            "model": "",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens
            }
        });
        return Ok(());
    }

    if let Some(embeddings) = body.get("embeddings").and_then(|e| e.as_array()).cloned() {
        let data: Vec<serde_json::Value> = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| json!({"object": "embedding", "embedding": embedding, "index": index}))
            .collect();
        *body = json!({
            "object": "list",
            "data": data,
            "model": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        });
        return Ok(());
    }

    Ok(())
}

/// Parse a Bedrock ConverseStream EventStream event into a `ChatCompletionChunk`.
///
/// Bedrock ConverseStream events:
/// - `messageStart` → role delta
/// - `contentBlockStart` → tool_use start (with toolUseId and name)
/// - `contentBlockDelta` → text delta or tool_use input delta
/// - `contentBlockStop` → (ignored)
/// - `messageStop` → finish_reason
/// - `metadata` → usage (emitted as a final chunk with empty delta)
///
/// Returns `Ok(None)` for events that don't map to a chunk (e.g. `contentBlockStop`).
///
/// **Known limitation:** The `id` field is hardcoded to `"bedrock-stream"` and
/// `model` is always `""` on every chunk.  Bedrock's ConverseStream protocol does
/// not include a request/response ID or model name in its event payloads, and
/// this parser is stateless so it cannot carry forward values from the original
/// request.  This differs from the OpenAI format where every chunk includes the
/// real `id` and `model`.
pub(crate) fn parse_bedrock_stream_event(event_type: &str, payload: &str) -> Result<Option<ChatCompletionChunk>> {
    use crate::error::LiterLlmError;
    use serde_json::json;

    let v: serde_json::Value = serde_json::from_str(payload).map_err(|e| LiterLlmError::Streaming {
        message: format!("Bedrock stream event parse error: {e}"),
    })?;

    let chunk_from_json = |chunk_json: serde_json::Value| -> Result<ChatCompletionChunk> {
        serde_json::from_value(chunk_json).map_err(|e| LiterLlmError::Streaming {
            message: format!("Bedrock chunk deserialization error: {e}"),
        })
    };

    match event_type {
        "messageStart" => {
            let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("assistant");
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {"role": role},
                    "finish_reason": null
                }]
            }))
            .map(Some)
        }
        "contentBlockStart" => {
            let index = v.get("contentBlockIndex").and_then(|i| i.as_u64()).unwrap_or(0);
            if let Some(tool_use) = v.pointer("/start/toolUse") {
                let tool_use_id = tool_use.get("toolUseId").and_then(|t| t.as_str()).unwrap_or("");
                let name = tool_use.get("name").and_then(|n| n.as_str()).unwrap_or("");
                chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": tool_use_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""}
                            }]
                        },
                        "finish_reason": null
                    }]
                }))
                .map(Some)
            } else {
                Ok(None)
            }
        }
        "contentBlockDelta" => {
            let index = v.get("contentBlockIndex").and_then(|i| i.as_u64()).unwrap_or(0);

            if let Some(text) = v.pointer("/delta/text").and_then(|t| t.as_str()) {
                return chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                    }]
                }))
                .map(Some);
            }

            if let Some(input_json) = v.pointer("/delta/toolUse/input").and_then(|i| i.as_str()) {
                return chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {"arguments": input_json}
                            }]
                        },
                        "finish_reason": null
                    }]
                }))
                .map(Some);
            }

            tracing::warn!(
                content_block_index = index,
                "Bedrock contentBlockDelta with unrecognized delta shape; skipping"
            );

            Ok(None)
        }
        "contentBlockStop" => Ok(None),
        "messageStop" => {
            let stop_reason = v.get("stopReason").and_then(|s| s.as_str()).unwrap_or("end_turn");
            let finish_reason = match stop_reason {
                "end_turn" => "stop",
                "tool_use" => "tool_calls",
                "max_tokens" => "length",
                "stop_sequence" => "stop",
                "content_filtered" | "guardrail_intervened" => "content_filter",
                _ => "stop",
            };
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }))
            .map(Some)
        }
        "metadata" => {
            let input_tokens = v.pointer("/usage/inputTokens").and_then(|t| t.as_u64()).unwrap_or(0);
            let output_tokens = v.pointer("/usage/outputTokens").and_then(|t| t.as_u64()).unwrap_or(0);
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }))
            .map(Some)
        }
        _ => Ok(None),
    }
}

/// Apply the cross-region inference profile prefix using the value cached at
/// construction time from the `BEDROCK_CROSS_REGION` environment variable.
///
/// When the prefix is set (e.g. `"us."`), the model ID
/// `anthropic.claude-3-sonnet-20240229-v1:0` becomes
/// `us.anthropic.claude-3-sonnet-20240229-v1:0`.
///
/// If the model already starts with the cross-region prefix, it is returned
/// unchanged to avoid double-prefixing.
impl BedrockProvider {
    fn apply_cross_region_prefix(&self, model: &str) -> String {
        match &self.cross_region_prefix {
            Some(prefix) => {
                if model.starts_with(prefix.as_str()) {
                    model.to_owned()
                } else {
                    format!("{prefix}{model}")
                }
            }
            None => model.to_owned(),
        }
    }
}

/// Legacy free function kept for existing tests. Reads the env var directly.
///
/// Production code uses [`BedrockProvider::apply_cross_region_prefix`] which
/// reads the env var once at construction time.
#[cfg(test)]
fn apply_cross_region_prefix(model: &str) -> String {
    match std::env::var("BEDROCK_CROSS_REGION") {
        Ok(region) if !region.is_empty() => {
            let prefix = format!("{region}.");
            if model.starts_with(&prefix) {
                model.to_owned()
            } else {
                format!("{prefix}{model}")
            }
        }
        _ => model.to_owned(),
    }
}

/// Compute AWS SigV4 signing headers using the `aws-sigv4` crate.
///
/// Each credential falls back to the standard AWS environment variable when
/// the corresponding explicit argument is `None`:
/// - `access_key_id` -> `AWS_ACCESS_KEY_ID` (required)
/// - `secret_access_key` -> `AWS_SECRET_ACCESS_KEY` (required)
/// - `session_token` -> `AWS_SESSION_TOKEN` (optional, for temporary credentials)
///
/// Returns a vector of `(header-name, header-value)` pairs to inject into the
/// outgoing HTTP request.
#[cfg(feature = "bedrock")]
fn sigv4_sign(
    method: &str,
    url: &str,
    body: &[u8],
    region: &str,
    access_key_id: Option<&str>,
    secret_access_key: Option<&str>,
    session_token: Option<&str>,
) -> Result<Vec<(String, String)>> {
    use aws_credential_types::Credentials;
    use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
    use aws_sigv4::sign::v4::SigningParams;

    let access_key = access_key_id
        .map(str::to_owned)
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .ok_or_else(|| LiterLlmError::BadRequest {
            message: "AWS access key ID is required for Bedrock requests: set it explicitly via config or \
                      the AWS_ACCESS_KEY_ID environment variable"
                .into(),
            status: 400,
        })?;
    let secret_key = secret_access_key
        .map(str::to_owned)
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .ok_or_else(|| LiterLlmError::BadRequest {
            message: "AWS secret access key is required for Bedrock requests: set it explicitly via config or \
                      the AWS_SECRET_ACCESS_KEY environment variable"
                .into(),
            status: 400,
        })?;
    let session_token = session_token
        .map(str::to_owned)
        .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok());

    let credentials = Credentials::new(access_key, secret_key, session_token, None, "env");

    let identity = credentials.into();

    let signing_settings = SigningSettings::default();
    let now = std::time::SystemTime::now();

    let params = SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("bedrock")
        .time(now)
        .settings(signing_settings)
        .build()
        .map_err(|e| LiterLlmError::BadRequest {
            message: format!("failed to build SigV4 signing params: {e}"),
            status: 400,
        })?;

    let signable = SignableRequest::new(
        method,
        url,
        std::iter::empty::<(&str, &str)>(),
        SignableBody::Bytes(body),
    )
    .map_err(|e| LiterLlmError::BadRequest {
        message: format!("failed to create signable request: {e}"),
        status: 400,
    })?;

    let signing_output = sign(signable, &params.into()).map_err(|e| LiterLlmError::BadRequest {
        message: format!("SigV4 signing failed: {e}"),
        status: 400,
    })?;

    let instructions = signing_output.output();
    let signed_headers: Vec<(String, String)> = instructions
        .headers()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect();

    Ok(signed_headers)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use serial_test::serial;

    use super::*;
    use crate::provider::Provider;
    use crate::types::chat::FinishReason;

    fn provider() -> BedrockProvider {
        // ~keep SAFETY: env vars are process-global; `#[serial]` on callers prevents races.
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
        // ~keep Explicit dummy credentials so `transform_request`'s per-request
        // `validate()` check (see #42) doesn't fail non-signing-focused tests
        // regardless of the ambient AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env state.
        BedrockProvider::new("us-east-1").with_credentials(
            Some("AKIATESTDUMMY".to_owned()),
            Some("test-dummy-secret".to_owned()),
            None,
        )
    }

    #[test]
    #[serial]
    fn from_config_prefers_explicit_region_over_env() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::set_var("AWS_DEFAULT_REGION", "us-west-2") };
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = BedrockProvider::from_config(Some("eu-central-1".to_owned()), None, None, None, None);
        assert_eq!(p.region(), "eu-central-1");
        unsafe { std::env::remove_var("AWS_DEFAULT_REGION") };
    }

    #[test]
    #[serial]
    fn from_config_falls_back_to_env_region_when_unset() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_DEFAULT_REGION") };
        unsafe { std::env::set_var("AWS_REGION", "ap-southeast-1") };
        let p = BedrockProvider::from_config(None, None, None, None, None);
        assert_eq!(p.region(), "ap-southeast-1");
        unsafe { std::env::remove_var("AWS_REGION") };
    }

    #[test]
    #[serial]
    fn from_config_falls_back_to_default_region() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_DEFAULT_REGION") };
        unsafe { std::env::remove_var("AWS_REGION") };
        let p = BedrockProvider::from_config(None, None, None, None, None);
        assert_eq!(p.region(), DEFAULT_REGION);
    }

    #[test]
    #[serial]
    #[cfg(feature = "bedrock")]
    fn with_credentials_overrides_env_for_signing() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::remove_var("AWS_SESSION_TOKEN") };
        let headers = sigv4_sign(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/converse",
            b"{}",
            "us-east-1",
            Some("AKIAEXPLICIT"),
            Some("explicit-secret"),
            Some("explicit-token"),
        )
        .expect("signing should succeed with explicit credentials");
        assert!(
            headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        );
    }

    #[test]
    #[serial]
    #[cfg(feature = "bedrock")]
    fn sigv4_sign_fails_without_any_credentials() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        let result = sigv4_sign(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/converse",
            b"{}",
            "us-east-1",
            None,
            None,
            None,
        );
        assert!(result.is_err(), "signing without credentials should fail");
    }

    #[test]
    #[serial]
    #[cfg(feature = "bedrock")]
    fn signing_headers_propagates_signing_failure_instead_of_returning_empty() {
        // ~keep Regression test for #42. `sigv4_sign_fails_without_any_credentials` proves the
        // signer errors; this proves `signing_headers` PROPAGATES that error rather than
        // swallowing it into an empty header vec, which is what sent unsigned requests. The
        // sibling test covering the empty-vec return is compiled out under this feature, so
        // without this test the with-feature path has no coverage at all.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
        let provider = BedrockProvider::new("us-east-1");
        let result = provider.signing_headers(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/converse",
            b"{}",
        );
        assert!(
            result.is_err(),
            "signing_headers must surface the signing failure, not return empty headers"
        );
    }

    #[test]
    #[serial]
    #[cfg(feature = "bedrock")]
    fn transform_request_fails_hard_without_credentials_rather_than_sending_unsigned() {
        // ~keep Regression test for #42: before the fix, `signing_headers` swallowed a
        // signing failure via `.unwrap_or_default()` and the request went out with no
        // Authorization header. `transform_request` must now hard-error before any
        // network I/O so an unsigned Bedrock request can never be sent.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
        let p = BedrockProvider::new("us-east-1");
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        let result = p.transform_request(&mut body);
        assert!(
            result.is_err(),
            "transform_request must hard-error when Bedrock has no credentials, not silently \
             succeed and let signing_headers send an unsigned request"
        );
    }

    #[test]
    #[serial]
    fn bearer_token_reads_env_when_no_explicit_credentials() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = BedrockProvider::new("us-east-1");
        assert_eq!(p.bearer_token().as_deref(), Some("ABSKtest"));
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn bearer_token_ignores_empty_env_value() {
        // ~keep An exported-but-blank variable is how a shell says "unset". Treating it as
        // a credential would send `Authorization: Bearer ` and mask real SigV4 credentials.
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "") };
        let p = BedrockProvider::new("us-east-1");
        assert!(p.bearer_token().is_none());
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn bearer_token_yields_to_explicit_credentials() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = BedrockProvider::new("us-east-1").with_credentials(
            Some("AKIAEXPLICIT".to_owned()),
            Some("explicit-secret".to_owned()),
            None,
        );
        assert!(
            p.bearer_token().is_none(),
            "an ambient env token must not override credentials set explicitly in config"
        );
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn bearer_token_survives_a_half_empty_credential_pair() {
        // ~keep `LlmConfig` turns a config that sets only `access_key_id` into
        // `secret_access_key: Some("")`, which must not count as an explicit pair.
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = BedrockProvider::new("us-east-1").with_credentials(
            Some("AKIAEXPLICIT".to_owned()),
            Some(String::new()),
            None,
        );
        assert_eq!(p.bearer_token().as_deref(), Some("ABSKtest"));
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn signing_headers_uses_bearer_token_without_signing() {
        // ~keep Ungated on purpose: the bearer path must hold in both builds.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = BedrockProvider::new("us-east-1");
        let headers = p
            .signing_headers(
                "POST",
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/converse",
                b"{}",
            )
            .expect("a bearer token needs no signing and cannot fail");
        assert_eq!(
            headers.len(),
            1,
            "bearer auth sends one header, not a signed set: {headers:?}"
        );
        let (name, value) = &headers[0];
        assert!(name.eq_ignore_ascii_case("authorization"));
        assert_eq!(value, "Bearer ABSKtest");
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn validate_accepts_bearer_token_without_access_keys() {
        // ~keep `validate()` re-runs from `transform_request` per request, so an
        // access-key-only check rejects a bearer request before any I/O.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = BedrockProvider::new("us-east-1");
        assert!(p.validate().is_ok());
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    #[cfg(feature = "bedrock")]
    fn signing_headers_prefers_explicit_credentials_over_bearer_token() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
        unsafe { std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "ABSKtest") };
        let p = provider();
        let headers = p
            .signing_headers(
                "POST",
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/converse",
                b"{}",
            )
            .expect("signing should succeed with the explicit dummy credentials");
        let authorization = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone())
            .expect("an Authorization header should be present");
        assert!(
            authorization.starts_with("AWS4-HMAC-SHA256"),
            "explicit credentials must still sign with SigV4, got: {authorization}"
        );
        unsafe { std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK") };
    }

    #[test]
    #[serial]
    fn validate_accepts_explicit_credentials_without_env() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("AWS_ACCESS_KEY_ID") };
        let p = BedrockProvider::from_config(
            Some("us-east-1".to_owned()),
            None,
            Some("AKIAEXPLICIT".to_owned()),
            Some("explicit-secret".to_owned()),
            None,
        );
        assert!(p.validate().is_ok());
    }

    #[test]
    #[serial]
    fn with_cross_region_prefix_normalizes_trailing_dot() {
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = BedrockProvider::new("us-east-1").with_cross_region_prefix(Some("us".to_owned()));
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert!(
            url.contains("us.anthropic.claude-3-sonnet"),
            "cross-region prefix should be applied: {url}"
        );
    }

    #[test]
    #[serial]
    fn build_url_chat_completions() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = provider();
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
    }

    #[test]
    #[serial]
    fn build_url_embeddings() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = provider();
        let url = p.build_url("/embeddings", "amazon.titan-embed-text-v1");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        );
    }

    #[test]
    #[serial]
    fn build_url_other_path() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = provider();
        let url = p.build_url("/models", "any-model");
        assert_eq!(url, "https://bedrock-runtime.us-east-1.amazonaws.com/models");
    }

    #[test]
    #[serial]
    fn build_url_eusc_region() {
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
        let p = BedrockProvider::new("eusc-de-east-1");
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.eusc-de-east-1.amazonaws.eu/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
    }

    #[test]
    #[serial]
    fn build_url_china_region() {
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
        let p = BedrockProvider::new("cn-north-1");
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.cn-north-1.amazonaws.com.cn/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
    }

    #[test]
    #[serial]
    fn build_url_base_url_override() {
        unsafe { std::env::set_var("BEDROCK_BASE_URL", "https://custom.endpoint.example.com") };
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = BedrockProvider::new("us-east-1");
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://custom.endpoint.example.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
    }

    #[test]
    #[serial]
    fn build_url_base_url_trailing_slash_trimmed() {
        unsafe { std::env::set_var("BEDROCK_BASE_URL", "https://custom.endpoint.example.com/") };
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = BedrockProvider::new("us-east-1");
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://custom.endpoint.example.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
    }

    #[test]
    #[serial]
    fn build_url_base_url_override_ignores_cross_region() {
        unsafe { std::env::set_var("BEDROCK_BASE_URL", "https://custom.endpoint.example.com") };
        unsafe { std::env::set_var("BEDROCK_CROSS_REGION", "eu") };
        let p = BedrockProvider::new("us-east-1");
        let url = p.build_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://custom.endpoint.example.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
        unsafe { std::env::remove_var("BEDROCK_BASE_URL") };
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
    }

    #[test]
    fn dns_suffix_standard_regions() {
        assert_eq!(dns_suffix_for_region("us-east-1"), "amazonaws.com");
        assert_eq!(dns_suffix_for_region("eu-west-1"), "amazonaws.com");
        assert_eq!(dns_suffix_for_region("us-gov-west-1"), "amazonaws.com");
    }

    #[test]
    fn dns_suffix_eusc_regions() {
        assert_eq!(dns_suffix_for_region("eusc-de-east-1"), "amazonaws.eu");
    }

    #[test]
    fn dns_suffix_china_regions() {
        assert_eq!(dns_suffix_for_region("cn-north-1"), "amazonaws.com.cn");
        assert_eq!(dns_suffix_for_region("cn-northwest-1"), "amazonaws.com.cn");
    }

    #[test]
    fn percent_encode_model_colon() {
        let encoded = percent_encode_model("anthropic.claude-3-sonnet-20240229-v1:0");
        assert!(
            encoded.contains("%3A"),
            "colon should be percent-encoded with uppercase hex: {encoded}"
        );
        assert!(!encoded.contains("%3a"), "lowercase hex must not appear: {encoded}");
        assert!(!encoded.contains(':'), "raw colon should not remain: {encoded}");
    }

    #[test]
    fn percent_encode_model_safe_chars() {
        let encoded = percent_encode_model("amazon.titan-embed-text-v1");
        assert_eq!(encoded, "amazon.titan-embed-text-v1");
    }

    #[test]
    #[serial]
    fn transform_request_basic_chat() {
        let p = provider();
        let mut body = json!({
            "model": "anthropic.claude-3-sonnet",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["system"][0]["text"], "You are helpful.");

        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "Hello!");

        assert_eq!(body["inferenceConfig"]["maxTokens"], 100);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.7);
    }

    /// Revert line: delete
    /// `super::validate_sampling_param_range(body, "temperature", "Bedrock", 0.0, 1.0)?;`
    /// in `transform_request` to make this test fail.
    #[test]
    #[serial]
    fn transform_request_rejects_temperature_above_bedrock_maximum() {
        let p = provider();
        let mut body = json!({
            "model": "anthropic.claude-3-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.5
        });

        let err = p
            .transform_request(&mut body)
            .expect_err("temperature above Bedrock's 1.0 maximum should be rejected");

        assert_eq!(err.status_code(), 400);
        let message = err.to_string();
        assert!(
            message.contains("temperature=1.5"),
            "error message should name the offending value: {message}"
        );
        assert!(
            message.contains("Bedrock"),
            "error message should name the provider: {message}"
        );
    }

    /// Revert line: delete
    /// `super::validate_sampling_param_range(body, "top_p", "Bedrock", 0.0, 1.0)?;`
    /// in `transform_request` to make this test fail.
    #[test]
    #[serial]
    fn transform_request_rejects_top_p_above_bedrock_maximum() {
        let p = provider();
        let mut body = json!({
            "model": "anthropic.claude-3-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 1.2
        });

        let err = p
            .transform_request(&mut body)
            .expect_err("top_p above Bedrock's 1.0 maximum should be rejected");

        assert_eq!(err.status_code(), 400);
        assert!(
            err.to_string().contains("top_p=1.2"),
            "error message should name the offending value: {err}"
        );
    }

    /// `max_completion_tokens` maps to `inferenceConfig.maxTokens` when `max_tokens` is absent.
    /// Previously untested even though the mapping itself predates this fix.
    #[test]
    #[serial]
    fn transform_request_max_completion_tokens_maps_to_max_tokens() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 512
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["inferenceConfig"]["maxTokens"], 512);
    }

    /// `service_tier` maps to Bedrock Converse's `serviceTier: {type}` for every value except
    /// `"auto"`, which has no Bedrock counterpart and is left unset (provider default).
    #[test]
    #[serial]
    fn transform_request_service_tier_maps_to_bedrock_service_tier() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "flex"
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["serviceTier"]["type"], "flex");
    }

    #[test]
    #[serial]
    fn transform_request_service_tier_auto_is_omitted() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "auto"
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert!(body.get("serviceTier").is_none());
    }

    /// `metadata` maps to Bedrock Converse's `requestMetadata`, the same string-to-string
    /// tag-map shape used for CloudTrail/CloudWatch filtering.
    #[test]
    #[serial]
    fn transform_request_metadata_maps_to_request_metadata() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"run": "nightly", "team": "platform"}
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["requestMetadata"]["run"], "nightly");
        assert_eq!(body["requestMetadata"]["team"], "platform");
    }

    /// `logprobs`/`top_logprobs`/`audio`/`web_search_options` have no Bedrock equivalent; the
    /// wholesale body rebuild already dropped them, this pins that they never leak onto the wire.
    #[test]
    #[serial]
    fn transform_request_unmappable_openai_only_fields_dropped() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5,
            "store": true,
            "prediction": {"type": "content", "content": "draft"},
            "audio": {"voice": "alloy", "format": "wav"},
            "web_search_options": {"search_context_size": "medium"}
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        for key in &[
            "logprobs",
            "top_logprobs",
            "store",
            "prediction",
            "audio",
            "web_search_options",
        ] {
            assert!(body.get(key).is_none(), "`{key}` must not be forwarded to Bedrock");
        }
    }

    #[test]
    #[serial]
    fn transform_request_with_tool_calls() {
        let p = provider();
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "content": "Sunny, 22°C"
                }
            ]
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let messages = body["messages"].as_array().expect("messages should be an array");
        assert_eq!(messages.len(), 3);

        let assistant = &messages[1];
        assert_eq!(assistant["role"], "assistant");
        let tool_use = &assistant["content"][0]["toolUse"];
        assert_eq!(tool_use["toolUseId"], "call_abc");
        assert_eq!(tool_use["name"], "get_weather");
        assert_eq!(tool_use["input"]["city"], "Berlin");

        let tool_result_msg = &messages[2];
        assert_eq!(tool_result_msg["role"], "user");
        let tool_result = &tool_result_msg["content"][0]["toolResult"];
        assert_eq!(tool_result["toolUseId"], "call_abc");
        assert_eq!(tool_result["status"], "success");
    }

    /// Regression test: before the shared `convert_content_to_bedrock_blocks` helper,
    /// a `Parts` tool result silently dropped to an empty string because the "tool"
    /// arm only read `content.as_str()`. This asserts the image part now reaches
    /// Bedrock as a native `image` content block instead of being lost.
    #[test]
    #[serial]
    fn transform_request_tool_result_image_part_maps_to_bedrock_image_block() {
        let p = provider();
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "Take a screenshot"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_shot",
                        "type": "function",
                        "function": {"name": "take_screenshot", "arguments": "{}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_shot",
                    "content": [
                        {"type": "text", "text": "Here is the screenshot"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
                    ]
                }
            ]
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let messages = body["messages"].as_array().expect("messages should be an array");
        let tool_result_msg = &messages[2];
        let result_content = tool_result_msg["content"][0]["toolResult"]["content"]
            .as_array()
            .expect("toolResult content should be an array");
        assert_eq!(result_content.len(), 2);
        assert_eq!(result_content[0], json!({"text": "Here is the screenshot"}));
        assert_eq!(
            result_content[1],
            json!({"image": {"format": "png", "source": {"bytes": "abc123"}}})
        );
    }

    #[test]
    #[serial]
    fn transform_request_tools_schema() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            }]
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let tools = body["toolConfig"]["tools"]
            .as_array()
            .expect("tools should be an array");
        assert_eq!(tools.len(), 1);
        let spec = &tools[0]["toolSpec"];
        assert_eq!(spec["name"], "search");
        assert_eq!(spec["description"], "Search the web");
        assert_eq!(spec["inputSchema"]["json"]["type"], "object");
    }

    #[test]
    #[serial]
    fn transform_response_basic() {
        let p = provider();
        let mut body = json!({
            "requestId": "req-123",
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, world!"}]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5
            }
        });

        p.transform_response(&mut body)
            .expect("transform_response should not fail");

        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["id"], "req-123");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello, world!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 10);
        assert_eq!(body["usage"]["completion_tokens"], 5);
        assert_eq!(body["usage"]["total_tokens"], 15);
    }

    #[test]
    #[serial]
    fn transform_response_tool_calls() {
        let p = provider();
        let mut body = json!({
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {
                            "toolUseId": "call_xyz",
                            "name": "get_weather",
                            "input": {"city": "Berlin"}
                        }}
                    ]
                }
            },
            "usage": {"inputTokens": 20, "outputTokens": 10}
        });

        p.transform_response(&mut body)
            .expect("transform_response should not fail");

        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tool_calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls should be an array");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_xyz");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
        let args: serde_json::Value = serde_json::from_str(
            tool_calls[0]["function"]["arguments"]
                .as_str()
                .expect("arguments should be a string"),
        )
        .expect("arguments should be valid JSON");
        assert_eq!(args["city"], "Berlin");
    }

    #[test]
    #[serial]
    fn transform_response_finish_reason_mapping() {
        let p = provider();

        for (bedrock_reason, expected_oai_reason) in [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("content_filtered", "content_filter"),
            ("guardrail_intervened", "content_filter"),
            ("unknown_future_reason", "stop"),
        ] {
            let mut body = json!({
                "stopReason": bedrock_reason,
                "output": {"message": {"role": "assistant", "content": [{"text": ""}]}},
                "usage": {"inputTokens": 0, "outputTokens": 0}
            });
            p.transform_response(&mut body)
                .expect("transform_response should not fail");
            assert_eq!(
                body["choices"][0]["finish_reason"], expected_oai_reason,
                "bedrock stopReason '{bedrock_reason}' should map to '{expected_oai_reason}'"
            );
        }
    }

    #[test]
    #[serial]
    fn strip_model_prefix() {
        let p = provider();
        assert_eq!(p.strip_model_prefix("bedrock/anthropic.claude-3"), "anthropic.claude-3");
        assert_eq!(p.strip_model_prefix("anthropic.claude-3"), "anthropic.claude-3");
    }

    #[test]
    #[serial]
    fn matches_model() {
        let p = provider();
        assert!(p.matches_model("bedrock/anthropic.claude-3"));
        assert!(!p.matches_model("anthropic.claude-3"));
        assert!(!p.matches_model("gpt-4"));
    }

    #[test]
    #[serial]
    fn stream_format_is_eventstream() {
        let p = provider();
        assert_eq!(p.stream_format(), StreamFormat::AwsEventStream);
    }

    #[test]
    #[serial]
    fn build_stream_url_chat_completions() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let p = provider();
        let url = p.build_stream_url("/chat/completions", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse-stream"
        );
    }

    #[test]
    #[serial]
    fn build_stream_url_non_chat_falls_back() {
        let p = provider();
        let url = p.build_stream_url("/embeddings", "amazon.titan-embed-text-v1");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
        );
    }

    #[test]
    fn parse_stream_event_message_start() {
        let chunk = parse_bedrock_stream_event("messageStart", r#"{"role":"assistant"}"#)
            .expect("parse should not fail")
            .expect("should yield a chunk");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn parse_stream_event_text_delta() {
        let chunk = parse_bedrock_stream_event(
            "contentBlockDelta",
            r#"{"contentBlockIndex":0,"delta":{"text":"Hello world"}}"#,
        )
        .expect("parse should not fail")
        .expect("should yield a chunk");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello world"));
    }

    #[test]
    fn parse_stream_event_tool_use_start() {
        let chunk = parse_bedrock_stream_event(
            "contentBlockStart",
            r#"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_123","name":"get_weather"}}}"#,
        )
        .expect("parse should not fail")
        .expect("should yield a chunk");
        let tc = &chunk.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls should be present")[0];
        assert_eq!(tc.id.as_deref(), Some("call_123"));
        assert_eq!(
            tc.function
                .as_ref()
                .expect("function should be present")
                .name
                .as_deref(),
            Some("get_weather")
        );
    }

    #[test]
    fn parse_stream_event_tool_use_input_delta() {
        let chunk = parse_bedrock_stream_event(
            "contentBlockDelta",
            r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":\"Berlin\"}"}}}"#,
        )
        .expect("parse should not fail")
        .expect("should yield a chunk");
        let tc = &chunk.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls should be present")[0];
        assert_eq!(
            tc.function
                .as_ref()
                .expect("function should be present")
                .arguments
                .as_deref(),
            Some("{\"city\":\"Berlin\"}")
        );
    }

    #[test]
    fn parse_stream_event_message_stop() {
        let chunk = parse_bedrock_stream_event("messageStop", r#"{"stopReason":"end_turn"}"#)
            .expect("parse should not fail")
            .expect("should yield a chunk");
        assert_eq!(chunk.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn parse_stream_event_metadata_usage() {
        let chunk = parse_bedrock_stream_event("metadata", r#"{"usage":{"inputTokens":42,"outputTokens":10}}"#)
            .expect("parse should not fail")
            .expect("should yield a chunk");
        let usage = chunk.usage.expect("usage should be present");
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 10);
    }

    #[test]
    fn parse_stream_event_content_block_stop_returns_none() {
        let result = parse_bedrock_stream_event("contentBlockStop", r#"{"contentBlockIndex":0}"#)
            .expect("parse should not fail");
        assert!(result.is_none());
    }

    #[test]
    fn parse_stream_event_unknown_returns_none() {
        let result = parse_bedrock_stream_event("futureEventType", r#"{}"#).expect("parse should not fail");
        assert!(result.is_none());
    }

    #[test]
    #[serial]
    fn transform_request_reasoning_effort_low() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "Think step by step."}],
            "reasoning_effort": "low",
            "max_tokens": 1000
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let amf = &body["additionalModelRequestFields"];
        assert_eq!(amf["thinking"]["type"], "enabled");
        assert_eq!(amf["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    #[serial]
    fn transform_request_reasoning_effort_medium() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "Think."}],
            "reasoning_effort": "medium"
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["additionalModelRequestFields"]["thinking"]["budget_tokens"], 4096);
    }

    #[test]
    #[serial]
    fn transform_request_reasoning_effort_high() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "Think hard."}],
            "reasoning_effort": "high"
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["additionalModelRequestFields"]["thinking"]["budget_tokens"], 16384);
    }

    #[test]
    #[serial]
    fn transform_request_no_reasoning_effort_omits_amf() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert!(body.get("additionalModelRequestFields").is_none());
    }

    #[test]
    #[serial]
    fn transform_request_document_content_part() {
        let p = provider();
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document."},
                    {
                        "type": "document",
                        "document": {
                            "data": "JVBERi0xLjQ=",
                            "media_type": "application/pdf"
                        }
                    }
                ]
            }]
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let content = body["messages"][0]["content"]
            .as_array()
            .expect("content should be an array");
        assert_eq!(content.len(), 2);

        assert_eq!(content[0]["text"], "Summarize this document.");

        let doc = &content[1]["document"];
        assert_eq!(doc["name"], "doc");
        assert_eq!(doc["format"], "pdf");
        assert_eq!(doc["source"]["bytes"], "JVBERi0xLjQ=");
    }

    #[test]
    #[serial]
    fn transform_request_document_csv_format() {
        let p = provider();
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "document": {
                            "data": "Y29sMSxjb2wy",
                            "media_type": "text/csv"
                        }
                    }
                ]
            }]
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let doc = &body["messages"][0]["content"][0]["document"];
        assert_eq!(doc["format"], "csv");
    }

    #[test]
    #[serial]
    fn transform_request_guardrails() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {
                "guardrailConfig": {
                    "guardrailIdentifier": "my-guardrail-id",
                    "guardrailVersion": "DRAFT",
                    "trace": "enabled"
                }
            }
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let gc = &body["guardrailConfig"];
        assert_eq!(gc["guardrailIdentifier"], "my-guardrail-id");
        assert_eq!(gc["guardrailVersion"], "DRAFT");
        assert_eq!(gc["trace"], "enabled");
    }

    #[test]
    #[serial]
    fn transform_request_no_guardrails_omits_config() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert!(body.get("guardrailConfig").is_none());
    }

    #[test]
    #[serial]
    fn transform_request_json_object_response_format() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "Give me JSON."}],
            "response_format": {"type": "json_object"}
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let system = body["system"].as_array().expect("system should be an array");
        let has_json_instruction = system
            .iter()
            .any(|s| s["text"].as_str().unwrap_or("").contains("valid JSON"));
        assert!(has_json_instruction, "should inject JSON instruction in system");
    }

    #[test]
    #[serial]
    fn transform_request_json_schema_response_format() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "Give me structured data."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "my_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        }
                    }
                }
            }
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let system = body["system"].as_array().expect("system should be an array");
        let json_instruction = system
            .iter()
            .find(|s| s["text"].as_str().unwrap_or("").contains("valid JSON"))
            .expect("JSON instruction block should be present");
        let text = json_instruction["text"].as_str().expect("text should be a string");
        assert!(
            text.contains("conforms to this schema"),
            "should include schema reference: {text}"
        );
        assert!(text.contains("\"name\""), "should include the schema content: {text}");
    }

    #[test]
    #[serial]
    fn transform_request_text_response_format_no_injection() {
        let p = provider();
        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "text"}
        });
        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert!(body.get("system").is_none());
    }

    #[test]
    #[serial]
    fn apply_cross_region_prefix_when_set() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::set_var("BEDROCK_CROSS_REGION", "us") };
        let result = super::apply_cross_region_prefix("anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(result, "us.anthropic.claude-3-sonnet-20240229-v1:0");
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
    }

    #[test]
    #[serial]
    fn apply_cross_region_prefix_no_double_prefix() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::set_var("BEDROCK_CROSS_REGION", "eu") };
        let result = super::apply_cross_region_prefix("eu.anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            result, "eu.anthropic.claude-3-sonnet-20240229-v1:0",
            "should not double-prefix"
        );
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
    }

    #[test]
    #[serial]
    fn apply_cross_region_prefix_unset() {
        // ~keep SAFETY: env vars are process-global; `#[serial]` ensures no parallel mutation.
        unsafe { std::env::remove_var("BEDROCK_CROSS_REGION") };
        let result = super::apply_cross_region_prefix("anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(result, "anthropic.claude-3-sonnet-20240229-v1:0");
    }

    #[test]
    fn reasoning_effort_budget_tokens() {
        assert_eq!(super::reasoning_effort_to_budget_tokens("low"), 1024);
        assert_eq!(super::reasoning_effort_to_budget_tokens("medium"), 4096);
        assert_eq!(super::reasoning_effort_to_budget_tokens("high"), 16384);
        assert_eq!(super::reasoning_effort_to_budget_tokens("unknown"), 4096);
    }

    #[test]
    fn format_from_media_type_extraction() {
        assert_eq!(super::format_from_media_type("application/pdf"), "pdf");
        assert_eq!(super::format_from_media_type("text/csv"), "csv");
        assert_eq!(
            super::format_from_media_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            "vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
        assert_eq!(super::format_from_media_type("pdf"), "pdf");
    }

    #[test]
    #[serial]
    fn transform_request_titan_embedding_input_is_not_mangled_into_empty_messages() {
        // ~keep Regression test: before the embed/chat branch, `transform_request`
        // ~keep unconditionally rebuilt the body as `{"messages": []}` for any request,
        // ~keep silently discarding an embedding request's `input` entirely.
        let p = provider();
        let mut body = json!({
            "model": "amazon.titan-embed-text-v1",
            "input": "hello world"
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["inputText"], "hello world");
        assert!(
            body.get("messages").is_none(),
            "embedding body must not gain a messages field"
        );
    }

    /// Titan's synchronous invoke takes one input, but OpenAI's contract is that
    /// `data[]` parallels `input[]`. Quietly embedding only the first text would
    /// hand back one vector for an N-element batch — misaligned embeddings that
    /// surface much later as bad retrieval, not as an error. Reject instead.
    #[test]
    #[serial]
    fn transform_request_titan_embedding_batch_input_is_rejected_not_truncated() {
        let p = provider();
        let mut body = json!({
            "model": "amazon.titan-embed-text-v2:0",
            "input": ["first", "second"]
        });

        let err = p
            .transform_request(&mut body)
            .expect_err("a batched Titan embedding request must be rejected, not silently truncated");

        assert_eq!(err.status_code(), 400);
        assert!(
            err.to_string().contains("single input per call"),
            "the error must explain the single-input limit, got: {err}"
        );
    }

    #[test]
    #[serial]
    fn transform_request_titan_embedding_single_element_array_is_accepted() {
        let p = provider();
        let mut body = json!({
            "model": "amazon.titan-embed-text-v2:0",
            "input": ["only"]
        });

        p.transform_request(&mut body)
            .expect("a single-element batch is within Titan's one-input limit");

        assert_eq!(body["inputText"], "only");
    }

    #[test]
    #[serial]
    fn transform_request_embedding_multimodal_input_is_rejected() {
        let p = provider();
        let mut body = json!({
            "model": "amazon.titan-embed-text-v2:0",
            "input": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
        });

        let error = p
            .transform_request(&mut body)
            .expect_err("multimodal input should be rejected");
        assert_eq!(error.status_code(), 400);
        assert!(error.to_string().contains("text input only"));
    }

    #[test]
    #[serial]
    fn transform_request_cohere_embedding_maps_to_texts_and_input_type() {
        let p = provider();
        let mut body = json!({
            "model": "cohere.embed-english-v3",
            "input": ["one", "two"]
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        let texts = body["texts"].as_array().expect("texts should be an array");
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "one");
        assert_eq!(texts[1], "two");
        assert_eq!(body["input_type"], "search_document");
    }

    #[test]
    #[serial]
    fn transform_request_unknown_embedding_model_is_rejected() {
        let p = provider();
        let mut body = json!({
            "model": "unknown-vendor.embed-v1",
            "input": "hello"
        });

        let err = p.transform_request(&mut body).unwrap_err();
        assert!(
            err.to_string().contains("unsupported Bedrock embedding model"),
            "got: {err}"
        );
    }

    /// Regression guard: the embed/chat dispatch must only trigger for
    /// input-without-messages bodies. A normal chat body (messages present) must
    /// still go through the unchanged Converse transform.
    #[test]
    #[serial]
    fn transform_request_chat_path_unaffected_by_embedding_branch() {
        let p = provider();
        let mut body = json!({
            "model": "anthropic.claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello!"}]
        });

        p.transform_request(&mut body)
            .expect("transform_request should not fail");

        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "Hello!");
        assert!(body.get("inputText").is_none());
    }

    #[test]
    #[serial]
    fn transform_response_titan_embedding_response_normalized() {
        let p = provider();
        let mut body = json!({
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 4
        });

        p.transform_response(&mut body)
            .expect("transform_response should not fail");

        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["embedding"], json!([0.1, 0.2, 0.3]));
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(body["usage"]["prompt_tokens"], 4);
    }

    #[test]
    #[serial]
    fn transform_response_cohere_embedding_response_normalized() {
        let p = provider();
        let mut body = json!({
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "id": "abc",
            "response_type": "embeddings_floats",
            "texts": ["one", "two"]
        });

        p.transform_response(&mut body)
            .expect("transform_response should not fail");

        let data = body["data"].as_array().expect("data should be an array");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["embedding"], json!([0.1, 0.2]));
        assert_eq!(data[1]["embedding"], json!([0.3, 0.4]));
        assert_eq!(data[1]["index"], 1);
    }

    /// Regression guard: a normal chat response (no `embedding`/`embeddings` key)
    /// must still go through the unchanged Converse response transform.
    #[test]
    #[serial]
    fn transform_response_chat_path_unaffected_by_embedding_branch() {
        let p = provider();
        let mut body = json!({
            "requestId": "req-456",
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hi"}]
                }
            },
            "usage": {"inputTokens": 1, "outputTokens": 1}
        });

        p.transform_response(&mut body)
            .expect("transform_response should not fail");

        assert_eq!(body["object"], "chat.completion");
        assert!(body.get("data").is_none());
    }
}
