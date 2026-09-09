//! Google Vertex AI ADC (Application Default Credentials) provider.
//!
//! Obtains a short-lived OAuth2 access token without a service-account JSON key
//! file, making it suitable for GKE Workload Identity, Cloud Run, and Compute
//! Engine deployments.
//!
//! # Token acquisition order
//!
//! 1. **GKE / Compute Engine metadata server** — sends a `GET` to
//!    `http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token`
//!    with `Metadata-Flavor: Google`.  This is the fastest path for pods running
//!    under Workload Identity.
//!
//! 2. **`gcp_auth` ADC discovery** — when the outbound policy is
//!    [`crate::provider::OutboundPolicy::Off`], falls back to
//!    `gcp_auth::provider()`. That transport may use a service-account
//!    `token_uri` that Liter cannot inspect, so the fallback is disabled while
//!    an outbound policy is active. The fixed metadata endpoint remains an
//!    explicit, narrowly scoped exception for GCP runtimes.
//!
//! Tokens are cached using the same `RwLock<Option<CachedToken>>` + 5-minute
//! pre-expiry buffer as [`super::vertex_oauth`].
//!
//! # Environment variables
//!
//! | Variable | Description |
//! |----------|-------------|
//! | `VERTEX_AI_SCOPE` | OAuth scope (defaults to `https://www.googleapis.com/auth/cloud-platform`) |
//! | `GOOGLE_APPLICATION_CREDENTIALS` | Path to a SA JSON key (used by `gcp_auth` only when the outbound policy is `Off`) |
//!
//! # Usage
//!
//! ```rust,ignore
//! use liter_llm::auth::vertex_adc::VertexAdcCredentialProvider;
//! use liter_llm::client::ClientConfigBuilder;
//! use std::sync::Arc;
//!
//! let provider = VertexAdcCredentialProvider::new();
//! let config = ClientConfigBuilder::new("")
//!     .credential_provider(Arc::new(provider))
//!     .build();
//! ```

use std::time::Instant;

use secrecy::SecretString;
use tokio::sync::RwLock;

use super::{Credential, CredentialProvider};
use crate::client::BoxFuture;
use crate::error::LiterLlmError;

/// Default OAuth2 scope for Google Cloud Platform / Vertex AI.
const DEFAULT_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// GKE / Compute Engine metadata server token endpoint.
const METADATA_TOKEN_URL: &str = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token";

/// Required header that the metadata server validates before returning a token.
const METADATA_FLAVOR_HEADER: &str = "Metadata-Flavor";
const METADATA_FLAVOR_VALUE: &str = "Google";

/// Metadata service requests are local and should either complete quickly or fall back.
const METADATA_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Minimum remaining lifetime before a cached token is considered expired.
const EXPIRY_BUFFER_SECS: u64 = 300;

/// Assumed GCP access-token lifetime for tokens obtained via the `gcp_auth` fallback.
///
/// GCP access tokens are issued with a 3600-second lifetime.  Using this as
/// the cache ceiling means our EXPIRY_BUFFER_SECS (300) guard triggers a
/// refresh after ~3300 s — safely before the actual expiry.
const GCP_TOKEN_LIFETIME_SECS: u64 = 3600;

/// Cached token with acquisition time and lifetime.
struct CachedToken {
    token: SecretString,
    acquired_at: Instant,
    expires_in_secs: u64,
}

impl CachedToken {
    /// Returns `true` if the token is still valid with the safety buffer applied.
    fn is_valid(&self) -> bool {
        let elapsed = self.acquired_at.elapsed().as_secs();
        elapsed + EXPIRY_BUFFER_SECS < self.expires_in_secs
    }
}

/// Google Vertex AI ADC credential provider.
///
/// Prefers the Compute Engine / GKE metadata server. When the outbound policy
/// is [`crate::provider::OutboundPolicy::Off`], it falls back to `gcp_auth`'s
/// full ADC discovery chain. Active policies disable that opaque fallback.
pub struct VertexAdcCredentialProvider {
    scope: String,
    /// Overridable metadata server base URL (set to the real 169.254.169.254
    /// address in production; injected during tests to point at a mock server).
    metadata_token_url: String,
    trusted_metadata_url: bool,
    /// When `false`, skip the `gcp_auth` fallback path.  Used in tests to
    /// exercise the error path without triggering real ADC discovery.
    use_gcp_auth_fallback: bool,
    cached: RwLock<Option<CachedToken>>,
    http_client: Option<reqwest::Client>,
}

impl std::fmt::Debug for VertexAdcCredentialProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexAdcCredentialProvider")
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl VertexAdcCredentialProvider {
    /// Create a provider with the default scope.
    ///
    /// The scope defaults to `https://www.googleapis.com/auth/cloud-platform`
    /// but can be overridden via the `VERTEX_AI_SCOPE` environment variable or
    /// by calling [`VertexAdcCredentialProvider::with_scope`].
    #[must_use]
    pub fn new() -> Self {
        let scope = std::env::var("VERTEX_AI_SCOPE").unwrap_or_else(|_| DEFAULT_SCOPE.to_owned());
        Self {
            scope,
            metadata_token_url: METADATA_TOKEN_URL.to_owned(),
            trusted_metadata_url: true,
            use_gcp_auth_fallback: true,
            cached: RwLock::new(None),
            // ~keep Vertex ADC must reach the link-local GCP metadata service. It is the explicit
            // ~keep trusted exception to the provider outbound policy, not a caller-controlled URL.
            http_client: None,
        }
    }

    /// Override the OAuth2 scope (default: `https://www.googleapis.com/auth/cloud-platform`).
    #[must_use]
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = scope.into();
        self
    }

    /// Override the HTTP client used for metadata server requests.
    ///
    /// This is a trusted transport override. The default client deliberately
    /// disables redirects for the fixed metadata-service exception.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Override the metadata server token URL.
    ///
    /// This is intended for testing — in production the URL is always
    /// `http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token`.
    #[must_use]
    pub fn with_metadata_url(metadata_base_url: impl Into<String>) -> Self {
        let scope = std::env::var("VERTEX_AI_SCOPE").unwrap_or_else(|_| DEFAULT_SCOPE.to_owned());
        let base = metadata_base_url.into();
        let metadata_token_url = format!(
            "{}/computeMetadata/v1/instance/service-accounts/default/token",
            base.trim_end_matches('/')
        );
        Self {
            scope,
            metadata_token_url,
            trusted_metadata_url: false,
            use_gcp_auth_fallback: true,
            cached: RwLock::new(None),
            // ~keep Build the normal guarded client lazily so this infallible constructor does
            // ~keep not hide TLS/client configuration errors.
            http_client: None,
        }
    }

    /// Disable the `gcp_auth` fallback path.
    ///
    /// When set, the provider returns an error instead of falling back to ADC
    /// discovery if the metadata server is unreachable or returns a non-success
    /// status.  Useful in tests that exercise the error path in isolation.
    #[must_use]
    pub fn without_gcp_auth_fallback(mut self) -> Self {
        self.use_gcp_auth_fallback = false;
        self
    }

    /// Attempt to fetch a token from the GKE / Compute Engine metadata server.
    ///
    /// Returns `None` when the metadata server is not reachable (i.e. we are
    /// not running on GCP) so the caller can fall back to ADC discovery.
    async fn fetch_from_metadata_server(&self) -> Result<Option<CachedToken>, LiterLlmError> {
        if !self.trusted_metadata_url {
            crate::provider::validate_outbound_url(&self.metadata_token_url).await?;
        }
        let http_client = match &self.http_client {
            Some(client) => client.clone(),
            None if self.trusted_metadata_url => metadata_http_client()?,
            None => crate::provider::authenticated_outbound_client()?,
        };
        let response = http_client
            .get(&self.metadata_token_url)
            .header(METADATA_FLAVOR_HEADER, METADATA_FLAVOR_VALUE)
            .send()
            .await
            .map_err(|error| {
                crate::provider::outbound_forbidden_from_reqwest(&error).unwrap_or_else(|| {
                    LiterLlmError::Authentication {
                        message: format!("Vertex ADC metadata request failed: {error}"),
                        status: 401,
                    }
                })
            });
        let response = match response {
            Ok(response) => response,
            Err(LiterLlmError::OutboundForbidden { url, reason }) => {
                return Err(LiterLlmError::OutboundForbidden { url, reason });
            }
            Err(_) => return Ok(None),
        };

        if !response.status().is_success() {
            tracing::warn!(
                status = response.status().as_u16(),
                "metadata server returned non-success status; will try gcp_auth ADC fallback"
            );
            return Ok(None);
        }

        let Ok(body) = response.text().await else {
            return Ok(None);
        };
        let Ok(parsed) = serde_json::from_str::<MetadataTokenResponse>(&body) else {
            return Ok(None);
        };

        tracing::info!("obtained access token from metadata server");

        Ok(Some(CachedToken {
            token: SecretString::from(parsed.access_token),
            acquired_at: Instant::now(),
            expires_in_secs: parsed.expires_in,
        }))
    }

    /// Fetch a token via the `gcp_auth` ADC discovery chain.
    ///
    /// This covers: `GOOGLE_APPLICATION_CREDENTIALS` file,
    /// `~/.config/gcloud/application_default_credentials.json`, metadata server
    /// (second attempt), and `gcloud auth print-access-token`.
    async fn fetch_from_gcp_auth(&self) -> Result<CachedToken, LiterLlmError> {
        if !matches!(crate::provider::current_policy(), crate::provider::OutboundPolicy::Off) {
            return Err(LiterLlmError::OutboundForbidden {
                url: "gcp-auth://adc-fallback".into(),
                reason: "gcp_auth fallback transport cannot enforce the active outbound policy".into(),
            });
        }
        let provider = gcp_auth::provider().await.map_err(|e| LiterLlmError::Authentication {
            message: format!("gcp_auth ADC discovery failed: {e}"),
            status: 401,
        })?;

        let scopes = &[self.scope.as_str()];
        let token = provider
            .token(scopes)
            .await
            .map_err(|e| LiterLlmError::Authentication {
                message: format!("gcp_auth token acquisition failed: {e}"),
                status: 401,
            })?;

        tracing::info!("obtained access token via gcp_auth ADC discovery");

        Ok(CachedToken {
            token: SecretString::from(token.as_str().to_owned()),
            acquired_at: Instant::now(),
            expires_in_secs: GCP_TOKEN_LIFETIME_SECS,
        })
    }

    /// Fetch a fresh token, preferring the metadata server over gcp_auth ADC.
    async fn fetch_token(&self) -> Result<CachedToken, LiterLlmError> {
        if let Some(cached) = self.fetch_from_metadata_server().await? {
            return Ok(cached);
        }

        if self.use_gcp_auth_fallback {
            tracing::debug!("metadata server not available; trying gcp_auth ADC discovery");
            self.fetch_from_gcp_auth().await
        } else {
            Err(LiterLlmError::Authentication {
                message: "Vertex AI ADC: metadata server unavailable and gcp_auth fallback is disabled".into(),
                status: 401,
            })
        }
    }
}

fn metadata_http_client() -> Result<reqwest::Client, LiterLlmError> {
    crate::ensure_crypto_provider();
    reqwest::Client::builder()
        // ~keep The metadata endpoint is an explicit link-local capability and must never
        // ~keep be routed through caller or environment-configured proxies.
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(METADATA_REQUEST_TIMEOUT)
        .build()
        .map_err(LiterLlmError::from)
}

impl Default for VertexAdcCredentialProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialProvider for VertexAdcCredentialProvider {
    fn resolve(&self) -> BoxFuture<'_, crate::error::Result<Credential>> {
        Box::pin(async move {
            {
                let guard = self.cached.read().await;
                if let Some(ref cached) = *guard
                    && cached.is_valid()
                {
                    tracing::debug!("returning cached Vertex AI ADC token");
                    return Ok(Credential::BearerToken(cached.token.clone()));
                }
            }

            let mut guard = self.cached.write().await;

            // ~keep Double-check after write-lock acquisition to avoid duplicate token fetches.
            if let Some(ref cached) = *guard
                && cached.is_valid()
            {
                tracing::debug!("returning cached Vertex AI ADC token (post-lock check)");
                return Ok(Credential::BearerToken(cached.token.clone()));
            }

            let fresh = self.fetch_token().await?;
            let token = fresh.token.clone();
            *guard = Some(fresh);

            Ok(Credential::BearerToken(token))
        })
    }
}

/// Deserialised response from the GCE metadata server token endpoint.
#[derive(serde::Deserialize)]
struct MetadataTokenResponse {
    access_token: String,
    expires_in: u64,
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use secrecy::SecretString;

    use super::*;

    #[test]
    fn cached_token_is_valid_with_plenty_of_time() {
        let cached = CachedToken {
            token: SecretString::from("tok".to_owned()),
            acquired_at: Instant::now(),
            expires_in_secs: 3600,
        };
        assert!(cached.is_valid());
    }

    #[test]
    fn cached_token_is_expired_at_zero_lifetime() {
        let cached = CachedToken {
            token: SecretString::from("tok".to_owned()),
            acquired_at: Instant::now(),
            expires_in_secs: 0,
        };
        assert!(!cached.is_valid());
    }

    #[test]
    fn cached_token_is_expired_within_buffer() {
        let cached = CachedToken {
            token: SecretString::from("tok".to_owned()),
            acquired_at: Instant::now(),
            expires_in_secs: 200,
        };
        assert!(!cached.is_valid());
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn default_scope_is_cloud_platform() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider = VertexAdcCredentialProvider::new();
        assert_eq!(provider.scope, DEFAULT_SCOPE);
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn scope_override_via_env_var() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", Some("https://custom.scope/"));
        let provider = VertexAdcCredentialProvider::new();
        assert_eq!(provider.scope, "https://custom.scope/");
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn with_scope_overrides_scope() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider = VertexAdcCredentialProvider::new().with_scope("https://my.scope/");
        assert_eq!(provider.scope, "https://my.scope/");
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn default_impl_equals_new() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider: VertexAdcCredentialProvider = Default::default();
        assert_eq!(provider.scope, DEFAULT_SCOPE);
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn with_metadata_url_appends_token_path() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider = VertexAdcCredentialProvider::with_metadata_url("http://127.0.0.1:12345");
        assert_eq!(
            provider.metadata_token_url,
            "http://127.0.0.1:12345/computeMetadata/v1/instance/service-accounts/default/token"
        );
        assert!(
            !provider.trusted_metadata_url,
            "caller-controlled overrides are never trusted"
        );
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn default_metadata_exception_is_exact_and_capability_scoped() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider = VertexAdcCredentialProvider::new();
        assert!(provider.trusted_metadata_url);
        assert_eq!(provider.metadata_token_url, METADATA_TOKEN_URL);
    }

    #[tokio::test]
    #[serial_test::serial(vertex_adc_env)]
    async fn fixed_metadata_client_ignores_environment_proxies() {
        const CHILD_MARKER: &str = "LITER_VERTEX_METADATA_PROXY_CHILD";
        if std::env::var_os(CHILD_MARKER).is_none() {
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tokio::process::Command::new(std::env::current_exe().expect("test executable"))
                    .args([
                        "auth::vertex_adc::tests::fixed_metadata_client_ignores_environment_proxies",
                        "--exact",
                        "--nocapture",
                    ])
                    .env(CHILD_MARKER, "1")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .expect("isolated metadata proxy test deadline")
            .expect("launch isolated metadata proxy test");
            assert!(output.status.success(), "isolated metadata proxy test: {output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed; 0 ignored"));
            return;
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind proxy listener");
        listener.set_nonblocking(true).expect("set proxy listener nonblocking");
        let proxy_url = format!("http://{}", listener.local_addr().expect("proxy listener address"));
        let _http_proxy = EnvGuard::new("HTTP_PROXY", Some(&proxy_url));
        let _http_proxy_lower = EnvGuard::new("http_proxy", Some(&proxy_url));
        let _all_proxy = EnvGuard::new("ALL_PROXY", Some(&proxy_url));
        let _all_proxy_lower = EnvGuard::new("all_proxy", Some(&proxy_url));
        let _no_proxy = EnvGuard::new("NO_PROXY", None);
        let _no_proxy_lower = EnvGuard::new("no_proxy", None);

        let client = metadata_http_client().expect("build metadata client");
        let _ = client.get(METADATA_TOKEN_URL).send().await;

        let accepted = listener.accept();
        assert!(
            matches!(accepted, Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "the fixed link-local metadata capability must bypass environment proxies"
        );
    }

    #[test]
    #[serial_test::serial(vertex_adc_env)]
    fn with_metadata_url_trailing_slash_is_normalised() {
        let _guard = EnvGuard::new("VERTEX_AI_SCOPE", None);
        let provider = VertexAdcCredentialProvider::with_metadata_url("http://127.0.0.1:12345/");
        assert_eq!(
            provider.metadata_token_url,
            "http://127.0.0.1:12345/computeMetadata/v1/instance/service-accounts/default/token"
        );
    }

    #[tokio::test]
    #[serial_test::serial(outbound_policy)]
    async fn caller_controlled_metadata_url_does_not_receive_trusted_exception() {
        use crate::provider::{OutboundPolicy, set_outbound_policy};

        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let provider = VertexAdcCredentialProvider::with_metadata_url("http://127.0.0.1:9").without_gcp_auth_fallback();
        let result = provider.resolve().await;
        set_outbound_policy(OutboundPolicy::Off);

        assert!(
            matches!(result, Err(LiterLlmError::OutboundForbidden { .. })),
            "only the fixed metadata endpoint may bypass DenyPrivate: {result:?}"
        );
    }

    #[tokio::test]
    #[serial_test::serial(outbound_policy)]
    async fn gcp_auth_fallback_is_disabled_under_active_policy() {
        use crate::provider::{OutboundPolicy, set_outbound_policy};

        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let provider = VertexAdcCredentialProvider::new();
        let result = provider.fetch_from_gcp_auth().await;
        set_outbound_policy(OutboundPolicy::Off);

        assert!(
            matches!(result, Err(LiterLlmError::OutboundForbidden { .. })),
            "uncontrolled gcp_auth token_uri transport must not bypass the policy"
        );
    }

    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvGuard {
        fn new(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var(key).ok();
            // ~keep SAFETY: EnvGuard tests are single-threaded and restore env vars on drop.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // ~keep SAFETY: same single-threaded EnvGuard invariant during drop.
            unsafe {
                match &self.original {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn live_metadata_server_or_adc_returns_bearer_token() {
        let provider = VertexAdcCredentialProvider::new();
        let credential = provider.resolve().await.expect("token acquisition failed");
        assert!(matches!(credential, Credential::BearerToken(_)));
    }
}
