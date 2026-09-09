//! Provider routing — weighted selection, dynamic discovery, and concurrency limits.
//!
//! # Overview
//!
//! This module provides three independent building blocks that compose to form
//! a full routing stack:
//!
//! - [`Weight`] — a saturating `u32` wrapper for canary and weighted-random
//!   weights; avoids NaN/Inf foot-guns from raw `f64` weights.
//! - [`UpstreamDiscover`] / [`StaticDiscover`] — a trait that abstracts over
//!   dynamic service discovery (etcd, file-watch, HTTP poll) and a built-in
//!   static implementation that seeds from a fixed list.
//! - [`DynamicRouter`] — a generic router over any `UpstreamDiscover` that
//!   pre-warms discovered services in a [`tower::ready_cache::ReadyCache`]
//!   so request-time setup cost is zero.
//! - [`Router`] — the original statically-configured router, retained for
//!   backward compatibility and as the default when dynamic discovery is not
//!   required.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use dashmap::DashMap;
use futures_core::Stream;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt};
use tower::Service;
use tower::discover::{Change, Discover};
use tower::limit::ConcurrencyLimit;
use tower::ready_cache::ReadyCache;

use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{LiterLlmError, Result};

/// An integer traffic weight in the range [0, [`u32::MAX`]].
///
/// Uses saturating conversion from `f64` so that NaN and negative values
/// clamp to 0 and `+Inf` clamps to `u32::MAX`.  This prevents canary
/// configurations with malformed YAML weights from causing panics or
/// undefined distribution behaviour.
///
/// # Example
///
/// ```
/// use liter_llm::tower::router::Weight;
///
/// assert_eq!(Weight::from_f64(1.0).as_u32(), 1);
/// assert_eq!(Weight::from_f64(f64::NAN).as_u32(), 0);
/// assert_eq!(Weight::from_f64(f64::INFINITY).as_u32(), u32::MAX);
/// assert_eq!(Weight::from_f64(-5.0).as_u32(), 0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Weight(u32);

impl Weight {
    /// Zero weight — the service receives no traffic.
    pub const ZERO: Weight = Weight(0);
    /// Default unit weight (corresponds to `1.0_f64`).
    pub const ONE: Weight = Weight(1);
    /// Maximum representable weight.
    pub const MAX: Weight = Weight(u32::MAX);

    /// Convert from an `f64` with saturating semantics.
    ///
    /// - NaN → 0
    /// - negative → 0
    /// - `+Inf` → [`u32::MAX`]
    /// - otherwise: `round(f)` clamped to `[0, u32::MAX]`
    #[must_use]
    pub fn from_f64(f: f64) -> Self {
        if f.is_nan() || f < 0.0 {
            Self::ZERO
        } else if f.is_infinite() {
            Self::MAX
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let w = f.round().min(f64::from(u32::MAX)) as u32;
            Self(w)
        }
    }

    /// Return the raw `u32` value.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl Default for Weight {
    fn default() -> Self {
        Self::ONE
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Routing strategy for selecting among multiple deployments.
#[derive(Clone)]
#[cfg_attr(alef, alef(skip))]
pub enum RoutingStrategy {
    /// Round-robin across all deployments in order.
    RoundRobin,
    /// Try deployments in order; advance to the next on a transient error.
    /// Propagates immediately on non-transient errors.
    Fallback,
    /// Route to the deployment with the lowest observed latency (exponential
    /// moving average).
    LatencyBased,
    /// Route to the cheapest deployment for the requested model using the
    /// embedded pricing registry.
    CostBased,
    /// Weighted random distribution across deployments.  Weights are
    /// normalised at request time; higher values receive proportionally
    /// more traffic.  Weights of 0 exclude the deployment entirely.
    WeightedRandom {
        /// One weight per deployment (must have the same length as the
        /// deployments vec).
        weights: Vec<Weight>,
    },
    /// Intent-based semantic routing.
    ///
    /// Calls the provided [`RouteClassifier`] cascade to determine which
    /// model should handle the request.  The classifier inspects the prompt
    /// text (and optionally system prompt / metadata) and returns a model ID.
    ///
    /// If the classifier returns `None` (all tiers defer) the router falls
    /// back to [`RoutingStrategy::RoundRobin`] across the available
    /// deployments so requests are never dropped.
    ///
    /// [`RouteClassifier`]: super::route_classify::RouteClassifier
    Semantic(Arc<dyn super::route_classify::RouteClassifier>),
}

impl std::fmt::Debug for RoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::Fallback => write!(f, "Fallback"),
            Self::LatencyBased => write!(f, "LatencyBased"),
            Self::CostBased => write!(f, "CostBased"),
            Self::WeightedRandom { weights } => f.debug_struct("WeightedRandom").field("weights", weights).finish(),
            Self::Semantic(_) => write!(f, "Semantic(…)"),
        }
    }
}

/// Tracks per-deployment latency using an exponential moving average.
#[derive(Debug)]
struct DeploymentMetrics {
    /// Exponential moving average of latency in seconds.
    latency_ema: f64,
    /// Number of requests seen (used to seed the EMA).
    request_count: u64,
}

impl Default for DeploymentMetrics {
    fn default() -> Self {
        Self {
            latency_ema: 0.0,
            request_count: 0,
        }
    }
}

impl DeploymentMetrics {
    /// Update the EMA with a new latency sample (in seconds).
    fn record_latency(&mut self, latency_secs: f64) {
        const ALPHA: f64 = 0.3;

        if self.request_count == 0 {
            self.latency_ema = latency_secs;
        } else {
            self.latency_ema = ALPHA * latency_secs + (1.0 - ALPHA) * self.latency_ema;
        }
        self.request_count += 1;
    }
}

/// Shared state tracking per-deployment metrics, keyed by deployment index.
#[cfg_attr(alef, alef(skip))]
pub struct RouterState {
    metrics: Arc<DashMap<usize, DeploymentMetrics>>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            metrics: Arc::new(DashMap::new()),
        }
    }
}

impl Clone for RouterState {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
        }
    }
}

/// A router that distributes [`LlmRequest`]s across multiple service
/// instances according to a [`RoutingStrategy`].
///
/// The inner deployments must be `Clone` so the router can hand out
/// independent service handles per call.  Use [`LlmService`] as the
/// deployment type when wrapping a [`crate::client::LlmClient`].
///
/// [`LlmService`]: super::service::LlmService
#[cfg_attr(alef, alef(skip))]
pub struct Router<S> {
    deployments: Vec<S>,
    strategy: RoutingStrategy,
    /// Monotonically incrementing counter used by [`RoutingStrategy::RoundRobin`].
    counter: Arc<AtomicUsize>,
    /// Per-deployment metrics (latency tracking, etc.).
    state: RouterState,
    /// Model identifier for each deployment, in the same order as `deployments`.
    ///
    /// Used by [`RoutingStrategy::Semantic`] so the classifier cascade is
    /// handed real model IDs instead of positional placeholders. Defaults to
    /// the deployment's positional index as a string (e.g. `"0"`) until
    /// [`Router::with_deployment_models`] is called.
    deployment_models: Vec<String>,
    /// Seeded PRNG for [`RoutingStrategy::WeightedRandom`].
    ///
    /// Created once in [`Router::new`] and shared across clones via `Arc`, so
    /// selections advance one generator instead of each deriving a threshold
    /// from `SystemTime::now()`: concurrent calls landing in the same clock
    /// tick used to compute the same threshold and pile onto the same
    /// deployment — exactly under the burst load weighted routing exists to
    /// spread, and predictably so.
    weighted_random_rng: Arc<Mutex<SmallRng>>,
}

impl<S> Router<S> {
    /// Create a new router.
    ///
    /// # Errors
    ///
    /// Returns [`LiterLlmError::BadRequest`] if `deployments` is empty — a
    /// router with no deployments cannot handle any request.
    ///
    /// For [`RoutingStrategy::WeightedRandom`], returns an error if the
    /// weights vector length does not match the number of deployments or
    /// if all weights are zero.
    pub fn new(deployments: Vec<S>, strategy: RoutingStrategy) -> Result<Self> {
        if deployments.is_empty() {
            return Err(LiterLlmError::BadRequest {
                message: "Router requires at least one deployment".into(),
                status: 400,
            });
        }
        if let RoutingStrategy::WeightedRandom { ref weights } = strategy {
            if weights.len() != deployments.len() {
                return Err(LiterLlmError::BadRequest {
                    message: format!(
                        "WeightedRandom: weights length ({}) must match deployments length ({})",
                        weights.len(),
                        deployments.len()
                    ),
                    status: 400,
                });
            }
            let total: u64 = weights.iter().map(|w| u64::from(w.as_u32())).sum();
            if total == 0 {
                return Err(LiterLlmError::BadRequest {
                    message: "WeightedRandom: total weight must be positive".into(),
                    status: 400,
                });
            }
        }
        let deployment_models = (0..deployments.len()).map(|i| i.to_string()).collect();
        Ok(Self {
            deployments,
            strategy,
            counter: Arc::new(AtomicUsize::new(0)),
            state: RouterState::new(),
            deployment_models,
            weighted_random_rng: Arc::new(Mutex::new(rand::make_rng::<SmallRng>())),
        })
    }

    /// Attach real model identifiers to each deployment, in the same order as
    /// the `deployments` vec passed to [`Router::new`].
    ///
    /// [`RoutingStrategy::Semantic`] hands these identifiers to the classifier
    /// cascade and maps its verdict back to a deployment by exact string
    /// match. Without this call, deployments are only addressable by their
    /// positional index as a string (e.g. `"0"`), which classifiers built
    /// around real model names (the common case) will never emit — their
    /// verdict is then silently discarded and the router falls back to
    /// round-robin.
    ///
    /// If `models.len()` does not match the deployment count, a warning is
    /// logged; extra entries are ignored and missing entries keep their
    /// positional-index fallback.
    #[must_use]
    pub fn with_deployment_models(mut self, models: Vec<String>) -> Self {
        let expected = self.deployments.len();
        if models.len() != expected {
            tracing::warn!(
                expected,
                got = models.len(),
                "Router::with_deployment_models: length mismatch; missing entries keep the positional-index fallback"
            );
        }
        for (i, model) in models.into_iter().take(expected).enumerate() {
            self.deployment_models[i] = model;
        }
        self
    }
}

impl<S: Clone> Clone for Router<S> {
    fn clone(&self) -> Self {
        Self {
            deployments: self.deployments.clone(),
            strategy: self.strategy.clone(),
            counter: Arc::clone(&self.counter),
            state: self.state.clone(),
            deployment_models: self.deployment_models.clone(),
            weighted_random_rng: Arc::clone(&self.weighted_random_rng),
        }
    }
}

impl<S> Service<LlmRequest> for Router<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = LiterLlmError;
    type Future = BoxFuture<'static, Result<LlmResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        match &self.strategy {
            RoutingStrategy::RoundRobin => self.call_round_robin(req),
            RoutingStrategy::Fallback => self.call_fallback(req),
            RoutingStrategy::LatencyBased => self.call_latency_based(req),
            RoutingStrategy::CostBased => self.call_cost_based(req),
            RoutingStrategy::WeightedRandom { weights } => self.call_weighted_random(weights, req),
            RoutingStrategy::Semantic(classifier) => self.call_semantic(classifier, req),
        }
    }
}

impl<S> Router<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    /// [`RoutingStrategy::RoundRobin`]: advance the shared counter and dispatch
    /// to the next deployment.
    fn call_round_robin(&self, req: LlmRequest) -> BoxFuture<'static, Result<LlmResponse>> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.deployments.len();
        let mut svc = self.deployments[idx].clone();
        Box::pin(async move { svc.call(req).await })
    }

    /// [`RoutingStrategy::Fallback`]: try deployments in order, advancing past
    /// transient errors and propagating any other error immediately.
    fn call_fallback(&self, req: LlmRequest) -> BoxFuture<'static, Result<LlmResponse>> {
        let deployments = self.deployments.clone();
        Box::pin(async move {
            let mut last_err: Option<LiterLlmError> = None;
            for mut svc in deployments {
                match svc.call(req.clone()).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) if e.is_transient() => {
                        tracing::warn!(
                            error = %e,
                            "deployment failed with transient error; trying next deployment"
                        );
                        last_err = Some(e);
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(last_err.unwrap_or(LiterLlmError::ServerError {
                message: "all deployments failed".into(),
                status: 500,
            }))
        })
    }

    /// [`RoutingStrategy::LatencyBased`]: dispatch to the deployment with the
    /// lowest observed latency EMA, then record this call's latency.
    fn call_latency_based(&self, req: LlmRequest) -> BoxFuture<'static, Result<LlmResponse>> {
        let state = self.state.clone();
        let n = self.deployments.len();

        let mut best_idx = 0;
        let mut best_ema = f64::MAX;
        for i in 0..n {
            let ema = state.metrics.get(&i).map_or(0.0, |m| m.latency_ema);
            if ema < best_ema {
                best_ema = ema;
                best_idx = i;
            }
        }

        let mut svc = self.deployments[best_idx].clone();
        let idx = best_idx;

        Box::pin(async move {
            let start = Instant::now();
            let result = svc.call(req).await;
            let latency = start.elapsed().as_secs_f64();

            state.metrics.entry(idx).or_default().record_latency(latency);

            result
        })
    }

    /// [`RoutingStrategy::CostBased`]: try deployments in order, logging the
    /// estimated cost of the first success.
    fn call_cost_based(&self, req: LlmRequest) -> BoxFuture<'static, Result<LlmResponse>> {
        let model = req.model().map(ToOwned::to_owned);
        let deployments = self.deployments.clone();

        Box::pin(async move {
            let mut last_err: Option<LiterLlmError> = None;
            for mut svc in deployments {
                match svc.call(req.clone()).await {
                    Ok(resp) => {
                        log_estimated_cost(model.as_deref(), &resp);
                        return Ok(resp);
                    }
                    Err(e) if e.is_transient() => {
                        last_err = Some(e);
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(last_err.unwrap_or(LiterLlmError::ServerError {
                message: "all deployments failed".into(),
                status: 500,
            }))
        })
    }

    /// [`RoutingStrategy::WeightedRandom`]: dispatch using weighted-random
    /// selection over `weights`.
    fn call_weighted_random(&self, weights: &[Weight], req: LlmRequest) -> BoxFuture<'static, Result<LlmResponse>> {
        let idx = {
            // ~keep Recover the guard on poison rather than unwrapping: a panic
            // ~keep elsewhere while the lock was held doesn't invalidate the RNG's
            // ~keep state, and a bad `.unwrap()` here would take down every future
            // ~keep call through this router over one unrelated poisoning panic.
            let mut rng = self
                .weighted_random_rng
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            weighted_random_select(weights, &mut *rng)
        };
        let mut svc = self.deployments[idx].clone();
        Box::pin(async move { svc.call(req).await })
    }

    /// [`RoutingStrategy::Semantic`]: ask the classifier cascade for a model ID
    /// and dispatch to the deployment serving that model, falling back to
    /// round-robin when the cascade defers or names a model this router has no
    /// deployment for.
    fn call_semantic(
        &self,
        classifier: &Arc<dyn super::route_classify::RouteClassifier>,
        req: LlmRequest,
    ) -> BoxFuture<'static, Result<LlmResponse>> {
        use super::route_classify::ClassifyContext;

        let classifier = Arc::clone(classifier);
        let deployments = self.deployments.clone();
        let counter = Arc::clone(&self.counter);
        let deployment_models = self.deployment_models.clone();

        let (prompt, system_prompt) = extract_semantic_prompt(&req);

        Box::pin(async move {
            let meta: HashMap<String, String> = HashMap::new();
            let ctx = ClassifyContext {
                prompt: &prompt,
                system_prompt: system_prompt.as_deref(),
                metadata: &meta,
                available_models: &deployment_models,
            };

            let verdict = classifier.classify(&ctx).await;
            let idx = resolve_semantic_index(verdict.as_deref(), &deployment_models, &counter);

            deployments[idx].clone().call(req).await
        })
    }
}

/// Log the estimated cost of a successful cost-based routing call, if pricing
/// data is available for the model.
fn log_estimated_cost(model: Option<&str>, resp: &LlmResponse) {
    if let (Some(model_name), Some(usage)) = (model, resp.usage())
        && let Some(cost) = crate::cost::completion_cost(model_name, usage.prompt_tokens, usage.completion_tokens)
    {
        tracing::debug!(model = %model_name, cost_usd = cost, "cost-based routing: estimated cost");
    }
}

/// Pull the latest user message text and an optional system prompt out of a
/// chat request for the semantic classifier cascade. Non-chat requests
/// (embeddings, image generation, etc.) have no prompt text to classify on.
fn extract_semantic_prompt(req: &LlmRequest) -> (String, Option<String>) {
    let LlmRequestKind::Chat(r) = &req.kind else {
        return (String::new(), None);
    };
    let prompt = r
        .messages
        .iter()
        .rev()
        .find_map(|m| {
            if let crate::types::Message::User(u) = m {
                match &u.content {
                    crate::types::UserContent::Text(t) => Some(t.clone()),
                    crate::types::UserContent::Parts(_) => None,
                }
            } else {
                None
            }
        })
        .unwrap_or_default();
    let system = r.messages.iter().find_map(|m| {
        if let crate::types::Message::System(s) = m {
            s.content.as_text()
        } else {
            None
        }
    });
    (prompt, system)
}

/// Map a classifier's model-ID verdict back to a deployment index by exact
/// string match against `deployment_models`. Falls back to round-robin (via
/// `counter`) when the classifier deferred (`None`) or named a model with no
/// matching deployment, so a request is never dropped.
fn resolve_semantic_index(verdict: Option<&str>, deployment_models: &[String], counter: &AtomicUsize) -> usize {
    verdict
        .and_then(|model_id| deployment_models.iter().position(|m| m == model_id))
        .unwrap_or_else(|| counter.fetch_add(1, Ordering::Relaxed) % deployment_models.len())
}

/// Select a deployment index using weighted random distribution.
///
/// Uses a simple linear scan with a random threshold.  For small deployment
/// counts (typical: 2-5) this is fast enough; no binary search needed.
///
/// Draws the threshold from `rng` rather than deriving it from the system
/// clock, so back-to-back calls sharing a caller-held generator (see
/// [`Router`]'s `weighted_random_rng` field) advance independently instead of
/// collapsing to the same index when they land in the same clock tick.
fn weighted_random_select(weights: &[Weight], rng: &mut impl Rng) -> usize {
    let total: u64 = weights.iter().map(|w| u64::from(w.as_u32())).sum();
    if total == 0 {
        return 0;
    }
    let threshold = rng.random_range(0..total);

    let mut cumulative: u64 = 0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += u64::from(w.as_u32());
        if threshold < cumulative {
            return i;
        }
    }
    weights.len() - 1
}

/// A typed extension of [`tower::discover::Discover`] for LLM upstream
/// services.
///
/// Implementors plug in their own discovery mechanism — file-based configs,
/// etcd watches, HTTP polling — and the [`DynamicRouter`] handles the rest.
/// The key type must be `String` so that provider names are human-readable in
/// logs and metrics.
///
/// # Object safety
///
/// `UpstreamDiscover` is **not** object-safe and **must not** be stored as
/// `dyn UpstreamDiscover`.  It is a generic bound used exclusively as a type
/// parameter for [`DynamicRouter<D>`].  All discovery implementations are
/// monomorphised at compile time.
///
/// If you need a runtime registry of heterogeneous discovery sources, wrap
/// each source in an `Arc<Mutex<Box<dyn …>>>` and poll them via a custom
/// `Stream` adapter — do not store them as `dyn UpstreamDiscover`.
///
/// # Note for 1.A integration
///
/// If the router encounters a discovery error, it wraps it in
/// [`RouterError::Discover`].  The 1.A error-consolidation workstream should
/// replace this local enum with the canonical error hierarchy.
pub trait UpstreamDiscover: Discover<Key = String> + Unpin + Send {}

impl<D> UpstreamDiscover for D where D: Discover<Key = String> + Unpin + Send {}

/// Errors produced exclusively by the router.
///
/// **Note**: 1.A owns error-type consolidation.  These codes start at 2000 so
/// they don't clash with the 1xxx range used by the existing
/// [`LiterLlmError`] variants.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(alef, alef(skip))]
pub enum RouterError {
    /// Discovery stream returned an error.
    #[error("discovery error (code 2001): {source}")]
    Discover {
        /// The underlying discovery stream error.
        source: tower::BoxError,
        /// Numeric code for cross-language error conversion.
        code: u32,
    },
    /// No ready upstream is available to serve the request.
    #[error("no ready upstream available (code 2002)")]
    NoReadyUpstream {
        /// Numeric code for cross-language error conversion.
        code: u32,
    },
}

impl RouterError {
    /// Numeric error code, suitable for FFI boundaries.
    #[must_use]
    pub fn code(&self) -> u32 {
        match self {
            Self::Discover { code, .. } | Self::NoReadyUpstream { code } => *code,
        }
    }
}

impl From<RouterError> for LiterLlmError {
    fn from(e: RouterError) -> Self {
        LiterLlmError::ServerError {
            message: e.to_string(),
            status: 503,
        }
    }
}

/// A [`tower::discover::Discover`]-compatible stream that wraps a fixed list
/// of named services.
///
/// In tower 0.5, `Discover` is a blanket impl over any type implementing
/// `TryStream<Ok = Change<K, S>, Error = E>`.  So `StaticDiscover` implements
/// `Stream<Item = Result<Change<String, S>, Infallible>>` which satisfies the
/// `TryStream` bound, making it auto-implement `Discover`.
///
/// Yields one [`Change::Insert`] per service, then signals end-of-stream.
/// This preserves the behaviour of the original [`Router`] while making
/// it composable with [`DynamicRouter`].
#[cfg_attr(alef, alef(skip))]
pub struct StaticDiscover<S> {
    keys: std::collections::VecDeque<String>,
    services: std::collections::VecDeque<S>,
}

impl<S> StaticDiscover<S> {
    /// Build a `StaticDiscover` from an iterable of `(name, service)` pairs.
    pub fn new(services: impl IntoIterator<Item = (String, S)>) -> Self {
        let (keys, services): (std::collections::VecDeque<_>, std::collections::VecDeque<_>) =
            services.into_iter().unzip();
        Self { keys, services }
    }
}

impl<S: Unpin> Unpin for StaticDiscover<S> {}

impl<S: Unpin> Stream for StaticDiscover<S> {
    type Item = std::result::Result<Change<String, S>, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match (self.keys.pop_front(), self.services.pop_front()) {
            (Some(key), Some(svc)) => Poll::Ready(Some(Ok(Change::Insert(key, svc)))),
            _ => Poll::Ready(None),
        }
    }
}

/// Default maximum concurrent in-flight requests per upstream provider.
///
/// Prevents a single slow provider from exhausting all Tokio permits.
/// Callers can override per-provider via [`ProviderConfig::concurrency_limit`].
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 256;

/// Per-provider configuration attached to each upstream in a
/// [`DynamicRouter`].
#[derive(Debug, Clone)]
#[cfg_attr(alef, alef(skip))]
pub struct ProviderConfig {
    /// Maximum concurrent requests allowed to this upstream.
    /// Defaults to [`DEFAULT_CONCURRENCY_LIMIT`].
    pub concurrency_limit: usize,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            concurrency_limit: DEFAULT_CONCURRENCY_LIMIT,
        }
    }
}

/// A router over a [`tower::discover::Discover`] stream of LLM upstreams.
///
/// Services discovered via `D` are pre-warmed in a
/// [`tower::ready_cache::ReadyCache`] so that request-path setup cost is
/// minimal.  Each service is also wrapped in a per-provider
/// [`tower::limit::ConcurrencyLimit`] to prevent one rogue upstream from
/// monopolising Tokio permits.
///
/// # Type parameters
///
/// - `D`: the discovery source; must implement [`UpstreamDiscover`].
/// - `S`: the underlying service type yielded by `D`.
///
/// # Usage
///
/// ```rust,ignore
/// use liter_llm::tower::router::{DynamicRouter, StaticDiscover};
/// use liter_llm::tower::service::LlmService;
///
/// let discover = StaticDiscover::new([
///     ("openai".into(), LlmService::new(openai_client)),
///     ("anthropic".into(), LlmService::new(anthropic_client)),
/// ]);
/// let router = DynamicRouter::new(discover);
/// ```
#[cfg_attr(alef, alef(skip))]
pub struct DynamicRouter<D>
where
    D: Discover<Key = String>,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError>,
{
    discover: D,
    /// Pre-warmed, ready services keyed by provider name.
    services: ReadyCache<String, ConcurrencyLimit<D::Service>, LlmRequest>,
    /// Per-provider configuration (concurrency limits, etc.).
    provider_configs: HashMap<String, ProviderConfig>,
    /// Monotonic counter for round-robin selection among currently-ready
    /// upstreams; see [`Self::call`].
    counter: AtomicUsize,
    _marker: PhantomData<LlmRequest>,
}

impl<D> fmt::Debug for DynamicRouter<D>
where
    D: Discover<Key = String> + fmt::Debug,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicRouter")
            .field("discover", &self.discover)
            .finish_non_exhaustive()
    }
}

impl<D> DynamicRouter<D>
where
    D: Discover<Key = String> + Unpin,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Send + Unpin + 'static,
    <D::Service as Service<LlmRequest>>::Future: Send + 'static,
    D::Error: Into<tower::BoxError>,
{
    /// Create a new `DynamicRouter` from a discovery source.
    ///
    /// Use [`StaticDiscover`] to preserve the behaviour of the original
    /// [`Router`] without external service discovery infrastructure.
    pub fn new(discover: D) -> Self {
        Self {
            discover,
            services: ReadyCache::default(),
            provider_configs: HashMap::new(),
            counter: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// Attach a per-provider [`ProviderConfig`] (concurrency limits, etc.).
    pub fn with_provider_config(mut self, key: impl Into<String>, config: ProviderConfig) -> Self {
        self.provider_configs.insert(key.into(), config);
        self
    }

    /// Return the number of upstream services currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.services.len()
    }

    /// Return `true` if no upstream services are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }

    /// Poll the discovery stream and apply any pending insertions/removals.
    fn update_from_discover(&mut self, cx: &mut Context<'_>) -> std::result::Result<(), RouterError> {
        loop {
            match Pin::new(&mut self.discover).poll_discover(cx) {
                Poll::Pending => return Ok(()),
                Poll::Ready(None) => return Ok(()),
                Poll::Ready(Some(Err(e))) => {
                    return Err(RouterError::Discover {
                        source: e.into(),
                        code: 2001,
                    });
                }
                Poll::Ready(Some(Ok(Change::Insert(key, svc)))) => {
                    let limit = self
                        .provider_configs
                        .get(&key)
                        .map_or(DEFAULT_CONCURRENCY_LIMIT, |c| c.concurrency_limit);
                    tracing::debug!(provider = %key, concurrency_limit = limit, "discovered new upstream");
                    self.services.push(key, ConcurrencyLimit::new(svc, limit));
                }
                Poll::Ready(Some(Ok(Change::Remove(key)))) => {
                    tracing::debug!(provider = %key, "upstream removed from discovery");
                    self.services.evict(&key);
                }
            }
        }
    }
}

/// Select the next ready-set index to dispatch to, rotating through
/// `[0, ready_len)` on every call.
///
/// Dispatching unconditionally to index 0 would pin all traffic on whichever
/// upstream currently occupies that slot in the [`ReadyCache`] instead of
/// distributing across the ready set; see [`DynamicRouter::call`].
fn next_ready_index(counter: &AtomicUsize, ready_len: usize) -> usize {
    counter.fetch_add(1, Ordering::Relaxed) % ready_len
}

impl<D> Service<LlmRequest> for DynamicRouter<D>
where
    D: Discover<Key = String> + Unpin + Send,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Send + Unpin + 'static,
    <D::Service as Service<LlmRequest>>::Future: Send + 'static,
    D::Error: Into<tower::BoxError>,
{
    type Response = LlmResponse;
    type Error = LiterLlmError;
    type Future = BoxFuture<'static, Result<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        if let Err(e) = self.update_from_discover(cx) {
            return Poll::Ready(Err(e.into()));
        }

        let _ = self.services.poll_pending(cx);

        if self.services.ready_len() > 0 {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let ready_len = self.services.ready_len();
        if ready_len == 0 {
            return Box::pin(async { Err(RouterError::NoReadyUpstream { code: 2002 }.into()) });
        }
        let index = next_ready_index(&self.counter, ready_len);
        let fut = self.services.call_ready_index(index, req);
        Box::pin(fut)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use futures_core::Stream;
    use rand::SeedableRng;

    use super::*;
    use crate::tower::service::LlmService;
    use crate::tower::tests_common::{MockClient, chat_req};
    use crate::tower::types::LlmRequest;

    #[test]
    fn weight_clamps_nan_to_zero() {
        assert_eq!(Weight::from_f64(f64::NAN).as_u32(), 0);
    }

    #[test]
    fn weight_clamps_negative_to_zero() {
        assert_eq!(Weight::from_f64(-1.0).as_u32(), 0);
        assert_eq!(Weight::from_f64(-f64::INFINITY).as_u32(), 0);
    }

    #[test]
    fn weight_clamps_inf_to_max() {
        assert_eq!(Weight::from_f64(f64::INFINITY).as_u32(), u32::MAX);
    }

    #[test]
    fn weight_rounds_normal_values() {
        assert_eq!(Weight::from_f64(1.0).as_u32(), 1);
        assert_eq!(Weight::from_f64(1.4).as_u32(), 1);
        assert_eq!(Weight::from_f64(1.5).as_u32(), 2);
        assert_eq!(Weight::from_f64(100.0).as_u32(), 100);
    }

    #[test]
    fn weight_default_is_one() {
        assert_eq!(Weight::default().as_u32(), 1);
    }

    /// Weight ratios (1:2:3) must still translate into proportional index
    /// space after the entropy source moved from `SystemTime` to a seeded
    /// PRNG — the weighting math itself is unchanged, only the threshold
    /// source is, so every index must still be reachable.
    #[test]
    fn weighted_random_selects_proportionally() {
        let weights = vec![Weight(1), Weight(2), Weight(3)];
        let mut rng = SmallRng::seed_from_u64(0xF00D);
        let mut counts = [0usize; 3];

        for _ in 0..600u64 {
            let idx = weighted_random_select(&weights, &mut rng);
            assert!(idx < 3, "index {idx} out of range");
            counts[idx] += 1;
        }

        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 0, "index {i} was never selected (counts: {counts:?})");
        }
    }

    /// Regression: the old implementation derived its threshold from
    /// `SystemTime::now().subsec_nanos()`, so calls issued back-to-back in a
    /// tight loop routinely landed in the same clock tick and computed the
    /// *same* threshold every time — collapsing all traffic onto one
    /// deployment during exactly the burst load weighted routing exists to
    /// spread. A per-router PRNG shared across calls (mirroring how
    /// `Router` holds one `weighted_random_rng` for its lifetime) must
    /// instead advance on every draw.
    ///
    /// Made non-flaky by asserting a property true with overwhelming
    /// probability rather than one that could fail by chance: with weights
    /// 1:2:3 (total 6), the *most* likely single index has probability 1/2,
    /// so the chance that 500 independent draws from a real advancing PRNG
    /// all land on it is `0.5^500` — indistinguishable from zero. Any
    /// implementation that reintroduces per-tick clustering (the actual bug
    /// this pins) would instead reliably produce a single repeated index.
    #[test]
    fn weighted_random_rapid_calls_do_not_all_collapse_to_one_index() {
        let weights = vec![Weight(1), Weight(2), Weight(3)];
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..500u64 {
            seen.insert(weighted_random_select(&weights, &mut rng));
        }

        assert!(
            seen.len() > 1,
            "500 rapid back-to-back calls all returned the same index — entropy source is not advancing per call"
        );
    }

    #[tokio::test]
    async fn latency_based_routes_to_fastest() {
        let deployments: Vec<LlmService<MockClient>> =
            vec![LlmService::new(MockClient::ok()), LlmService::new(MockClient::ok())];

        let mut router = Router::new(deployments, RoutingStrategy::LatencyBased).expect("non-empty deployments");

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok());

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn cost_based_falls_through_on_transient_error() {
        let deployments: Vec<LlmService<MockClient>> = vec![
            LlmService::new(MockClient::failing_service_unavailable()),
            LlmService::new(MockClient::ok()),
        ];

        let mut router = Router::new(deployments, RoutingStrategy::CostBased).expect("non-empty deployments");

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "should fall through to second deployment");
    }

    #[tokio::test]
    async fn weighted_random_selects_valid_deployment() {
        let deployments: Vec<LlmService<MockClient>> = vec![
            LlmService::new(MockClient::ok()),
            LlmService::new(MockClient::ok()),
            LlmService::new(MockClient::ok()),
        ];

        let mut router = Router::new(
            deployments,
            RoutingStrategy::WeightedRandom {
                weights: vec![Weight(1), Weight(2), Weight(3)],
            },
        )
        .expect("non-empty deployments");

        for _ in 0..20 {
            let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
            assert!(resp.is_ok());
        }
    }

    #[tokio::test]
    async fn weighted_random_rejects_mismatched_weights() {
        let deployments: Vec<LlmService<MockClient>> =
            vec![LlmService::new(MockClient::ok()), LlmService::new(MockClient::ok())];

        let result = Router::new(
            deployments,
            RoutingStrategy::WeightedRandom {
                weights: vec![Weight(1)],
            },
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn weighted_random_rejects_zero_total_weight() {
        let deployments: Vec<LlmService<MockClient>> = vec![LlmService::new(MockClient::ok())];

        let result = Router::new(
            deployments,
            RoutingStrategy::WeightedRandom {
                weights: vec![Weight::ZERO],
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn weighted_random_select_returns_valid_index() {
        let weights = vec![Weight(1), Weight(2), Weight(3)];
        let mut rng = SmallRng::seed_from_u64(0xABCD);
        for _ in 0..100 {
            let idx = weighted_random_select(&weights, &mut rng);
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn deployment_metrics_ema_updates() {
        let mut m = DeploymentMetrics::default();
        m.record_latency(1.0);
        assert!(
            (m.latency_ema - 1.0).abs() < 1e-9,
            "first sample should set EMA directly"
        );

        m.record_latency(0.0);
        assert!(
            (m.latency_ema - 0.7).abs() < 1e-9,
            "EMA should be 0.7 after second sample"
        );
    }

    /// A `Stream`-based discover that drains from a pre-built VecDeque.
    /// In tower 0.5, `Discover` is a blanket impl over `TryStream`, so
    /// implementing `Stream` here is sufficient.
    struct VecDiscover {
        items: VecDeque<std::result::Result<Change<String, LlmService<MockClient>>, std::convert::Infallible>>,
    }

    impl VecDiscover {
        fn new(services: Vec<(String, LlmService<MockClient>)>) -> Self {
            Self {
                items: services.into_iter().map(|(k, v)| Ok(Change::Insert(k, v))).collect(),
            }
        }
    }

    impl Stream for VecDiscover {
        type Item = std::result::Result<Change<String, LlmService<MockClient>>, std::convert::Infallible>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.items.pop_front())
        }
    }

    impl Unpin for VecDiscover {}

    #[tokio::test]
    async fn dynamic_router_warms_ready_cache() {
        let discover = VecDiscover::new(vec![
            ("openai".into(), LlmService::new(MockClient::ok())),
            ("anthropic".into(), LlmService::new(MockClient::ok())),
        ]);

        let mut router = DynamicRouter::new(discover);

        futures_util::future::poll_fn(|cx| match router.poll_ready(cx) {
            Poll::Ready(Ok(())) => Poll::Ready(()),
            Poll::Ready(Err(e)) => panic!("unexpected error: {e}"),
            Poll::Pending => Poll::Pending,
        })
        .await;

        assert!(!router.is_empty(), "at least one upstream should be ready");
    }

    #[tokio::test]
    async fn dynamic_router_evicts_stale() {
        /// A stream that inserts then immediately removes a service.
        struct InsertThenRemoveDiscover {
            step: usize,
        }

        impl Stream for InsertThenRemoveDiscover {
            type Item = std::result::Result<Change<String, LlmService<MockClient>>, std::convert::Infallible>;

            fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                let step = self.step;
                self.step += 1;
                match step {
                    0 => Poll::Ready(Some(Ok(Change::Insert(
                        "openai".into(),
                        LlmService::new(MockClient::ok()),
                    )))),
                    1 => Poll::Ready(Some(Ok(Change::Remove("openai".into())))),
                    _ => Poll::Ready(None),
                }
            }
        }

        impl Unpin for InsertThenRemoveDiscover {}

        let discover = InsertThenRemoveDiscover { step: 0 };
        let mut router = DynamicRouter::new(discover);

        let mut noop_cx = std::task::Context::from_waker(futures_util::task::noop_waker_ref());
        let _ = router.poll_ready(&mut noop_cx);

        assert_eq!(router.len(), 0, "evicted service should be removed");
    }

    #[tokio::test]
    async fn concurrency_limit_rejects_at_max() {
        #[derive(Clone)]
        struct BlockingService {
            call_count: Arc<AtomicUsize>,
        }

        impl Service<LlmRequest> for BlockingService {
            type Response = LlmResponse;
            type Error = LiterLlmError;
            type Future = BoxFuture<'static, Result<LlmResponse>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _req: LlmRequest) -> Self::Future {
                self.call_count.fetch_add(1, AtomicOrdering::SeqCst);
                Box::pin(std::future::pending())
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let inner = BlockingService {
            call_count: Arc::clone(&counter),
        };

        let mut limited = ConcurrencyLimit::new(inner, 1);

        assert!(
            futures_util::future::poll_fn(|cx| limited.poll_ready(cx)).await.is_ok(),
            "first poll_ready should be ok"
        );

        let _held_fut = limited.call(LlmRequest::ListModels());

        let mut noop_cx = std::task::Context::from_waker(futures_util::task::noop_waker_ref());
        let poll = limited.poll_ready(&mut noop_cx);
        assert!(
            poll.is_pending(),
            "second poll_ready should be Pending when limit=1 and one request is in-flight"
        );
    }

    #[tokio::test]
    async fn static_discover_yields_all_services() {
        let mut discover = StaticDiscover::new(vec![
            ("a".to_owned(), LlmService::new(MockClient::ok())),
            ("b".to_owned(), LlmService::new(MockClient::ok())),
        ]);

        let mut noop_cx = std::task::Context::from_waker(futures_util::task::noop_waker_ref());

        let first = Pin::new(&mut discover).poll_next(&mut noop_cx);
        assert!(matches!(first, Poll::Ready(Some(Ok(Change::Insert(ref k, _)))) if k == "a"));

        let second = Pin::new(&mut discover).poll_next(&mut noop_cx);
        assert!(matches!(second, Poll::Ready(Some(Ok(Change::Insert(ref k, _)))) if k == "b"));

        let third = Pin::new(&mut discover).poll_next(&mut noop_cx);
        assert!(matches!(third, Poll::Ready(None)));
    }

    /// A classifier that always routes to deployment index `target`.
    struct FixedIndexClassifier {
        target: String,
    }

    impl crate::tower::route_classify::RouteClassifier for FixedIndexClassifier {
        fn classify<'a>(
            &'a self,
            _ctx: &'a crate::tower::route_classify::ClassifyContext<'a>,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
            let target = self.target.clone();
            Box::pin(async move { Some(target) })
        }
    }

    /// A classifier that always defers (returns None).
    struct DeferringClassifier;

    impl crate::tower::route_classify::RouteClassifier for DeferringClassifier {
        fn classify<'a>(
            &'a self,
            _ctx: &'a crate::tower::route_classify::ClassifyContext<'a>,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
            Box::pin(async move { None })
        }
    }

    /// [`RoutingStrategy::Semantic`] routes to the deployment index returned
    /// by the classifier.
    ///
    /// Deployment 0 wraps a `failing_rate_limited` client and deployment 1
    /// wraps an `ok` client.  The classifier always routes to index "1",
    /// so the request must succeed.
    #[tokio::test]
    async fn router_semantic_strategy_uses_classifier() {
        let deployments: Vec<LlmService<MockClient>> = vec![
            LlmService::new(MockClient::failing_rate_limited()),
            LlmService::new(MockClient::ok()),
        ];
        let classifier = Arc::new(FixedIndexClassifier { target: "1".into() });
        let mut router = Router::new(deployments, RoutingStrategy::Semantic(classifier)).expect("valid router");

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "classifier should have routed to the ok deployment");
    }

    /// When the classifier returns `None` (defers), the router falls back to
    /// round-robin so the request is never dropped.
    #[tokio::test]
    async fn router_semantic_strategy_fallback_to_round_robin_when_classifier_defers() {
        let deployments: Vec<LlmService<MockClient>> =
            vec![LlmService::new(MockClient::ok()), LlmService::new(MockClient::ok())];
        let classifier = Arc::new(DeferringClassifier);
        let mut router = Router::new(deployments, RoutingStrategy::Semantic(classifier)).expect("valid router");

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "fallback round-robin should handle the request");
    }

    /// A classifier that records the `available_models` slice it was given,
    /// then defers. Used to prove the router hands the classifier real model
    /// IDs rather than stringified positional indices.
    struct RecordingClassifier {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl crate::tower::route_classify::RouteClassifier for RecordingClassifier {
        fn classify<'a>(
            &'a self,
            ctx: &'a crate::tower::route_classify::ClassifyContext<'a>,
        ) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
            *self.seen.lock().expect("mutex not poisoned") = ctx.available_models.to_vec();
            Box::pin(async move { None })
        }
    }

    /// Regression test for the "semantic routing is silently inert" bug: the
    /// classifier must see the deployments' real model IDs (as configured via
    /// [`Router::with_deployment_models`]), not `["0", "1", ...]` positional
    /// placeholders that no real-world classifier would ever return.
    #[tokio::test]
    async fn router_semantic_strategy_passes_real_model_ids_to_classifier() {
        let deployments: Vec<LlmService<MockClient>> =
            vec![LlmService::new(MockClient::ok()), LlmService::new(MockClient::ok())];
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let classifier = Arc::new(RecordingClassifier {
            seen: Arc::clone(&seen),
        });
        let mut router = Router::new(deployments, RoutingStrategy::Semantic(classifier))
            .expect("valid router")
            .with_deployment_models(vec!["gpt-4o".into(), "claude-3-5-sonnet".into()]);

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok());
        assert_eq!(
            *seen.lock().expect("mutex not poisoned"),
            vec!["gpt-4o".to_string(), "claude-3-5-sonnet".to_string()],
            "classifier must see real model IDs, not positional index placeholders"
        );
    }

    /// Regression test for the "verdict parsed back as an index" bug: the
    /// classifier returns a real model *name* (not `"1"`), and the router must
    /// resolve it to the matching deployment by name — not silently discard
    /// the verdict because it fails to parse as `usize`.
    ///
    /// Deployment 0 wraps a `failing_rate_limited` client; deployment 1 (named
    /// `"claude-3-5-sonnet"`) wraps an `ok` client. Only a genuine name-based
    /// resolution reaches the working deployment.
    #[tokio::test]
    async fn router_semantic_strategy_routes_by_real_model_id() {
        let deployments: Vec<LlmService<MockClient>> = vec![
            LlmService::new(MockClient::failing_rate_limited()),
            LlmService::new(MockClient::ok()),
        ];
        let classifier = Arc::new(FixedIndexClassifier {
            target: "claude-3-5-sonnet".into(),
        });
        let mut router = Router::new(deployments, RoutingStrategy::Semantic(classifier))
            .expect("valid router")
            .with_deployment_models(vec!["gpt-4o".into(), "claude-3-5-sonnet".into()]);

        let resp = router.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(
            resp.is_ok(),
            "verdict 'claude-3-5-sonnet' should resolve to deployment 1 by name"
        );
    }

    /// Regression test for "`DynamicRouter::call` always dispatches to ready
    /// index 0": the selection helper must rotate through the ready set
    /// instead of returning a constant.
    #[test]
    fn next_ready_index_rotates_through_ready_set() {
        let counter = AtomicUsize::new(0);
        assert_eq!(next_ready_index(&counter, 3), 0);
        assert_eq!(
            next_ready_index(&counter, 3),
            1,
            "second call must not be pinned to index 0"
        );
        assert_eq!(next_ready_index(&counter, 3), 2);
        assert_eq!(
            next_ready_index(&counter, 3),
            0,
            "wraps back around after a full rotation"
        );
    }
}
