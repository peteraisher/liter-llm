//! Idempotency-Key dedup layer.
//!
//! [`IdempotencyLayer`] implements the OpenAI `Idempotency-Key` header
//! convention for Tower services.  When a request carries an
//! [`LlmRequest::idempotency_key`][crate::tower::types::LlmRequest::idempotency_key],
//! the layer enforces three semantics:
//!
//! 1. **First request** — forwarded to the inner service.  On success the
//!    response is stored in the [`IdempotencyStore`].
//! 2. **Repeat request, same body** — the stored response is returned without
//!    invoking the inner service (within TTL).
//! 3. **Repeat request, different body, same key** — returns
//!    [`LiterLlmError::IdempotencyConflict`][crate::error::LiterLlmError::IdempotencyConflict].
//!
//! If a request with the same key is currently in-flight (the first request
//! has not yet returned a response), the layer returns
//! [`LiterLlmError::IdempotencyInFlight`][crate::error::LiterLlmError::IdempotencyInFlight]
//! immediately so the caller can retry after a short delay.  This avoids
//! sleep-polling inside the library and keeps Tokio task lifetimes bounded.
//!
//! # Default TTL
//!
//! The default TTL is **24 hours**, matching the OpenAI `Idempotency-Key`
//! convention.  Use [`IdempotencyLayer::with_ttl`] to override.
//!
//! # Storage
//!
//! [`InMemoryIdempotencyStore`] is the default backend.  It uses a
//! [`dashmap::DashMap`] with per-entry TTL checked on every read.  Implement
//! [`IdempotencyStore`] to plug in Redis, DynamoDB, or any other backend.
//!
//! # Layer order
//!
//! Place `IdempotencyLayer` **outermost** — before singleflight and caching —
//! so that repeat requests short-circuit before any cache interaction:
//!
//! ```text
//! IdempotencyLayer → SingleflightLayer → NegativeCacheLayer → CacheLayer → Upstream
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use liter_llm::tower::{IdempotencyLayer, InMemoryIdempotencyStore, LlmService};
//! use tower::ServiceBuilder;
//!
//! let store = InMemoryIdempotencyStore::default();
//! let svc = ServiceBuilder::new()
//!     .layer(IdempotencyLayer::new(store))
//!     .service(LlmService::new(client));
//!
//! let request = LlmRequest::Chat(chat_req).with_idempotency_key("req-abc-123");
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tower::{Layer, Service};

use crate::client::BoxFuture;
use crate::error::LiterLlmError;
use crate::error::Result as LiterResult;
use crate::tower::cache::CachedResponse;
use crate::tower::types::{LlmRequest, LlmRequestKind, LlmResponse};

/// Fixed seeds for the `ahash` [`RandomState`] used by body hashing.
///
/// These constants MUST NOT be changed once idempotency entries have been
/// persisted, as a seed change would invalidate stored hashes.
const IDEM_HASH_SEED_0: u64 = 0x6964_656d_706f_7465;
const IDEM_HASH_SEED_1: u64 = 0x6e63_795f_6861_7368;
const IDEM_HASH_SEED_2: u64 = 0x5f73_6565_6430_5f76;
const IDEM_HASH_SEED_3: u64 = 0x315f_6c6c_6d00_0000;

/// Process-global deterministic [`ahash::RandomState`] for body hashing.
///
/// Constructed once from compile-time-fixed seeds so the same body always
/// produces the same hash across process restarts and Rust version upgrades.
/// This makes it safe to use in distributed stores (Redis, DynamoDB) where
/// multiple processes must agree on the hash value.
fn idem_random_state() -> &'static ahash::RandomState {
    use std::sync::OnceLock;
    static STATE: OnceLock<ahash::RandomState> = OnceLock::new();
    // ~keep `with_seeds`, not `generate_with`: the latter folds a per-process random
    // ~keep component into the seeds, so the digest was stable only WITHIN one process
    // ~keep (the OnceLock hid it). Every restart and every replica produced a different
    // ~keep hash for the same body — which silently breaks the distributed-store use
    // ~keep case documented above, turning a repeat request into a spurious conflict.
    STATE.get_or_init(|| {
        ahash::RandomState::with_seeds(IDEM_HASH_SEED_0, IDEM_HASH_SEED_1, IDEM_HASH_SEED_2, IDEM_HASH_SEED_3)
    })
}

/// Upper bound on the serialised-body prefix embedded in a body hash.
const BODY_HASH_PREFIX_BYTES: usize = 64;

/// Largest index `<= max_bytes` that is a UTF-8 character boundary in `s`.
///
/// `str` slicing panics when an index lands inside a multi-byte character, and
/// `serde_json` emits non-ASCII text raw rather than `\u`-escaped, so a plain
/// non-English prompt puts one across a fixed byte offset. ~keep
fn char_boundary_at_or_before(s: &str, max_bytes: usize) -> usize {
    let mut cut = s.len().min(max_bytes);
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    cut
}

/// Compute a stable body hash for the request.
///
/// Only `kind` is hashed — `tenant_id` and `idempotency_key` are infra
/// metadata and must not affect the body identity check.
///
/// Uses [`ahash::RandomState`] with four fixed compile-time seeds so the
/// hash is identical across process restarts, distributed nodes, and Rust
/// versions.  The body string prefix is embedded in the output for extra
/// collision resistance, so a hash collision yields a spurious
/// `IdempotencyConflict` rather than silent data corruption.
///
/// Returns `None` for request variants that cannot be serialised (should
/// never happen in practice — all variants derive `serde::Serialize`).
fn compute_body_hash(request: &LlmRequest) -> Option<String> {
    // ~keep Hash only the provider payload so infra metadata does not affect idempotency.
    let json = serde_json::to_string(&request.kind).ok()?;

    let h = idem_random_state().hash_one(&json);
    // ~keep Embed a JSON prefix so hash collisions cause conflicts, not silent corruption.
    let cut = char_boundary_at_or_before(&json, BODY_HASH_PREFIX_BYTES);
    Some(format!("{h:016x}:{}", &json[..cut]))
}

/// An entry in the idempotency store.
#[derive(Clone)]
pub struct IdempotencyEntry {
    /// Hash of the canonical request body at the time of first insertion.
    pub body_hash: String,
    /// The stored response.  `None` while the first request is still in-flight.
    pub response: Option<CachedResponse>,
    /// Wall-clock instant at which this entry was created.
    pub inserted_at: Instant,
    /// Effective TTL for this entry.
    pub ttl: Duration,
}

impl std::fmt::Debug for IdempotencyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdempotencyEntry")
            .field("body_hash", &self.body_hash)
            .field("has_response", &self.response.is_some())
            .field("inserted_at", &self.inserted_at)
            .field("ttl", &self.ttl)
            .finish()
    }
}

impl IdempotencyEntry {
    fn is_expired(&self) -> bool {
        self.inserted_at.elapsed() > self.ttl
    }
}

/// Error type for [`IdempotencyStore`] operations.
#[derive(Debug, thiserror::Error)]
pub enum IdempotencyStoreError {
    /// A backend-specific error occurred.
    #[error("idempotency store backend error: {0}")]
    Backend(String),
}

/// Pluggable backing store for the idempotency layer.
///
/// The default in-process implementation is [`InMemoryIdempotencyStore`].
/// Implement this trait to provide distributed idempotency coordination via
/// Redis, DynamoDB, or any other backend.
///
/// All methods return pinned boxed futures so the trait is object-safe and
/// can be used behind `Arc<dyn IdempotencyStore>`.
pub trait IdempotencyStore: Send + Sync + 'static {
    /// Look up an existing entry by idempotency key.
    ///
    /// Returns `None` on a miss (key never seen or TTL expired).
    fn get<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<IdempotencyEntry>, IdempotencyStoreError>> + Send + 'a>>;

    /// Insert a placeholder entry for `key` if none exists yet.
    ///
    /// Returns `Ok(true)` when this caller won the insertion race (it is the
    /// writer — the caller proceeds to invoke the inner service).
    /// Returns `Ok(false)` when a concurrent inserter beat this caller (the
    /// caller should re-read the entry and act accordingly).
    fn try_insert<'a>(
        &'a self,
        key: &'a str,
        body_hash: &'a str,
        ttl: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, IdempotencyStoreError>> + Send + 'a>>;

    /// Finalise an in-flight entry by storing the inner service's response.
    ///
    /// Called by the writer after the inner service returns successfully.
    /// A failed inner call must NOT call `store_response`; the placeholder
    /// entry will expire naturally so subsequent callers can retry.
    fn store_response<'a>(
        &'a self,
        key: &'a str,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>>;

    /// Remove the placeholder entry for `key`.
    ///
    /// Called by the writer when the inner service fails, so subsequent
    /// callers do not observe a stale in-flight entry.  Implementations that
    /// do not support explicit removal may rely on TTL expiry instead.
    fn remove<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>>;
}

/// In-memory idempotency store backed by a [`DashMap`].
///
/// Per-entry TTLs are checked lazily on every `get` call; there is no
/// background expiry task, so `try_insert` additionally reclaims room once
/// the store reaches its capacity (see `Self::DEFAULT_MAX_ENTRIES` and the
/// `with_max_entries` builder).
///
/// # Concurrency
///
/// `DashMap` provides lock-striped concurrent access.  `try_insert` uses an
/// atomic `entry()` operation to guarantee that exactly one concurrent caller
/// wins the insertion race.
pub struct InMemoryIdempotencyStore {
    map: DashMap<String, IdempotencyEntry>,
    max_entries: usize,
}

impl Default for InMemoryIdempotencyStore {
    fn default() -> Self {
        Self {
            map: DashMap::new(),
            max_entries: Self::DEFAULT_MAX_ENTRIES,
        }
    }
}

impl InMemoryIdempotencyStore {
    /// Default cap on stored idempotency entries.
    ///
    /// The key is a tenant-prefixed, CLIENT-SUPPLIED `Idempotency-Key` header,
    /// so without a cap a client sending unique keys grows the store without
    /// bound for the full TTL (24h by default) — expiry is only checked lazily
    /// on `get`, so nothing is reclaimed unless something is looked up again.
    /// Matches the cap used by the other unbounded-key-cardinality caches in
    /// this crate (`ClassifierVerdictCache`, the metrics attribute caches). ~keep
    pub const DEFAULT_MAX_ENTRIES: usize = 4096;

    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the cap on stored idempotency entries.
    #[must_use]
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries.max(1);
        self
    }

    /// Reclaim room for a new entry once the store is at capacity.
    ///
    /// First drops entries whose TTL has elapsed. Only if that is
    /// insufficient does it evict further — and even then, only entries that
    /// are both unexpired AND already finalised (`response.is_some()`).
    ///
    /// CRITICAL: an unexpired entry with `response: None` is a placeholder
    /// backing a request that is still in flight (see
    /// [`IdempotencyInFlightGuard`]); evicting it would let a second caller
    /// with the same key miss the in-flight check and re-run the request
    /// against upstream, turning a correct dedup into a duplicate call. This
    /// method never removes such an entry — only `is_expired()` entries and
    /// entries with a stored `response` are eviction candidates. ~keep
    fn make_room(&self) {
        if self.map.len() < self.max_entries {
            return;
        }

        // ~keep Reclaim genuinely expired entries first; placeholders are included
        // ~keep here too, but only once their own TTL has elapsed, which is the
        // ~keep existing (correct) bound on how long an in-flight request may block.
        self.map.retain(|_, entry| !entry.is_expired());

        if self.map.len() < self.max_entries {
            return;
        }

        // ~keep Oldest-first, not DashMap iteration order: evicting a finalised entry
        // ~keep means a client retrying that key re-runs the request upstream, so the
        // ~keep entry closest to expiring is the one whose loss costs least.
        let mut candidates: Vec<(Instant, String)> = self
            .map
            .iter()
            .filter(|entry| entry.response.is_some())
            .map(|entry| (entry.inserted_at, entry.key().clone()))
            .collect();
        candidates.sort_unstable_by_key(|(inserted_at, _)| *inserted_at);

        let mut evicted = 0usize;
        for (_, key) in candidates {
            if self.map.len() < self.max_entries {
                break;
            }
            if self.map.remove(&key).is_some() {
                evicted += 1;
            }
        }

        if evicted > 0 {
            tracing::warn!(
                cap = self.max_entries,
                evicted,
                "idempotency store reached its cap; evicted completed entries early"
            );
        }
    }
}

impl IdempotencyStore for InMemoryIdempotencyStore {
    fn get<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<IdempotencyEntry>, IdempotencyStoreError>> + Send + 'a>> {
        let result = self
            .map
            .get(key)
            .and_then(|entry| if entry.is_expired() { None } else { Some(entry.clone()) });
        Box::pin(std::future::ready(Ok(result)))
    }

    fn try_insert<'a>(
        &'a self,
        key: &'a str,
        body_hash: &'a str,
        ttl: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, IdempotencyStoreError>> + Send + 'a>> {
        use dashmap::mapref::entry::Entry;

        // ~keep Reclaim room, if any is needed, before taking the shard lock via
        // ~keep entry() below — a Vacant match here is exactly the case that grows
        // ~keep the map by one entry.
        self.make_room();

        let inserted = match self.map.entry(key.to_owned()) {
            Entry::Vacant(slot) => {
                slot.insert(IdempotencyEntry {
                    body_hash: body_hash.to_owned(),
                    response: None,
                    inserted_at: Instant::now(),
                    ttl,
                });
                true
            }
            Entry::Occupied(entry) => {
                // ~keep Expired idempotency entries are replaced atomically so one caller wins the retry.
                if entry.get().is_expired() {
                    entry.replace_entry(IdempotencyEntry {
                        body_hash: body_hash.to_owned(),
                        response: None,
                        inserted_at: Instant::now(),
                        ttl,
                    });
                    true
                } else {
                    false
                }
            }
        };
        Box::pin(std::future::ready(Ok(inserted)))
    }

    fn store_response<'a>(
        &'a self,
        key: &'a str,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>> {
        if let Some(mut entry) = self.map.get_mut(key) {
            entry.response = Some(response);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn remove<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>> {
        self.map.remove(key);
        Box::pin(std::future::ready(Ok(())))
    }
}

/// Tower [`Layer`] that deduplicates requests sharing the same `Idempotency-Key`.
///
/// See [module documentation][self] for semantics and layer order.
#[cfg_attr(alef, alef(skip))]
pub struct IdempotencyLayer<S: IdempotencyStore> {
    store: Arc<S>,
    ttl: Duration,
}

impl<S: IdempotencyStore> IdempotencyLayer<S> {
    /// Create a new layer with the default 24-hour TTL.
    #[must_use]
    pub fn new(store: S) -> Self {
        Self::with_ttl(store, Duration::from_secs(24 * 60 * 60))
    }

    /// Create a new layer with an explicit TTL.
    #[must_use]
    pub fn with_ttl(store: S, ttl: Duration) -> Self {
        Self {
            store: Arc::new(store),
            ttl,
        }
    }
}

impl<S: IdempotencyStore> Clone for IdempotencyLayer<S> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

impl<I, S: IdempotencyStore> Layer<I> for IdempotencyLayer<S> {
    type Service = IdempotencyService<I, S>;

    fn layer(&self, inner: I) -> Self::Service {
        IdempotencyService {
            inner,
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

/// Tower service produced by [`IdempotencyLayer`].
#[cfg_attr(alef, alef(skip))]
pub struct IdempotencyService<I, S: IdempotencyStore> {
    inner: I,
    store: Arc<S>,
    ttl: Duration,
}

impl<I: Clone, S: IdempotencyStore> Clone for IdempotencyService<I, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

/// RAII guard that releases an in-flight idempotency placeholder if the
/// writer's future is dropped (cancelled) before it finalises the entry via
/// `store_response` or `remove`.
///
/// # Why this exists
///
/// `try_insert` writes a placeholder entry (`response: None`) that blocks
/// every other caller with the same key behind `IdempotencyInFlight` until
/// either the writer finalises it or the full TTL (24h by default) elapses.
/// Before this guard, a cancelled writer future — client disconnect, a
/// request timeout, an aborted hedge loser (see
/// [`crate::tower::hedge::HedgeService`]), or any other future drop — never
/// reached the `store_response`/`remove` call, leaving the placeholder stuck
/// for the full TTL. Every other caller with that key would receive
/// `IdempotencyInFlight` for up to 24 hours even though no request was
/// actually still running.
///
/// The guard is created immediately after this caller wins the insertion
/// race and is disarmed only after the entry has actually been finalised
/// (success, failure, or non-cacheable response), so it also cleans up a
/// cancellation that happens mid-finalisation.
struct IdempotencyInFlightGuard<S: IdempotencyStore> {
    store: Arc<S>,
    key: String,
    disarmed: bool,
}

impl<S: IdempotencyStore> IdempotencyInFlightGuard<S> {
    fn new(store: Arc<S>, key: String) -> Self {
        Self {
            store,
            key,
            disarmed: false,
        }
    }

    /// Prevent the guard from removing the entry on drop.
    ///
    /// Call this only after the entry has been finalised (via
    /// `store_response` or `remove`) so a genuine cancellation between
    /// insertion and finalisation is still caught.
    fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl<S: IdempotencyStore> Drop for IdempotencyInFlightGuard<S> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }

        let store = Arc::clone(&self.store);
        let key = std::mem::take(&mut self.key);

        // ~keep IdempotencyStore::remove is async; Drop is sync, so spawn a best-effort cleanup task —
        // ~keep same tokio::runtime::Handle::try_current() convention as hooks.rs's CancellationGuard.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(e) = store.remove(&key).await {
                    tracing::warn!(
                        error = %e,
                        "idempotency: failed to release in-flight entry after cancellation"
                    );
                }
            });
        } else {
            tracing::warn!(
                "idempotency: no Tokio runtime available to release in-flight entry after cancellation; \
                 entry will remain blocked until its TTL expires"
            );
        }
    }
}

impl<I, S> Service<LlmRequest> for IdempotencyService<I, S>
where
    I: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Clone + Send + 'static,
    I::Future: Send + 'static,
    S: IdempotencyStore,
{
    type Response = LlmResponse;
    type Error = LiterLlmError;
    type Future = BoxFuture<'static, LiterResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<LiterResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: LlmRequest) -> Self::Future {
        // ~keep Tower contract: consume the polled-ready instance and leave a fresh standby clone.
        let standby = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, standby);

        let store = Arc::clone(&self.store);
        let ttl = self.ttl;

        Box::pin(async move {
            let Some(ref raw_key) = request.idempotency_key.clone() else {
                return inner.call(request).await;
            };

            // ~keep Tenant-scope idempotency keys so guessable keys cannot leak responses across tenants.
            let tenant_prefix = request.tenant_id.as_ref().map(|t| t.as_ref()).unwrap_or("_");
            let key = format!("{tenant_prefix}:{raw_key}");

            let body_hash = match compute_body_hash(&request) {
                Some(h) => h,
                None => {
                    return inner.call(request).await;
                }
            };

            if let Some(entry) = store.get(&key).await.map_err(store_err)? {
                if entry.body_hash != body_hash {
                    // ~keep Report the raw user-facing key, not the tenant-scoped internal store key.
                    return Err(LiterLlmError::IdempotencyConflict { key: raw_key.clone() });
                }
                if let Some(cached) = entry.response {
                    return cached.into_llm_response();
                }
                return Err(LiterLlmError::IdempotencyInFlight { key: raw_key.clone() });
            }

            let inserted = store.try_insert(&key, &body_hash, ttl).await.map_err(store_err)?;

            // ~keep If expiry wins this race, an extra upstream call is safer than blocking the caller.
            if !inserted && let Some(entry) = store.get(&key).await.map_err(store_err)? {
                if entry.body_hash != body_hash {
                    return Err(LiterLlmError::IdempotencyConflict { key: raw_key.clone() });
                }
                if let Some(cached) = entry.response {
                    return cached.into_llm_response();
                }
                return Err(LiterLlmError::IdempotencyInFlight { key: raw_key.clone() });
            }

            // ~keep This caller is the writer from this point on; guard against the future being
            // ~keep dropped (cancelled) before the entry below is finalised, so a stuck IdempotencyInFlight
            // ~keep placeholder never survives past this call — see IdempotencyInFlightGuard's docs.
            let mut guard = IdempotencyInFlightGuard::new(Arc::clone(&store), key.clone());

            let result = inner.call(request).await;

            match &result {
                Ok(resp) => {
                    let cached = match resp {
                        LlmResponse::Chat(r) => Some(CachedResponse::Chat(r.clone())),
                        LlmResponse::Embed(r) => Some(CachedResponse::Embed(r.clone())),
                        // ~keep Non-cacheable responses remove the placeholder; consumed streams are not replayable.
                        _ => None,
                    };
                    if let Some(cached_resp) = cached {
                        if let Err(error) = store.store_response(&key, cached_resp).await {
                            // ~keep The caller still gets its real response either way; a failed
                            // ~keep write only means the idempotency guarantee is lost for the next
                            // ~keep caller with this key, which must not fail silently.
                            tracing::warn!(
                                %error,
                                "idempotency: failed to persist response; duplicate requests with this key may re-run"
                            );
                        }
                    } else if let Err(error) = store.remove(&key).await {
                        tracing::warn!(%error, "idempotency: failed to remove non-cacheable placeholder entry");
                    }
                }
                Err(_) => {
                    if let Err(error) = store.remove(&key).await {
                        tracing::warn!(%error, "idempotency: failed to remove placeholder entry after inner failure");
                    }
                }
            }
            guard.disarm();

            result
        })
    }
}

/// Map an [`IdempotencyStoreError`] to [`LiterLlmError::InternalError`].
#[inline]
fn store_err(e: IdempotencyStoreError) -> LiterLlmError {
    LiterLlmError::InternalError {
        message: format!("idempotency store: {e}"),
    }
}

/// Helper: return true only for request variants whose responses are cacheable.
///
/// Used by `IdempotencyService` to decide whether to store or discard the
/// inner service's response.
#[must_use]
#[allow(dead_code)]
pub(crate) fn is_cacheable_kind(kind: &LlmRequestKind) -> bool {
    matches!(kind, LlmRequestKind::Chat(_) | LlmRequestKind::Embed(_))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use super::*;
    use crate::error::LiterLlmError;
    use crate::tower::service::LlmService;

    /// The body hash is documented as safe for distributed stores, which requires
    /// every process and replica to agree on it. A `OnceLock` makes any seeding
    /// strategy look stable within one process, so the only way to detect a
    /// per-process random component is to build two states independently and
    /// compare — `generate_with` fails this, `with_seeds` passes.
    #[test]
    fn body_hash_seeding_is_identical_across_independent_constructions() {
        let build =
            || ahash::RandomState::with_seeds(IDEM_HASH_SEED_0, IDEM_HASH_SEED_1, IDEM_HASH_SEED_2, IDEM_HASH_SEED_3);

        assert_eq!(
            build().hash_one("the-same-body"),
            build().hash_one("the-same-body"),
            "two independently constructed states must agree, or the hash is only stable \
             within a single process and cannot back a shared idempotency store"
        );
    }
    use crate::tower::tests_common::{MockClient, chat_req};
    use crate::tower::types::{LlmRequest, LlmResponse};

    fn make_layer() -> IdempotencyLayer<InMemoryIdempotencyStore> {
        IdempotencyLayer::new(InMemoryIdempotencyStore::new())
    }

    fn req_with_key(model: &str, key: &str) -> LlmRequest {
        LlmRequest::Chat(chat_req(model)).with_idempotency_key(key)
    }

    #[tokio::test]
    async fn store_get_returns_none_on_miss() {
        let store = InMemoryIdempotencyStore::new();
        let result = store.get("missing-key").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn store_try_insert_wins_first_caller() {
        let store = InMemoryIdempotencyStore::new();
        let inserted = store.try_insert("k1", "hash1", Duration::from_secs(60)).await.unwrap();
        assert!(inserted, "first caller must win insertion");

        let second = store.try_insert("k1", "hash1", Duration::from_secs(60)).await.unwrap();
        assert!(!second, "second caller must lose insertion race");
    }

    #[tokio::test]
    async fn store_try_insert_wins_after_expiry() {
        let store = InMemoryIdempotencyStore::new();
        store.try_insert("k2", "hash", Duration::from_nanos(1)).await.unwrap();

        tokio::time::sleep(Duration::from_millis(2)).await;

        let inserted = store.try_insert("k2", "hash", Duration::from_secs(60)).await.unwrap();
        assert!(inserted, "insertion after TTL expiry must succeed");
    }

    #[tokio::test]
    async fn store_get_returns_none_for_expired_entry() {
        let store = InMemoryIdempotencyStore::new();
        store.try_insert("k3", "hash", Duration::from_nanos(1)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        let result = store.get("k3").await.unwrap();
        assert!(result.is_none(), "expired entry must not be returned");
    }

    #[tokio::test]
    async fn store_store_response_populates_entry() {
        let store = InMemoryIdempotencyStore::new();
        store.try_insert("k4", "hash", Duration::from_secs(60)).await.unwrap();
        let resp = CachedResponse::Chat(crate::tower::tests_common::make_chat_response("gpt-4"));
        store.store_response("k4", resp).await.unwrap();

        let entry = store.get("k4").await.unwrap().expect("entry must exist");
        assert!(entry.response.is_some(), "response must be populated");
    }

    /// The store's key is a tenant-prefixed, CLIENT-SUPPLIED `Idempotency-Key`
    /// header, with expiry checked lazily only on `get` — nothing sweeps stale
    /// entries in the background. Before `make_room`, a client sending unique
    /// keys grew the map without bound for the full 24h default TTL. Each
    /// insertion here is immediately finalised via `store_response` (as the
    /// real `IdempotencyService` does once the inner call returns), so every
    /// prior entry is a legitimate eviction candidate; this drives insertions
    /// well past the cap and asserts the map never exceeds it.
    #[tokio::test]
    async fn store_try_insert_is_bounded_by_max_entries() {
        const CAP: usize = 8;
        let store = InMemoryIdempotencyStore::new().with_max_entries(CAP);

        for i in 0..CAP * 10 {
            let key = format!("key-{i}");
            store.try_insert(&key, "hash", Duration::from_secs(60)).await.unwrap();
            let resp = CachedResponse::Chat(crate::tower::tests_common::make_chat_response("gpt-4"));
            store.store_response(&key, resp).await.unwrap();
        }

        assert!(
            store.map.len() <= CAP,
            "store must stay within its cap; held {} entries with a cap of {CAP}",
            store.map.len()
        );
    }

    /// Evicting a finalised entry makes a client retrying that key re-run the
    /// request upstream — a duplicate LLM call and a duplicate charge. So when
    /// eviction is unavoidable, it must fall on the entry closest to expiring,
    /// not on whichever key `DashMap` happens to yield first. Pins oldest-first
    /// ordering: the newest entry must outlive the oldest.
    #[tokio::test]
    async fn store_make_room_evicts_the_oldest_finalised_entry_first() {
        const CAP: usize = 4;
        let store = InMemoryIdempotencyStore::new().with_max_entries(CAP);

        for i in 0..CAP {
            let key = format!("aged-{i}");
            store.try_insert(&key, "hash", Duration::from_secs(60)).await.unwrap();
            let resp = CachedResponse::Chat(crate::tower::tests_common::make_chat_response("gpt-4"));
            store.store_response(&key, resp).await.unwrap();
            // ~keep Instant has coarse resolution on some platforms; separate the
            // ~keep insertions so the ordering under test is actually observable.
            tokio::time::sleep(Duration::from_millis(2)).await;
        }

        store
            .try_insert("newcomer", "hash", Duration::from_secs(60))
            .await
            .unwrap();

        assert!(
            store.get("aged-0").await.unwrap().is_none(),
            "the oldest finalised entry must be the one evicted"
        );
        assert!(
            store.get(&format!("aged-{}", CAP - 1)).await.unwrap().is_some(),
            "the newest finalised entry must survive while an older one is available to evict"
        );
    }

    /// Reaching the cap must first reclaim entries that are merely stale
    /// rather than immediately discarding entries that are still within TTL.
    #[tokio::test]
    async fn store_make_room_reclaims_expired_entries_before_evicting_live_ones() {
        const CAP: usize = 4;
        let store = InMemoryIdempotencyStore::new().with_max_entries(CAP);

        for i in 0..CAP {
            let key = format!("stale-{i}");
            store.try_insert(&key, "hash", Duration::from_nanos(1)).await.unwrap();
        }
        assert_eq!(store.map.len(), CAP);

        tokio::time::sleep(Duration::from_millis(5)).await;

        store
            .try_insert("fresh", "hash", Duration::from_secs(60))
            .await
            .unwrap();

        assert_eq!(
            store.map.len(),
            1,
            "the expired entries must be reclaimed, leaving only the fresh one"
        );
        assert!(
            store.get("fresh").await.unwrap().is_some(),
            "the fresh entry must survive"
        );
    }

    /// CRITICAL regression: an unexpired entry with `response: None` is a
    /// placeholder backing a request that is still in flight
    /// ([`IdempotencyInFlightGuard`]). If `make_room` ever evicted it to make
    /// space, a second caller with the same key would miss the in-flight
    /// check and re-run the request against upstream — turning a correct
    /// dedup into a duplicate upstream call. This test fills the store with
    /// unexpired, unfinalised placeholders up to the cap, then forces another
    /// insertion past the cap, and asserts every placeholder is still present
    /// with `response` still `None`.
    #[tokio::test]
    async fn store_make_room_never_evicts_an_unexpired_in_flight_placeholder() {
        const CAP: usize = 4;
        let store = InMemoryIdempotencyStore::new().with_max_entries(CAP);

        for i in 0..CAP {
            let key = format!("in-flight-{i}");
            let inserted = store.try_insert(&key, "hash", Duration::from_secs(60)).await.unwrap();
            assert!(inserted, "placeholder {i} must be inserted");
        }

        // ~keep Force another insertion attempt at the cap; make_room runs but must
        // ~keep find no evictable candidate since every entry is an unexpired placeholder.
        store
            .try_insert("in-flight-extra", "hash", Duration::from_secs(60))
            .await
            .unwrap();

        for i in 0..CAP {
            let key = format!("in-flight-{i}");
            let entry = store
                .get(&key)
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("in-flight placeholder {key} must not be evicted"));
            assert!(
                entry.response.is_none(),
                "placeholder {key} must remain unfinalised, not silently dropped and re-inserted"
            );
        }

        // ~keep The store exceeds its cap here by exactly the placeholders it correctly
        // ~keep refused to evict; that is the documented trade-off (correctness over a
        // ~keep strict cap) — see make_room's doc comment.
        assert!(
            store.map.len() > CAP,
            "extra placeholder only fits by exceeding the cap"
        );
    }

    #[tokio::test]
    async fn store_remove_deletes_entry() {
        let store = InMemoryIdempotencyStore::new();
        store.try_insert("k5", "hash", Duration::from_secs(60)).await.unwrap();
        store.remove("k5").await.unwrap();
        let result = store.get("k5").await.unwrap();
        assert!(result.is_none(), "removed entry must not be present");
    }

    #[tokio::test]
    async fn first_request_hits_inner() {
        let layer = make_layer();
        let client = MockClient::ok();
        let call_count = Arc::clone(&client.call_count);
        let mut svc = layer.layer(LlmService::new(client));

        let result = svc.call(req_with_key("gpt-4", "key-001")).await;
        assert!(result.is_ok(), "first request must succeed");
        assert_eq!(call_count.load(Ordering::SeqCst), 1, "inner must be called once");
    }

    #[tokio::test]
    async fn repeat_same_key_same_body_returns_cached() {
        let layer = make_layer();
        let client = MockClient::ok();
        let call_count = Arc::clone(&client.call_count);
        let mut svc = layer.layer(LlmService::new(client));

        svc.call(req_with_key("gpt-4", "key-002"))
            .await
            .expect("first call must succeed");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        let result = svc.call(req_with_key("gpt-4", "key-002")).await;
        assert!(result.is_ok(), "second call must succeed");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "inner must NOT be called on second request with same key+body"
        );
    }

    #[tokio::test]
    async fn repeat_same_key_different_body_returns_conflict() {
        let layer = make_layer();
        let client = MockClient::ok();
        let mut svc = layer.layer(LlmService::new(client));

        svc.call(req_with_key("gpt-4", "key-003"))
            .await
            .expect("first call must succeed");

        let result = svc.call(req_with_key("gpt-3.5-turbo", "key-003")).await;
        assert!(
            matches!(result, Err(LiterLlmError::IdempotencyConflict { .. })),
            "different body for same key must return IdempotencyConflict; got {result:?}"
        );
    }

    #[tokio::test]
    async fn no_key_passes_through() {
        let layer = make_layer();
        let client = MockClient::ok();
        let call_count = Arc::clone(&client.call_count);
        let mut svc = layer.layer(LlmService::new(client));

        let result = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(result.is_ok(), "request without key must succeed");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "inner must be called for keyless request"
        );
    }

    #[tokio::test]
    async fn inner_error_does_not_cache() {
        let layer = make_layer();
        let client = MockClient::failing_rate_limited();
        let call_count = Arc::clone(&client.call_count);
        let mut svc = layer.layer(LlmService::new(client));

        let first = svc.call(req_with_key("gpt-4", "key-err")).await;
        assert!(first.is_err(), "first call must fail");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        let second = svc.call(req_with_key("gpt-4", "key-err")).await;
        assert!(second.is_err(), "second call must also fail (same inner error)");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "inner must be called again after first failed call"
        );
    }

    #[tokio::test]
    #[ignore = "moka time-mocking not available; TTL expiry tested via InMemoryIdempotencyStore unit tests"]
    async fn ttl_expiry_allows_new_invocation() {}

    #[tokio::test]
    async fn different_keys_are_independent() {
        let layer = make_layer();
        let client = MockClient::ok();
        let call_count = Arc::clone(&client.call_count);
        let mut svc = layer.layer(LlmService::new(client));

        svc.call(req_with_key("gpt-4", "key-A"))
            .await
            .expect("call A must succeed");
        svc.call(req_with_key("gpt-4", "key-B"))
            .await
            .expect("call B must succeed");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "different keys must both hit inner"
        );

        svc.call(req_with_key("gpt-4", "key-A"))
            .await
            .expect("repeat A must succeed");
        svc.call(req_with_key("gpt-4", "key-B"))
            .await
            .expect("repeat B must succeed");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "repeated calls with same key+body must not hit inner"
        );
    }

    #[tokio::test]
    async fn returned_response_matches_original() {
        let layer = make_layer();
        let client = MockClient::ok();
        let mut svc = layer.layer(LlmService::new(client));

        let first = svc
            .call(req_with_key("gpt-4", "key-content"))
            .await
            .expect("first call");
        let first_model = match &first {
            LlmResponse::Chat(r) => r.model.clone(),
            _ => panic!("expected Chat response"),
        };

        let second = svc
            .call(req_with_key("gpt-4", "key-content"))
            .await
            .expect("second call");
        let second_model = match &second {
            LlmResponse::Chat(r) => r.model.clone(),
            _ => panic!("expected Chat response"),
        };

        assert_eq!(first_model, second_model, "cached response must match original");
    }

    /// `compute_body_hash` must return the same value on every call for the
    /// same request, even when constructed from independent instances.
    /// The old `DefaultHasher` used a randomized seed (Rust 1.36+), so two
    /// different process runs (or, in distributed setups, two different nodes)
    /// could produce different hashes for the same request body, breaking
    /// distributed idempotency coordination.
    #[test]
    fn idempotency_body_hash_deterministic_across_instances() {
        let req = LlmRequest::Chat(chat_req("gpt-4"));

        let hashes: Vec<_> = (0..10).map(|_| compute_body_hash(&req)).collect();

        let first = hashes[0].as_ref().expect("hash must be Some");
        for (i, h) in hashes.iter().enumerate() {
            assert_eq!(
                h.as_ref().expect("hash must be Some"),
                first,
                "hash #{i} differs from hash #0 — ahash seed is not fixed"
            );
        }
    }

    /// `compute_body_hash` embeds a byte-bounded prefix of the serialised body.
    /// `serde_json` emits non-ASCII text raw rather than `\u`-escaped, so a
    /// request whose payload puts a multi-byte character across the cut point
    /// used to panic on the string slice — a reachable panic in the request
    /// path, reached by an ordinary non-English prompt.  Sweeping the leading
    /// padding walks the character across the boundary whatever the exact
    /// serialised layout is; the `straddled` guard fails the test if a future
    /// layout change means no case exercises the boundary any more.
    #[test]
    fn body_hash_survives_multibyte_char_on_prefix_boundary() {
        use crate::types::common::{Message, UserMessage};

        let mut straddled = false;

        for pad in 0..BODY_HASH_PREFIX_BYTES {
            let mut chat = chat_req("gpt-4");
            chat.messages = vec![Message::User(UserMessage {
                content: format!("{}🌍 доброе утро", "a".repeat(pad)).into(),
                name: None,
            })];

            let request = LlmRequest::Chat(chat);
            let json = serde_json::to_string(&request.kind).expect("request must serialise");
            if json.len() > BODY_HASH_PREFIX_BYTES && !json.is_char_boundary(BODY_HASH_PREFIX_BYTES) {
                straddled = true;
            }

            let hash = compute_body_hash(&request).unwrap_or_else(|| panic!("hash must be Some (pad={pad})"));

            assert!(
                hash.contains(':'),
                "hash must keep the `<digest>:<prefix>` shape (pad={pad}), got {hash}"
            );
        }

        assert!(
            straddled,
            "no padding put a multi-byte character across byte {BODY_HASH_PREFIX_BYTES} — \
             the serialised layout changed and this test no longer covers the panic"
        );
    }

    /// Two requests with the same idempotency key but different tenant IDs must
    /// not share the same cached response.  Before the fix, the store key was
    /// the raw idempotency key with no tenant prefix, so tenant B could observe
    /// tenant A's cached response if they happened to use the same key string.
    #[tokio::test]
    async fn idempotency_tenant_scoped_keys_dont_collide() {
        use crate::tower::types::LlmResponse;

        let store = Arc::new(InMemoryIdempotencyStore::new());
        let layer_a = IdempotencyLayer::new(InMemoryIdempotencyStore::new());
        let layer_b = IdempotencyLayer::new(InMemoryIdempotencyStore::new());
        let _ = (store, layer_a, layer_b);

        let shared_store = Arc::new(InMemoryIdempotencyStore::new());
        let make_layer_shared = || IdempotencyLayer {
            store: Arc::clone(&shared_store),
            ttl: Duration::from_secs(60),
        };

        let client_a = MockClient::ok();
        let call_count_a = Arc::clone(&client_a.call_count);
        let mut svc_a = make_layer_shared().layer(LlmService::new(client_a));

        let client_b = MockClient::ok();
        let call_count_b = Arc::clone(&client_b.call_count);
        let mut svc_b = make_layer_shared().layer(LlmService::new(client_b));

        let req_a = LlmRequest::Chat(chat_req("gpt-4"))
            .with_idempotency_key("shared-key")
            .with_tenant_id("tenant-A");
        let req_b = LlmRequest::Chat(chat_req("gpt-4"))
            .with_idempotency_key("shared-key")
            .with_tenant_id("tenant-B");

        let resp_a = svc_a.call(req_a.clone()).await.expect("tenant A first call");
        assert!(matches!(resp_a, LlmResponse::Chat(_)));
        assert_eq!(call_count_a.load(Ordering::SeqCst), 1, "inner called for tenant A");

        let resp_b = svc_b.call(req_b.clone()).await.expect("tenant B first call");
        assert!(matches!(resp_b, LlmResponse::Chat(_)));
        assert_eq!(
            call_count_b.load(Ordering::SeqCst),
            1,
            "inner called for tenant B (no cross-tenant hit)"
        );

        svc_a.call(req_a).await.expect("tenant A repeat");
        assert_eq!(
            call_count_a.load(Ordering::SeqCst),
            1,
            "inner NOT called on tenant A repeat"
        );

        svc_b.call(req_b).await.expect("tenant B repeat");
        assert_eq!(
            call_count_b.load(Ordering::SeqCst),
            1,
            "inner NOT called on tenant B repeat"
        );
    }

    /// Regression for "idempotency key stuck in-flight for 24h if the request
    /// is cancelled": before `IdempotencyInFlightGuard` existed, aborting the
    /// writer's future (client disconnect, timeout, hedge-loser cancellation,
    /// ...) while it awaited `inner.call()` left the placeholder entry
    /// (`response: None`) in the store. Every subsequent caller with the same
    /// key would receive `IdempotencyInFlight` for the full TTL — 24h by
    /// default — even though no request was actually still running.
    ///
    /// This test aborts the writer's task mid-flight, then polls a second
    /// caller with the same key: it must eventually stop seeing
    /// `IdempotencyInFlight` and succeed, proving the guard released the
    /// placeholder instead of leaving it stuck for the TTL.
    #[tokio::test]
    async fn cancelled_writer_releases_in_flight_placeholder() {
        use std::sync::atomic::AtomicUsize;

        /// First call pends forever (to be aborted); every later call
        /// succeeds immediately, so a released placeholder is observable
        /// without the test itself hanging.
        #[derive(Clone)]
        struct PendingThenOkService {
            call_count: Arc<AtomicUsize>,
        }

        impl tower::Service<LlmRequest> for PendingThenOkService {
            type Response = LlmResponse;
            type Error = LiterLlmError;
            type Future = crate::client::BoxFuture<'static, crate::error::Result<LlmResponse>>;

            fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<crate::error::Result<()>> {
                std::task::Poll::Ready(Ok(()))
            }

            fn call(&mut self, _req: LlmRequest) -> Self::Future {
                let attempt = self.call_count.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if attempt == 0 {
                        std::future::pending::<()>().await;
                        unreachable!("first attempt must be aborted before completing");
                    }
                    Ok(LlmResponse::Chat(crate::tower::tests_common::make_chat_response(
                        "gpt-4",
                    )))
                })
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let inner_client = PendingThenOkService {
            call_count: Arc::clone(&call_count),
        };
        let layer = IdempotencyLayer::with_ttl(InMemoryIdempotencyStore::new(), Duration::from_secs(24 * 60 * 60));
        let svc = layer.layer(inner_client);

        let mut svc_for_writer = svc.clone();
        let handle = tokio::spawn(async move {
            let _ = svc_for_writer.call(req_with_key("gpt-4", "cancel-key")).await;
        });

        // Wait until the writer has won try_insert and is parked inside inner.call().
        for _ in 0..200 {
            if call_count.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "writer must have reached inner.call before it is aborted"
        );

        handle.abort();
        let _ = handle.await;

        let mut svc2 = svc;
        let mut released = false;
        for _ in 0..200 {
            match svc2.call(req_with_key("gpt-4", "cancel-key")).await {
                Err(LiterLlmError::IdempotencyInFlight { .. }) => {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                Ok(_) => {
                    released = true;
                    break;
                }
                Err(other) => panic!("unexpected error while polling for release: {other:?}"),
            }
        }

        assert!(
            released,
            "cancelled writer must release the in-flight placeholder, not block callers for the full TTL"
        );
    }
}
