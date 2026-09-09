//! Singleflight deduplication middleware.
//!
//! Under concurrent bursts, multiple callers may issue identical requests
//! simultaneously.  Without coordination, each caller independently hits
//! the upstream LLM provider, multiplying cost and saturating rate limits.
//!
//! [`SingleflightLayer`] collapses concurrent identical requests into a single
//! upstream call.  The *leader* — the first caller for a given key — performs
//! the real work; all subsequent *followers* await the leader's result and
//! receive the same value.
//!
//! # Design
//!
//! The [`SingleflightCoordinator`] trait is the extension point.  The default
//! implementation ([`InMemorySingleflight`]) uses a [`dashmap::DashMap`] of
//! Tokio broadcast channels.  Broadcast (rather than a single `oneshot`) lets
//! an arbitrary number of followers subscribe without any follower needing to
//! hold a unique receiver slot.
//!
//! `tokio::sync::broadcast::Sender::subscribe` (and `Receiver::resubscribe`)
//! position a new receiver at the *current tail*: neither replays a value
//! that was already sent.  A follower that subscribed strictly before the
//! leader's `send` always sees the result; a follower that only reaches the
//! map after the leader's round has fully finished sees a vacant entry and
//! becomes the leader of a fresh round.  What must never happen is a
//! follower subscribing to the channel *after* its one-shot value was sent
//! but *before* the map entry is removed — it would then wait on a second
//! message that never arrives and, once the sender drops, surface a
//! spurious `RecvError::Closed`.
//!
//! **That window is still open.**  `complete` sends and then removes as two
//! separate steps, so a `join` landing between them subscribes past the value.
//! Serialising both under one `DashMap` shard lock via `entry` was tried and
//! did not survive its own test, so it was reverted rather than shipped
//! unverified — do not re-apply that change without a test that actually
//! reproduces the race first.  The window is two statements wide and needs a
//! burst to hit, but it is real; see the LOCAL register entry.
//!
//! # Recommended layer order
//!
//! See [`crate::tower::cache`] module documentation for the full recommended
//! layer composition order.
//!
//! # Panics
//!
//! `SingleflightService` does not panic in normal operation.  `unwrap` calls
//! inside the implementation are guarded by invariants documented in `SAFETY`
//! comments.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::DashMap;
use tokio::sync::broadcast;
use tower::{Layer, Service};

use super::cache::{CachedResponse, record_cache_state};
use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{LiterLlmError, Result};
use crate::observability::usage::CacheState;

type InFlightMap = Arc<DashMap<u64, broadcast::Sender<SingleflightResult>>>;

/// The value broadcast from a singleflight leader to all followers.
///
/// The error value is shared so every follower receives the same upstream
/// failure without cloning the underlying error.
pub type SingleflightResult = std::result::Result<CachedResponse, Arc<LiterLlmError>>;

/// Outcome of [`SingleflightCoordinator::join`].
///
/// - A [`SingleflightHandle::Leader`] performs the upstream call and delivers
///   the result by calling the `complete` closure.
/// - A [`SingleflightHandle::Follower`] awaits the leader's result via the
///   broadcast receiver.
pub enum SingleflightHandle {
    /// First caller for this key.  Caller is responsible for performing the
    /// upstream work and signalling completion via `complete`.
    Leader {
        /// Deliver the result to all waiting followers.
        ///
        /// Calling `complete` is mandatory.  Dropping it without calling causes
        /// all followers to receive a `RecvError` (channel closed), which the
        /// `SingleflightService` maps to an `InternalError`.
        complete: Box<dyn FnOnce(SingleflightResult) + Send>,
    },
    /// Subsequent caller.  Awaits the leader's broadcast result.
    Follower {
        /// Receiver for the leader's result.  Call `.await` to block until the
        /// leader completes.
        recv: broadcast::Receiver<SingleflightResult>,
    },
}

/// Pluggable singleflight coordination strategy.
///
/// Implement this trait to provide distributed singleflight coordination (e.g.
/// via Redis `SET NX` / pub-sub) without modifying library code.
///
/// The default in-process implementation is [`InMemorySingleflight`].
#[cfg_attr(alef, alef(skip))]
pub trait SingleflightCoordinator: Send + Sync + 'static {
    /// Register the caller's interest in `key`.
    ///
    /// Returns a [`SingleflightHandle`] that indicates whether this caller is
    /// the leader (must do upstream work) or a follower (must await the leader).
    fn join<'a>(&'a self, key: u64) -> Pin<Box<dyn Future<Output = SingleflightHandle> + Send + 'a>>;
}

/// In-memory singleflight coordinator backed by a [`DashMap`] of broadcast channels.
///
/// Each in-flight key maps to a `broadcast::Sender<SingleflightResult>`.  The
/// first caller for a key creates the sender (becoming the leader).  Subsequent
/// callers subscribe to the same sender (becoming followers).  When the leader
/// calls `complete`, the result is broadcast to all subscribers.
///
/// Entries are removed from the map by the `complete` closure immediately after
/// broadcasting, so that the next distinct request for the same key starts a
/// fresh singleflight round.
#[cfg_attr(alef, alef(skip))]
pub struct InMemorySingleflight {
    /// Shared in-flight map, wrapped in `Arc` so it can be moved into the
    /// `complete` closure without lifetime constraints.
    ///
    /// A broadcast channel capacity of 1 is sufficient: the channel carries a
    /// single result event.  Late subscribers (followers that join after the
    /// leader completes) receive the stored value from the channel's ring buffer.
    in_flight: InFlightMap,
}

impl Default for InMemorySingleflight {
    fn default() -> Self {
        Self {
            in_flight: Arc::new(DashMap::new()),
        }
    }
}

impl InMemorySingleflight {
    /// Create a new coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl SingleflightCoordinator for InMemorySingleflight {
    fn join<'a>(&'a self, key: u64) -> Pin<Box<dyn Future<Output = SingleflightHandle> + Send + 'a>> {
        Box::pin(async move {
            use dashmap::mapref::entry::Entry;

            match self.in_flight.entry(key) {
                Entry::Vacant(slot) => {
                    let (tx, _) = broadcast::channel::<SingleflightResult>(1);
                    slot.insert(tx.clone());

                    // ~keep `complete` must own the map so cleanup outlives the coordinator borrow.
                    let map = Arc::clone(&self.in_flight);

                    // ~keep LeaderDropGuard removes abandoned entries so followers receive Closed, not a hang.
                    let guard = LeaderDropGuard {
                        map: Arc::clone(&map),
                        key,
                        disarmed: false,
                    };

                    let complete = Box::new(move |result: SingleflightResult| {
                        let mut g = guard;
                        g.disarmed = true;

                        // ~keep A `join` landing between the send and the removal subscribes past
                        // ~keep the value and later sees RecvError::Closed. Serialising the two under
                        // ~keep one shard lock looked like the fix but did not hold up under test —
                        // ~keep see the LOCAL register entry. Left as-is deliberately rather than
                        // ~keep shipping an unverified change to a concurrency primitive.
                        let _ = tx.send(result);
                        map.remove(&key);
                    });

                    SingleflightHandle::Leader { complete }
                }
                Entry::Occupied(entry) => {
                    let recv = entry.get().subscribe();
                    SingleflightHandle::Follower { recv }
                }
            }
        })
    }
}

/// RAII guard that removes a singleflight key from the in-flight map when
/// the leader's `complete` closure is dropped without being called.
///
/// This handles the case where a leader task is cancelled (e.g. via
/// `JoinHandle::abort()`) before it can call `complete`.  Without this guard,
/// the `broadcast::Sender` stored in the DashMap would outlive the leader's
/// owned sender copy, preventing the channel from closing and causing followers
/// to hang indefinitely.
///
/// When the guard's `Drop` runs (armed), it removes the map entry holding
/// the `broadcast::Sender`.  Combined with the leader's `tx` going out of
/// scope, all sender clones are freed, and the channel closes.  Followers
/// then receive `RecvError::Closed`.
struct LeaderDropGuard {
    map: InFlightMap,
    key: u64,
    disarmed: bool,
}

impl Drop for LeaderDropGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.map.remove(&self.key);
        }
    }
}

/// Tower [`Layer`] that collapses concurrent identical requests into one
/// upstream call via a [`SingleflightCoordinator`].
#[cfg_attr(alef, alef(skip))]
pub struct SingleflightLayer<C: SingleflightCoordinator> {
    coordinator: Arc<C>,
}

impl<C: SingleflightCoordinator> SingleflightLayer<C> {
    /// Create a new singleflight layer with the given coordinator.
    #[must_use]
    pub fn new(coordinator: Arc<C>) -> Self {
        Self { coordinator }
    }
}

impl<C: SingleflightCoordinator, S> Layer<S> for SingleflightLayer<C> {
    type Service = SingleflightService<C, S>;

    fn layer(&self, inner: S) -> Self::Service {
        SingleflightService {
            coordinator: Arc::clone(&self.coordinator),
            inner,
        }
    }
}

/// Tower service produced by [`SingleflightLayer`].
#[cfg_attr(alef, alef(skip))]
pub struct SingleflightService<C: SingleflightCoordinator, S> {
    coordinator: Arc<C>,
    inner: S,
}

impl<C: SingleflightCoordinator, S: Clone> Clone for SingleflightService<C, S> {
    fn clone(&self) -> Self {
        Self {
            coordinator: Arc::clone(&self.coordinator),
            inner: self.inner.clone(),
        }
    }
}

/// Derive the singleflight key from a request.
///
/// Only `Chat` and `Embed` requests are deduplicated; other variants are
/// passed through without coordination.  Returns `None` for non-cacheable
/// variants.
///
/// # Tenant isolation
///
/// `LlmRequest::tenant_id` is folded into the hash alongside the request
/// body.  Without this, two virtual keys posting byte-identical bodies would
/// collapse to a single upstream call: the second caller would become a
/// `Follower`, so the Budget/Cost/guardrail layers downstream of this one
/// would never run for it, leaving its spend unrecorded and its budget cap
/// unenforced — a cross-tenant response leak.  This mirrors `strategy_key` in
/// `cache.rs`, which reads `req.tenant_id()` for the same reason.
///
/// `req.tenant_id` is `Option<TenantId>`, and both `Option` and `TenantId`
/// derive `Hash`.  The derived `Hash` for an enum feeds the variant
/// discriminant into the hasher before any contained data, so `None`,
/// `Some(TenantId(String::new()))` and `Some(TenantId("none".into()))` are
/// guaranteed to hash differently even though two of them share string
/// content — `None` can never collide with a tenant literally named `""` or
/// `"none"`. This is not a sentinel-string convention a real tenant name
/// could accidentally reproduce; it relies on the discriminant, not a
/// reserved value in the same space as tenant names.
///
/// # Hash stability
///
/// `DefaultHasher` (SipHash with fixed, non-randomized keys) is not a
/// cross-process or cross-version stable digest — it is only guaranteed
/// consistent within a single running process, and may change between Rust
/// standard library versions.  That is sufficient here: [`InMemorySingleflight`]'s
/// map is process-local and reset on every restart, so keys are never
/// compared across processes or persisted.
fn singleflight_key(req: &LlmRequest) -> Option<u64> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let json = match &req.kind {
        LlmRequestKind::Chat(r) => serde_json::to_string(r).ok()?,
        LlmRequestKind::Embed(r) => serde_json::to_string(r).ok()?,
        _ => return None,
    };
    let mut hasher = DefaultHasher::new();
    req.tenant_id.hash(&mut hasher);
    json.hash(&mut hasher);
    Some(hasher.finish())
}

impl<C, S> Service<LlmRequest> for SingleflightService<C, S>
where
    C: SingleflightCoordinator,
    S: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = LiterLlmError;
    type Future = BoxFuture<'static, Result<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let key = singleflight_key(&req);

        let Some(key) = key else {
            let fut = self.inner.call(req);
            #[allow(clippy::redundant_async_block)]
            return Box::pin(async move { fut.await });
        };

        let coordinator = Arc::clone(&self.coordinator);

        // ~keep The leader must consume the poll_ready slot; followers may drop it without calling.
        // ~keep Leave a fresh un-readied clone for the next poll_ready/call cycle.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        Box::pin(async move {
            match coordinator.join(key).await {
                SingleflightHandle::Leader { complete } => {
                    // ~keep Leader is the sole caller, preserving one call per poll_ready.
                    let result = inner.call(req).await;
                    let sf_result: SingleflightResult = match &result {
                        Ok(resp) => match resp {
                            LlmResponse::Chat(r) => Ok(CachedResponse::Chat(r.clone())),
                            LlmResponse::Embed(r) => Ok(CachedResponse::Embed(r.clone())),
                            _ => Err(Arc::new(LiterLlmError::InternalError {
                                message: "singleflight: non-cacheable response variant in leader".into(),
                            })),
                        },
                        // ~keep Preserve error class for followers even though LiterLlmError is not Clone.
                        Err(e) => Err(Arc::new(e.to_singleflight_error())),
                    };
                    complete(sf_result);
                    result
                }
                SingleflightHandle::Follower { mut recv } => {
                    // ~keep Followers never call the readied service; dropping without call is allowed.
                    drop(inner);
                    match recv.recv().await {
                        Ok(Ok(cached)) => {
                            record_cache_state(CacheState::ExactHit);
                            cached.into_llm_response()
                        }
                        Ok(Err(arc_err)) => {
                            // ~keep Preserve error variant even when broadcast leaves multiple Arc refs.
                            Err(Arc::try_unwrap(arc_err).unwrap_or_else(|arc| arc.to_singleflight_error()))
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::debug!(skipped = n, "singleflight follower lagged; resubscribing");
                            let mut rx2 = recv.resubscribe();
                            match rx2.recv().await {
                                Ok(Ok(cached)) => {
                                    record_cache_state(CacheState::ExactHit);
                                    cached.into_llm_response()
                                }
                                Ok(Err(arc_err)) => {
                                    Err(Arc::try_unwrap(arc_err).unwrap_or_else(|arc| arc.to_singleflight_error()))
                                }
                                Err(_) => Err(LiterLlmError::InternalError {
                                    message: "singleflight: follower lagged and retry also failed".into(),
                                }),
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => Err(LiterLlmError::InternalError {
                            message: "singleflight: leader closed channel without sending a result".into(),
                        }),
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::tower::service::LlmService;
    use crate::tower::tests_common::{MockClient, chat_req};
    use crate::tower::types::LlmRequest;

    /// A slow inner service that introduces an artificial delay so that all
    /// concurrent callers can arrive at the singleflight coordinator before the
    /// leader completes.
    ///
    /// Without a delay, `MockClient` returns synchronously and the leader
    /// completes before follower tasks are scheduled, defeating deduplication.
    #[derive(Clone)]
    struct SlowClient {
        inner: MockClient,
        delay: std::time::Duration,
    }

    impl SlowClient {
        fn ok_with_delay(delay: std::time::Duration) -> Self {
            Self {
                inner: MockClient::ok(),
                delay,
            }
        }
    }

    impl crate::client::LlmClient for SlowClient {
        fn chat(
            &self,
            req: crate::types::ChatCompletionRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::ChatCompletionResponse>> {
            let delay = self.delay;
            let inner_fut = self.inner.chat(req);
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                inner_fut.await
            })
        }

        fn chat_stream(
            &self,
            req: crate::types::ChatCompletionRequest,
        ) -> crate::client::BoxFuture<
            '_,
            crate::error::Result<
                crate::client::BoxStream<'static, crate::error::Result<crate::types::ChatCompletionChunk>>,
            >,
        > {
            self.inner.chat_stream(req)
        }

        fn embed(
            &self,
            req: crate::types::EmbeddingRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::EmbeddingResponse>> {
            self.inner.embed(req)
        }

        fn list_models(&self) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::ModelsListResponse>> {
            self.inner.list_models()
        }

        fn image_generate(
            &self,
            req: crate::types::image::CreateImageRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::image::ImagesResponse>> {
            self.inner.image_generate(req)
        }

        fn speech(
            &self,
            req: crate::types::audio::CreateSpeechRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<bytes::Bytes>> {
            self.inner.speech(req)
        }

        fn transcribe(
            &self,
            req: crate::types::audio::CreateTranscriptionRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::audio::TranscriptionResponse>> {
            self.inner.transcribe(req)
        }

        fn moderate(
            &self,
            req: crate::types::moderation::ModerationRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::moderation::ModerationResponse>> {
            self.inner.moderate(req)
        }

        fn rerank(
            &self,
            req: crate::types::rerank::RerankRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::rerank::RerankResponse>> {
            self.inner.rerank(req)
        }

        fn search(
            &self,
            req: crate::types::search::SearchRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::search::SearchResponse>> {
            self.inner.search(req)
        }

        fn ocr(
            &self,
            req: crate::types::ocr::OcrRequest,
        ) -> crate::client::BoxFuture<'_, crate::error::Result<crate::types::ocr::OcrResponse>> {
            self.inner.ocr(req)
        }
    }

    /// Spawn `n` concurrent requests for the same key via *independent service clones*
    /// that share an `Arc<InMemorySingleflight>`, then assert inner was called exactly once.
    ///
    /// Using independent clones is critical: a single `&mut self` service can only
    /// handle one request at a time (Tower's contract), so sharing a single service
    /// behind a `Mutex` would serialize all calls and defeat singleflight.  Each clone
    /// calls `poll_ready` + `call` independently, but the shared coordinator collapses
    /// them into one upstream call.
    ///
    /// A slow inner service ensures all 100 tasks arrive at the coordinator
    /// while the leader is still awaiting its upstream call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_leader_runs_upstream_once_under_burst() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(100));

        let handles: Vec<_> = (0..100)
            .map(|_| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    use tower::Service as _;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 100, "all 100 callers should get a successful response");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "inner service must be called exactly once under burst; got {calls}"
        );
    }

    /// 10 concurrent requests via independent service clones all receive the same result.
    ///
    /// Uses `SlowClient` (50 ms delay) so all 10 tasks reach the coordinator as
    /// followers before the leader's upstream call completes.  Without the delay
    /// the leader may complete before followers subscribe, causing spurious second
    /// leader rounds.
    ///
    /// Defect 3 fix: the `models.iter().all(|m| m == first)` check alone cannot
    /// fail — `MockClient::chat` returns `make_chat_response(&req.model)` and every
    /// task sends the same model name, so that assertion holds whether or not
    /// deduplication actually happened.  The decisive addition is `call_count`: if
    /// singleflight failed to collapse the burst (e.g. dedup silently disabled),
    /// every one of the 10 callers would hit `SlowClient` independently and
    /// `calls` would be 10, not 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_followers_get_same_result() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let models: Vec<String> = results
            .into_iter()
            .map(|join_result| {
                let llm_resp = join_result
                    .expect("task did not panic")
                    .expect("service call succeeded");
                match llm_resp {
                    LlmResponse::Chat(r) => r.model,
                    _ => panic!("expected Chat response"),
                }
            })
            .collect();

        let first = &models[0];
        assert!(
            models.iter().all(|m| m == first),
            "all followers must receive the same result"
        );

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "followers must actually be deduplicated, not merely coincide on a fixed mock \
             response; inner service called {calls} times, expected exactly 1"
        );
    }

    /// When the leader returns an error, all followers receive that error.
    ///
    /// A `SlowClient` with a 50 ms delay ensures all 10 tasks subscribe as followers
    /// before the leader's future resolves — otherwise the fast `MockClient` would
    /// complete before followers arrive, causing multiple "leader" rounds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_leader_error_propagates_to_followers() {
        let inner_client = MockClient::failing_rate_limited();
        let slow_client = SlowClient {
            inner: inner_client,
            delay: std::time::Duration::from_millis(50),
        };
        let call_count = Arc::clone(&slow_client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let svc = layer.layer(LlmService::new(slow_client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let error_count = results.iter().filter(|r| r.as_ref().unwrap().is_err()).count();

        assert_eq!(error_count, 10, "all callers must receive the leader's error");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "inner should be called exactly once under singleflight; got {calls}"
        );
    }

    /// Followers must never invoke `inner.call` — only the leader does.
    ///
    /// Wire a slow mock with a call counter, fire 10 concurrent requests for the
    /// same key, and assert the inner counter is exactly 1 (the leader) even though
    /// all 10 callers received a successful response.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_follower_does_not_call_inner_service() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    use tower::Service as _;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 10, "all 10 callers should succeed");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "inner service must be called exactly once (leader only); followers must not call it; got {calls}"
        );
    }

    /// Requests with distinct keys must not be deduplicated — each key triggers its
    /// own upstream call.
    ///
    /// Fire 10 concurrent requests with 10 different model names (which produces
    /// 10 different cache keys) and assert the inner service call counter equals 10.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_concurrent_keys_dont_dedupe() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(20));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10u32)
            .map(|i| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    use tower::Service as _;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req(&format!("gpt-4-model-{i}")))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 10, "all 10 distinct-key callers should succeed");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 10,
            "each distinct key must produce its own upstream call; got {calls}"
        );
    }

    /// 100 concurrent callers for the same key must collapse to exactly one
    /// inner call; all 100 must receive the identical leader response.
    ///
    /// Semantically identical to `singleflight_leader_runs_upstream_once_under_burst`
    /// but explicitly named per pass-3 requirements and asserts response identity.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_n100_burst_one_inner_call_only() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(100));
        let handles: Vec<_> = (0..100)
            .map(|_| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;

        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 100, "all 100 callers should get a successful response");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(calls, 1, "inner service called {calls} times; expected exactly 1");

        let models: Vec<String> = results
            .into_iter()
            .map(|r| match r.unwrap().unwrap() {
                LlmResponse::Chat(resp) => resp.model,
                _ => panic!("expected Chat response"),
            })
            .collect();
        let first = &models[0];
        assert!(
            models.iter().all(|m| m == first),
            "all 100 callers must receive identical responses"
        );
    }

    /// When the leader's future is cancelled (aborted via JoinHandle) before it
    /// calls `complete`, followers must receive an error rather than hanging.
    ///
    /// Protocol:
    /// 1. Leader joins coordinator (gets `Leader` handle), then signals via `ready_tx`
    ///    that it has registered, then parks on a `Semaphore` that is never released.
    /// 2. Main task waits for `ready_tx`, then spawns 10 followers that each subscribe
    ///    and wait, then waits for all followers to be parked on `recv.recv()` via a
    ///    `Barrier`, then aborts the leader.
    /// 3. `LeaderDropGuard` removes the map entry → channel closes →
    ///    followers receive `RecvError::Closed`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_leader_cancelled_followers_receive_cancellation() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let key: u64 = 0xDEAD_BEEF;

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
        let all_subscribed = Arc::new(tokio::sync::Barrier::new(11));

        let leader_handle = tokio::spawn({
            let coordinator = Arc::clone(&coordinator);
            async move {
                let handle = coordinator.join(key).await;
                match handle {
                    SingleflightHandle::Leader { complete: _complete } => {
                        let _ = ready_tx.send(());
                        std::future::pending::<()>().await;
                    }
                    SingleflightHandle::Follower { .. } => panic!("first join must be Leader"),
                }
            }
        });

        ready_rx.await.expect("leader must signal readiness");

        let follower_handles: Vec<_> = (0..10)
            .map(|_| {
                let coordinator = Arc::clone(&coordinator);
                let barrier = Arc::clone(&all_subscribed);
                tokio::spawn(async move {
                    let recv = match coordinator.join(key).await {
                        SingleflightHandle::Follower { recv } => recv,
                        SingleflightHandle::Leader { .. } => panic!("subsequent joins must be Follower"),
                    };
                    barrier.wait().await;
                    let mut recv = recv;
                    recv.recv().await
                })
            })
            .collect();

        all_subscribed.wait().await;

        leader_handle.abort();
        let _ = leader_handle.await;

        for handle in follower_handles {
            let result = handle.await.expect("follower task must not panic");
            assert!(
                matches!(result, Err(tokio::sync::broadcast::error::RecvError::Closed)),
                "follower must receive RecvError::Closed when leader is cancelled; got {result:?}"
            );
        }
    }

    /// When the leader's inner service returns `RateLimited`, all followers
    /// must receive an error whose variant is `RateLimited` — not a downgraded
    /// `InternalError`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_leader_error_broadcast_to_followers() {
        let inner_client = MockClient::failing_rate_limited();
        let slow_client = SlowClient {
            inner: inner_client,
            delay: std::time::Duration::from_millis(50),
        };
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let svc = layer.layer(LlmService::new(slow_client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;

        for (i, result) in results.into_iter().enumerate() {
            let err = result
                .unwrap_or_else(|e| panic!("task {i} panicked: {e}"))
                .expect_err("all callers must receive an error");

            assert!(
                matches!(err, LiterLlmError::RateLimited { .. }),
                "caller {i} got {err:?}; expected RateLimited (variant must be preserved across broadcast)"
            );
        }
    }

    /// A follower that subscribes BEFORE the leader calls `complete` must
    /// receive the result — not a `RecvError::Closed`.
    ///
    /// This does not exercise the send-before-remove race (see
    /// `singleflight_late_joiner_during_complete_never_sees_spurious_closed_error`
    /// below): the follower here subscribes long before `complete` runs, so
    /// inverting `complete` to `map.remove(&key); tx.send(result);` would not
    /// fail this test either way.
    #[tokio::test]
    async fn singleflight_follower_subscribed_before_complete_gets_result() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let key: u64 = 0xC0FF_EE00;

        let complete = match coordinator.join(key).await {
            SingleflightHandle::Leader { complete } => complete,
            SingleflightHandle::Follower { .. } => panic!("first join must be Leader"),
        };

        let mut recv = match coordinator.join(key).await {
            SingleflightHandle::Follower { recv } => recv,
            SingleflightHandle::Leader { .. } => panic!("second join must be Follower"),
        };

        complete(Ok(CachedResponse::Chat(
            crate::tower::tests_common::make_chat_response("gpt-4"),
        )));

        let received = recv.recv().await.expect("follower must receive leader result");
        assert!(received.is_ok(), "follower must receive success result");
    }

    /// Defect 1 regression: two callers posting byte-identical bodies but
    /// different `tenant_id` must NOT collapse into one upstream call. Before
    /// the fix, `singleflight_key` hashed only the inner `ChatCompletionRequest`
    /// — `LlmRequest::tenant_id` is a sibling field that never reached the
    /// hasher — so the second caller silently became a `Follower`: the
    /// Budget/Cost/guardrail layers downstream of this one never ran for it,
    /// leaving its spend unrecorded and its budget cap unenforced, while it was
    /// stamped `CacheState::ExactHit` on what was actually a live upstream call
    /// for a different tenant.
    ///
    /// Revert line: reverting `singleflight_key` to hash only
    /// `serde_json::to_string(r)` (dropping the `req.tenant_id.hash(&mut
    /// hasher);` line) makes this test fail, because both distinct-tenant
    /// requests would collapse to `calls == 1`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_cross_tenant_identical_body_does_not_dedupe() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let tenants = ["tenant-a", "tenant-b"];
        let handles: Vec<_> = tenants
            .iter()
            .map(|tenant| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                let tenant = (*tenant).to_owned();
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    let req = LlmRequest::Chat(chat_req("gpt-4")).with_tenant_id(tenant);
                    svc.call(req).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 2, "both distinct-tenant callers should succeed");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 2,
            "identical bodies from different tenants must NOT dedupe; got {calls} upstream call(s)"
        );
    }

    /// Companion to the cross-tenant test above: identical requests from the
    /// SAME tenant must still collapse to one upstream call. This guards
    /// against a "fix" that folds tenant into the key in a way that always
    /// differs between calls (e.g. mixing in a per-call nonce instead of the
    /// actual tenant identity), which would silently disable singleflight
    /// altogether rather than fixing the isolation bug.
    ///
    /// Revert line: replacing `req.tenant_id.hash(&mut hasher);` with
    /// anything that varies per call (e.g. a freshly generated nonce, or the
    /// current time) makes this test fail, because `calls` would be 10, not 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_same_tenant_identical_requests_still_dedupe() {
        let client = SlowClient::ok_with_delay(std::time::Duration::from_millis(50));
        let call_count = Arc::clone(&client.inner.call_count);
        let coordinator = Arc::new(InMemorySingleflight::new());
        let layer = SingleflightLayer::new(Arc::clone(&coordinator));

        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let svc = layer.layer(LlmService::new(client.clone()));
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    let mut svc = svc;
                    futures_util::future::poll_fn(|cx| svc.poll_ready(cx)).await.unwrap();
                    let req = LlmRequest::Chat(chat_req("gpt-4")).with_tenant_id("tenant-a");
                    svc.call(req).await
                })
            })
            .collect();

        let results: Vec<_> = futures_util::future::join_all(handles).await;
        let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert_eq!(success_count, 10, "all same-tenant callers should succeed");

        let calls = call_count.load(Ordering::SeqCst);
        assert_eq!(
            calls, 1,
            "identical same-tenant requests must still dedupe; got {calls} upstream call(s)"
        );
    }

    /// Defect 1: `None` tenant must not collide with a tenant literally named
    /// `""` or `"none"` for an otherwise-identical request body.
    ///
    /// `Option<TenantId>`'s derived `Hash` feeds the enum discriminant into
    /// the hasher before any contained string, so `None`, `Some("")` and
    /// `Some("none")` are guaranteed to hash differently even though two of
    /// them share string content — this relies on the discriminant, not on a
    /// sentinel string a real tenant name could accidentally reproduce.
    ///
    /// Revert line: reverting `singleflight_key` to hash only
    /// `serde_json::to_string(r)` (pre-fix) makes this test fail, because all
    /// three keys collapse to the same hash.
    #[test]
    fn singleflight_key_none_tenant_is_unambiguous_vs_empty_and_literal_none() {
        let none_req = LlmRequest::Chat(chat_req("gpt-4"));
        let empty_req = LlmRequest::Chat(chat_req("gpt-4")).with_tenant_id("");
        let literal_req = LlmRequest::Chat(chat_req("gpt-4")).with_tenant_id("none");

        let none_key = singleflight_key(&none_req).expect("chat requests are keyable");
        let empty_key = singleflight_key(&empty_req).expect("chat requests are keyable");
        let literal_key = singleflight_key(&literal_req).expect("chat requests are keyable");

        assert_ne!(none_key, empty_key, "None tenant must not collide with tenant \"\"");
        assert_ne!(
            none_key, literal_key,
            "None tenant must not collide with tenant \"none\""
        );
        assert_ne!(
            empty_key, literal_key,
            "tenant \"\" must not collide with tenant \"none\""
        );
    }
}
