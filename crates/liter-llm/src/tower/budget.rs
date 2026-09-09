//! Budget enforcement middleware.
//!
//! [`BudgetLayer`] wraps any [`Service<LlmRequest>`] and enforces spending
//! limits (global and per-model) in USD.  Cost is calculated after each
//! successful response using [`crate::cost::completion_cost`] and accumulated
//! atomically in [`BudgetState`].
//!
//! Two enforcement modes are supported:
//!
//! - **Hard** — pre-request check rejects with [`LiterLlmError::BudgetExceeded`]
//!   when the accumulated spend is at or above the configured limit.  Note that
//!   hard enforcement is **best-effort** under concurrent load: because cost is
//!   recorded after the response, concurrent in-flight requests may collectively
//!   overshoot the limit.  See [`check_budget`] for details.
//! - **Soft** — requests are never rejected; a `tracing::warn!` is emitted when
//!   the limit is exceeded.
//!
//! # Pluggable ledger
//!
//! [`BudgetLedger`] is the extension point for custom per-key / per-user cost
//! tracking and multi-dimensional budgets.  The built-in [`InMemoryBudgetLedger`]
//! tracks spend across the global, per-model, per-tenant, per-user, and
//! per-API-key dimensions using sliding-window accumulators backed by
//! [`DashMap`]s.  Supply any type implementing [`BudgetLedger`] to plug in a
//! database-backed or remote ledger.
//!
//! # Example
//!
//! ```rust,ignore
//! use liter_llm::tower::{BudgetConfig, BudgetLayer, BudgetState, Enforcement, LlmService};
//! use tower::ServiceBuilder;
//! use std::sync::Arc;
//!
//! let state = Arc::new(BudgetState::new());
//! let config = BudgetConfig {
//!     global_limit: Some(10.0),
//!     model_limits: Default::default(),
//!     enforcement: Enforcement::Hard,
//! };
//!
//! let client = liter_llm::DefaultClient::new(cfg, None)?;
//! let service = ServiceBuilder::new()
//!     .layer(BudgetLayer::new(config, Arc::clone(&state)))
//!     .service(LlmService::new(client));
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use tower::{Layer, Service};

use super::cost::observe_stream_usage;
use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::cost;
use crate::error::{LiterLlmError, Result};
use crate::types::Usage;

/// The dimension along which a budget rejection was triggered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetDimension {
    /// Cumulative spend across all dimensions.
    Global,
    /// Spend for a specific model.
    Model(String),
    /// Spend for a tenant (organisation-level grouping).
    Tenant(String),
    /// Spend for an individual end-user.
    User(String),
    /// Spend for a specific API key.
    ApiKey(String),
}

/// Decision returned by [`BudgetLedger::check`].
#[derive(Debug, Clone)]
pub enum BudgetVerdict {
    /// The request may proceed.
    Allow,
    /// The request should be rejected because a budget limit was exceeded.
    Reject {
        /// Human-readable reason.
        reason: String,
        /// Which limit was triggered.
        dimension: BudgetDimension,
    },
}

/// Contextual metadata passed to [`BudgetLedger::record`] after a successful
/// completion.
pub struct CostRecordContext<'a> {
    /// The model name (e.g. `"gpt-4"`).
    pub model: &'a str,
    /// The provider name (e.g. `"openai"`).
    pub provider: &'a str,
    /// Optional organisation / tenant identifier.
    pub tenant_id: Option<&'a str>,
    /// Optional end-user identifier.
    pub user_id: Option<&'a str>,
    /// Optional API-key identifier (not the raw secret — an opaque handle).
    pub api_key_id: Option<&'a str>,
    /// Actual cost of this call in US dollars.
    pub cost_usd: f64,
    /// Number of prompt (input) tokens consumed.
    pub tokens_in: u64,
    /// Number of completion (output) tokens consumed.
    pub tokens_out: u64,
    /// Wall-clock time at which the response was received.
    pub timestamp: SystemTime,
}

/// Contextual metadata passed to [`BudgetLedger::check`] before a call is
/// dispatched.  Identical to [`CostRecordContext`] except that `cost_usd`,
/// `tokens_in`, and `tokens_out` are not yet known.
pub struct CostCheckContext<'a> {
    /// The model name (e.g. `"gpt-4"`).
    pub model: &'a str,
    /// The provider name (e.g. `"openai"`).
    pub provider: &'a str,
    /// Optional organisation / tenant identifier.
    pub tenant_id: Option<&'a str>,
    /// Optional end-user identifier.
    pub user_id: Option<&'a str>,
    /// Optional API-key identifier (not the raw secret — an opaque handle).
    pub api_key_id: Option<&'a str>,
    /// Wall-clock time at which the pre-flight check is performed.
    pub timestamp: SystemTime,
}

/// A point-in-time snapshot of cumulative spend across all tracked dimensions.
///
/// Used for observability dashboards and as the primitive for chargeback-ready
/// CSV export via [`InMemoryBudgetLedger::export_csv`].  The `limits_*` fields
/// carry the configured caps so that helpers such as [`should_hedge`] can make
/// limit-aware decisions without requiring access to ledger internals.
#[derive(Debug, Clone, Default)]
pub struct BudgetSnapshot {
    /// Total spend across all dimensions, in USD.
    pub global_spend_usd: f64,
    /// Per-model spend, keyed by model name, in USD.
    pub per_model: HashMap<String, f64>,
    /// Per-tenant spend, keyed by tenant identifier, in USD.
    pub per_tenant: HashMap<String, f64>,
    /// Per-user spend, keyed by user identifier, in USD.
    pub per_user: HashMap<String, f64>,
    /// Per-API-key spend, keyed by API-key identifier, in USD.
    pub per_api_key: HashMap<String, f64>,
    /// Configured global spending cap in USD, if any.
    pub limit_global: Option<f64>,
    /// Configured per-user spending caps in USD.
    pub limits_per_user: HashMap<String, f64>,
    /// Configured per-API-key spending caps in USD.
    pub limits_per_api_key: HashMap<String, f64>,
    /// Configured per-tenant spending caps in USD.
    pub limits_per_tenant: HashMap<String, f64>,
}

/// Pluggable cost-tracking and budget-enforcement backend.
///
/// Implement this trait to plug in a database-backed, Redis-backed, or remote
/// ledger.  The built-in implementation is [`InMemoryBudgetLedger`].
///
/// # Object safety
///
/// The trait is object-safe; you can store it as `Arc<dyn BudgetLedger>`.
pub trait BudgetLedger: Send + Sync + 'static {
    /// Record the cost of a successful call against all relevant ledgers.
    ///
    /// This is called **after** the inner service returns a successful response.
    /// Implementations must be non-blocking; long-running work should be
    /// spawned as a background task.
    fn record<'a>(&'a self, ctx: &'a CostRecordContext<'a>) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    /// Check whether the *next* call would exceed any configured budget limit.
    ///
    /// This is called **before** the inner service is invoked.  Return
    /// [`BudgetVerdict::Reject`] to short-circuit the call without forwarding
    /// to the upstream provider.
    fn check<'a>(&'a self, ctx: &'a CostCheckContext<'a>) -> Pin<Box<dyn Future<Output = BudgetVerdict> + Send + 'a>>;

    /// Return a point-in-time snapshot of all tracked spend dimensions.
    ///
    /// Callers use this for dashboards and for the cost-aware rate-limiter.
    fn snapshot(&self) -> BudgetSnapshot;
}

/// Sliding-window accumulator for a single budget dimension.
///
/// Each dimension (global, model, tenant, user, API-key) maintains its own
/// pair of `(spend_microcents, window_start)`.  When the window elapses the
/// counters are atomically zeroed so that the limit applies fresh each period.
///
/// All values are stored in **microcents** (`USD × 1_000_000`) to avoid
/// floating-point atomics while retaining sub-cent precision.
#[derive(Debug)]
struct WindowEntry {
    /// Accumulated spend in microcents (USD × 1_000_000).
    spend_mc: AtomicU64,
    /// Epoch seconds at which the current window started.
    window_start_secs: AtomicU64,
    /// Window duration in seconds.
    window_secs: u64,
}

impl WindowEntry {
    fn new(window: Duration) -> Self {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            spend_mc: AtomicU64::new(0),
            window_start_secs: AtomicU64::new(now_secs),
            window_secs: window.as_secs(),
        }
    }

    /// Return current spend in USD, resetting if the window has elapsed.
    ///
    /// Uses a `compare_exchange` CAS so that under concurrent calls exactly one
    /// thread wins the rollover.  The winner subtracts the snapshot of
    /// `spend_mc` taken **before** the CAS (the old-window accumulation), so
    /// that any concurrent `fetch_add` calls that land after the snapshot —
    /// whether before or after the CAS — are preserved in the counter.
    /// Threads that lose the CAS simply re-read the counter.
    fn spend_usd(&self, now: SystemTime) -> f64 {
        let now_secs = now.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
        let start = self.window_start_secs.load(Ordering::Acquire);
        if now_secs.saturating_sub(start) >= self.window_secs {
            // ~keep Snapshot before CAS so racing increments after this point are preserved.
            let old_mc = self.spend_mc.load(Ordering::Acquire);

            // ~keep Only the CAS winner performs rollover; losers keep the winner's reset.
            if self
                .window_start_secs
                .compare_exchange(start, now_secs, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // ~keep Subtract only the old-window amount so new-window racing increments remain.
                self.spend_mc.fetch_sub(old_mc, Ordering::AcqRel);
            }
        }
        microcents_to_usd(self.spend_mc.load(Ordering::Acquire))
    }

    /// Add `usd` to this entry, respecting the sliding window.
    fn add(&self, usd: f64, now: SystemTime) {
        let _ = self.spend_usd(now);
        self.spend_mc.fetch_add(usd_to_microcents(usd), Ordering::AcqRel);
    }
}

/// Per-dimension limits configuration used by [`InMemoryBudgetLedger`].
#[derive(Debug, Clone, Default)]
pub struct DimensionLimits {
    /// Global spending cap in USD.  `None` means unlimited.
    pub global: Option<f64>,
    /// Per-model spending caps in USD.
    pub per_model: HashMap<String, f64>,
    /// Per-tenant spending caps in USD.
    pub per_tenant: HashMap<String, f64>,
    /// Per-user spending caps in USD.
    pub per_user: HashMap<String, f64>,
    /// Per-API-key spending caps in USD.
    pub per_api_key: HashMap<String, f64>,
}

/// In-memory [`BudgetLedger`] backed by [`DashMap`]s with sliding-window reset.
///
/// Use [`InMemoryBudgetLedger::new`] for full control or
/// [`InMemoryBudgetLedger::from_config`] to build from an existing
/// [`BudgetConfig`] (for backward compatibility).
///
/// `limits` lives behind an [`ArcSwap`] rather than a plain field so that
/// [`InMemoryBudgetLedger::update_limits`] can hot-swap the configured caps —
/// e.g. on a config reload — without touching the per-tenant/per-user/
/// per-API-key spend already accumulated in the `DashMap`s below. This keeps
/// the read path (`check`/`snapshot`) lock-free: a swap is a single atomic
/// pointer load, matching the concurrency style the sliding-window
/// [`WindowEntry`] accumulators already use.
///
/// # Bounded dimensions
///
/// `per_user` is keyed by the request's free-text `user` field and
/// `per_api_key` by a caller-supplied API-key identifier — both are
/// attacker-controlled, arbitrary-cardinality inputs, so they are capped at
/// `max_principal_entries` entries (see [`Self::with_max_principal_entries`]
/// to override). `per_model` and `per_tenant` are keyed by operator-controlled
/// values (the deployment's configured model list / tenant roster) with
/// naturally bounded cardinality, so they are deliberately left uncapped —
/// capping them would add eviction-policy risk (see below) for a dimension
/// that was never the actual memory-exhaustion vector.
///
/// # Why eviction never touches recorded spend
///
/// This is a *spend ledger*, not a cache: an entry's value is money already
/// billed against a principal. Evicting a live (non-zero-spend) entry to make
/// room for a newcomer would silently forgive that spend, which is a budget
/// bypass — a caller could burn through their limit, get evicted by
/// cardinality pressure from other callers, and come back with a fresh
/// budget. So the cap (`entry_add_capped`) only ever reclaims entries
/// whose recorded spend reads as exactly zero, and once no such entries
/// remain it **declines to track new principals** rather than evict a live
/// one — the call still succeeds and global/per-model spend is still
/// recorded, but that one principal's per-user/per-API-key budget is
/// unenforced until capacity frees up. This is a known, deliberate trade-off:
/// bounded memory always wins, at the cost of enforcement granularity (never
/// of already-recorded spend) under sustained cardinality pressure.
#[derive(Debug)]
pub struct InMemoryBudgetLedger {
    limits: ArcSwap<DimensionLimits>,
    window: Duration,
    global: Arc<WindowEntry>,
    per_model: Arc<DashMap<String, WindowEntry>>,
    per_tenant: Arc<DashMap<String, WindowEntry>>,
    per_user: Arc<DashMap<String, WindowEntry>>,
    per_api_key: Arc<DashMap<String, WindowEntry>>,
    max_principal_entries: usize,
}

impl InMemoryBudgetLedger {
    /// Default cap on distinct entries tracked by `per_user` and
    /// `per_api_key`.
    ///
    /// Both maps are keyed by attacker-controlled, arbitrary-cardinality input
    /// (the request's free-text `user` field / a caller-supplied API-key id),
    /// so without a cap a caller can grow either map without bound simply by
    /// varying that field across requests — unbounded memory growth.
    /// `per_model`/`per_tenant` are operator-controlled and naturally
    /// bounded, so they are not capped. ~keep
    pub const DEFAULT_MAX_PRINCIPAL_ENTRIES: usize = 100_000;

    /// Create a new ledger with explicit limits and a shared window duration.
    ///
    /// The `window` controls how long spend is accumulated before the
    /// per-dimension counters reset (e.g. `Duration::from_secs(86400)` for
    /// daily budgets).
    #[must_use]
    pub fn new(limits: DimensionLimits, window: Duration) -> Self {
        Self {
            global: Arc::new(WindowEntry::new(window)),
            per_model: Arc::new(DashMap::new()),
            per_tenant: Arc::new(DashMap::new()),
            per_user: Arc::new(DashMap::new()),
            per_api_key: Arc::new(DashMap::new()),
            limits: ArcSwap::from_pointee(limits),
            window,
            max_principal_entries: Self::DEFAULT_MAX_PRINCIPAL_ENTRIES,
        }
    }

    /// Override the cap on tracked `per_user` / `per_api_key` entries.
    ///
    /// Opt-in escape hatch for deployments with either a much larger or much
    /// smaller expected principal cardinality than
    /// [`Self::DEFAULT_MAX_PRINCIPAL_ENTRIES`].
    #[must_use]
    pub fn with_max_principal_entries(mut self, max_principal_entries: usize) -> Self {
        self.max_principal_entries = max_principal_entries.max(1);
        self
    }

    /// Replace the configured per-dimension limits in place, preserving every
    /// sliding-window spend entry already accumulated.
    ///
    /// This is the fix for a config-reload-time budget reset: rebuilding a
    /// fresh [`InMemoryBudgetLedger`] to pick up new limits used to discard
    /// all `DashMap` spend entries along with the stale limits. Swapping only
    /// the `limits` pointer leaves `global`/`per_model`/`per_tenant`/
    /// `per_user`/`per_api_key` untouched, so month-to-date spend survives a
    /// reload.
    ///
    /// # Behavior on lowered limits
    ///
    /// If a limit drops below a dimension's already-accumulated spend, the
    /// next [`BudgetLedger::check`] call sees the (unchanged) spend against
    /// the new, lower limit and rejects immediately — existing spend is never
    /// forgiven. This is deliberate: silently resetting spend on a limit
    /// change is the exact failure mode this method exists to close.
    ///
    /// # Behavior on removed dimension keys
    ///
    /// A tenant/user/API-key dropped from `limits` (e.g. a virtual key
    /// removed from config) keeps its `DashMap` window entry — only the
    /// enforced cap disappears, so the key becomes unconstrained until a
    /// limit is configured for it again. Spend history is retained in case
    /// the same key is re-added later, rather than being silently discarded.
    pub fn update_limits(&self, limits: DimensionLimits) {
        self.limits.store(Arc::new(limits));
    }

    /// Build from a legacy [`BudgetConfig`].
    ///
    /// Global and per-model limits from `config` are mapped directly.
    /// Tenant, user, and API-key limits are left empty.
    /// The sliding window defaults to 30 days (a calendar month approximation).
    #[must_use]
    pub fn from_config(config: &BudgetConfig) -> Self {
        let limits = DimensionLimits {
            global: config.global_limit,
            per_model: config.model_limits.clone(),
            ..Default::default()
        };
        Self::new(limits, Duration::from_secs(30 * 24 * 3600))
    }

    /// Export a CSV of the current spend snapshot to `writer`.
    ///
    /// The CSV has two columns: `dimension,spend_usd`.  Each tracked key is
    /// emitted as one row.  Designed for cron-job extraction into a chargeback
    /// pipeline.
    ///
    /// # Errors
    ///
    /// Returns `Err(io::Error)` if writing to `writer` fails.
    pub fn export_csv(&self, mut writer: impl io::Write) -> io::Result<()> {
        let snap = self.snapshot();
        writeln!(writer, "dimension,spend_usd")?;
        writeln!(writer, "global,{}", snap.global_spend_usd)?;
        for (model, spend) in &snap.per_model {
            writeln!(writer, "model:{model},{spend}")?;
        }
        for (tenant, spend) in &snap.per_tenant {
            writeln!(writer, "tenant:{tenant},{spend}")?;
        }
        for (user, spend) in &snap.per_user {
            writeln!(writer, "user:{user},{spend}")?;
        }
        for (key, spend) in &snap.per_api_key {
            writeln!(writer, "api_key:{key},{spend}")?;
        }
        Ok(())
    }

    /// Reset all dimension counters to zero (useful for tests and manual overrides).
    pub fn reset(&self) {
        let now = SystemTime::now();
        let zero_secs = SystemTime::UNIX_EPOCH
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.global.spend_mc.store(0, Ordering::Relaxed);
        self.global.window_start_secs.store(zero_secs, Ordering::Relaxed);
        let _ = self.global.spend_usd(now);

        self.per_model.clear();
        self.per_tenant.clear();
        self.per_user.clear();
        self.per_api_key.clear();
    }

    fn entry_spend(map: &DashMap<String, WindowEntry>, key: &str, now: SystemTime) -> f64 {
        map.get(key).map(|e| e.spend_usd(now)).unwrap_or(0.0)
    }

    fn entry_add(map: &DashMap<String, WindowEntry>, key: &str, usd: f64, window: Duration, now: SystemTime) {
        map.entry(key.to_owned())
            .or_insert_with(|| WindowEntry::new(window))
            .add(usd, now);
    }

    /// Like [`Self::entry_add`], but bounds `map` at `max_entries` — used for
    /// `per_user` / `per_api_key`, whose keys are attacker-controlled.
    ///
    /// An already-tracked key always gets its spend recorded, regardless of
    /// how full `map` is: the cap only ever gates *new* keys, never starves an
    /// existing principal of accounting. See the [`InMemoryBudgetLedger`]
    /// doc comment for why a full map declines new principals instead of
    /// evicting a live one.
    fn entry_add_capped(
        map: &DashMap<String, WindowEntry>,
        dimension_name: &'static str,
        key: &str,
        usd: f64,
        window: Duration,
        now: SystemTime,
        max_entries: usize,
    ) {
        if !map.contains_key(key) && map.len() >= max_entries {
            Self::reclaim_zero_spend_entries(map, dimension_name, now);
        }

        if !map.contains_key(key) && map.len() >= max_entries {
            // ~keep Refuse to allocate tracking state for a brand-new principal rather
            // ~keep than evict a live one to make room — see the InMemoryBudgetLedger
            // ~keep doc comment. Global/per-model spend for this call is still recorded
            // ~keep by the caller; only this one dimension's enforcement is affected.
            tracing::warn!(
                dimension = dimension_name,
                cap = max_entries,
                "budget ledger at capacity; declining to track new principal, spend for this call was not recorded"
            );
            return;
        }

        map.entry(key.to_owned())
            .or_insert_with(|| WindowEntry::new(window))
            .add(usd, now);
    }

    /// Remove entries from `map` whose recorded spend reads as exactly zero,
    /// reclaiming capacity without ever discarding a live spend record.
    fn reclaim_zero_spend_entries(map: &DashMap<String, WindowEntry>, dimension_name: &'static str, now: SystemTime) {
        let mut evicted_live_spend_usd = 0.0_f64;

        map.retain(|_, entry| {
            let spend = entry.spend_usd(now);
            if spend > 0.0 {
                return true;
            }
            // ~keep Defensive invariant, not expected to ever be non-zero: this branch is
            // ~keep reached only when `spend <= 0.0`, so `evicted_live_spend_usd` should
            // ~keep never accumulate anything. Kept as a canary — if this ever fires, a
            // ~keep future change broke the "eviction never touches live spend" guarantee
            // ~keep this ledger relies on to avoid becoming a budget bypass.
            evicted_live_spend_usd += spend;
            false
        });

        if evicted_live_spend_usd > 0.0 {
            tracing::warn!(
                dimension = dimension_name,
                evicted_spend_usd = evicted_live_spend_usd,
                "budget ledger eviction reclaimed entries carrying non-zero spend; budget enforcement was weakened"
            );
        }
    }

    fn check_limit(spend: f64, limit: f64, dimension: BudgetDimension, key: &str) -> Option<BudgetVerdict> {
        if spend >= limit {
            Some(BudgetVerdict::Reject {
                reason: format!("{key} budget exceeded: spent ${spend:.6}, limit ${limit:.6}"),
                dimension,
            })
        } else {
            None
        }
    }
}

impl BudgetLedger for InMemoryBudgetLedger {
    fn record<'a>(&'a self, ctx: &'a CostRecordContext<'a>) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let now = ctx.timestamp;
            self.global.add(ctx.cost_usd, now);
            Self::entry_add(&self.per_model, ctx.model, ctx.cost_usd, self.window, now);
            if let Some(tenant) = ctx.tenant_id {
                Self::entry_add(&self.per_tenant, tenant, ctx.cost_usd, self.window, now);
            }
            if let Some(user) = ctx.user_id {
                Self::entry_add_capped(
                    &self.per_user,
                    "user",
                    user,
                    ctx.cost_usd,
                    self.window,
                    now,
                    self.max_principal_entries,
                );
            }
            if let Some(key) = ctx.api_key_id {
                Self::entry_add_capped(
                    &self.per_api_key,
                    "api_key",
                    key,
                    ctx.cost_usd,
                    self.window,
                    now,
                    self.max_principal_entries,
                );
            }

            #[cfg(feature = "otel")]
            {
                use super::metrics;
                metrics::record_budget_spend(
                    ctx.model,
                    ctx.provider,
                    ctx.tenant_id,
                    ctx.user_id,
                    ctx.api_key_id,
                    ctx.cost_usd,
                );
            }
        })
    }

    fn check<'a>(&'a self, ctx: &'a CostCheckContext<'a>) -> Pin<Box<dyn Future<Output = BudgetVerdict> + Send + 'a>> {
        Box::pin(async move {
            let now = ctx.timestamp;
            // ~keep load_full (owned Arc) rather than load (thread-local Guard) because this
            // ~keep async block must produce a Send future; an owned Arc<DimensionLimits> is
            // ~keep unambiguously Send, avoiding any question about Guard's Send-ness here.
            let limits = self.limits.load_full();

            if let Some(limit) = limits.global {
                let spend = self.global.spend_usd(now);
                if let Some(v) = Self::check_limit(spend, limit, BudgetDimension::Global, "global") {
                    return v;
                }
            }

            if let Some(&limit) = limits.per_model.get(ctx.model) {
                let spend = Self::entry_spend(&self.per_model, ctx.model, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::Model(ctx.model.to_owned()),
                    &format!("model:{}", ctx.model),
                ) {
                    return v;
                }
            }

            if let Some(tenant) = ctx.tenant_id
                && let Some(&limit) = limits.per_tenant.get(tenant)
            {
                let spend = Self::entry_spend(&self.per_tenant, tenant, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::Tenant(tenant.to_owned()),
                    &format!("tenant:{tenant}"),
                ) {
                    return v;
                }
            }

            if let Some(user) = ctx.user_id
                && let Some(&limit) = limits.per_user.get(user)
            {
                let spend = Self::entry_spend(&self.per_user, user, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::User(user.to_owned()),
                    &format!("user:{user}"),
                ) {
                    return v;
                }
            }

            if let Some(key) = ctx.api_key_id
                && let Some(&limit) = limits.per_api_key.get(key)
            {
                let spend = Self::entry_spend(&self.per_api_key, key, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::ApiKey(key.to_owned()),
                    &format!("api_key:{key}"),
                ) {
                    return v;
                }
            }

            BudgetVerdict::Allow
        })
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let now = SystemTime::now();
        let limits = self.limits.load();

        let global_spend_usd = self.global.spend_usd(now);

        let per_model = self
            .per_model
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_tenant = self
            .per_tenant
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_user = self
            .per_user
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_api_key = self
            .per_api_key
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        BudgetSnapshot {
            global_spend_usd,
            per_model,
            per_tenant,
            per_user,
            per_api_key,
            limit_global: limits.global,
            limits_per_user: limits.per_user.clone(),
            limits_per_api_key: limits.per_api_key.clone(),
            limits_per_tenant: limits.per_tenant.clone(),
        }
    }
}

/// Advise the hedge layer wiring whether to issue a speculative duplicate
/// request for the given pre-flight context.
///
/// Returns `false` (suppress hedging) when issuing a second speculative copy
/// of the request would push any budget dimension over its limit.  The hedge
/// wiring callsite should consult this before enabling the hedge policy.
///
/// # Parameters
///
/// * `ledger` — the live budget ledger to consult.
/// * `ctx` — pre-flight context identifying the user / key / model.
/// * `estimated_cost_usd` — expected cost of **one** copy of the request.  A
///   hedged call doubles this cost, so the check uses `2 × estimated_cost`.
/// * `safety_margin_pct` — fraction of each limit to reserve before blocking
///   hedging (e.g. `0.10` stops hedging when spend would exceed 90 % of the
///   limit).  Must be in `[0.0, 1.0)`.
///
/// # Logic
///
/// For each budget dimension that is both tracked in the ledger snapshot and
/// has a configured limit on `ledger`, hedging is suppressed when:
///
/// ```text
/// current_spend + 2 × estimated_cost  >=  limit × (1 − safety_margin_pct)
/// ```
///
/// Returns `true` only if **all** applicable dimensions have sufficient
/// headroom for two copies of the call.
#[must_use]
pub fn should_hedge<L: BudgetLedger>(
    ledger: &L,
    ctx: &CostCheckContext<'_>,
    estimated_cost_usd: f64,
    safety_margin_pct: f64,
) -> bool {
    let snap = ledger.snapshot();
    let hedge_cost = 2.0 * estimated_cost_usd;
    let margin = safety_margin_pct.clamp(0.0, 0.999);

    let has_headroom = |spend: f64, limit: f64| -> bool {
        let effective_limit = limit * (1.0 - margin);
        spend + hedge_cost < effective_limit
    };

    if let Some(global_limit) = snap.limit_global
        && !has_headroom(snap.global_spend_usd, global_limit)
    {
        return false;
    }

    if let Some(user) = ctx.user_id
        && let Some(&user_limit) = snap.limits_per_user.get(user)
    {
        let user_spend = snap.per_user.get(user).copied().unwrap_or(0.0);
        if !has_headroom(user_spend, user_limit) {
            return false;
        }
    }

    if let Some(key) = ctx.api_key_id
        && let Some(&key_limit) = snap.limits_per_api_key.get(key)
    {
        let key_spend = snap.per_api_key.get(key).copied().unwrap_or(0.0);
        if !has_headroom(key_spend, key_limit) {
            return false;
        }
    }

    if let Some(tenant) = ctx.tenant_id
        && let Some(&tenant_limit) = snap.limits_per_tenant.get(tenant)
    {
        let tenant_spend = snap.per_tenant.get(tenant).copied().unwrap_or(0.0);
        if !has_headroom(tenant_spend, tenant_limit) {
            return false;
        }
    }

    true
}

/// Derive the OpenTelemetry GenAI `gen_ai.system` provider prefix from a
/// model identifier (e.g. `"openai"` from `"openai/gpt-4o"`), matching the
/// convention [`crate::tower::tracing::TracingService`] uses. Returns `""`
/// when the model has no `<provider>/` prefix.
pub(crate) fn provider_of(model: &str) -> &str {
    model.split_once('/').map_or("", |(prefix, _)| prefix)
}

/// Extract the end-user identifier from a `Chat`/`ChatStream`/`Embed`
/// request's `user` field.
///
/// Returns `None` for request kinds that carry no `user` field (image,
/// audio, moderation, etc.) or when the field is unset.
pub(crate) fn user_id_of(req: &LlmRequest) -> Option<&str> {
    match &req.kind {
        LlmRequestKind::Chat(r) | LlmRequestKind::ChatStream(r) => r.user.as_deref(),
        LlmRequestKind::Embed(r) => r.user.as_deref(),
        _ => None,
    }
}

/// Tower [`Layer`] that enforces and records spend via a pluggable
/// [`BudgetLedger`], adding per-tenant / per-user / per-API-key budget
/// dimensions on top of what [`BudgetLayer`] provides.
///
/// # Why this layer exists
///
/// [`BudgetLedger`] (and its default [`InMemoryBudgetLedger`] implementation)
/// is a fully-built, independently tested trait for multi-dimensional spend
/// tracking — but nothing in the Tower stack ever constructed a `Service`
/// around it: [`BudgetLayer`] only ever touches the simpler [`BudgetState`]
/// atomic counters, which track just the global and per-model dimensions.
/// `BudgetLedgerLayer` is the missing wiring. Compose it alongside (or
/// instead of) [`BudgetLayer`] to get tenant/user-scoped enforcement and
/// recording.
///
/// # Context extraction
///
/// - `model` / `provider` come from [`LlmRequest::model`] (`provider` is the
///   `<provider>/` prefix, see [`provider_of`]).
/// - `tenant_id` comes from [`LlmRequest::tenant_id`].
/// - `user_id` comes from the `user` field on `Chat`/`ChatStream`/`Embed`
///   requests (see [`user_id_of`]).
/// - `api_key_id` is always `None` — [`LlmRequest`] does not currently carry
///   an API-key identifier anywhere in its public surface. This dimension is
///   therefore inert (never checked, never recorded) until a caller extends
///   `LlmRequest` (or a wrapping layer) with one.
///
/// # Streaming
///
/// Like [`BudgetLayer`], the pre-flight check applies uniformly to every
/// request kind. Post-response recording uses
/// [`observe_stream_usage`][crate::tower::cost::observe_stream_usage] so
/// `ChatStream` responses are recorded once the stream completes instead of
/// being silently skipped — recording happens on a spawned task since
/// [`BudgetLedger::record`] is async but the stream's completion callback is
/// synchronous.
#[cfg_attr(alef, alef(skip))]
pub struct BudgetLedgerLayer<L: BudgetLedger> {
    ledger: Arc<L>,
    enforcement: Enforcement,
}

impl<L: BudgetLedger> BudgetLedgerLayer<L> {
    /// Create a new layer backed by `ledger`.
    // ~keep Redundant with the `alef(skip)` on the type itself: alef does not propagate a
    // ~keep type-level skip to that type's impl blocks, so it still reports this generic
    // ~keep constructor as an unrepresentable public item and fails generation outright.
    #[cfg_attr(alef, alef(skip))]
    #[must_use]
    pub fn new(ledger: Arc<L>, enforcement: Enforcement) -> Self {
        Self { ledger, enforcement }
    }
}

impl<L: BudgetLedger, S> Layer<S> for BudgetLedgerLayer<L> {
    type Service = BudgetLedgerService<L, S>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetLedgerService {
            inner,
            ledger: Arc::clone(&self.ledger),
            enforcement: self.enforcement,
        }
    }
}

/// Tower service produced by [`BudgetLedgerLayer`].
#[cfg_attr(alef, alef(skip))]
pub struct BudgetLedgerService<L: BudgetLedger, S> {
    inner: S,
    ledger: Arc<L>,
    enforcement: Enforcement,
}

impl<L: BudgetLedger, S: Clone> Clone for BudgetLedgerService<L, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            ledger: Arc::clone(&self.ledger),
            enforcement: self.enforcement,
        }
    }
}

impl<L, S> Service<LlmRequest> for BudgetLedgerService<L, S>
where
    L: BudgetLedger,
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
        let model = req.model().unwrap_or("unknown").to_owned();
        let provider = provider_of(&model).to_owned();
        let tenant_id = req.tenant_id().map(|t| t.as_ref().to_owned());
        let user_id = user_id_of(&req).map(str::to_owned);
        let ledger = Arc::clone(&self.ledger);
        let enforcement = self.enforcement;

        // ~keep The pre-flight check is async (ledger-backed), so it must run inside the returned future,
        // ~keep before `inner.call(req)`. Consume the polled-ready instance and leave a fresh standby clone,
        // ~keep matching the Tower contract other layers in this crate follow (e.g. CacheService, HedgeService).
        let standby = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, standby);

        Box::pin(async move {
            let check_ctx = CostCheckContext {
                model: &model,
                provider: &provider,
                tenant_id: tenant_id.as_deref(),
                user_id: user_id.as_deref(),
                api_key_id: None,
                timestamp: SystemTime::now(),
            };

            if enforcement == Enforcement::Hard
                && let BudgetVerdict::Reject { reason, dimension } = ledger.check(&check_ctx).await
            {
                let model_field = match &dimension {
                    BudgetDimension::Model(m) => Some(m.clone()),
                    _ => None,
                };
                return Err(LiterLlmError::BudgetExceeded {
                    message: reason,
                    model: model_field,
                });
            }

            let resp = inner.call(req).await?;

            match resp {
                LlmResponse::ChatStream(stream) => {
                    let ledger_for_completion = Arc::clone(&ledger);
                    let model_c = model.clone();
                    let provider_c = provider.clone();
                    let tenant_c = tenant_id.clone();
                    let user_c = user_id.clone();
                    let wrapped = observe_stream_usage(stream, move |usage| {
                        let Some(usage) = usage else { return };
                        let Some(usd) = cost::completion_cost(&model_c, usage.prompt_tokens, usage.completion_tokens)
                        else {
                            return;
                        };
                        // ~keep BudgetLedger::record is async but this callback runs synchronously inside
                        // ~keep poll_next; spawn so recording never blocks the caller draining the stream.
                        // ~keep Guard against "no current runtime" if the stream is drained outside Tokio (matches
                        // ~keep the tokio::runtime::Handle::try_current() convention used by hooks.rs's CancellationGuard).
                        let record = async move {
                            let ctx = CostRecordContext {
                                model: &model_c,
                                provider: &provider_c,
                                tenant_id: tenant_c.as_deref(),
                                user_id: user_c.as_deref(),
                                api_key_id: None,
                                cost_usd: usd,
                                tokens_in: usage.prompt_tokens,
                                tokens_out: usage.completion_tokens,
                                timestamp: SystemTime::now(),
                            };
                            ledger_for_completion.record(&ctx).await;
                        };
                        if let Ok(handle) = tokio::runtime::Handle::try_current() {
                            handle.spawn(record);
                        } else {
                            tracing::warn!(
                                "budget ledger: no Tokio runtime available to record streamed usage; spend was not recorded"
                            );
                        }
                    });
                    Ok(LlmResponse::ChatStream(wrapped))
                }
                other => {
                    if let Some(usage) = other.usage()
                        && let Some(usd) = cost::completion_cost(&model, usage.prompt_tokens, usage.completion_tokens)
                    {
                        let ctx = CostRecordContext {
                            model: &model,
                            provider: &provider,
                            tenant_id: tenant_id.as_deref(),
                            user_id: user_id.as_deref(),
                            api_key_id: None,
                            cost_usd: usd,
                            tokens_in: usage.prompt_tokens,
                            tokens_out: usage.completion_tokens,
                            timestamp: SystemTime::now(),
                        };
                        ledger.record(&ctx).await;
                    }
                    Ok(other)
                }
            }
        })
    }
}

/// How budget limits are enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Enforcement {
    /// Reject requests that would exceed the budget with
    /// [`LiterLlmError::BudgetExceeded`].
    Hard,
    /// Allow requests through but emit a `tracing::warn!` when the budget is
    /// exceeded.
    Soft,
}

/// Configuration for budget enforcement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetConfig {
    /// Maximum total spend across all models, in USD.  `None` means unlimited.
    pub global_limit: Option<f64>,
    /// Per-model spending limits in USD.  Models not listed here are only
    /// constrained by `global_limit`.
    pub model_limits: HashMap<String, f64>,
    /// Whether to reject requests or merely warn when a limit is exceeded.
    pub enforcement: Enforcement,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            global_limit: None,
            model_limits: HashMap::new(),
            enforcement: Enforcement::Hard,
        }
    }
}

/// Shared, thread-safe budget accumulator.
///
/// All values are stored in **microcents** (USD * 1_000_000) as `AtomicU64` to
/// avoid floating-point atomics while retaining sub-cent precision.
#[derive(Debug)]
pub struct BudgetState {
    /// Total spend across all models (microcents).
    global_spend: AtomicU64,
    /// Per-model spend (microcents).
    model_spend: DashMap<String, AtomicU64>,
}

impl BudgetState {
    /// Create a new, zeroed budget state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            global_spend: AtomicU64::new(0),
            model_spend: DashMap::new(),
        }
    }

    /// Return the total global spend in USD.
    #[must_use]
    pub fn global_spend(&self) -> f64 {
        microcents_to_usd(self.global_spend.load(Ordering::Relaxed))
    }

    /// Return the spend for a specific model in USD, or `0.0` if the model has
    /// not been seen.
    #[must_use]
    pub fn model_spend(&self, model: &str) -> f64 {
        self.model_spend
            .get(model)
            .map(|v| microcents_to_usd(v.load(Ordering::Relaxed)))
            .unwrap_or(0.0)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.global_spend.store(0, Ordering::Relaxed);
        self.model_spend.clear();
    }

    /// Add `usd` to the global and per-model counters.
    fn record(&self, model: &str, usd: f64) {
        let mc = usd_to_microcents(usd);
        self.global_spend.fetch_add(mc, Ordering::Relaxed);
        self.model_spend
            .entry(model.to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(mc, Ordering::Relaxed);
    }
}

#[cfg_attr(alef, alef(skip))]
impl Default for BudgetState {
    fn default() -> Self {
        Self::new()
    }
}

fn usd_to_microcents(usd: f64) -> u64 {
    if usd <= 0.0 {
        return 0;
    }
    (usd * 1_000_000.0).round() as u64
}

fn microcents_to_usd(mc: u64) -> f64 {
    mc as f64 / 1_000_000.0
}

/// Tower [`Layer`] that enforces spending budgets.
#[cfg_attr(alef, alef(skip))]
pub struct BudgetLayer {
    config: BudgetConfig,
    state: Arc<BudgetState>,
}

#[cfg_attr(alef, alef(skip))]
impl BudgetLayer {
    /// Create a new budget layer with the given configuration and shared state.
    ///
    /// The caller retains an `Arc<BudgetState>` for runtime introspection
    /// (e.g. dashboard queries, manual resets).
    #[must_use]
    pub fn new(config: BudgetConfig, state: Arc<BudgetState>) -> Self {
        Self { config, state }
    }
}

impl<S> Layer<S> for BudgetLayer {
    type Service = BudgetService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetService {
            inner,
            config: self.config.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

/// Tower service produced by [`BudgetLayer`].
#[cfg_attr(alef, alef(skip))]
pub struct BudgetService<S> {
    inner: S,
    config: BudgetConfig,
    state: Arc<BudgetState>,
}

impl<S: Clone> Clone for BudgetService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> Service<LlmRequest> for BudgetService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = LiterLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = LiterLlmError;
    type Future = BoxFuture<'static, Result<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let model = req.model().unwrap_or("unknown").to_owned();
        let config = self.config.clone();
        let state = Arc::clone(&self.state);

        if config.enforcement == Enforcement::Hard
            && let Some(err) = check_budget(&config, &state, &model)
        {
            return Box::pin(async move { Err(err) });
        }

        let fut = self.inner.call(req);

        Box::pin(async move {
            let resp = fut.await?;

            match resp {
                // ~keep LlmResponse::usage() always returns None for ChatStream (usage isn't known until the
                // ~keep stream completes), so recording must happen in the stream's completion callback instead.
                LlmResponse::ChatStream(stream) => {
                    let model_for_completion = model.clone();
                    let state_for_completion = Arc::clone(&state);
                    let config_for_completion = config.clone();
                    let wrapped = observe_stream_usage(stream, move |usage| {
                        record_usage(
                            &config_for_completion,
                            &state_for_completion,
                            &model_for_completion,
                            usage.as_ref(),
                        );
                    });
                    Ok(LlmResponse::ChatStream(wrapped))
                }
                other => {
                    record_usage(&config, &state, &model, other.usage());
                    Ok(other)
                }
            }
        })
    }
}

/// Compute the cost of `usage` and record it against `state`, emitting soft
/// enforcement warnings if configured. No-op when `usage` is `None` or the
/// model has no pricing data.
///
/// Shared by the non-streaming response path (usage known immediately) and
/// the `ChatStream` completion callback (usage only known once the stream is
/// fully consumed).
fn record_usage(config: &BudgetConfig, state: &BudgetState, model: &str, usage: Option<&Usage>) {
    let Some(usage) = usage else { return };
    let Some(usd) = cost::completion_cost(model, usage.prompt_tokens, usage.completion_tokens) else {
        return;
    };

    state.record(model, usd);

    if config.enforcement == Enforcement::Soft {
        emit_soft_warnings(config, state, model);
    }
}

/// Check whether the current spend exceeds any configured limit.  Returns
/// `Some(LiterLlmError)` if the budget is exceeded under hard enforcement.
///
/// **Concurrency note:** This check is best-effort under concurrent load.
/// Because the budget is checked (read) before the request and recorded
/// (write) after the response, concurrent requests may all pass the
/// pre-flight check before any of them record their cost.  This means
/// hard enforcement can slightly overshoot the configured limit by up to
/// `N * max_single_request_cost` where `N` is the number of concurrent
/// in-flight requests.  For strict dollar-accurate enforcement, use an
/// external budget service with transactional semantics.
fn check_budget(config: &BudgetConfig, state: &BudgetState, model: &str) -> Option<LiterLlmError> {
    if let Some(limit) = config.global_limit
        && state.global_spend() >= limit
    {
        return Some(LiterLlmError::BudgetExceeded {
            message: format!(
                "global budget exceeded: spent ${:.6}, limit ${:.6}",
                state.global_spend(),
                limit,
            ),
            model: None,
        });
    }

    if let Some(&limit) = config.model_limits.get(model)
        && state.model_spend(model) >= limit
    {
        return Some(LiterLlmError::BudgetExceeded {
            message: format!(
                "model {model} budget exceeded: spent ${:.6}, limit ${:.6}",
                state.model_spend(model),
                limit,
            ),
            model: Some(model.to_owned()),
        });
    }

    None
}

/// Emit `tracing::warn!` messages for any exceeded limits (soft mode).
fn emit_soft_warnings(config: &BudgetConfig, state: &BudgetState, model: &str) {
    if let Some(limit) = config.global_limit
        && state.global_spend() >= limit
    {
        tracing::warn!(
            spend = state.global_spend(),
            limit,
            "global budget exceeded (soft enforcement)"
        );
    }

    if let Some(&limit) = config.model_limits.get(model)
        && state.model_spend(model) >= limit
    {
        tracing::warn!(
            model,
            spend = state.model_spend(model),
            limit,
            "model budget exceeded (soft enforcement)"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::tower::service::LlmService;
    use crate::tower::tests_common::{MockClient, chat_req};
    use crate::tower::types::LlmRequest;

    /// Helper: build a budget layer + service with the given config.
    fn build_service(config: BudgetConfig, state: Arc<BudgetState>) -> BudgetService<LlmService<MockClient>> {
        let layer = BudgetLayer::new(config, state);
        let inner = LlmService::new(MockClient::ok());
        layer.layer(inner)
    }

    #[tokio::test]
    async fn hard_enforcement_rejects_when_global_limit_exceeded() {
        let state = Arc::new(BudgetState::new());
        state.global_spend.store(usd_to_microcents(10.0), Ordering::Relaxed);

        let config = BudgetConfig {
            global_limit: Some(5.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };

        let mut svc = build_service(config, state);
        let err = svc
            .call(LlmRequest::Chat(chat_req("gpt-4")))
            .await
            .expect_err("should reject over-budget request");
        assert!(matches!(err, LiterLlmError::BudgetExceeded { .. }));
    }

    #[tokio::test]
    async fn hard_enforcement_rejects_when_model_limit_exceeded() {
        let state = Arc::new(BudgetState::new());
        state
            .model_spend
            .entry("gpt-4".to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .store(usd_to_microcents(2.0), Ordering::Relaxed);

        let mut limits = HashMap::new();
        limits.insert("gpt-4".into(), 1.0);

        let config = BudgetConfig {
            global_limit: None,
            model_limits: limits,
            enforcement: Enforcement::Hard,
        };

        let mut svc = build_service(config, state);
        let err = svc
            .call(LlmRequest::Chat(chat_req("gpt-4")))
            .await
            .expect_err("should reject over-budget model request");

        match &err {
            LiterLlmError::BudgetExceeded { model, .. } => {
                assert_eq!(model.as_deref(), Some("gpt-4"));
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn hard_enforcement_allows_requests_under_limit() {
        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(100.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };

        let mut svc = build_service(config, state);
        let resp = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "request under budget should succeed");
    }

    #[tokio::test]
    async fn soft_enforcement_allows_requests_over_global_limit() {
        let state = Arc::new(BudgetState::new());
        state.global_spend.store(usd_to_microcents(100.0), Ordering::Relaxed);

        let config = BudgetConfig {
            global_limit: Some(5.0),
            enforcement: Enforcement::Soft,
            ..Default::default()
        };

        let mut svc = build_service(config, state);
        let resp = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "soft mode should never reject");
    }

    #[tokio::test]
    async fn soft_enforcement_allows_requests_over_model_limit() {
        let state = Arc::new(BudgetState::new());
        state
            .model_spend
            .entry("gpt-4".to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .store(usd_to_microcents(10.0), Ordering::Relaxed);

        let mut limits = HashMap::new();
        limits.insert("gpt-4".into(), 1.0);

        let config = BudgetConfig {
            global_limit: None,
            model_limits: limits,
            enforcement: Enforcement::Soft,
        };

        let mut svc = build_service(config, state);
        let resp = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(resp.is_ok(), "soft mode should never reject");
    }

    #[tokio::test]
    async fn accumulates_cost_after_response() {
        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(100.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };

        let mut svc = build_service(config, Arc::clone(&state));
        svc.call(LlmRequest::Chat(chat_req("gpt-4")))
            .await
            .expect("service call should not fail");

        assert!(state.global_spend() > 0.0, "global spend should be recorded");
        assert!(state.model_spend("gpt-4") > 0.0, "model spend should be recorded");
    }

    #[tokio::test]
    async fn per_model_limits_are_independent() {
        let state = Arc::new(BudgetState::new());
        state
            .model_spend
            .entry("gpt-4".to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .store(usd_to_microcents(5.0), Ordering::Relaxed);

        let mut limits = HashMap::new();
        limits.insert("gpt-4".into(), 1.0);

        let config = BudgetConfig {
            global_limit: None,
            model_limits: limits,
            enforcement: Enforcement::Hard,
        };

        let mut svc = build_service(config, state);

        let err = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(err.is_err(), "gpt-4 should be rejected");

        let ok = svc.call(LlmRequest::Chat(chat_req("gpt-3.5-turbo"))).await;
        assert!(ok.is_ok(), "gpt-3.5-turbo should not be limited");
    }

    #[tokio::test]
    async fn reset_clears_all_counters() {
        let state = Arc::new(BudgetState::new());
        state.global_spend.store(usd_to_microcents(50.0), Ordering::Relaxed);
        state
            .model_spend
            .entry("gpt-4".to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .store(usd_to_microcents(25.0), Ordering::Relaxed);

        assert!(state.global_spend() > 0.0);
        assert!(state.model_spend("gpt-4") > 0.0);

        state.reset();

        assert_eq!(state.global_spend(), 0.0, "global spend should be zero after reset");
        assert_eq!(
            state.model_spend("gpt-4"),
            0.0,
            "model spend should be zero after reset"
        );
    }

    #[tokio::test]
    async fn reset_allows_previously_blocked_requests() {
        let state = Arc::new(BudgetState::new());
        state.global_spend.store(usd_to_microcents(10.0), Ordering::Relaxed);

        let config = BudgetConfig {
            global_limit: Some(5.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };

        let mut svc = build_service(config, Arc::clone(&state));

        let err = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(err.is_err());

        state.reset();
        let ok = svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await;
        assert!(ok.is_ok(), "should succeed after reset");
    }

    #[tokio::test]
    async fn unlimited_config_allows_all_requests() {
        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig::default();

        let mut svc = build_service(config, state);
        for _ in 0..20 {
            assert!(svc.call(LlmRequest::Chat(chat_req("gpt-4"))).await.is_ok());
        }
    }

    #[tokio::test]
    async fn propagates_inner_service_errors() {
        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(100.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };

        let layer = BudgetLayer::new(config, state);
        let inner = LlmService::new(MockClient::failing_timeout());
        let mut svc = layer.layer(inner);

        let err = svc
            .call(LlmRequest::Chat(chat_req("gpt-4")))
            .await
            .expect_err("should propagate inner error");
        assert!(matches!(err, LiterLlmError::Timeout));
    }

    #[tokio::test]
    async fn budget_ledger_records_per_key_and_per_user() {
        let limits = DimensionLimits::default();
        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        let ctx1 = CostRecordContext {
            model: "gpt-4",
            provider: "openai",
            tenant_id: Some("acme"),
            user_id: Some("alice"),
            api_key_id: Some("key-1"),
            cost_usd: 0.10,
            tokens_in: 1000,
            tokens_out: 500,
            timestamp: SystemTime::now(),
        };
        ledger.record(&ctx1).await;

        let ctx2 = CostRecordContext {
            model: "gpt-4",
            provider: "openai",
            tenant_id: Some("acme"),
            user_id: Some("bob"),
            api_key_id: Some("key-2"),
            cost_usd: 0.20,
            tokens_in: 2000,
            tokens_out: 1000,
            timestamp: SystemTime::now(),
        };
        ledger.record(&ctx2).await;

        let snap = ledger.snapshot();
        assert!(
            (snap.global_spend_usd - 0.30).abs() < 1e-9,
            "global: {}",
            snap.global_spend_usd
        );
        assert!((snap.per_model["gpt-4"] - 0.30).abs() < 1e-9);
        assert!((snap.per_tenant["acme"] - 0.30).abs() < 1e-9);
        assert!((snap.per_user["alice"] - 0.10).abs() < 1e-9);
        assert!((snap.per_user["bob"] - 0.20).abs() < 1e-9);
        assert!((snap.per_api_key["key-1"] - 0.10).abs() < 1e-9);
        assert!((snap.per_api_key["key-2"] - 0.20).abs() < 1e-9);
    }

    /// `per_user` is keyed by the request's free-text, attacker-controlled
    /// `user` field, so without a cap a caller grows the map without bound
    /// simply by varying that field per request. This pins the cap: recording
    /// spend for far more distinct users than the configured cap must never
    /// let `per_user` grow past it.
    #[tokio::test]
    async fn budget_ledger_per_user_map_is_bounded_by_max_principal_entries() {
        const CAP: usize = 8;
        let ledger = InMemoryBudgetLedger::new(DimensionLimits::default(), Duration::from_secs(3600))
            .with_max_principal_entries(CAP);

        for i in 0..CAP * 10 {
            let user = format!("user-{i}");
            ledger
                .record(&CostRecordContext {
                    model: "gpt-4",
                    provider: "openai",
                    tenant_id: None,
                    user_id: Some(&user),
                    api_key_id: None,
                    cost_usd: 0.01,
                    tokens_in: 10,
                    tokens_out: 5,
                    timestamp: SystemTime::now(),
                })
                .await;
        }

        let len = ledger.per_user.len();
        assert!(
            len <= CAP,
            "per_user must stay within its cap; held {len} entries with a cap of {CAP}"
        );
    }

    /// Same regression as above, but for `per_api_key` — the other
    /// caller-supplied, unbounded-cardinality dimension.
    #[tokio::test]
    async fn budget_ledger_per_api_key_map_is_bounded_by_max_principal_entries() {
        const CAP: usize = 8;
        let ledger = InMemoryBudgetLedger::new(DimensionLimits::default(), Duration::from_secs(3600))
            .with_max_principal_entries(CAP);

        for i in 0..CAP * 10 {
            let key = format!("key-{i}");
            ledger
                .record(&CostRecordContext {
                    model: "gpt-4",
                    provider: "openai",
                    tenant_id: None,
                    user_id: None,
                    api_key_id: Some(&key),
                    cost_usd: 0.01,
                    tokens_in: 10,
                    tokens_out: 5,
                    timestamp: SystemTime::now(),
                })
                .await;
        }

        let len = ledger.per_api_key.len();
        assert!(
            len <= CAP,
            "per_api_key must stay within its cap; held {len} entries with a cap of {CAP}"
        );
    }

    /// Zero-spend entries carry no enforcement value, so reaching the cap
    /// must first reclaim them before declining to track a new principal —
    /// the same "reclaim before giving up" shape as
    /// `ClassifierVerdictCache::put_cached`, but keyed on recorded spend
    /// rather than TTL expiry.
    #[tokio::test]
    async fn budget_ledger_cap_reclaims_zero_spend_entries_before_declining_new_principals() {
        const CAP: usize = 4;
        let ledger = InMemoryBudgetLedger::new(DimensionLimits::default(), Duration::from_secs(3600))
            .with_max_principal_entries(CAP);

        for i in 0..CAP {
            let user = format!("zero-spend-{i}");
            ledger
                .record(&CostRecordContext {
                    model: "gpt-4",
                    provider: "openai",
                    tenant_id: None,
                    user_id: Some(&user),
                    api_key_id: None,
                    cost_usd: 0.0,
                    tokens_in: 0,
                    tokens_out: 0,
                    timestamp: SystemTime::now(),
                })
                .await;
        }
        assert_eq!(ledger.per_user.len(), CAP, "setup should fill the ledger to its cap");

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("real-spender"),
                api_key_id: None,
                cost_usd: 5.0,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        assert_eq!(
            ledger.per_user.len(),
            1,
            "the zero-spend entries must be reclaimed, leaving only the new spender"
        );
        assert!((ledger.snapshot().per_user["real-spender"] - 5.0).abs() < 1e-9);
    }

    /// The core safety property this cap must uphold: filling the ledger to
    /// capacity with unrelated new principals must never reset (via eviction)
    /// a principal's already-recorded, non-zero spend. If it did, the
    /// bounded-memory fix would itself become a budget-bypass — a caller
    /// could spend heavily, get displaced by cardinality pressure from other
    /// keys, and come back with a silently fresh budget.
    #[tokio::test]
    async fn budget_ledger_cap_never_resets_an_existing_principals_spend() {
        const CAP: usize = 4;
        let ledger = InMemoryBudgetLedger::new(DimensionLimits::default(), Duration::from_secs(3600))
            .with_max_principal_entries(CAP);

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 42.0,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;
        assert!((ledger.snapshot().per_user["alice"] - 42.0).abs() < 1e-9);

        // Flood with far more distinct, non-zero-spend principals than the cap allows.
        for i in 0..CAP * 10 {
            let user = format!("flood-{i}");
            ledger
                .record(&CostRecordContext {
                    model: "gpt-4",
                    provider: "openai",
                    tenant_id: None,
                    user_id: Some(&user),
                    api_key_id: None,
                    cost_usd: 1.0,
                    tokens_in: 10,
                    tokens_out: 5,
                    timestamp: SystemTime::now(),
                })
                .await;
        }

        let spend_after_flood = ledger.snapshot().per_user.get("alice").copied();
        assert!(
            matches!(spend_after_flood, Some(v) if (v - 42.0).abs() < 1e-9),
            "alice's already-recorded spend must survive cap pressure from other principals, got {spend_after_flood:?}"
        );

        // An already-tracked principal must keep accumulating spend even while
        // the ledger sits at capacity -- the cap must gate only new principals.
        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 8.0,
                tokens_in: 10,
                tokens_out: 5,
                timestamp: SystemTime::now(),
            })
            .await;
        assert!(
            (ledger.snapshot().per_user["alice"] - 50.0).abs() < 1e-9,
            "an already-tracked principal must keep accumulating spend once the ledger is at capacity"
        );
    }

    /// Regression for the reset-on-reload bug: `update_limits` must swap only
    /// the caps, not the accumulated spend. Before the fix, the only way to
    /// change limits was to rebuild the whole ledger, which zeroed every
    /// `DashMap` entry along with it — this test fails against that
    /// rebuild-the-ledger behaviour because the post-update spend would read
    /// back as 0.0 instead of the pre-update amount.
    #[tokio::test]
    async fn update_limits_preserves_accumulated_spend() {
        let mut limits = DimensionLimits::default();
        limits.per_user.insert("alice".to_owned(), 100.0);
        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 7.50,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;
        assert!((ledger.snapshot().per_user["alice"] - 7.50).abs() < 1e-9);

        let mut new_limits = DimensionLimits::default();
        new_limits.per_user.insert("alice".to_owned(), 200.0);
        ledger.update_limits(new_limits);

        let snap = ledger.snapshot();
        assert!(
            (snap.per_user["alice"] - 7.50).abs() < 1e-9,
            "spend must survive update_limits, got {}",
            snap.per_user["alice"]
        );
        assert_eq!(snap.limits_per_user.get("alice"), Some(&200.0));

        let verdict = ledger
            .check(&CostCheckContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                timestamp: SystemTime::now(),
            })
            .await;
        assert!(
            matches!(verdict, BudgetVerdict::Allow),
            "spend $7.50 is well under the new $200 limit, expected Allow, got {verdict:?}"
        );
    }

    /// Lowering a limit below already-accumulated spend must reject the very
    /// next request rather than silently forgiving the existing spend — the
    /// opposite failure mode (forgiving spend) is the reset-on-reload bug in
    /// disguise.
    #[tokio::test]
    async fn update_limits_lowering_below_spend_rejects_immediately() {
        let mut limits = DimensionLimits::default();
        limits.per_user.insert("alice".to_owned(), 100.0);
        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 5.0,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        let mut lowered = DimensionLimits::default();
        lowered.per_user.insert("alice".to_owned(), 1.0);
        ledger.update_limits(lowered);

        let verdict = ledger
            .check(&CostCheckContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                timestamp: SystemTime::now(),
            })
            .await;
        match verdict {
            BudgetVerdict::Reject { dimension, .. } => {
                assert!(matches!(dimension, BudgetDimension::User(ref u) if u == "alice"));
            }
            BudgetVerdict::Allow => panic!("lowering the limit below existing spend must reject, got Allow"),
        }
    }

    /// A tenant dropped from the limits map on reload must keep its spend
    /// history: enforcement lifts (no configured cap means no rejection) but
    /// the `DashMap` window entry is retained, so re-adding the same tenant
    /// later does not silently reset it to zero.
    #[tokio::test]
    async fn update_limits_retains_spend_for_removed_tenant() {
        let mut limits = DimensionLimits::default();
        limits.per_tenant.insert("acme".to_owned(), 10.0);
        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: Some("acme"),
                user_id: None,
                api_key_id: None,
                cost_usd: 3.0,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        // ~keep "acme" is absent from the new limits map entirely, simulating removal from config.
        ledger.update_limits(DimensionLimits::default());

        let verdict = ledger
            .check(&CostCheckContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: Some("acme"),
                user_id: None,
                api_key_id: None,
                timestamp: SystemTime::now(),
            })
            .await;
        assert!(
            matches!(verdict, BudgetVerdict::Allow),
            "no configured limit means no enforcement, expected Allow, got {verdict:?}"
        );

        let snap = ledger.snapshot();
        assert!(
            (snap.per_tenant["acme"] - 3.0).abs() < 1e-9,
            "spend history for a removed tenant must be retained, got {:?}",
            snap.per_tenant.get("acme")
        );
    }

    #[tokio::test]
    async fn budget_ledger_rejects_when_user_limit_exceeded() {
        let mut limits = DimensionLimits::default();
        limits.per_user.insert("alice".to_owned(), 0.05);

        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 0.10,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        let verdict = ledger
            .check(&CostCheckContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                timestamp: SystemTime::now(),
            })
            .await;

        match verdict {
            BudgetVerdict::Reject { dimension, .. } => {
                assert!(
                    matches!(dimension, BudgetDimension::User(ref u) if u == "alice"),
                    "expected User(alice) dimension, got {dimension:?}"
                );
            }
            BudgetVerdict::Allow => panic!("expected Reject, got Allow"),
        }
    }

    #[tokio::test]
    async fn budget_ledger_resets_at_window_boundary() {
        let limits = DimensionLimits {
            global: Some(100.0),
            ..Default::default()
        };
        let window = Duration::from_secs(1);
        let ledger = InMemoryBudgetLedger::new(limits, window);

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: None,
                api_key_id: None,
                cost_usd: 50.0,
                tokens_in: 1_000_000,
                tokens_out: 0,
                timestamp: SystemTime::now(),
            })
            .await;

        assert!(ledger.snapshot().global_spend_usd > 0.0);

        let future = SystemTime::now() + Duration::from_secs(2);

        let spend_after_window = ledger.global.spend_usd(future);
        assert_eq!(spend_after_window, 0.0, "spend should reset to 0 after window boundary");
    }

    #[tokio::test]
    async fn budget_snapshot_csv_export_round_trips() {
        let ledger = InMemoryBudgetLedger::new(DimensionLimits::default(), Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: Some("tenant-x"),
                user_id: Some("user-y"),
                api_key_id: Some("key-z"),
                cost_usd: 1.23,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        let mut csv_bytes: Vec<u8> = Vec::new();
        ledger.export_csv(&mut csv_bytes).expect("CSV export must not fail");
        let csv = String::from_utf8(csv_bytes).expect("CSV must be valid UTF-8");

        assert!(csv.starts_with("dimension,spend_usd\n"), "missing header: {csv}");

        let mut found_global = false;
        let mut found_model = false;
        let mut found_tenant = false;
        let mut found_user = false;
        let mut found_key = false;

        for line in csv.lines().skip(1) {
            let parts: Vec<&str> = line.splitn(2, ',').collect();
            assert_eq!(parts.len(), 2, "malformed CSV line: {line}");
            let dimension = parts[0];
            let spend: f64 = parts[1].parse().expect("spend must be a float");

            match dimension {
                "global" => {
                    assert!((spend - 1.23).abs() < 1e-6, "global spend mismatch: {spend}");
                    found_global = true;
                }
                "model:gpt-4" => {
                    assert!((spend - 1.23).abs() < 1e-6);
                    found_model = true;
                }
                "tenant:tenant-x" => {
                    assert!((spend - 1.23).abs() < 1e-6);
                    found_tenant = true;
                }
                "user:user-y" => {
                    assert!((spend - 1.23).abs() < 1e-6);
                    found_user = true;
                }
                "api_key:key-z" => {
                    assert!((spend - 1.23).abs() < 1e-6);
                    found_key = true;
                }
                _ => {}
            }
        }

        assert!(found_global, "global row missing from CSV");
        assert!(found_model, "model row missing from CSV");
        assert!(found_tenant, "tenant row missing from CSV");
        assert!(found_user, "user row missing from CSV");
        assert!(found_key, "api_key row missing from CSV");
    }

    /// Spawn 100 threads each calling `add($0.10)` exactly at the window
    /// boundary and assert the total is $10.00, not less.
    ///
    /// The CAS in `spend_usd` guarantees exactly one thread resets the window;
    /// the other 99 threads see the already-zeroed counter but still add their
    /// $0.10 contribution via `fetch_add`.  Without the CAS fix, both threads
    /// that race on the boundary would zero `spend_mc` independently, causing
    /// each other's prior `add` to be dropped.
    #[test]
    fn window_rollover_under_concurrent_threads_does_not_undercount() {
        use std::sync::Barrier;
        use std::thread;

        let entry = Arc::new(WindowEntry::new(Duration::from_secs(1)));

        let future_now = SystemTime::now() + Duration::from_secs(2);

        let barrier = Arc::new(Barrier::new(100));
        let mut handles = Vec::with_capacity(100);

        for _ in 0..100 {
            let entry_clone = Arc::clone(&entry);
            let barrier_clone = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_clone.wait();
                entry_clone.add(0.10, future_now);
            }));
        }

        for h in handles {
            h.join().expect("thread must not panic");
        }

        let total = microcents_to_usd(entry.spend_mc.load(Ordering::Acquire));
        assert!(
            (total - 10.0_f64).abs() < 1e-4,
            "expected $10.00 total, got ${total:.6} — window rollover race caused under-counting"
        );
    }

    /// Bug 6 fix: 200 parallel `add($0.10)` calls at a rollover boundary must
    /// total exactly $20.00 — no contribution lost due to TOCTOU.
    #[test]
    fn budget_window_rollover_no_torn_read() {
        use std::sync::Barrier;
        use std::thread;

        let entry = Arc::new(WindowEntry::new(Duration::from_secs(1)));
        let future_now = SystemTime::now() + Duration::from_secs(2);

        const WRITERS: usize = 200;
        let barrier = Arc::new(Barrier::new(WRITERS));
        let mut handles = Vec::with_capacity(WRITERS);
        for _ in 0..WRITERS {
            let e = Arc::clone(&entry);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                e.add(0.10, future_now);
            }));
        }
        for h in handles {
            h.join().expect("writer must not panic");
        }
        let total = microcents_to_usd(entry.spend_mc.load(Ordering::Acquire));
        assert!(
            (total - 20.0_f64).abs() < 1e-4,
            "expected $20.00 total after 200 concurrent adds at rollover; got ${total:.6}"
        );
    }

    /// $10 user budget, $9.50 spend, estimated_cost=$0.50, safety_margin=0.10
    /// → effective limit = $10 × 0.90 = $9.00.
    /// $9.50 + 2×$0.50 = $10.50 ≥ $9.00 → hedging must be suppressed.
    #[tokio::test]
    async fn should_hedge_respects_user_budget() {
        let mut limits = DimensionLimits::default();
        limits.per_user.insert("alice".to_owned(), 10.0);

        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 9.50,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        let ctx = CostCheckContext {
            model: "gpt-4",
            provider: "openai",
            tenant_id: None,
            user_id: Some("alice"),
            api_key_id: None,
            timestamp: SystemTime::now(),
        };

        let result = should_hedge(&ledger, &ctx, 0.50, 0.10);
        assert!(
            !result,
            "hedging should be suppressed when user spend + 2×cost would exceed 90% of budget"
        );
    }

    /// Same $10 user budget but only $1.00 spend.
    /// $1.00 + 2×$0.50 = $2.00 < $9.00 → hedging must be allowed.
    #[tokio::test]
    async fn should_hedge_allows_when_far_below_budget() {
        let mut limits = DimensionLimits::default();
        limits.per_user.insert("alice".to_owned(), 10.0);

        let ledger = InMemoryBudgetLedger::new(limits, Duration::from_secs(3600));

        ledger
            .record(&CostRecordContext {
                model: "gpt-4",
                provider: "openai",
                tenant_id: None,
                user_id: Some("alice"),
                api_key_id: None,
                cost_usd: 1.00,
                tokens_in: 100,
                tokens_out: 50,
                timestamp: SystemTime::now(),
            })
            .await;

        let ctx = CostCheckContext {
            model: "gpt-4",
            provider: "openai",
            tenant_id: None,
            user_id: Some("alice"),
            api_key_id: None,
            timestamp: SystemTime::now(),
        };

        let result = should_hedge(&ledger, &ctx, 0.50, 0.10);
        assert!(
            result,
            "hedging should be allowed when user spend + 2×cost is well below 90% of budget"
        );
    }

    /// A minimal inner `Service` that returns a `ChatStream` response carrying
    /// usage on its final chunk, so `BudgetService`'s streaming-accounting path
    /// can be exercised without pulling in the full `LlmClient` mock surface.
    #[derive(Clone)]
    struct StreamingUsageService {
        prompt_tokens: u64,
        completion_tokens: u64,
    }

    fn usage_chunk(model: &str, usage: Option<Usage>) -> crate::types::ChatCompletionChunk {
        crate::types::ChatCompletionChunk {
            id: "chunk".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: model.into(),
            choices: vec![],
            usage,
            system_fingerprint: None,
            service_tier: None,
        }
    }

    struct ChunkStream(std::collections::VecDeque<crate::types::ChatCompletionChunk>);

    impl futures_core::Stream for ChunkStream {
        type Item = Result<crate::types::ChatCompletionChunk>;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(self.0.pop_front().map(Ok))
        }
    }

    impl tower::Service<LlmRequest> for StreamingUsageService {
        type Response = LlmResponse;
        type Error = LiterLlmError;
        type Future = BoxFuture<'static, Result<LlmResponse>>;

        fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: LlmRequest) -> Self::Future {
            let model = req.model().unwrap_or("gpt-4").to_owned();
            let usage = Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.completion_tokens,
                total_tokens: self.prompt_tokens + self.completion_tokens,
                prompt_tokens_details: None,
            };
            Box::pin(async move {
                let chunks =
                    std::collections::VecDeque::from([usage_chunk(&model, None), usage_chunk(&model, Some(usage))]);
                let stream: crate::client::BoxStream<'static, Result<crate::types::ChatCompletionChunk>> =
                    Box::pin(ChunkStream(chunks));
                Ok(LlmResponse::ChatStream(stream))
            })
        }
    }

    /// Regression for the "streaming bypasses budget accounting" bug:
    /// `LlmResponse::usage()` always returns `None` for `ChatStream`, so a
    /// naive post-response check (`resp.usage()`) never sees the tokens a
    /// streamed call actually consumed, and `BudgetState` is never updated —
    /// a caller could stream unlimited tokens through a budget-limited
    /// endpoint at zero recorded cost.
    #[tokio::test]
    async fn budget_service_records_cost_for_streamed_response() {
        use futures_util::StreamExt as _;

        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(1000.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };
        let layer = BudgetLayer::new(config, Arc::clone(&state));
        let mut svc = layer.layer(StreamingUsageService {
            prompt_tokens: 1000,
            completion_tokens: 500,
        });

        let resp = svc
            .call(LlmRequest::ChatStream(chat_req("gpt-4")))
            .await
            .expect("streamed call should succeed");
        let LlmResponse::ChatStream(mut stream) = resp else {
            panic!("expected a ChatStream response");
        };

        // Drain the stream fully so the completion callback (which records cost) fires.
        while stream.next().await.is_some() {}

        assert!(
            state.global_spend() > 0.0,
            "cost of a streamed response must be recorded in global spend"
        );
        assert!(
            state.model_spend("gpt-4") > 0.0,
            "cost of a streamed response must be recorded per-model"
        );
    }

    /// Spend must still be recorded when the caller abandons the stream.
    ///
    /// Accounting is settle-only, and settlement used to happen exclusively in
    /// the stream's terminal poll — so a client that started a stream and
    /// dropped it consumed real provider tokens (the whole completion is
    /// generated and buffered before the caller sees a byte) while the ledger
    /// stayed at zero.  Repeating that in a loop is unmetered usage.
    #[tokio::test]
    async fn abandoned_stream_still_records_spend() {
        use futures_util::StreamExt as _;

        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(1000.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };
        let layer = BudgetLayer::new(config, Arc::clone(&state));
        let mut svc = layer.layer(StreamingUsageService {
            prompt_tokens: 1000,
            completion_tokens: 500,
        });

        let resp = svc
            .call(LlmRequest::ChatStream(chat_req("gpt-4")))
            .await
            .expect("streamed call should succeed");
        let LlmResponse::ChatStream(mut stream) = resp else {
            panic!("expected a ChatStream response");
        };

        // ~keep Take one chunk and walk away — the usage-bearing final chunk is never polled.
        let _ = stream.next().await;
        drop(stream);

        assert!(
            state.global_spend() > 0.0,
            "spend must be recorded even though the stream was dropped before it ended"
        );
        assert!(
            state.model_spend("gpt-4") > 0.0,
            "per-model spend must be recorded for an abandoned stream"
        );
    }

    /// Dropping without polling at all must also settle.
    #[tokio::test]
    async fn stream_dropped_without_a_single_poll_records_spend() {
        let state = Arc::new(BudgetState::new());
        let config = BudgetConfig {
            global_limit: Some(1000.0),
            enforcement: Enforcement::Hard,
            ..Default::default()
        };
        let layer = BudgetLayer::new(config, Arc::clone(&state));
        let mut svc = layer.layer(StreamingUsageService {
            prompt_tokens: 1000,
            completion_tokens: 500,
        });

        let resp = svc
            .call(LlmRequest::ChatStream(chat_req("gpt-4")))
            .await
            .expect("streamed call should succeed");
        let LlmResponse::ChatStream(stream) = resp else {
            panic!("expected a ChatStream response");
        };

        drop(stream);

        assert!(
            state.global_spend() > 0.0,
            "spend must be recorded for a stream that was never polled"
        );
    }
}
