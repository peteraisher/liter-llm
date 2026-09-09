use std::task::{Context, Poll};

use tower::Layer;
use tower::Service;
use tracing::Instrument as _;

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{LiterLlmError, Result};
use crate::types::FinishReason;

/// Tower [`Layer`] that wraps a service with OpenTelemetry GenAI semantic
/// convention tracing spans.
///
/// Each call creates a [`tracing::info_span`] named `"gen_ai"` with the
/// following attributes:
///
/// - `gen_ai.operation.name` — `"chat"`, `"embeddings"`, or `"list_models"`.
/// - `gen_ai.request.model` — the model name from the request, or `""` for
///   [`LlmRequest::ListModels`].
/// - `gen_ai.system` — the provider prefix extracted from the model name (e.g.
///   `"openai"` for `"openai/gpt-4"`), or `""` when absent.
/// - `gen_ai.usage.input_tokens` — populated on successful chat / embed
///   responses where usage data is present.
/// - `gen_ai.usage.output_tokens` — populated on successful chat responses.
/// - `gen_ai.response.id` — the completion ID from the response.
/// - `gen_ai.response.model` — the actual model used (may differ from requested).
/// - `gen_ai.response.finish_reasons` — space-separated finish reasons from
///   all choices (e.g. `"stop"`).
/// - `error.type` — set to the error variant name if the inner service returns
///   an error.
#[cfg_attr(alef, alef(skip))]
pub struct TracingLayer;

impl<S> Layer<S> for TracingLayer {
    type Service = TracingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TracingService { inner }
    }
}

/// Tower service produced by [`TracingLayer`].
#[cfg_attr(alef, alef(skip))]
pub struct TracingService<S> {
    inner: S,
}

impl<S> Clone for TracingService<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for TracingService<S>
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
        let operation_name = req.operation_name();
        let model_str = req.model().unwrap_or("");
        let system = model_str.split_once('/').map_or("", |(prefix, _)| prefix);
        let model = model_str.to_owned();

        let span = tracing::info_span!(
            "gen_ai",
            gen_ai.operation.name = operation_name,
            gen_ai.request.model = %model,
            gen_ai.system = system,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.usage.cost = tracing::field::Empty,
            gen_ai.response.finish_reasons = tracing::field::Empty,
            error.type = tracing::field::Empty,
        );

        let fut = self.inner.call(req);

        Box::pin(
            async move {
                match fut.await {
                    Ok(resp) => {
                        record_response(&tracing::Span::current(), &resp);
                        Ok(resp)
                    }
                    Err(e) => {
                        tracing::Span::current().record("error.type", e.error_type());
                        Err(e)
                    }
                }
            }
            .instrument(span),
        )
    }
}

/// Re-export `tracing_opentelemetry` when the `otel` feature is active.
///
/// This lets callers compose a subscriber that exports spans to an
/// OpenTelemetry collector without taking a direct dependency on the crate:
///
/// ```rust,ignore
/// use liter_llm::tower::tracing::otel::tracing_opentelemetry::OpenTelemetryLayer;
/// ```
#[cfg(feature = "otel")]
pub use tracing_opentelemetry;

/// Re-export `opentelemetry` when the `otel` feature is active.
///
/// Provides access to tracer/provider types needed to build a full
/// OpenTelemetry pipeline (e.g. `opentelemetry::global::tracer`).
#[cfg(feature = "otel")]
pub use opentelemetry;

/// Record span attributes from the response according to GenAI semantic conventions.
fn record_response(span: &tracing::Span, resp: &LlmResponse) {
    match resp {
        LlmResponse::Chat(r) => {
            span.record("gen_ai.response.id", r.id.as_str());
            span.record("gen_ai.response.model", r.model.as_str());

            let finish_reasons = finish_reasons_str(r.choices.iter().map(|c| c.finish_reason.as_ref()));
            if !finish_reasons.is_empty() {
                span.record("gen_ai.response.finish_reasons", finish_reasons.as_str());
            }
        }
        LlmResponse::Embed(r) => {
            span.record("gen_ai.response.model", r.model.as_str());
        }
        LlmResponse::ChatStream(_)
        | LlmResponse::ListModels(_)
        | LlmResponse::ImageGenerate(_)
        | LlmResponse::Speech(_)
        | LlmResponse::Transcribe(_)
        | LlmResponse::Moderate(_)
        | LlmResponse::Rerank(_)
        | LlmResponse::Search(_)
        | LlmResponse::Ocr(_) => {}
    }

    if let Some(usage) = resp.usage() {
        span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", usage.completion_tokens);
    }
}

/// Build a space-separated string of finish reason names from an iterator of
/// optional [`FinishReason`] values.  `None` entries are skipped.
///
/// Optimised for the common single-choice case: when there is exactly one
/// reason, the static `&str` is returned directly as an owned `String` without
/// an intermediate `Vec` or repeated `push_str` calls.
fn finish_reasons_str<'a>(reasons: impl Iterator<Item = Option<&'a FinishReason>>) -> String {
    let first = reasons.filter_map(|r| r.map(finish_reason_name));
    let mut iter = first.peekable();
    let Some(first_name) = iter.next() else {
        return String::new();
    };
    if iter.peek().is_none() {
        return first_name.to_owned();
    }
    iter.fold(first_name.to_owned(), |mut acc, name| {
        acc.push(' ');
        acc.push_str(name);
        acc
    })
}

/// Map a [`FinishReason`] variant to its GenAI semantic convention string.
const fn finish_reason_name(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::FunctionCall => "function_call",
        FinishReason::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    use super::*;
    use crate::tower::service::LlmService;
    use crate::tower::tests_common::{MockClient, chat_req};
    use crate::tower::types::LlmRequest;

    /// A captured span: its declared field-name schema (from [`Metadata`],
    /// which lists every field named in the `tracing::info_span!` macro
    /// invocation, including `tracing::field::Empty`-valued ones) plus
    /// whatever field values have actually been recorded on it so far, either
    /// at creation or via a later `Span::record` call.
    #[derive(Clone)]
    struct CapturedSpan {
        name: &'static str,
        declared_fields: Vec<&'static str>,
        values: HashMap<String, String>,
        // ~keep Retained so `RecordingSubscriber::current_span` (needed for
        // ~keep `tracing::Span::current()` to resolve at all — see that impl)
        // ~keep can hand back the `'static` metadata reference `Current::new`
        // ~keep requires, without a second lookup path.
        metadata: &'static Metadata<'static>,
    }

    /// Records string-ish representations of every field value visited, keyed
    /// by field name, so tests can assert on them without needing a real OTel
    /// exporter or a `tracing-subscriber` dependency (this crate does not
    /// depend on `tracing-subscriber`).
    #[derive(Default)]
    struct FieldRecorder {
        values: HashMap<String, String>,
    }

    impl Visit for FieldRecorder {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.values.insert(field.name().to_owned(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.values.insert(field.name().to_owned(), value.to_owned());
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.values.insert(field.name().to_owned(), value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.values.insert(field.name().to_owned(), value.to_string());
        }
    }

    /// Shared handle into a [`RecordingSubscriber`]'s captured state.
    ///
    /// ~keep `tracing::subscriber::set_default` takes the subscriber by value
    /// ~keep and wraps it in a `Dispatch`, which does not hand the value back
    /// ~keep out — so the only way to inspect what was recorded after the call
    /// ~keep under test finishes is to keep a second, cloned handle to the
    /// ~keep shared state alongside the one the subscriber itself holds.
    #[derive(Clone, Default)]
    struct SpanCapture {
        next_id: Arc<AtomicU64>,
        spans: Arc<Mutex<HashMap<u64, CapturedSpan>>>,
        // ~keep A stack, not a single slot: `Instrumented::poll` brackets each
        // ~keep poll of the wrapped future with `enter`/`exit`, and nested
        // ~keep spans (not exercised here, but a correct `Subscriber` must
        // ~keep still support them) push/pop rather than overwrite.
        entered: Arc<Mutex<Vec<(Id, &'static Metadata<'static>)>>>,
    }

    impl SpanCapture {
        /// Return the single span created while this capture was active.
        /// Panics if zero or more than one span was created, since every test
        /// below drives exactly one `TracingService::call`.
        fn only_span(&self) -> CapturedSpan {
            let spans = self.spans.lock().expect("test mutex must not be poisoned");
            assert_eq!(spans.len(), 1, "expected exactly one span to have been created");
            spans.values().next().cloned().expect("checked len == 1 above")
        }

        fn metadata_for(&self, id: &Id) -> Option<&'static Metadata<'static>> {
            self.spans
                .lock()
                .expect("test mutex must not be poisoned")
                .get(&id.into_u64())
                .map(|span| span.metadata)
        }
    }

    /// Minimal hand-rolled [`Subscriber`] that records span creation metadata
    /// and every subsequent `Span::record` call. Written by hand instead of
    /// pulling in `tracing-subscriber` (not a dependency of this crate, and
    /// out of scope to add here) — this only needs to observe span metadata
    /// and field values, not do any real collection/export.
    struct RecordingSubscriber {
        capture: SpanCapture,
    }

    impl Subscriber for RecordingSubscriber {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, attrs: &Attributes<'_>) -> Id {
            let id = self.capture.next_id.fetch_add(1, Ordering::SeqCst) + 1;

            let mut visitor = FieldRecorder::default();
            attrs.record(&mut visitor);

            let metadata = attrs.metadata();
            let captured = CapturedSpan {
                name: metadata.name(),
                declared_fields: metadata.fields().iter().map(|f| f.name()).collect(),
                values: visitor.values,
                metadata,
            };

            self.capture
                .spans
                .lock()
                .expect("test mutex must not be poisoned")
                .insert(id, captured);

            Id::from_u64(id)
        }

        fn record(&self, span: &Id, values: &Record<'_>) {
            let mut visitor = FieldRecorder::default();
            values.record(&mut visitor);

            let mut spans = self.capture.spans.lock().expect("test mutex must not be poisoned");
            if let Some(captured) = spans.get_mut(&span.into_u64()) {
                captured.values.extend(visitor.values);
            }
        }

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
        fn event(&self, _event: &Event<'_>) {}

        // ~keep `tracing::Span::current()` — which `TracingService::call` relies
        // ~keep on for both the success (`record_response`) and error
        // ~keep (`error.type`) paths — resolves via `Dispatch::current_span`,
        // ~keep which defaults to "unknown" unless a `Subscriber` tracks entry
        // ~keep state itself. Without `enter`/`exit`/`current_span` below,
        // ~keep `Span::current()` would silently return a no-op span and every
        // ~keep `.record(...)` call in production code would vanish, making
        // ~keep every test in this module pass regardless of whether recording
        // ~keep actually works — exactly the kind of test defect #7 flags.
        fn enter(&self, span: &Id) {
            if let Some(metadata) = self.capture.metadata_for(span) {
                self.capture
                    .entered
                    .lock()
                    .expect("test mutex must not be poisoned")
                    .push((span.clone(), metadata));
            }
        }

        fn exit(&self, span: &Id) {
            let mut entered = self.capture.entered.lock().expect("test mutex must not be poisoned");
            if entered.last().is_some_and(|(id, _)| id == span) {
                entered.pop();
            }
        }

        fn current_span(&self) -> tracing_core::span::Current {
            let entered = self.capture.entered.lock().expect("test mutex must not be poisoned");
            match entered.last() {
                Some((id, metadata)) => tracing_core::span::Current::new(id.clone(), metadata),
                None => tracing_core::span::Current::none(),
            }
        }
    }

    /// Pin the `gen_ai` span's name and its full declared field-key schema.
    /// Both are documented, GenAI-semantic-convention public API (this
    /// module's doc comment), and by the crate's `tracing-product-surface`
    /// convention span names and field keys are semver-relevant — yet nothing
    /// previously asserted on either, so renaming the span or dropping a
    /// field declaration broke no test.
    ///
    /// Revert line: renaming the `"gen_ai"` literal in this file's
    /// `tracing::info_span!` call (in `TracingService::call`) to any other
    /// string makes the first assertion fail; deleting any one of that same
    /// macro invocation's `tracing::field::Empty` field declarations shrinks
    /// `declared_fields` and makes the second assertion fail.
    #[tokio::test]
    async fn gen_ai_span_declares_stable_name_and_field_keys() {
        let capture = SpanCapture::default();
        let subscriber = RecordingSubscriber {
            capture: capture.clone(),
        };
        let _guard = tracing::subscriber::set_default(subscriber);

        let inner = LlmService::new(MockClient::ok());
        let mut svc = TracingLayer.layer(inner);
        let _resp = svc
            .call(LlmRequest::Chat(chat_req("openai/gpt-4")))
            .await
            .expect("should succeed");

        let span = capture.only_span();
        assert_eq!(span.name, "gen_ai");

        let mut fields = span.declared_fields;
        fields.sort_unstable();
        assert_eq!(
            fields,
            vec![
                "error.type",
                "gen_ai.operation.name",
                "gen_ai.request.model",
                "gen_ai.response.finish_reasons",
                "gen_ai.response.id",
                "gen_ai.response.model",
                "gen_ai.system",
                "gen_ai.usage.cost",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
            ]
        );
    }

    /// Verify the recorded field *values* on a successful response, and that
    /// no message bodies, keys, or tokens ever reach a span field — only the
    /// bounded set asserted here. A prior review confirmed `tracing.rs`
    /// records no secrets (`error.type` uses the bounded `&'static str` from
    /// `error_type()`); this pins the success-path field set so a future
    /// change that adds, say, a raw request/response body field would be
    /// caught by the exact-match style used here rather than silently
    /// expanding what this module puts on the wire.
    ///
    /// Revert line: commenting out `record_response`'s body (or any of its
    /// individual `span.record(...)` calls) in `TracingService::call` makes
    /// the corresponding `assert_eq!` below observe `None` instead of the
    /// expected value.
    #[tokio::test]
    async fn gen_ai_span_records_response_fields_on_success() {
        let capture = SpanCapture::default();
        let subscriber = RecordingSubscriber {
            capture: capture.clone(),
        };
        let _guard = tracing::subscriber::set_default(subscriber);

        let inner = LlmService::new(MockClient::ok());
        let mut svc = TracingLayer.layer(inner);
        let _resp = svc
            .call(LlmRequest::Chat(chat_req("openai/gpt-4")))
            .await
            .expect("should succeed");

        let span = capture.only_span();
        let get = |key: &str| span.values.get(key).map(String::as_str);

        assert_eq!(get("gen_ai.operation.name"), Some("chat"));
        assert_eq!(get("gen_ai.system"), Some("openai"));
        assert_eq!(get("gen_ai.request.model"), Some("openai/gpt-4"));
        assert_eq!(get("gen_ai.response.id"), Some("test-id"));
        assert_eq!(get("gen_ai.response.model"), Some("openai/gpt-4"));
        assert_eq!(get("gen_ai.response.finish_reasons"), Some("stop"));
        assert_eq!(get("gen_ai.usage.input_tokens"), Some("10"));
        assert_eq!(get("gen_ai.usage.output_tokens"), Some("5"));
        assert!(
            !span.values.contains_key("error.type"),
            "error.type must not be recorded on a successful response"
        );
    }

    /// Verify `error.type` is recorded with the bounded `&'static str` from
    /// [`crate::error::LiterLlmError::error_type`] on failure, and that no
    /// response fields leak onto a span that never got a response.
    ///
    /// Revert line: deleting the
    /// `tracing::Span::current().record("error.type", e.error_type());` call
    /// in `TracingService::call`'s `Err` arm makes the first assertion
    /// observe `None` instead of `Some("Timeout")`.
    #[tokio::test]
    async fn gen_ai_span_records_error_type_on_failure() {
        let capture = SpanCapture::default();
        let subscriber = RecordingSubscriber {
            capture: capture.clone(),
        };
        let _guard = tracing::subscriber::set_default(subscriber);

        let inner = LlmService::new(MockClient::failing_timeout());
        let mut svc = TracingLayer.layer(inner);
        let _err = svc
            .call(LlmRequest::Chat(chat_req("openai/gpt-4")))
            .await
            .expect_err("should fail");

        let span = capture.only_span();
        assert_eq!(span.values.get("error.type").map(String::as_str), Some("Timeout"));
        assert!(
            !span.values.contains_key("gen_ai.response.id"),
            "response fields must not be recorded on the error path"
        );
    }
}
