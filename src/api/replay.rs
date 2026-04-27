//! Phase 2.5 Slice D.3 — per-session ring buffer + `Last-Event-ID` replay.
//!
//! Architecture: **replay-only, not live re-attach.** On client disconnect
//! the upstream agent is cancelled (Slice D.2 semantics) — the agent dies
//! immediately, so any frames not already stored in the ring are lost
//! forever. The ring retains frames generated for the *current turn only*;
//! starting a fresh turn (`POST /chat/stream/:id` with no `Last-Event-ID`)
//! clears the ring before streaming.
//!
//! On reconnect with `Last-Event-ID: N`, the handler yields every stored
//! frame with `id > N`, then a terminal `event: replay_complete` frame,
//! and closes the stream. The client is expected to treat `replay_complete`
//! as end-of-stream (no agent to resume against).
//!
//! Concurrency: `ReplayBuffers` is `Arc<Mutex<HashMap<session_id,
//! Arc<Mutex<ReplayRing>>>>>`. Lock order is always outer → inner; the
//! outer lock is released the moment we clone the inner `Arc`. Both are
//! `std::sync::Mutex` because they are never held across `.await`.
//!
//! No TTL / no GC: each `ReplayRing` is tied to a session id string and
//! persists for the process lifetime. Memory bound: cap × (frame size) ×
//! active sessions. At default cap 256 with ~200-byte frames, 100 live
//! sessions cost ~5 MB — fine for a single-process daemon.

use axum::http::HeaderMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// A single SSE frame captured as it left `sse_output_stream`. `id`
/// matches the `id:` written to the wire; `event` is the event type
/// name ("token", "done", ...); `data` is the raw JSON-string payload
/// — i.e. the same bytes that followed `data: ` on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredFrame {
    pub id: u64,
    pub event: String,
    pub data: String,
}

/// Bounded FIFO of stored frames. At capacity the oldest frame is
/// evicted. Default cap 256 covers ~2–3 typical turns at typical
/// granularity; beyond that, the client is too far behind to replay
/// coherently and should start a fresh turn.
#[derive(Debug)]
pub struct ReplayRing {
    frames: VecDeque<StoredFrame>,
    cap: usize,
}

impl ReplayRing {
    pub const DEFAULT_CAP: usize = 256;

    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAP)
    }

    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(cap.min(Self::DEFAULT_CAP)),
            cap,
        }
    }

    pub fn push(&mut self, frame: StoredFrame) {
        if self.frames.len() >= self.cap {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Frames strictly newer than `last_id`, in wire order (oldest first).
    #[must_use]
    pub fn frames_after(&self, last_id: u64) -> Vec<StoredFrame> {
        self.frames
            .iter()
            .filter(|f| f.id > last_id)
            .cloned()
            .collect()
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl Default for ReplayRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-wide mapping from `session_id` to its replay ring. Stored in
/// `ApiState`; cloned by `axum::State<ApiState>` through the `Arc`.
pub type ReplayBuffers = Arc<Mutex<HashMap<String, Arc<Mutex<ReplayRing>>>>>;

#[must_use]
pub fn new_buffers() -> ReplayBuffers {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Get the ring for `session_id`, creating one if absent. The returned
/// `Arc` can be held across `.await` safely — the inner `Mutex` is only
/// locked for push/query in synchronous sections.
pub fn get_or_create_ring(buffers: &ReplayBuffers, session_id: &str) -> Arc<Mutex<ReplayRing>> {
    let mut map = buffers.lock().expect("reconnect_buffers poisoned");
    map.entry(session_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(ReplayRing::new())))
        .clone()
}

/// Parse `Last-Event-ID` from request headers. Missing, non-ASCII, or
/// non-numeric values yield `None` — the handler then treats the request
/// as a fresh turn rather than a replay.
#[must_use]
pub fn extract_last_event_id(headers: &HeaderMap) -> Option<u64> {
    headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64, event: &str, data: &str) -> StoredFrame {
        StoredFrame {
            id,
            event: event.into(),
            data: data.into(),
        }
    }

    #[test]
    fn ring_push_and_frames_after_in_order() {
        let mut ring = ReplayRing::new();
        ring.push(frame(1, "token", r#"{"content":"a"}"#));
        ring.push(frame(2, "token", r#"{"content":"b"}"#));
        ring.push(frame(3, "done", r#"{"response":"ab"}"#));

        let tail = ring.frames_after(1);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].id, 2);
        assert_eq!(tail[1].id, 3);
        assert_eq!(tail[1].event, "done");
    }

    #[test]
    fn ring_frames_after_empty_when_caught_up() {
        let mut ring = ReplayRing::new();
        ring.push(frame(1, "token", "{}"));
        ring.push(frame(2, "done", "{}"));

        assert!(
            ring.frames_after(2).is_empty(),
            "last_id == highest → no tail"
        );
        assert!(
            ring.frames_after(99).is_empty(),
            "last_id > highest → no tail"
        );
    }

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let mut ring = ReplayRing::with_capacity(3);
        ring.push(frame(1, "token", "a"));
        ring.push(frame(2, "token", "b"));
        ring.push(frame(3, "token", "c"));
        ring.push(frame(4, "token", "d"));

        assert_eq!(ring.len(), 3);
        let all = ring.frames_after(0);
        let ids: Vec<u64> = all.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![2, 3, 4], "frame 1 must have been evicted");
    }

    #[test]
    fn ring_clear_resets_len_and_tail() {
        let mut ring = ReplayRing::new();
        ring.push(frame(1, "token", "a"));
        ring.push(frame(2, "done", "b"));
        assert_eq!(ring.len(), 2);

        ring.clear();
        assert_eq!(ring.len(), 0);
        assert!(ring.is_empty());
        assert!(ring.frames_after(0).is_empty());
    }

    #[test]
    fn get_or_create_ring_returns_same_arc_per_session() {
        let buffers = new_buffers();
        let a = get_or_create_ring(&buffers, "sess-1");
        let b = get_or_create_ring(&buffers, "sess-1");
        assert!(Arc::ptr_eq(&a, &b), "same id must return the same Arc");

        a.lock().unwrap().push(frame(1, "token", "x"));
        assert_eq!(
            b.lock().unwrap().len(),
            1,
            "writes through one handle are visible through the other"
        );
    }

    #[test]
    fn get_or_create_ring_isolates_distinct_sessions() {
        let buffers = new_buffers();
        let a = get_or_create_ring(&buffers, "sess-a");
        let b = get_or_create_ring(&buffers, "sess-b");
        assert!(!Arc::ptr_eq(&a, &b));

        a.lock().unwrap().push(frame(1, "token", "x"));
        assert_eq!(a.lock().unwrap().len(), 1);
        assert_eq!(
            b.lock().unwrap().len(),
            0,
            "distinct rings do not share state"
        );
    }

    #[test]
    fn extract_last_event_id_variants() {
        let mut h = HeaderMap::new();
        assert_eq!(extract_last_event_id(&h), None, "missing header");

        h.insert("last-event-id", "42".parse().unwrap());
        assert_eq!(extract_last_event_id(&h), Some(42));

        h.insert("last-event-id", "  17  ".parse().unwrap());
        assert_eq!(extract_last_event_id(&h), Some(17), "whitespace trimmed");

        h.insert("last-event-id", "not-a-number".parse().unwrap());
        assert_eq!(extract_last_event_id(&h), None, "non-numeric → None");

        h.insert("last-event-id", "-5".parse().unwrap());
        assert_eq!(
            extract_last_event_id(&h),
            None,
            "negative → None (u64 parse fails)"
        );
    }
}
