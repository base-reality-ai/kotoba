//! v0.2 end-to-end acceptance test for the learning loop.
//!
//! This test proves that the planner → conversation → recorder loop
//! closes against a real wiki on disk, without needing live model
//! calls. It runs against a tempdir-rooted `.dm/wiki/` and exercises
//! the same pure helpers `kotoba session` calls in production:
//!
//! - `host_caps::plan_session_in` against an empty wiki → starter brief.
//! - `host_caps::record_session_in` against a synthesized transcript
//!   that mirrors what dm would persist after a real Yuki conversation.
//! - `host_caps::plan_session_in` again, post-record, to confirm the
//!   wiki updates flow back into the next cycle's brief.
//!
//! The test is the v0.2 milestone gate: if it passes, the loop closes.
//!
//! Test surface uses the `#[path = "..."]` include trick (mirroring
//! `tests/host_caps.rs`) because `host_caps`/`domain` are bin-only
//! modules. The integration test compiles them as a private crate
//! and asserts against their `pub(crate)` helpers.

#[path = "../src/host_caps.rs"]
#[allow(dead_code)]
mod host_caps;

// `domain.rs` already carries an inner `#![allow(dead_code)]` (some types
// land in v0.2 but only the v0.3 planner/recorder agents will read them).
// No outer attribute needed here — duplicating it triggers
// `clippy::duplicated_attributes`.
#[path = "../src/domain.rs"]
mod domain;

use std::path::Path;
use tempfile::TempDir;

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, content).unwrap();
}

fn yuki_seed() -> String {
    // Mirrors the kotoba v0.1 Yuki persona seed shape — frontmatter
    // with `sessions_count: 0` plus a body section the planner reads
    // for signature topics. The recorder will rewrite this page on
    // call to bump sessions_count + append a session log entry.
    "---
title: Yuki
type: entity
entity_kind: persona
layer: host
sessions_count: 0
---

# Yuki

## Signature topics

- Self-introduction
- Café ordering
- Weekend hobbies

## Sessions log

"
    .to_string()
}

fn synthetic_transcript() -> &'static str {
    // Three vocab introductions matching the recorder's heuristic
    // (Japanese token + parenthesized kana + English gloss after a
    // " means " marker), plus two distinct learner-flagged struggles.
    // Mixed roles ("Learner" / "User") to verify the role recognizer
    // stays loose. The leading "Yuki: 系" and "Assistant:" lines are
    // assistant utterances and should not produce struggle entries.
    "\
Yuki: Welcome back! Let's review some basics today.
Yuki: New word: 学校 (がっこう) means school.
Yuki: New word: 猫 (ねこ) means cat.
Yuki: New word: 犬 (いぬ) means dog.
Learner: I don't know は vs が?
Yuki: Great question — particles are tricky. Let's slow down.
User: what is て-form?
Assistant: what is ignored because assistant is not the learner role.
Yuki: お疲れ様でした！
"
}

#[test]
fn empty_wiki_brief_is_starter_shaped() {
    let tmp = TempDir::new().unwrap();
    let today = chrono::NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();

    let brief = host_caps::plan_session_in(tmp.path(), "Yuki", 3, today);

    // Header + date present.
    assert!(brief.contains("Session brief — 2026-04-27"), "{}", brief);
    // No persona registered yet — fallback line must be present.
    assert!(brief.contains("(none registered"), "{}", brief);
    // Empty wiki → starter focus + no-struggle hint.
    assert!(
        brief.contains("Self-introduction and greetings"),
        "{}",
        brief
    );
    assert!(brief.contains("first session"), "{}", brief);
    assert!(brief.contains("No recent struggles"), "{}", brief);
}

#[test]
fn record_session_writes_three_vocab_one_struggle_page_and_bumps_persona() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    write(
        &root.join(".dm/wiki/entities/Persona/Yuki.md"),
        &yuki_seed(),
    );

    // Pin a session timestamp so struggles file path + persona log line
    // are deterministic.
    let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);

    let summary = host_caps::record_session_in(root, synthetic_transcript(), "Yuki", now).unwrap();

    // Recorder summary: 3 vocab + 2 struggles + sessions_count == 1.
    assert_eq!(summary.vocabulary_count, 3, "summary: {:?}", summary);
    assert_eq!(summary.struggle_count, 2, "summary: {:?}", summary);
    assert_eq!(summary.sessions_count, 1, "summary: {:?}", summary);

    // Three vocabulary entities exist on disk with the expected kanji,
    // kana, and meaning fields.
    let vocab_dir = root.join(".dm/wiki/entities/Vocabulary");
    let school = std::fs::read_to_string(vocab_dir.join("学校.md")).unwrap();
    assert!(school.contains("- **Kanji:** 学校"), "{}", school);
    assert!(school.contains("- **Kana:** がっこう"), "{}", school);
    assert!(school.contains("- **Meaning:** school"), "{}", school);

    let cat = std::fs::read_to_string(vocab_dir.join("猫.md")).unwrap();
    assert!(cat.contains("- **Kana:** ねこ"), "{}", cat);
    assert!(cat.contains("- **Meaning:** cat"), "{}", cat);

    let dog = std::fs::read_to_string(vocab_dir.join("犬.md")).unwrap();
    assert!(dog.contains("- **Kana:** いぬ"), "{}", dog);
    assert!(dog.contains("- **Meaning:** dog"), "{}", dog);

    // The struggles synthesis page exists for today and contains both
    // learner-flagged confusions.
    let struggles =
        std::fs::read_to_string(root.join(".dm/wiki/synthesis/struggles-2026-04-27.md")).unwrap();
    assert!(struggles.contains("**は vs が**"), "{}", struggles);
    assert!(struggles.contains("**て-form**"), "{}", struggles);
    // Assistant-prefixed lookalike must NOT appear as a struggle.
    assert!(!struggles.contains("ignored because"), "{}", struggles);

    // Persona page bumped + a sessions log entry was appended with the
    // exact timestamp + the topics the recorder inferred from struggles.
    let persona = std::fs::read_to_string(root.join(".dm/wiki/entities/Persona/Yuki.md")).unwrap();
    assert!(persona.contains("sessions_count: 1"), "{}", persona);
    assert!(persona.contains("2026-04-27 12:30:00"), "{}", persona);
    assert!(
        persona.contains("words introduced: 学校 / がっこう (school)"),
        "{}",
        persona
    );
    assert!(
        persona.contains("struggles flagged: は vs が, て-form"),
        "{}",
        persona
    );
}

#[test]
fn loop_closes_planner_after_recorder_reflects_wiki_state() {
    // The acceptance shape: plan, record, plan again — second brief
    // must materially differ because the wiki now has content. This is
    // the user-visible proof that the loop closes.
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    write(
        &root.join(".dm/wiki/entities/Persona/Yuki.md"),
        &yuki_seed(),
    );

    let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let today = now.date_naive();

    // Pre-record brief: no vocab, no struggles, persona at session 1.
    let pre = host_caps::plan_session_in(root, "Yuki", 3, today);
    assert!(pre.contains("Yuki (session 1)"), "{}", pre);
    assert!(
        pre.contains("Wiki has no vocabulary yet"),
        "pre-brief should flag empty vocab: {}",
        pre
    );

    // Run the recorder against the synthesized transcript.
    host_caps::record_session_in(root, synthetic_transcript(), "Yuki", now).unwrap();

    // Post-record brief: persona advanced to session 2, vocab section
    // now lists at least one of the freshly-recorded entries, and the
    // struggles surface in the grammar/patterns block.
    let post = host_caps::plan_session_in(root, "Yuki", 3, today);
    assert!(
        post.contains("Yuki (session 2)"),
        "post-brief should reflect bumped sessions_count: {}",
        post
    );
    assert!(!post.contains("Wiki has no vocabulary yet"), "{}", post);
    assert!(
        post.contains("学校") || post.contains("猫") || post.contains("犬"),
        "post-brief should surface at least one recorded vocab entry: {}",
        post
    );
    assert!(
        post.contains("は vs が") || post.contains("て-form"),
        "post-brief should surface at least one recorded struggle: {}",
        post
    );
}

#[test]
fn record_session_creates_persona_when_none_seeded() {
    // Operator might invoke `kotoba session --persona Hiro` before
    // Hiro has a persona seed; the recorder must create the page so
    // subsequent planner reads find it. This complements the v0.1
    // Yuki-seed flow tested above.
    let tmp = TempDir::new().unwrap();
    let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);

    let summary =
        host_caps::record_session_in(tmp.path(), "Yuki: 猫 (ねこ) means cat.", "Hiro", now)
            .unwrap();

    assert_eq!(summary.sessions_count, 1);
    let hiro =
        std::fs::read_to_string(tmp.path().join(".dm/wiki/entities/Persona/Hiro.md")).unwrap();
    assert!(hiro.contains("title: Hiro"), "{}", hiro);
    assert!(hiro.contains("sessions_count: 1"), "{}", hiro);
    assert!(hiro.contains("## Sessions log"), "{}", hiro);
}

#[test]
fn record_session_attributes_segments_after_persona_switch() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    write(
        &root.join(".dm/wiki/entities/Persona/Yuki.md"),
        &yuki_seed(),
    );
    write(
        &root.join(".dm/wiki/entities/Persona/Hiro.md"),
        "---
title: Hiro
type: entity
entity_kind: persona
layer: host
sessions_count: 0
---

# Hiro

## Signature topics

- Weekend plans

## Sessions log

",
    );

    let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let transcript = format!(
        "\
Yuki: New word: 学校 (がっこう) means school.
Learner: I don't know は vs が?
tool: {}Hiro'{}
Hiro: New word: 猫 (ねこ) means cat.
User: what is タメ口?
",
        host_caps::PERSONA_LOADED_PREFIX,
        host_caps::PERSONA_LOADED_AFTER_NAME
    );

    let summary = host_caps::record_session_in(root, &transcript, "Yuki", now).unwrap();
    assert_eq!(summary.persona, "Yuki, Hiro");
    assert_eq!(summary.vocabulary_count, 2, "summary: {:?}", summary);
    assert_eq!(summary.struggle_count, 2, "summary: {:?}", summary);

    let yuki = std::fs::read_to_string(root.join(".dm/wiki/entities/Persona/Yuki.md")).unwrap();
    assert!(yuki.contains("sessions_count: 1"), "{}", yuki);
    assert!(
        yuki.contains("words introduced: 学校 / がっこう (school)"),
        "{}",
        yuki
    );
    assert!(yuki.contains("struggles flagged: は vs が"), "{}", yuki);
    assert!(!yuki.contains("タメ口"), "{}", yuki);

    let hiro = std::fs::read_to_string(root.join(".dm/wiki/entities/Persona/Hiro.md")).unwrap();
    assert!(hiro.contains("sessions_count: 1"), "{}", hiro);
    assert!(
        hiro.contains("words introduced: 猫 / ねこ (cat)"),
        "{}",
        hiro
    );
    assert!(hiro.contains("struggles flagged: タメ口"), "{}", hiro);
    assert!(!hiro.contains("は vs が"), "{}", hiro);
}
