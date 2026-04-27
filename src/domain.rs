//! Kotoba domain types — the host-layer model for Japanese learning.
//!
//! These types describe the shape of host-layer content; the actual
//! durable storage is the wiki under `.dm/wiki/` (entity pages with
//! `layer: host` frontmatter). Wiki tools (`wiki_search`, `wiki_lookup`)
//! and the host capabilities defined in [`crate::host_caps`] read and
//! write against these shapes.
//!
//! See `DM.md` and `.dm/wiki/concepts/learning-loop.md` for how the
//! entities compose into the planner → conversation → recorder loop.
//!
//! v0.1 only consumes [`Mastery`] (used by host_caps when seeding new
//! entries). The richer types (Vocabulary, Kanji, GrammarPoint, Persona,
//! Session, Example, ImmersionMode) are the durable shape the v0.2
//! planner/recorder agent chains will read+write — they're defined now
//! so the wiki schema, integration tests, and downstream tooling have
//! a single source of truth.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A vocabulary word the learner has encountered.
///
/// Wiki page lives at `.dm/wiki/entities/Vocabulary/<slug>.md` with
/// `layer: host` frontmatter; this struct mirrors the durable shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Kanji form when present (e.g. "学校"); empty when kana-only word.
    pub kanji: String,
    /// Kana reading (hiragana or katakana, e.g. "がっこう").
    pub kana: String,
    /// Romaji transliteration for early learners (e.g. "gakkou").
    pub romaji: String,
    /// English meaning(s).
    pub meaning: String,
    /// Part of speech: noun / godan-verb / ichidan-verb / i-adjective /
    /// na-adjective / particle / adverb / expression / counter.
    pub pos: String,
    /// JLPT level 1-5 (5 = beginner) or `None` if unranked.
    pub jlpt: Option<u8>,
    /// Example sentences in Japanese with English glosses.
    pub examples: Vec<Example>,
    /// How well the learner has internalized this word.
    pub mastery: Mastery,
    /// ISO-8601 timestamp of the most recent session that used this word,
    /// `None` if never practiced (only introduced).
    pub last_practiced: Option<String>,
}

/// A single kanji character — first-class because kanji learning is the
/// central memory challenge of Japanese.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kanji {
    /// The character itself, e.g. "学".
    pub character: String,
    /// On'yomi readings (Chinese-derived), e.g. ["ガク"].
    pub onyomi: Vec<String>,
    /// Kun'yomi readings (native Japanese), e.g. ["まな(ぶ)"].
    pub kunyomi: Vec<String>,
    /// Primary English meaning(s).
    pub meaning: String,
    /// Constituent radicals (e.g. "学" decomposes to ⺍ + 冖 + 子).
    pub radicals: Vec<String>,
    /// JLPT level 1-5 or `None`.
    pub jlpt: Option<u8>,
    /// Common compound words this kanji appears in.
    pub compounds: Vec<String>,
    /// Optional mnemonic the learner has attached (often more memorable
    /// than the etymology).
    pub mnemonic: Option<String>,
    pub mastery: Mastery,
}

/// A grammar concept — particles, conjugation patterns, register rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarPoint {
    /// Concept name, e.g. "te-form", "wa vs ga", "polite-past tense".
    pub name: String,
    /// Plain-language explanation.
    pub explanation: String,
    /// Worked examples in Japanese with English glosses.
    pub examples: Vec<Example>,
    /// Common errors learners make with this point.
    pub common_errors: Vec<String>,
    /// Related grammar points the learner may want to revisit alongside.
    pub related: Vec<String>,
    pub mastery: Mastery,
}

/// A conversational AI character. Personas accumulate lore over time —
/// session count, topics covered, in-character details — so the learner
/// has a continuing relationship rather than starting from scratch each
/// conversation.
///
/// The persona's system prompt is constructed from the wiki entity page
/// at `.dm/wiki/entities/Persona/<name>.md` plus the most recent session
/// summary, giving the LLM enough context to stay in character and pick
/// up the relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// Display name (English form), e.g. "Yuki".
    pub name: String,
    /// Native-script form, e.g. "ゆうき" or "雪".
    pub native_name: String,
    /// Role: tutor / conversation-partner / drill-instructor / cultural-guide.
    pub role: String,
    /// Politeness register the persona uses: casual / polite / honorific.
    pub register: String,
    /// Dialect or accent, e.g. "standard Tokyo", "Kansai", "Hokkaido".
    pub dialect: String,
    /// Topics this persona is well-suited to discuss.
    pub signature_topics: Vec<String>,
    /// Whether the persona scaffolds (mixes English glosses, accepts
    /// romaji) or runs strict immersion (Japanese only).
    pub immersion_mode: ImmersionMode,
    /// Total sessions conducted with this persona.
    pub sessions_count: u32,
}

/// A learning session — one conversation between learner and persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// ISO-8601 timestamp the session started.
    pub started_at: String,
    /// Persona the learner conversed with.
    pub persona: String,
    /// Topic(s) covered.
    pub topics: Vec<String>,
    /// Words the persona introduced this session (may be new to the wiki).
    pub words_introduced: Vec<String>,
    /// Words from the wiki the learner practiced this session.
    pub words_practiced: Vec<String>,
    /// Things the learner stumbled on — feed for the next planner cycle.
    pub struggles: Vec<String>,
    /// Approximate duration in minutes; `None` if not tracked.
    pub duration_minutes: Option<u32>,
}

/// A bilingual example sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub japanese: String,
    pub english: String,
}

/// How well the learner has internalized a vocabulary word, kanji, or
/// grammar point. Used by the planner to weight what to introduce or
/// review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mastery {
    /// First seen but not yet recognized in isolation.
    Introduced,
    /// Recognized when shown but not produced.
    Recognized,
    /// Recalled with effort, with prompting.
    Recalled,
    /// Used spontaneously in conversation.
    Used,
    /// Internalized — stable across sessions, used naturally.
    Internalized,
}

impl Mastery {
    /// Numeric weight for planner sorting (lower = higher priority for
    /// review).
    pub fn priority(self) -> u8 {
        match self {
            Mastery::Introduced => 1,
            Mastery::Recognized => 2,
            Mastery::Recalled => 3,
            Mastery::Used => 4,
            Mastery::Internalized => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImmersionMode {
    /// Persona speaks Japanese only, will rephrase in simpler Japanese
    /// when the learner is stuck. For learners with a stable base.
    Strict,
    /// Persona speaks Japanese with English glosses for new words and
    /// will translate on request. Forgiving for any starting level —
    /// the v0.1 default.
    Scaffolded,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mastery_priority_orders_introduced_first() {
        assert!(Mastery::Introduced.priority() < Mastery::Internalized.priority());
        assert!(Mastery::Recognized.priority() < Mastery::Used.priority());
    }

    #[test]
    fn vocabulary_round_trips_through_json() {
        let v = Vocabulary {
            kanji: "学校".into(),
            kana: "がっこう".into(),
            romaji: "gakkou".into(),
            meaning: "school".into(),
            pos: "noun".into(),
            jlpt: Some(5),
            examples: vec![Example {
                japanese: "学校に行きます。".into(),
                english: "I go to school.".into(),
            }],
            mastery: Mastery::Recognized,
            last_practiced: None,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: Vocabulary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kanji, "学校");
        assert_eq!(back.mastery, Mastery::Recognized);
    }

    #[test]
    fn persona_immersion_mode_serializes_snake_case() {
        let json = serde_json::to_string(&ImmersionMode::Scaffolded).unwrap();
        assert_eq!(json, "\"scaffolded\"");
    }
}
