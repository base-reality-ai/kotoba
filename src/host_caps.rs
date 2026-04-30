//! Kotoba host capabilities — the host tools dm gets registered when
//! kotoba boots in host mode.
//!
//! Each tool here is a thin bridge between the dm tool registry and
//! kotoba's wiki layer. They write to (or read from) the host-layer
//! wiki entity pages so that kotoba's "second brain" is always the
//! durable wiki, not in-process state.
//!
//! Tools must carry the `host_` prefix per [`dark_matter::host`] —
//! the registry's `register_host` call enforces this and rejects
//! collisions with kernel tools or MCP tools.
//!
//! Wiki reads/writes route through [`dark_matter::wiki::Wiki`]; we
//! avoid duplicating its parsing logic here.

use anyhow::Result;
use async_trait::async_trait;
use dark_matter::config;
use dark_matter::conversation;
use dark_matter::host::HostCapabilities;
use dark_matter::ollama::client::OllamaClient;
use dark_matter::tools::registry;
use dark_matter::tools::{Tool, ToolResult};
use dark_matter::wiki::{Layer, PageType, Wiki, WikiPage};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::domain::Mastery;

/// Locate the project root. Defaults to the current working dir; tools
/// that need to write to `.dm/wiki/` consult this once per call.
fn project_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Slugify a string for use as a wiki entity filename.
///
/// Japanese characters are preserved verbatim so a kanji-form word like
/// "学校" produces `学校.md`. ASCII whitespace becomes `-`; characters
/// outside `[A-Za-z0-9_\-぀-鿿]` are dropped. Keeps filesystem
/// portability without losing the native-script identity.
pub(crate) fn slugify(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('-');
        } else if matches!(ch, '\u{3040}'..='\u{9fff}') {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "untitled".into()
    } else {
        out
    }
}

// ---------------------------------------------------------------------------
// host_invoke_persona
// ---------------------------------------------------------------------------

pub(crate) const PERSONA_LOADED_PREFIX: &str = "Persona '";
pub(crate) const PERSONA_LOADED_AFTER_NAME: &str =
    " loaded. The next agent turn should speak in this voice. Persona definition follows:";

/// Switches the active conversational persona. Reads the named
/// persona's wiki entity page (`.dm/wiki/entities/Persona/<name>.md`)
/// and surfaces it as the system-prompt-shaped response so the next
/// agent invocation in the chain takes on that voice.
pub struct InvokePersonaTool;

#[async_trait]
impl Tool for InvokePersonaTool {
    fn name(&self) -> &'static str {
        "host_invoke_persona"
    }

    fn description(&self) -> &'static str {
        "Switch the active conversational persona for the next turn. Reads the persona's wiki entity (e.g. Yuki) and returns its system-prompt-shaped frontmatter + body so the next chain node speaks in that voice."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_invoke_persona to switch the conversational voice mid-session \
             (e.g. operator types `/persona Hiro`). The tool returns the persona's \
             wiki entity body which becomes the next turn's persona context.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Persona name (matches Persona/<name>.md in the wiki). E.g. \"Yuki\"."
                }
            },
            "required": ["name"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `name` field"))?;

        let root = project_root();
        let persona_path = root
            .join(".dm/wiki/entities/Persona")
            .join(format!("{}.md", slugify(name)));

        if !persona_path.exists() {
            return Ok(ToolResult {
                content: format!(
                    "Persona '{}' not found at {}. Try: list the wiki's Persona directory or create the entity first.",
                    name,
                    persona_path.display()
                ),
                is_error: true,
            });
        }

        let body = std::fs::read_to_string(&persona_path)?;
        Ok(ToolResult {
            content: format!(
                "{}{}'{}\n\n{}",
                PERSONA_LOADED_PREFIX, name, PERSONA_LOADED_AFTER_NAME, body
            ),
            is_error: false,
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// host_log_vocabulary
// ---------------------------------------------------------------------------

/// Persists a newly-encountered vocabulary word to the wiki as a
/// host-layer entity page. If a page for the word already exists it
/// is updated rather than overwritten — preserving prior examples and
/// mastery progression.
pub struct LogVocabularyTool;

#[async_trait]
impl Tool for LogVocabularyTool {
    fn name(&self) -> &'static str {
        "host_log_vocabulary"
    }

    fn description(&self) -> &'static str {
        "Record a Japanese vocabulary word in the wiki (host-layer). Use whenever a new word is introduced in a session, or when an existing word's example/mastery should be updated."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_log_vocabulary the moment you introduce a new Japanese word \
             to the learner, or when revisiting an old one. Always pass at least \
             `kana` and `meaning`; add `kanji`, `romaji`, `pos`, `jlpt`, and an \
             example sentence when known. The wiki upserts on slug, so re-logging \
             the same word does not duplicate — it refreshes mastery context.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "kanji": {"type": "string", "description": "Kanji form (or empty for kana-only words)."},
                "kana": {"type": "string", "description": "Kana reading."},
                "romaji": {"type": "string", "description": "Romaji transliteration."},
                "meaning": {"type": "string", "description": "English meaning(s)."},
                "pos": {"type": "string", "description": "Part of speech (noun/godan-verb/ichidan-verb/i-adjective/na-adjective/particle/expression)."},
                "jlpt": {"type": "integer", "description": "JLPT level 1-5 (5 = beginner). Optional."},
                "example_japanese": {"type": "string", "description": "An example sentence in Japanese. Optional."},
                "example_english": {"type": "string", "description": "English gloss for the example. Optional."}
            },
            "required": ["kana", "meaning"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let kanji = args.get("kanji").and_then(|v| v.as_str()).unwrap_or("");
        let kana = args
            .get("kana")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `kana` field"))?;
        let romaji = args.get("romaji").and_then(|v| v.as_str()).unwrap_or("");
        let meaning = args
            .get("meaning")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `meaning` field"))?;
        let pos = args
            .get("pos")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let jlpt = args.get("jlpt").and_then(|v| v.as_u64());
        let example_jp = args.get("example_japanese").and_then(|v| v.as_str());
        let example_en = args.get("example_english").and_then(|v| v.as_str());

        let root = project_root();
        let page_path = log_vocabulary_in(
            &root,
            VocabularyInput {
                kanji,
                kana,
                romaji,
                meaning,
                pos,
                jlpt,
                example_japanese: example_jp,
                example_english: example_en,
            },
        )?;

        Ok(ToolResult {
            content: format!(
                "Vocabulary logged: {} ({}, \"{}\") → {}",
                if kanji.is_empty() { kana } else { kanji },
                kana,
                meaning,
                page_path
                    .strip_prefix(&root)
                    .unwrap_or(&page_path)
                    .display()
            ),
            is_error: false,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct VocabularyInput<'a> {
    kanji: &'a str,
    kana: &'a str,
    romaji: &'a str,
    meaning: &'a str,
    pos: &'a str,
    jlpt: Option<u64>,
    example_japanese: Option<&'a str>,
    example_english: Option<&'a str>,
}

fn log_vocabulary_in(root: &Path, input: VocabularyInput<'_>) -> Result<PathBuf> {
    // Use kanji form as slug if present; fall back to kana otherwise.
    let slug = slugify(if input.kanji.is_empty() {
        input.kana
    } else {
        input.kanji
    });

    let title = if input.kanji.is_empty() {
        input.kana
    } else {
        input.kanji
    };
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // Body markdown — same shape as the pre-canonical-9259acb raw write.
    // JLPT level used to live in the frontmatter; WikiPage's structured
    // fields don't have a slot for it, so it moves into the body where
    // it remains discoverable by wiki_search and renders the same to
    // a human reader.
    let mut body = String::new();
    body.push_str(&format!("# {}\n\n", title));
    if !input.kanji.is_empty() {
        body.push_str(&format!("- **Kanji:** {}\n", input.kanji));
    }
    body.push_str(&format!("- **Kana:** {}\n", input.kana));
    if !input.romaji.is_empty() {
        body.push_str(&format!("- **Romaji:** {}\n", input.romaji));
    }
    body.push_str(&format!("- **Meaning:** {}\n", input.meaning));
    body.push_str(&format!("- **Part of speech:** {}\n", input.pos));
    if let Some(level) = input.jlpt {
        body.push_str(&format!("- **JLPT:** N{}\n", level));
    }
    body.push_str(&format!("- **Mastery:** {:?}\n\n", Mastery::Introduced));

    if let (Some(jp), Some(en)) = (input.example_japanese, input.example_english) {
        body.push_str("## Examples\n\n");
        body.push_str(&format!("- {}\n  - _{}_\n", jp, en));
    }

    let page = WikiPage {
        title: title.to_string(),
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated: now,
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],

        extras: std::collections::BTreeMap::new(),
        body,
    };

    let one_liner = format!("{} ({}) — {}", title, input.kana, input.meaning);
    let relative_path = format!("entities/Vocabulary/{}.md", slug);

    let wiki = Wiki::open(root)?;
    wiki.register_host_page(&relative_path, &page, &one_liner)?;

    Ok(root.join(".dm/wiki").join(&relative_path))
}

// ---------------------------------------------------------------------------
// host_log_kanji
// ---------------------------------------------------------------------------

/// Persists a kanji entity (separate from vocabulary because kanji
/// learning is the central memory challenge of Japanese — they
/// deserve first-class wiki pages).
pub struct LogKanjiTool;

#[async_trait]
impl Tool for LogKanjiTool {
    fn name(&self) -> &'static str {
        "host_log_kanji"
    }

    fn description(&self) -> &'static str {
        "Record a kanji character in the wiki (host-layer) with its readings, radicals, and meaning. Use whenever a new kanji is introduced in conversation or when adding a memorable mnemonic."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_log_kanji whenever you introduce a kanji character or attach \
             a mnemonic. Pass on'yomi/kun'yomi readings, radicals, and JLPT level \
             when known. Mnemonics dramatically improve retention — include one \
             whenever you can think of a memorable visual story.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "character": {"type": "string", "description": "The kanji character itself, e.g. \"学\"."},
                "meaning": {"type": "string", "description": "Primary English meaning(s)."},
                "onyomi": {"type": "array", "items": {"type": "string"}, "description": "On'yomi readings (Chinese-derived). E.g. [\"ガク\"]."},
                "kunyomi": {"type": "array", "items": {"type": "string"}, "description": "Kun'yomi readings (native Japanese). E.g. [\"まな(ぶ)\"]."},
                "radicals": {"type": "array", "items": {"type": "string"}, "description": "Constituent radicals."},
                "jlpt": {"type": "integer", "description": "JLPT level 1-5. Optional."},
                "mnemonic": {"type": "string", "description": "Optional memorable story for retention."}
            },
            "required": ["character", "meaning"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let character = args
            .get("character")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `character` field"))?;
        let meaning = args
            .get("meaning")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `meaning` field"))?;

        let onyomi: Vec<String> = args
            .get("onyomi")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let kunyomi: Vec<String> = args
            .get("kunyomi")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let radicals: Vec<String> = args
            .get("radicals")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let jlpt = args.get("jlpt").and_then(|v| v.as_u64());
        let mnemonic = args.get("mnemonic").and_then(|v| v.as_str());

        let root = project_root();
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

        // Body markdown — same shape as the pre-9259acb raw write.
        // JLPT was previously a frontmatter line; WikiPage's structured
        // fields don't carry it, so it moves into the body where
        // wiki_search still finds it.
        let mut body = String::new();
        body.push_str(&format!("# {}\n\n", character));
        body.push_str(&format!("- **Meaning:** {}\n", meaning));
        if !onyomi.is_empty() {
            body.push_str(&format!("- **On'yomi:** {}\n", onyomi.join(", ")));
        }
        if !kunyomi.is_empty() {
            body.push_str(&format!("- **Kun'yomi:** {}\n", kunyomi.join(", ")));
        }
        if !radicals.is_empty() {
            body.push_str(&format!("- **Radicals:** {}\n", radicals.join(" + ")));
        }
        if let Some(level) = jlpt {
            body.push_str(&format!("- **JLPT:** N{}\n", level));
        }
        body.push_str(&format!("- **Mastery:** {:?}\n", Mastery::Introduced));
        if let Some(m) = mnemonic {
            body.push_str(&format!("\n## Mnemonic\n\n{}\n", m));
        }

        let page = WikiPage {
            title: character.to_string(),
            page_type: PageType::Entity,
            layer: Layer::Host,
            sources: vec![],
            last_updated: now,
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],

            extras: std::collections::BTreeMap::new(),
            body,
        };

        let one_liner = format!("{} — {}", character, meaning);
        let relative_path = format!("entities/Kanji/{}.md", slugify(character));
        let page_path = root.join(".dm/wiki").join(&relative_path);

        let wiki = Wiki::open(&root)?;
        wiki.register_host_page(&relative_path, &page, &one_liner)?;

        Ok(ToolResult {
            content: format!(
                "Kanji logged: {} (\"{}\") → {}",
                character,
                meaning,
                page_path
                    .strip_prefix(&root)
                    .unwrap_or(&page_path)
                    .display()
            ),
            is_error: false,
        })
    }
}

// ---------------------------------------------------------------------------
// host_record_struggle
// ---------------------------------------------------------------------------

/// Records a learner's stumbling point — a vocabulary word, grammar
/// concept, or idiomatic phrase that confused them — into a session-
/// scoped synthesis page. The next planner cycle reads these to weight
/// review priorities.
pub struct RecordStruggleTool;

#[async_trait]
impl Tool for RecordStruggleTool {
    fn name(&self) -> &'static str {
        "host_record_struggle"
    }

    fn description(&self) -> &'static str {
        "Flag a topic the learner stumbled on — confusion between similar words, mis-conjugation, particle confusion, etc. The planner agent uses these to prioritize what to review in the next session."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_record_struggle the moment a learner shows clear confusion: \
             mis-conjugating a verb, picking the wrong particle, mixing similar \
             words. Be concrete in `what_got_confused` (\"used wa instead of ga\") \
             — the next session's planner uses these strings to weight review.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "What was struggled with — vocabulary word, grammar point, idiom, etc."},
                "what_got_confused": {"type": "string", "description": "Specific confusion (\"used wa instead of ga\", \"mixed up similar-sounding words\", etc.)."}
            },
            "required": ["topic", "what_got_confused"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let topic = args
            .get("topic")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `topic` field"))?;
        let what = args
            .get("what_got_confused")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `what_got_confused` field"))?;

        let root = project_root();
        let now = chrono::Utc::now();
        let page_path = record_struggle_in(&root, topic, what, now)?;

        Ok(ToolResult {
            content: format!(
                "Struggle recorded: {} — {}. Today's struggles → {}",
                topic,
                what,
                page_path
                    .strip_prefix(&root)
                    .unwrap_or(&page_path)
                    .display()
            ),
            is_error: false,
        })
    }
}

fn record_struggle_in(
    root: &Path,
    topic: &str,
    what: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<PathBuf> {
    let date = now.format("%Y-%m-%d").to_string();
    let timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let entry = format!("- **{}** — {} _(logged {})_\n", topic, what, timestamp);

    let relative_path = format!("synthesis/struggles-{}.md", date);
    let wiki = Wiki::open(root)?;

    let mut page = match wiki.read_page(&relative_path) {
        Ok(existing) => existing,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WikiPage {
            title: format!("Struggles {}", date),
            page_type: PageType::Synthesis,
            layer: Layer::Host,
            sources: vec![],
            last_updated: timestamp.clone(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],

            extras: std::collections::BTreeMap::new(),
            body: format!("# Struggles for {}\n\n", date),
        },
        Err(e) => return Err(e.into()),
    };

    page.body.push_str(&entry);
    page.last_updated = timestamp;

    let one_liner = format!("Struggles for {}", date);
    wiki.register_host_page(&relative_path, &page, &one_liner)?;

    Ok(root.join(".dm/wiki").join(&relative_path))
}

// ---------------------------------------------------------------------------
// host_quiz_me
// ---------------------------------------------------------------------------

/// Pulls vocabulary entities from the host-layer wiki for review.
/// Filters to those with low mastery and lists them; the next agent
/// turn can use the list to drill the learner.
pub struct QuizMeTool;

#[async_trait]
impl Tool for QuizMeTool {
    fn name(&self) -> &'static str {
        "host_quiz_me"
    }

    fn description(&self) -> &'static str {
        "Pull vocabulary entries from the wiki for review. Returns up to `count` entries, prioritizing words at lower mastery levels (Introduced → Recognized → Recalled)."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_quiz_me when the learner asks to review or you detect they \
             want a drill. Don't dump the raw output — work the words back into \
             the conversation as natural prompts (\"what's the word for 'school' \
             again?\") rather than reading them aloud as a flashcard list.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Maximum entries to return. Default 5.", "default": 5},
                "topic": {"type": "string", "description": "Optional topic filter (matches against meaning/POS). Omit for any topic."}
            }
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let count = args.get("count").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let topic_filter = args.get("topic").and_then(|v| v.as_str());

        let root = project_root();
        let dir = root.join(".dm/wiki/entities/Vocabulary");
        if !dir.exists() {
            return Ok(ToolResult {
                content: "No vocabulary entries yet. Try: have a session with a persona to introduce some words first.".into(),
                is_error: false,
            });
        }

        let mut entries: Vec<(String, String)> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let body = std::fs::read_to_string(entry.path())?;
            if let Some(filter) = topic_filter {
                if !body.contains(filter) {
                    continue;
                }
            }
            let title = entry
                .file_name()
                .to_string_lossy()
                .trim_end_matches(".md")
                .to_string();
            entries.push((title, body));
            if entries.len() >= count {
                break;
            }
        }

        if entries.is_empty() {
            return Ok(ToolResult {
                content: format!(
                    "No vocabulary matching topic={:?} found. The wiki has {} entries total.",
                    topic_filter,
                    std::fs::read_dir(&dir)?.count()
                ),
                is_error: false,
            });
        }

        let mut report = format!("Quiz selection ({} word(s)):\n\n", entries.len());
        for (title, body) in entries {
            report.push_str(&format!("## {}\n\n{}\n\n---\n\n", title, body));
        }

        Ok(ToolResult {
            content: report,
            is_error: false,
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// build_persona_system_prompt
// ---------------------------------------------------------------------------

/// Marker text injected into the persona system prompt to make wiki recall
/// explicit. Tests pin against this to assert the recall-guidance section
/// is present; if it changes, both halves move together.
#[allow(dead_code)]
pub(crate) const PERSONA_RECALL_GUIDANCE_MARKER: &str = "## Wiki recall (the second brain)";

/// Compose the persona system prompt that kotoba hands to the dm session via
/// `--system`.
///
/// dm itself appends a `<tool_usage>` block listing every registered tool's
/// hint (including `wiki_search` / `wiki_lookup`) — see canonical
/// `tools::registry::system_prompt_hints_for`. That covers tool *availability*
/// for the model. This helper covers tool *usage policy* the persona is
/// expected to follow: the persona must call `wiki_search` BEFORE answering
/// questions about previously encountered material, so the second brain
/// actually compounds across sessions.
///
/// Without this, models often introduce a word fresh even though the learner
/// already saw it last session — the kernel-side `<tool_usage>` block tells
/// them the tool exists, but not when to reach for it.
#[allow(dead_code)]
pub fn build_persona_system_prompt(persona: &str, persona_body: &str, brief: &str) -> String {
    let mut out = String::with_capacity(persona_body.len() + brief.len() + 512);
    out.push_str(&format!(
        "You are the Japanese-learning persona '{}'. The persona's wiki entity follows.\n\n",
        persona
    ));
    out.push_str(persona_body);
    if !persona_body.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(PERSONA_RECALL_GUIDANCE_MARKER);
    out.push_str(
        "\n\nThe wiki at `.dm/wiki/` accumulates every word, kanji, struggle, \
         and prior session this learner has encountered. Before answering a \
         question about a word, kanji, grammar point, or topic the learner \
         mentions, call `wiki_search` (or `wiki_lookup` for a known page \
         path) and ground your reply in the prior entry. If the search \
         returns nothing, treat the material as fresh and log it via \
         `host_log_vocabulary` / `host_log_kanji` so the next session sees \
         it. Recall first, introduce only when truly new.\n\n",
    );
    out.push_str("--- Planner brief for this session ---\n\n");
    out.push_str(brief);
    out
}

// ---------------------------------------------------------------------------
// host_plan_session
// ---------------------------------------------------------------------------

/// Reads the host-layer wiki and synthesizes a pre-session brief for the
/// next conversation: today's focus, words to surface (sorted by mastery),
/// grammar points carried over from recent struggles, and a conversational
/// hook for the persona to open with.
///
/// v0.2 acceptance: rule-based + lightly templated. ML-driven prioritization
/// is v0.3 fodder. The brief is written so the persona's system prompt can
/// concatenate it directly as scaffolding without further rewriting.
pub struct PlanSessionTool;

#[async_trait]
impl Tool for PlanSessionTool {
    fn name(&self) -> &'static str {
        "host_plan_session"
    }

    fn description(&self) -> &'static str {
        "Pre-session planner. Reads the wiki (vocabulary mastery, recent struggles, the active persona's lore) and returns a structured markdown brief: today's focus topic, 3-5 words to surface for review, 1-2 grammar points to revisit, and one conversational hook the persona can open with. Call this before launching a learning session; pass the result into the persona's system prompt."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_plan_session at the start of a learning session before the \
             persona opens. Pass the active `persona` and an optional \
             `recent_struggle_days` window. The returned brief is meant to be \
             quoted into the persona's opening turn — don't paraphrase it away.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "Active persona name (matches Persona/<name>.md). Defaults to \"Yuki\".",
                    "default": "Yuki"
                },
                "recent_struggle_days": {
                    "type": "integer",
                    "description": "How many days back to include struggle entries from. Default 3.",
                    "default": 3
                }
            }
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let persona = args
            .get("persona")
            .and_then(|v| v.as_str())
            .unwrap_or("Yuki")
            .to_string();
        let recent_days = args
            .get("recent_struggle_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as i64;

        let root = project_root();
        let today = chrono::Utc::now().date_naive();
        let planner_model = planner_model_from_env();
        let brief =
            plan_session_in_with_optional_llm(&root, &persona, recent_days, today, &planner_model)
                .await?;

        Ok(ToolResult {
            content: brief,
            is_error: false,
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

/// Compact summary of a single Vocabulary wiki page used for planner sorting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VocabSummary {
    /// File stem (typically the kanji form, e.g. "学校").
    pub title: String,
    /// Kana reading parsed from the body's `- **Kana:**` line. Empty if absent.
    pub kana: String,
    /// English meaning parsed from the body. Empty if absent.
    pub meaning: String,
    pub mastery: Mastery,
}

/// Compact summary of a Persona wiki page used to template the brief.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersonaSummary {
    pub name: String,
    pub sessions_count: u32,
    pub signature_topics: Vec<String>,
}

/// Wiki state the planner needs, independent of whether the renderer is the
/// current rule template or a future single-shot model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannerWikiState {
    pub requested_persona: String,
    pub today: chrono::NaiveDate,
    pub persona: Option<PersonaSummary>,
    pub vocab: Vec<VocabSummary>,
    pub struggles: Vec<String>,
}

/// Parse the leading YAML-ish frontmatter block (between two `---` lines).
/// Returns key→value pairs. Values are trimmed; types are stringly. Returns
/// an empty map if no frontmatter is present.
pub(crate) fn parse_frontmatter(body: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let mut lines = body.lines();
    let Some(first) = lines.next() else {
        return out;
    };
    if first.trim() != "---" {
        return out;
    }
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

/// Find a `- **Field:** value` line in body and return the value, trimmed.
fn body_field(body: &str, field: &str) -> Option<String> {
    let needle = format!("- **{}:**", field);
    body.lines()
        .find_map(|l| l.trim().strip_prefix(&needle).map(|v| v.trim().to_string()))
}

fn parse_mastery(s: &str) -> Mastery {
    match s.trim() {
        "Recognized" => Mastery::Recognized,
        "Recalled" => Mastery::Recalled,
        "Used" => Mastery::Used,
        "Internalized" => Mastery::Internalized,
        _ => Mastery::Introduced,
    }
}

pub(crate) fn read_vocab_entries(root: &std::path::Path) -> Vec<VocabSummary> {
    let dir = root.join(".dm/wiki/entities/Vocabulary");
    let Ok(read) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in read.flatten() {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let kana = body_field(&body, "Kana").unwrap_or_default();
        let meaning = body_field(&body, "Meaning").unwrap_or_default();
        let mastery = body_field(&body, "Mastery")
            .map(|s| parse_mastery(&s))
            .unwrap_or(Mastery::Introduced);
        out.push(VocabSummary {
            title,
            kana,
            meaning,
            mastery,
        });
    }
    out
}

pub(crate) fn read_persona_summary(root: &std::path::Path, name: &str) -> Option<PersonaSummary> {
    let path = root
        .join(".dm/wiki/entities/Persona")
        .join(format!("{}.md", slugify(name)));
    let body = std::fs::read_to_string(&path).ok()?;
    let fm = parse_frontmatter(&body);
    let page_body = markdown_body_without_frontmatter(&body);
    let sessions_count = parse_persona_sessions_count(page_body)
        .or_else(|| fm.get("sessions_count").and_then(|v| v.parse::<u32>().ok()))
        .unwrap_or(0);
    // Pull bullet items under "## Signature topics" until next `##` heading.
    let mut topics = Vec::new();
    let mut in_section = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") {
            in_section = trimmed.eq_ignore_ascii_case("## Signature topics");
            continue;
        }
        if in_section {
            if let Some(item) = trimmed.strip_prefix("- ") {
                topics.push(item.to_string());
            }
        }
    }
    Some(PersonaSummary {
        name: name.to_string(),
        sessions_count,
        signature_topics: topics,
    })
}

/// Read struggle synthesis pages whose filename date is within `days_back`
/// of `today` (inclusive on both ends). Returns the lines starting with `-`
/// from each matching page, in newest-first date order.
pub(crate) fn read_recent_struggles(
    root: &std::path::Path,
    days_back: i64,
    today: chrono::NaiveDate,
) -> Vec<String> {
    let dir = root.join(".dm/wiki/synthesis");
    let Ok(read) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut dated: Vec<(chrono::NaiveDate, std::path::PathBuf)> = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(date_str) = stem.strip_prefix("struggles-") else {
            continue;
        };
        let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") else {
            continue;
        };
        let age = (today - date).num_days();
        if (0..=days_back).contains(&age) {
            dated.push((date, path));
        }
    }
    dated.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = Vec::new();
    for (_, path) in dated {
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("- ") {
                out.push(trimmed.to_string());
            }
        }
    }
    out
}

pub(crate) fn build_session_brief(
    persona: Option<&PersonaSummary>,
    vocab: &[VocabSummary],
    struggles: &[String],
    today: chrono::NaiveDate,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Session brief — {}\n\n", today));

    let persona_label = match persona {
        Some(p) => format!("**Persona:** {} (session {})", p.name, p.sessions_count + 1),
        None => "**Persona:** (none registered — running first session)".to_string(),
    };
    s.push_str(&persona_label);
    s.push_str("\n\n");

    // Today's focus: rotate through the persona's signature topics by
    // sessions_count; fall back to a starter topic when the wiki is bare.
    let focus = match persona {
        Some(p) if !p.signature_topics.is_empty() => {
            let idx = (p.sessions_count as usize) % p.signature_topics.len();
            p.signature_topics[idx].clone()
        }
        _ => "Self-introduction and greetings (starter topic — no persona lore yet)".to_string(),
    };
    s.push_str(&format!("## Today's focus\n\n{}\n\n", focus));

    // Words to surface: lowest-mastery first, take up to 5. Ties broken by
    // title for determinism.
    s.push_str("## Words to surface\n\n");
    if vocab.is_empty() {
        s.push_str(
            "_Wiki has no vocabulary yet — this looks like the first session. The persona should introduce 3-5 starter words organically._\n\n",
        );
    } else {
        let mut sorted = vocab.to_vec();
        sorted.sort_by(|a, b| {
            a.mastery
                .priority()
                .cmp(&b.mastery.priority())
                .then_with(|| a.title.cmp(&b.title))
        });
        for v in sorted.iter().take(5) {
            let kana = if v.kana.is_empty() {
                String::new()
            } else {
                format!(" ({})", v.kana)
            };
            let meaning = if v.meaning.is_empty() {
                String::new()
            } else {
                format!(" — {}", v.meaning)
            };
            s.push_str(&format!(
                "- **{}**{}{} _(mastery: {:?})_\n",
                v.title, kana, meaning, v.mastery
            ));
        }
        s.push('\n');
    }

    // Grammar to revisit: pulled from recent struggles, capped at 2.
    s.push_str("## Grammar / patterns to revisit\n\n");
    if struggles.is_empty() {
        s.push_str("_No recent struggles flagged. Persona may introduce a new grammar point if the conversation calls for it._\n\n");
    } else {
        for line in struggles.iter().take(2) {
            s.push_str(line);
            s.push('\n');
        }
        if struggles.len() > 2 {
            s.push_str(&format!(
                "\n_+{} more struggle(s) on file; see `.dm/wiki/synthesis/struggles-*.md`._\n",
                struggles.len() - 2
            ));
        }
        s.push('\n');
    }

    // Conversational hook — templated based on whether we know the persona.
    s.push_str("## Conversational hook\n\n");
    let hook = match persona {
        Some(p) if p.sessions_count == 0 => format!(
            "> 「はじめまして！{}です。今日は{}について話しましょうか。」\n",
            p.name, focus
        ),
        Some(_) => format!(
            "> 「お久しぶりです！前回の続きで、{}を一緒に練習しませんか。」\n",
            focus
        ),
        None => format!(
            "> Open with a greeting in です/ます register and propose: \"{}\".\n",
            focus
        ),
    };
    s.push_str(&hook);
    s.push('\n');

    s
}

pub(crate) fn collect_planner_wiki_state(
    root: &std::path::Path,
    persona_name: &str,
    recent_days: i64,
    today: chrono::NaiveDate,
) -> PlannerWikiState {
    PlannerWikiState {
        requested_persona: persona_name.to_string(),
        today,
        persona: read_persona_summary(root, persona_name),
        vocab: read_vocab_entries(root),
        struggles: read_recent_struggles(root, recent_days, today),
    }
}

#[allow(dead_code)]
pub(crate) fn build_planner_prompt(state: &PlannerWikiState) -> String {
    let mut s = String::new();
    s.push_str("You are kotoba's pre-session planner.\n");
    s.push_str("Produce a concise markdown session brief with exactly these sections:\n");
    s.push_str("- Session brief header with date\n");
    s.push_str("- Persona\n");
    s.push_str("- Today's focus\n");
    s.push_str("- Words to surface\n");
    s.push_str("- Grammar / patterns to revisit\n");
    s.push_str("- Conversational hook\n\n");

    s.push_str("## Planner input\n\n");
    s.push_str(&format!("- Date: {}\n", state.today));
    s.push_str(&format!(
        "- Requested persona: {}\n",
        state.requested_persona
    ));
    match &state.persona {
        Some(persona) => {
            s.push_str(&format!(
                "- Persona wiki entity: {} (sessions_count: {})\n",
                persona.name, persona.sessions_count
            ));
            if persona.signature_topics.is_empty() {
                s.push_str("- Signature topics: none recorded\n");
            } else {
                s.push_str(&format!(
                    "- Signature topics: {}\n",
                    persona.signature_topics.join("; ")
                ));
            }
        }
        None => s.push_str("- Persona wiki entity: missing\n"),
    }

    s.push_str("\n## Vocabulary candidates\n\n");
    if state.vocab.is_empty() {
        s.push_str("- none recorded\n");
    } else {
        let mut vocab = state.vocab.clone();
        vocab.sort_by(|a, b| {
            a.mastery
                .priority()
                .cmp(&b.mastery.priority())
                .then_with(|| a.title.cmp(&b.title))
        });
        for item in vocab.iter().take(10) {
            let kana = if item.kana.is_empty() {
                String::new()
            } else {
                format!(" ({})", item.kana)
            };
            let meaning = if item.meaning.is_empty() {
                String::new()
            } else {
                format!(" — {}", item.meaning)
            };
            s.push_str(&format!(
                "- {}{}{} [mastery: {:?}]\n",
                item.title, kana, meaning, item.mastery
            ));
        }
    }

    s.push_str("\n## Recent struggles\n\n");
    if state.struggles.is_empty() {
        s.push_str("- none recorded\n");
    } else {
        for struggle in &state.struggles {
            s.push_str(struggle);
            s.push('\n');
        }
    }

    s
}

fn env_flag_enabled(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn planner_model_from_env() -> String {
    std::env::var("KOTOBA_PLANNER_MODEL").unwrap_or_else(|_| "claude-opus".to_string())
}

fn recorder_model_from_env() -> String {
    std::env::var("KOTOBA_RECORDER_MODEL").unwrap_or_else(|_| "claude-opus".to_string())
}

fn ollama_base_url() -> String {
    if let Ok(raw) = std::env::var("OLLAMA_HOST") {
        return format!("http://{}/api", config::normalize_host(&raw));
    }
    config::Config::load()
        .map(|cfg| cfg.ollama_base_url())
        .unwrap_or_else(|_| "http://localhost:11434/api".to_string())
}

pub(crate) async fn plan_session_in_with_optional_llm(
    root: &std::path::Path,
    persona_name: &str,
    recent_days: i64,
    today: chrono::NaiveDate,
    planner_model: &str,
) -> Result<String> {
    let state = collect_planner_wiki_state(root, persona_name, recent_days, today);
    if !env_flag_enabled("KOTOBA_PLANNER_USE_LLM") {
        return Ok(build_session_brief_from_state(&state));
    }

    let prompt = build_planner_prompt(&state);
    let config_dir = root.join(".dm");
    std::fs::create_dir_all(&config_dir)?;
    let client = OllamaClient::new(ollama_base_url(), planner_model.to_string());
    let registry = registry::default_registry(
        "kotoba-planner",
        &config_dir,
        client.base_url(),
        client.model(),
        "",
    );
    let capture = conversation::run_conversation_capture_in_config_dir(
        &prompt,
        "kotoba-planner",
        &client,
        &registry,
        &config_dir,
    )
    .await?;
    Ok(capture.text)
}

pub(crate) fn build_session_brief_from_state(state: &PlannerWikiState) -> String {
    build_session_brief(
        state.persona.as_ref(),
        &state.vocab,
        &state.struggles,
        state.today,
    )
}

/// Pure top-level entry point used both by [`PlanSessionTool::call`] and by
/// unit tests. Wikiless invocations and partial-wiki invocations both yield
/// a sensible brief — never an error.
#[allow(dead_code)]
pub(crate) fn plan_session_in(
    root: &std::path::Path,
    persona_name: &str,
    recent_days: i64,
    today: chrono::NaiveDate,
) -> String {
    let state = collect_planner_wiki_state(root, persona_name, recent_days, today);
    build_session_brief_from_state(&state)
}

// ---------------------------------------------------------------------------
// host_record_session
// ---------------------------------------------------------------------------

/// Post-session recorder. Reads a saved transcript, extracts lightweight
/// evidence of new vocabulary and learner struggles, updates the wiki, and
/// appends a summary entry to the active persona's session log.
pub struct RecordSessionTool;

#[async_trait]
impl Tool for RecordSessionTool {
    fn name(&self) -> &'static str {
        "host_record_session"
    }

    fn description(&self) -> &'static str {
        "Post-session recorder. Given a transcript and persona name, records newly introduced Japanese vocabulary, flags learner struggles, appends a session log entry to the persona page, and increments that persona's sessions_count."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use host_record_session AFTER a session ends, with the full \
             transcript and the active persona. It bulk-extracts vocab and \
             struggles and bumps the persona's session counter. Run it once per \
             session; running mid-session double-counts.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Full session transcript to summarize into the wiki."
                },
                "persona": {
                    "type": "string",
                    "description": "Persona name whose wiki entity should receive the session entry. Defaults to \"Yuki\".",
                    "default": "Yuki"
                }
            },
            "required": ["transcript"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let transcript = args
            .get("transcript")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required `transcript` field"))?;
        let persona = args
            .get("persona")
            .and_then(|v| v.as_str())
            .unwrap_or("Yuki");

        let root = project_root();
        let now = chrono::Utc::now();
        let recorder_model = recorder_model_from_env();
        let summary =
            record_session_in_with_optional_llm(&root, transcript, persona, now, &recorder_model)
                .await?;

        Ok(ToolResult {
            content: summary.to_markdown(),
            is_error: false,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DetectedVocabulary {
    pub kanji: String,
    pub kana: String,
    pub meaning: String,
    pub example_japanese: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DetectedStruggle {
    pub topic: String,
    pub what_got_confused: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionRecordSummary {
    pub persona: String,
    pub vocabulary_count: usize,
    pub struggle_count: usize,
    pub sessions_count: u32,
    pub words: Vec<String>,
    pub struggles: Vec<String>,
}

impl SessionRecordSummary {
    fn to_markdown(&self) -> String {
        let words = if self.words.is_empty() {
            "none".to_string()
        } else {
            self.words.join(", ")
        };
        let struggles = if self.struggles.is_empty() {
            "none".to_string()
        } else {
            self.struggles.join(", ")
        };
        format!(
            "Session recorded for {}.\n- Vocabulary logged: {} ({})\n- Struggles flagged: {} ({})\n- Persona sessions_count: {}",
            self.persona,
            self.vocabulary_count,
            words,
            self.struggle_count,
            struggles,
            self.sessions_count
        )
    }
}

pub(crate) fn record_session_in(
    root: &Path,
    transcript: &str,
    persona: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<SessionRecordSummary> {
    let segments = split_persona_segments(transcript, persona);
    if segments.len() > 1 {
        let mut summaries = Vec::new();
        for segment in segments {
            summaries.push(record_persona_segment_in(
                root,
                &segment.transcript,
                &segment.persona,
                now,
            )?);
        }
        return Ok(merge_segment_summaries(summaries, persona));
    }

    record_persona_segment_in(root, transcript, persona, now)
}

pub(crate) async fn record_session_in_with_optional_llm(
    root: &Path,
    transcript: &str,
    persona: &str,
    now: chrono::DateTime<chrono::Utc>,
    recorder_model: &str,
) -> Result<SessionRecordSummary> {
    if !env_flag_enabled("KOTOBA_RECORDER_USE_LLM") {
        return record_session_in(root, transcript, persona, now);
    }

    let prompt = build_recorder_prompt(transcript, persona);
    let config_dir = root.join(".dm");
    std::fs::create_dir_all(&config_dir)?;
    let client = OllamaClient::new(ollama_base_url(), recorder_model.to_string());
    let registry = registry::default_registry(
        "kotoba-recorder",
        &config_dir,
        client.base_url(),
        client.model(),
        "",
    );
    let capture = conversation::run_conversation_capture_in_config_dir(
        &prompt,
        "kotoba-recorder",
        &client,
        &registry,
        &config_dir,
    )
    .await?;

    record_session_in(root, &capture.text, persona, now)
}

#[allow(dead_code)]
pub(crate) fn build_recorder_prompt(transcript: &str, persona: &str) -> String {
    format!(
        "You are kotoba's post-session recorder.\n\
         Convert the transcript into concise evidence lines that kotoba's \
         rule-based recorder can ingest.\n\n\
         Output only lines in these forms when supported by the transcript:\n\
         - Recorder: New word: <Japanese> (<kana>) means <English gloss>.\n\
         - Learner: I don't know <topic>.\n\n\
         Use the active persona name only as context; do not invent vocabulary, \
         readings, meanings, or struggles. If nothing should be recorded, \
         output exactly: Recorder: nothing to record.\n\n\
         Active persona: {persona}\n\n\
         Transcript:\n{transcript}\n"
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PersonaTranscriptSegment {
    pub persona: String,
    pub transcript: String,
}

pub(crate) fn split_persona_segments(
    transcript: &str,
    initial_persona: &str,
) -> Vec<PersonaTranscriptSegment> {
    let mut segments = Vec::new();
    let mut current_persona = initial_persona.to_string();
    let mut current_lines: Vec<String> = Vec::new();

    for line in transcript.lines() {
        if let Some(next_persona) = persona_switch_from_line(line) {
            push_persona_segment(&mut segments, &current_persona, &current_lines);
            current_persona = next_persona;
            current_lines.clear();
            continue;
        }
        current_lines.push(line.to_string());
    }
    push_persona_segment(&mut segments, &current_persona, &current_lines);

    if segments.is_empty() {
        segments.push(PersonaTranscriptSegment {
            persona: initial_persona.to_string(),
            transcript: String::new(),
        });
    }
    segments
}

fn push_persona_segment(
    segments: &mut Vec<PersonaTranscriptSegment>,
    persona: &str,
    lines: &[String],
) {
    let transcript = lines.join("\n").trim().to_string();
    if transcript.is_empty() {
        return;
    }
    if let Some(last) = segments.last_mut() {
        if last.persona == persona {
            if !last.transcript.ends_with('\n') {
                last.transcript.push('\n');
            }
            last.transcript.push_str(&transcript);
            return;
        }
    }
    segments.push(PersonaTranscriptSegment {
        persona: persona.to_string(),
        transcript,
    });
}

fn persona_switch_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("/persona ") {
        return first_persona_token(rest);
    }

    if let Some(rest) = trimmed
        .strip_prefix("Learner: /persona ")
        .or_else(|| trimmed.strip_prefix("User: /persona "))
        .or_else(|| trimmed.strip_prefix("Operator: /persona "))
    {
        return first_persona_token(rest);
    }

    let (_, rest) = trimmed.split_once(':').unwrap_or(("", trimmed));
    let rest = rest.trim();
    let after = rest.strip_prefix(PERSONA_LOADED_PREFIX)?;
    let (name, suffix) = after.split_once('\'')?;
    if suffix.starts_with(PERSONA_LOADED_AFTER_NAME) {
        first_persona_token(name)
    } else {
        None
    }
}

fn first_persona_token(raw: &str) -> Option<String> {
    raw.trim()
        .split_whitespace()
        .next()
        .map(|s| s.trim_matches(|c: char| c == '"' || c == '\'' || c == '.'))
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
}

fn merge_segment_summaries(
    summaries: Vec<SessionRecordSummary>,
    fallback_persona: &str,
) -> SessionRecordSummary {
    let mut personas = Vec::new();
    let mut vocabulary_count = 0;
    let mut struggle_count = 0;
    let mut sessions_count = 0;
    let mut words = Vec::new();
    let mut struggles = Vec::new();

    for summary in summaries {
        if !personas.contains(&summary.persona) {
            personas.push(summary.persona.clone());
        }
        vocabulary_count += summary.vocabulary_count;
        struggle_count += summary.struggle_count;
        sessions_count = summary.sessions_count;
        words.extend(summary.words);
        struggles.extend(summary.struggles);
    }

    SessionRecordSummary {
        persona: if personas.is_empty() {
            fallback_persona.to_string()
        } else {
            personas.join(", ")
        },
        vocabulary_count,
        struggle_count,
        sessions_count,
        words,
        struggles,
    }
}

fn record_persona_segment_in(
    root: &Path,
    transcript: &str,
    persona: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<SessionRecordSummary> {
    let vocab = detect_vocabulary(transcript);
    let struggles = detect_struggles(transcript);

    for item in &vocab {
        log_vocabulary_in(
            root,
            VocabularyInput {
                kanji: &item.kanji,
                kana: &item.kana,
                romaji: "",
                meaning: &item.meaning,
                pos: "unknown",
                jlpt: None,
                example_japanese: Some(&item.example_japanese),
                example_english: Some(&item.meaning),
            },
        )?;
    }

    for item in &struggles {
        record_struggle_in(root, &item.topic, &item.what_got_confused, now)?;
    }

    let sessions_count = append_persona_session_entry(root, persona, &vocab, &struggles, now)?;

    Ok(SessionRecordSummary {
        persona: persona.to_string(),
        vocabulary_count: vocab.len(),
        struggle_count: struggles.len(),
        sessions_count,
        words: vocab
            .iter()
            .map(|v| {
                if v.kanji == v.kana {
                    format!("{} ({})", v.kana, v.meaning)
                } else {
                    format!("{} / {} ({})", v.kanji, v.kana, v.meaning)
                }
            })
            .collect(),
        struggles: struggles.iter().map(|s| s.topic.clone()).collect(),
    })
}

pub(crate) fn detect_vocabulary(transcript: &str) -> Vec<DetectedVocabulary> {
    let mut out = Vec::new();
    for line in transcript.lines() {
        let Some(japanese) = first_japanese_token(line) else {
            continue;
        };
        let Some(meaning) = extract_english_gloss(line) else {
            continue;
        };
        let kana = extract_parenthesized_kana_after(line, &japanese).unwrap_or_else(|| {
            if japanese.chars().all(is_kana) {
                japanese.clone()
            } else {
                String::new()
            }
        });
        if kana.is_empty() {
            continue;
        }
        let candidate = DetectedVocabulary {
            kanji: japanese,
            kana,
            meaning,
            example_japanese: line.trim().to_string(),
        };
        if !out
            .iter()
            .any(|v: &DetectedVocabulary| v.kanji == candidate.kanji && v.kana == candidate.kana)
        {
            out.push(candidate);
        }
    }
    out
}

pub(crate) fn detect_struggles(transcript: &str) -> Vec<DetectedStruggle> {
    let mut out = Vec::new();
    for line in transcript.lines() {
        let Some(utterance) = learner_utterance(line) else {
            continue;
        };
        let lower = utterance.to_ascii_lowercase();
        let marker = [
            "i don't know ",
            "i dont know ",
            "what is ",
            "what's ",
            "why ",
        ]
        .into_iter()
        .find(|marker| lower.starts_with(marker));
        let Some(marker) = marker else {
            continue;
        };
        let topic = utterance[marker.len()..]
            .trim()
            .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == '"' || c == '\'')
            .to_string();
        if topic.is_empty() {
            continue;
        }
        let item = DetectedStruggle {
            topic,
            what_got_confused: format!("Learner said: {}", utterance.trim()),
        };
        if !out.iter().any(|s: &DetectedStruggle| s.topic == item.topic) {
            out.push(item);
        }
    }
    out
}

fn append_persona_session_entry(
    root: &Path,
    persona: &str,
    vocab: &[DetectedVocabulary],
    struggles: &[DetectedStruggle],
    now: chrono::DateTime<chrono::Utc>,
) -> Result<u32> {
    let timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let relative_path = format!("entities/Persona/{}.md", slugify(persona));
    let page_path = root.join(".dm/wiki").join(&relative_path);
    let raw_existing = std::fs::read_to_string(&page_path).ok();
    let wiki = Wiki::open(root)?;

    let mut page = match wiki.read_page(&relative_path) {
        Ok(existing) => existing,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => new_persona_page(persona, &timestamp),
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            let raw = raw_existing.as_deref().ok_or(e)?;
            legacy_persona_page_from_raw(persona, raw, &timestamp)
        }
        Err(e) => return Err(e.into()),
    };

    let current_count = parse_persona_sessions_count(&page.body)
        .or_else(|| {
            raw_existing
                .as_deref()
                .and_then(|raw| parse_frontmatter(raw).get("sessions_count").cloned())
                .and_then(|v| v.parse::<u32>().ok())
        })
        .unwrap_or(0);
    let next_count = current_count + 1;

    let mut updated = set_body_sessions_count(&page.body, next_count);
    if !updated.contains("## Sessions log") {
        updated.push_str("\n## Sessions log\n\n");
    }
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    let topics = infer_topics(vocab, struggles);
    let words = if vocab.is_empty() {
        "none".to_string()
    } else {
        vocab
            .iter()
            .map(|v| {
                if v.kanji == v.kana {
                    format!("{} ({})", v.kana, v.meaning)
                } else {
                    format!("{} / {} ({})", v.kanji, v.kana, v.meaning)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let flagged = if struggles.is_empty() {
        "none".to_string()
    } else {
        struggles
            .iter()
            .map(|s| s.topic.clone())
            .collect::<Vec<_>>()
            .join(", ")
    };
    updated.push_str(&format!(
        "\n- **{}** — Topics: {}; words introduced: {}; struggles flagged: {}.\n",
        timestamp, topics, words, flagged
    ));
    page.body = updated;
    page.last_updated = timestamp;
    page.page_type = PageType::Entity;
    page.layer = Layer::Host;

    let one_liner = format!("{} persona — {} recorded session(s)", persona, next_count);
    wiki.register_host_page(&relative_path, &page, &one_liner)?;
    Ok(next_count)
}

fn new_persona_page(persona: &str, last_updated: &str) -> WikiPage {
    WikiPage {
        title: persona.to_string(),
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated: last_updated.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],

        extras: std::collections::BTreeMap::new(),
        body: format!("# {}\n\n- **Sessions:** 0\n\n## Sessions log\n\n", persona),
    }
}

fn legacy_persona_page_from_raw(persona: &str, raw: &str, last_updated: &str) -> WikiPage {
    let fm = parse_frontmatter(raw);
    let title = fm
        .get("title")
        .cloned()
        .unwrap_or_else(|| persona.to_string());
    let last_updated = fm
        .get("last_updated")
        .cloned()
        .unwrap_or_else(|| last_updated.to_string());
    WikiPage {
        title,
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated,
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],

        extras: std::collections::BTreeMap::new(),
        body: markdown_body_without_frontmatter(raw).to_string(),
    }
}

fn markdown_body_without_frontmatter(markdown: &str) -> &str {
    let Some(rest) = markdown.strip_prefix("---\n") else {
        return markdown;
    };
    let Some(end) = rest.find("\n---\n") else {
        return markdown;
    };
    &rest[end + "\n---\n".len()..]
}

fn parse_persona_sessions_count(body: &str) -> Option<u32> {
    body.lines().find_map(|line| {
        let trimmed = line.trim();
        let rest = trimmed
            .strip_prefix("- **Sessions:**")
            .or_else(|| trimmed.strip_prefix("**Sessions:**"))?;
        rest.trim().parse::<u32>().ok()
    })
}

fn set_body_sessions_count(body: &str, count: u32) -> String {
    let replacement = format!("- **Sessions:** {}", count);
    let mut lines: Vec<String> = body.lines().map(ToString::to_string).collect();
    for line in &mut lines {
        let trimmed = line.trim();
        if trimmed.starts_with("- **Sessions:**") || trimmed.starts_with("**Sessions:**") {
            *line = replacement;
            let mut out = lines.join("\n");
            if body.ends_with('\n') {
                out.push('\n');
            }
            return out;
        }
    }

    let insert_at = lines
        .iter()
        .position(|line| line.trim_start().starts_with("# "))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    lines.insert(insert_at, String::new());
    lines.insert(insert_at + 1, replacement);
    let mut out = lines.join("\n");
    if !out.ends_with("\n\n") {
        out.push_str("\n\n");
    }
    out
}

fn infer_topics(vocab: &[DetectedVocabulary], struggles: &[DetectedStruggle]) -> String {
    if !struggles.is_empty() {
        return struggles
            .iter()
            .map(|s| s.topic.clone())
            .collect::<Vec<_>>()
            .join(", ");
    }
    if !vocab.is_empty() {
        return "new vocabulary".to_string();
    }
    "general conversation".to_string()
}

fn learner_utterance(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let (speaker, rest) = trimmed.split_once(':')?;
    let speaker = speaker.trim().to_ascii_lowercase();
    if matches!(
        speaker.as_str(),
        "learner" | "user" | "operator" | "student" | "me"
    ) {
        Some(rest.trim())
    } else {
        None
    }
}

fn first_japanese_token(line: &str) -> Option<String> {
    let mut current = String::new();
    for ch in line.chars() {
        if is_japanese(ch) {
            current.push(ch);
        } else if !current.is_empty() {
            break;
        }
    }
    if current.is_empty() {
        None
    } else {
        Some(current)
    }
}

fn extract_parenthesized_kana_after(line: &str, japanese: &str) -> Option<String> {
    let start = line.find(japanese)? + japanese.len();
    let rest = &line[start..];
    let open = rest.find('(')?;
    let close = rest[open + 1..].find(')')? + open + 1;
    let candidate = rest[open + 1..close].trim();
    if !candidate.is_empty() && candidate.chars().all(is_kana) {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn extract_english_gloss(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    let markers = [" means ", " meaning ", " = ", " - ", " — "];
    for marker in markers {
        if let Some(idx) = lower.find(marker) {
            let raw = &line[idx + marker.len()..];
            let gloss = raw
                .trim()
                .trim_matches(|c: char| c == '.' || c == '"' || c == '\'')
                .to_string();
            if gloss.chars().any(|c| c.is_ascii_alphabetic()) {
                return Some(gloss);
            }
        }
    }
    None
}

fn is_japanese(ch: char) -> bool {
    matches!(ch, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}')
}

fn is_kana(ch: char) -> bool {
    matches!(ch, '\u{3040}'..='\u{30ff}')
}

// ---------------------------------------------------------------------------
// HostCapabilities impl — registered at startup
// ---------------------------------------------------------------------------

/// The host capabilities kotoba registers with the dm kernel.
pub struct KotobaCapabilities;

impl HostCapabilities for KotobaCapabilities {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(InvokePersonaTool),
            Box::new(LogVocabularyTool),
            Box::new(LogKanjiTool),
            Box::new(RecordStruggleTool),
            Box::new(QuizMeTool),
            Box::new(PlanSessionTool),
            Box::new(RecordSessionTool),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_register_seven_tools_with_host_prefix() {
        let caps = KotobaCapabilities;
        let tools = caps.tools();
        assert_eq!(tools.len(), 7);
        for tool in &tools {
            assert!(
                tool.name().starts_with("host_"),
                "tool {} missing host_ prefix",
                tool.name()
            );
        }
        let names: Vec<&'static str> = tools.iter().map(|t| t.name()).collect();
        assert!(
            names.contains(&"host_plan_session"),
            "expected host_plan_session among {:?}",
            names
        );
        assert!(
            names.contains(&"host_record_session"),
            "expected host_record_session among {:?}",
            names
        );
    }

    #[test]
    fn slugify_preserves_kanji_and_kana() {
        assert_eq!(slugify("学校"), "学校");
        assert_eq!(slugify("ありがとう"), "ありがとう");
    }

    #[test]
    fn slugify_replaces_whitespace() {
        assert_eq!(slugify("hello world"), "hello-world");
    }

    #[test]
    fn slugify_strips_unsafe_chars() {
        assert_eq!(slugify("foo/bar?"), "foobar");
    }

    #[test]
    fn slugify_falls_back_for_empty() {
        assert_eq!(slugify(""), "untitled");
        assert_eq!(slugify("///"), "untitled");
    }

    // -------- planner: pure-helper tests against a tempdir wiki --------

    use tempfile::TempDir;

    fn write(path: &std::path::Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn vocab_md(title: &str, kana: &str, meaning: &str, mastery: &str) -> String {
        format!(
            "---\ntitle: {title}\ntype: entity\nentity_kind: vocabulary\nlayer: host\nlast_updated: 2026-04-27 10:00:00\n---\n\n# {title}\n\n- **Kanji:** {title}\n- **Kana:** {kana}\n- **Romaji:** romaji\n- **Meaning:** {meaning}\n- **Part of speech:** noun\n- **Mastery:** {mastery}\n"
        )
    }

    fn yuki_md(sessions: u32) -> String {
        format!(
            "---\ntitle: Yuki\ntype: entity\nentity_kind: persona\nlayer: host\nsessions_count: {}\n---\n\n# Yuki\n\n## Signature topics\n\n- Self-introduction\n- Café ordering\n- Weekend hobbies\n",
            sessions
        )
    }

    #[test]
    fn plan_session_empty_wiki_returns_starter_brief() {
        let tmp = TempDir::new().unwrap();
        let today = chrono::NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();
        let brief = plan_session_in(tmp.path(), "Yuki", 3, today);
        assert!(brief.contains("Session brief — 2026-04-27"));
        assert!(brief.contains("(none registered"));
        assert!(brief.contains("first session"));
        assert!(brief.contains("Self-introduction and greetings"));
        assert!(brief.contains("No recent struggles"));
    }

    #[test]
    fn plan_session_vocab_only_sorts_by_mastery_and_caps_at_five() {
        let tmp = TempDir::new().unwrap();
        let vocab_dir = tmp.path().join(".dm/wiki/entities/Vocabulary");
        // 6 entries; lowest-mastery should appear first, the 6th gets dropped.
        write(
            &vocab_dir.join("学校.md"),
            &vocab_md("学校", "がっこう", "school", "Used"),
        );
        write(
            &vocab_dir.join("猫.md"),
            &vocab_md("猫", "ねこ", "cat", "Introduced"),
        );
        write(
            &vocab_dir.join("犬.md"),
            &vocab_md("犬", "いぬ", "dog", "Recognized"),
        );
        write(
            &vocab_dir.join("水.md"),
            &vocab_md("水", "みず", "water", "Internalized"),
        );
        write(
            &vocab_dir.join("本.md"),
            &vocab_md("本", "ほん", "book", "Recalled"),
        );
        write(
            &vocab_dir.join("車.md"),
            &vocab_md("車", "くるま", "car", "Introduced"),
        );

        let today = chrono::NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();
        let brief = plan_session_in(tmp.path(), "Yuki", 3, today);

        let words_section = brief
            .split("## Words to surface")
            .nth(1)
            .expect("words section present")
            .split("## Grammar")
            .next()
            .unwrap();

        // 5 bullet entries, no more.
        let bullets: Vec<&str> = words_section
            .lines()
            .filter(|l| l.starts_with("- "))
            .collect();
        assert_eq!(bullets.len(), 5, "expected 5 bullets, got {:?}", bullets);

        // Lowest-mastery entries lead. Both 猫 and 車 are Introduced; tie
        // broken by title (猫 sorts before 車 by codepoint).
        assert!(bullets[0].contains("猫"), "bullet[0] = {}", bullets[0]);
        assert!(bullets[1].contains("車"), "bullet[1] = {}", bullets[1]);
        assert!(bullets[2].contains("犬"), "bullet[2] = {}", bullets[2]);

        // The Internalized entry (水) is the lowest priority; with 6 inputs
        // and a cap of 5, it must be the one that drops.
        assert!(
            !words_section.contains("水"),
            "Internalized entry should not make the cut: {}",
            words_section
        );
    }

    #[test]
    fn plan_session_mixed_wiki_includes_struggles_and_persona_topic() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join(".dm/wiki/entities/Persona/Yuki.md"),
            &yuki_md(2),
        );
        write(
            &tmp.path().join(".dm/wiki/entities/Vocabulary/猫.md"),
            &vocab_md("猫", "ねこ", "cat", "Introduced"),
        );
        // Today's struggles (in window).
        write(
            &tmp.path()
                .join(".dm/wiki/synthesis/struggles-2026-04-27.md"),
            "---\ntitle: Struggles 2026-04-27\n---\n\n- **wa vs ga** — confused topic vs subject\n- **te-form** — used masu-stem instead\n",
        );
        // Out-of-window struggle (4 days back, default window is 3).
        write(
            &tmp.path()
                .join(".dm/wiki/synthesis/struggles-2026-04-23.md"),
            "---\ntitle: Struggles 2026-04-23\n---\n\n- **stale entry** — should not appear\n",
        );

        let today = chrono::NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();
        let brief = plan_session_in(tmp.path(), "Yuki", 3, today);

        // Persona is recognized and session count incremented for display.
        assert!(brief.contains("Yuki (session 3)"), "brief was: {}", brief);
        // sessions_count=2 mod 3 topics → index 2 → "Weekend hobbies".
        assert!(brief.contains("Weekend hobbies"), "brief was: {}", brief);

        // Recent struggles surface; stale ones do not.
        assert!(brief.contains("wa vs ga"));
        assert!(brief.contains("te-form"));
        assert!(!brief.contains("stale entry"));

        // Vocab entry is present.
        assert!(brief.contains("猫"));
        assert!(brief.contains("ねこ"));
    }

    #[test]
    fn planner_prompt_uses_same_wiki_state_as_rule_brief() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join(".dm/wiki/entities/Persona/Yuki.md"),
            &yuki_md(1),
        );
        write(
            &tmp.path().join(".dm/wiki/entities/Vocabulary/猫.md"),
            &vocab_md("猫", "ねこ", "cat", "Introduced"),
        );
        write(
            &tmp.path()
                .join(".dm/wiki/synthesis/struggles-2026-04-27.md"),
            "---\ntitle: Struggles 2026-04-27\n---\n\n- **te-form** — used masu-stem instead\n",
        );

        let today = chrono::NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();
        let state = collect_planner_wiki_state(tmp.path(), "Yuki", 3, today);
        let prompt = build_planner_prompt(&state);
        let brief = build_session_brief_from_state(&state);

        assert!(prompt.contains("You are kotoba's pre-session planner."));
        assert!(prompt.contains("- Date: 2026-04-27"));
        assert!(prompt.contains("- Requested persona: Yuki"));
        assert!(prompt.contains("Yuki (sessions_count: 1)"));
        assert!(prompt.contains("Self-introduction; Café ordering; Weekend hobbies"));
        assert!(prompt.contains("猫 (ねこ) — cat [mastery: Introduced]"));
        assert!(prompt.contains("**te-form**"));

        assert!(brief.contains("Yuki (session 2)"));
        assert!(brief.contains("Café ordering"));
        assert!(brief.contains("猫"));
        assert!(brief.contains("te-form"));
    }

    #[test]
    fn parse_frontmatter_handles_missing_block() {
        let body = "# heading\n\nno frontmatter here\n";
        assert!(parse_frontmatter(body).is_empty());
    }

    #[test]
    fn parse_frontmatter_extracts_simple_keys() {
        let body = "---\ntitle: Yuki\nsessions_count: 7\n---\n\n# body\n";
        let fm = parse_frontmatter(body);
        assert_eq!(fm.get("title"), Some(&"Yuki".to_string()));
        assert_eq!(fm.get("sessions_count"), Some(&"7".to_string()));
    }

    #[test]
    fn detect_vocabulary_finds_japanese_tokens_with_english_glosses() {
        let transcript = "\
Yuki: New word: 猫 (ねこ) means cat.
Yuki: 学校 (がっこう) = school
Yuki: This line has no Japanese word.
";

        let found = detect_vocabulary(transcript);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].kanji, "猫");
        assert_eq!(found[0].kana, "ねこ");
        assert_eq!(found[0].meaning, "cat");
        assert_eq!(found[1].kanji, "学校");
        assert_eq!(found[1].kana, "がっこう");
        assert_eq!(found[1].meaning, "school");
    }

    #[test]
    fn detect_struggles_finds_learner_confusion_markers() {
        let transcript = "\
Learner: I don't know は vs が?
Yuki: Let's slow down.
User: what is て-form?
Assistant: what is ignored because assistant is not the learner.
";

        let found = detect_struggles(transcript);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].topic, "は vs が");
        assert_eq!(found[1].topic, "て-form");
    }

    #[test]
    fn read_persona_summary_prefers_body_sessions_count() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join(".dm/wiki/entities/Persona/Yuki.md"),
            "---\ntitle: Yuki\ntype: entity\nlayer: host\nsessions_count: 1\nlast_updated: 2026-04-27\n---\n# Yuki\n\n- **Sessions:** 4\n\n## Signature topics\n\n- Café ordering\n",
        );

        let summary = read_persona_summary(tmp.path(), "Yuki").unwrap();
        assert_eq!(summary.sessions_count, 4);
    }

    #[test]
    fn split_persona_segments_recognizes_shared_loaded_marker() {
        let transcript = format!(
            "\
Yuki: New word: 学校 (がっこう) means school.
tool: {}Hiro'{}
Hiro: New word: 猫 (ねこ) means cat.
",
            PERSONA_LOADED_PREFIX, PERSONA_LOADED_AFTER_NAME
        );

        let segments = split_persona_segments(&transcript, "Yuki");
        assert_eq!(segments.len(), 2, "{segments:?}");
        assert_eq!(segments[0].persona, "Yuki");
        assert!(segments[0].transcript.contains("学校"));
        assert_eq!(segments[1].persona, "Hiro");
        assert!(segments[1].transcript.contains("猫"));
    }

    #[test]
    fn record_session_updates_vocab_struggles_and_persona() {
        let tmp = TempDir::new().unwrap();
        write(
            &tmp.path().join(".dm/wiki/entities/Persona/Yuki.md"),
            &yuki_md(0),
        );
        let transcript = "\
Learner: I don't know は vs が?
Yuki: New word: 猫 (ねこ) means cat.
Yuki: New word: 学校 (がっこう) means school.
User: what is て-form?
";
        let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        let summary = record_session_in(tmp.path(), transcript, "Yuki", now).unwrap();
        assert_eq!(summary.vocabulary_count, 2);
        assert_eq!(summary.struggle_count, 2);
        assert_eq!(summary.sessions_count, 1);

        let neko =
            std::fs::read_to_string(tmp.path().join(".dm/wiki/entities/Vocabulary/猫.md")).unwrap();
        assert!(neko.contains("- **Kana:** ねこ"));
        assert!(neko.contains("- **Meaning:** cat"));

        let school =
            std::fs::read_to_string(tmp.path().join(".dm/wiki/entities/Vocabulary/学校.md"))
                .unwrap();
        assert!(school.contains("- **Kana:** がっこう"));
        assert!(school.contains("- **Meaning:** school"));

        let struggles = std::fs::read_to_string(
            tmp.path()
                .join(".dm/wiki/synthesis/struggles-2026-04-27.md"),
        )
        .unwrap();
        assert!(struggles.contains("**は vs が**"));
        assert!(struggles.contains("**て-form**"));

        let persona =
            std::fs::read_to_string(tmp.path().join(".dm/wiki/entities/Persona/Yuki.md")).unwrap();
        assert!(persona.contains("- **Sessions:** 1"));
        assert!(!persona.contains("sessions_count: 1"));
        assert!(persona.contains("2026-04-27 12:30:00"));
        assert!(persona.contains("words introduced: 猫 / ねこ (cat), 学校 / がっこう (school)"));
        assert!(persona.contains("struggles flagged: は vs が, て-form"));
    }

    #[test]
    fn record_session_creates_missing_persona_page() {
        let tmp = TempDir::new().unwrap();
        let now = chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        let summary =
            record_session_in(tmp.path(), "Yuki: こんにちは - hello", "Aki", now).unwrap();

        assert_eq!(summary.sessions_count, 1);
        let persona =
            std::fs::read_to_string(tmp.path().join(".dm/wiki/entities/Persona/Aki.md")).unwrap();
        assert!(persona.contains("title: Aki"));
        assert!(persona.contains("- **Sessions:** 1"));
        assert!(persona.contains("## Sessions log"));
    }
}
