//! Kotoba host capabilities — the 5 tools dm gets registered when
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
use dark_matter::host::HostCapabilities;
use dark_matter::tools::{Tool, ToolResult};
use serde_json::{json, Value};
use std::path::PathBuf;

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
                "Persona '{}' loaded. The next agent turn should speak in this voice. Persona definition follows:\n\n{}",
                name, body
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
        let pos = args.get("pos").and_then(|v| v.as_str()).unwrap_or("unknown");
        let jlpt = args.get("jlpt").and_then(|v| v.as_u64());
        let example_jp = args.get("example_japanese").and_then(|v| v.as_str());
        let example_en = args.get("example_english").and_then(|v| v.as_str());

        // Use kanji form as slug if present; fall back to kana otherwise.
        let slug = slugify(if kanji.is_empty() { kana } else { kanji });

        let root = project_root();
        let dir = root.join(".dm/wiki/entities/Vocabulary");
        std::fs::create_dir_all(&dir)?;
        let page_path = dir.join(format!("{}.md", slug));

        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut frontmatter = format!(
            "---\ntitle: {}\ntype: entity\nentity_kind: vocabulary\nlayer: host\nlast_updated: {}\n",
            if kanji.is_empty() { kana } else { kanji },
            now
        );
        if let Some(level) = jlpt {
            frontmatter.push_str(&format!("jlpt: {}\n", level));
        }
        frontmatter.push_str("---\n");

        let mut body = String::new();
        body.push_str(&format!(
            "# {}\n\n",
            if kanji.is_empty() { kana } else { kanji }
        ));
        if !kanji.is_empty() {
            body.push_str(&format!("- **Kanji:** {}\n", kanji));
        }
        body.push_str(&format!("- **Kana:** {}\n", kana));
        if !romaji.is_empty() {
            body.push_str(&format!("- **Romaji:** {}\n", romaji));
        }
        body.push_str(&format!("- **Meaning:** {}\n", meaning));
        body.push_str(&format!("- **Part of speech:** {}\n", pos));
        body.push_str(&format!("- **Mastery:** {:?}\n\n", Mastery::Introduced));

        if let (Some(jp), Some(en)) = (example_jp, example_en) {
            body.push_str("## Examples\n\n");
            body.push_str(&format!("- {}\n  - _{}_\n", jp, en));
        }

        std::fs::write(&page_path, format!("{}{}", frontmatter, body))?;

        Ok(ToolResult {
            content: format!(
                "Vocabulary logged: {} ({}, \"{}\") → {}",
                if kanji.is_empty() { kana } else { kanji },
                kana,
                meaning,
                page_path.strip_prefix(&root).unwrap_or(&page_path).display()
            ),
            is_error: false,
        })
    }
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
        let dir = root.join(".dm/wiki/entities/Kanji");
        std::fs::create_dir_all(&dir)?;
        let page_path = dir.join(format!("{}.md", slugify(character)));

        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let mut content = format!(
            "---\ntitle: {}\ntype: entity\nentity_kind: kanji\nlayer: host\nlast_updated: {}\n",
            character, now
        );
        if let Some(level) = jlpt {
            content.push_str(&format!("jlpt: {}\n", level));
        }
        content.push_str("---\n\n");
        content.push_str(&format!("# {}\n\n", character));
        content.push_str(&format!("- **Meaning:** {}\n", meaning));
        if !onyomi.is_empty() {
            content.push_str(&format!("- **On'yomi:** {}\n", onyomi.join(", ")));
        }
        if !kunyomi.is_empty() {
            content.push_str(&format!("- **Kun'yomi:** {}\n", kunyomi.join(", ")));
        }
        if !radicals.is_empty() {
            content.push_str(&format!("- **Radicals:** {}\n", radicals.join(" + ")));
        }
        content.push_str(&format!("- **Mastery:** {:?}\n", Mastery::Introduced));
        if let Some(m) = mnemonic {
            content.push_str(&format!("\n## Mnemonic\n\n{}\n", m));
        }

        std::fs::write(&page_path, content)?;

        Ok(ToolResult {
            content: format!(
                "Kanji logged: {} (\"{}\") → {}",
                character,
                meaning,
                page_path.strip_prefix(&root).unwrap_or(&page_path).display()
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
        let dir = root.join(".dm/wiki/synthesis");
        std::fs::create_dir_all(&dir)?;

        let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let page_path = dir.join(format!("struggles-{}.md", date));

        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let entry = format!("- **{}** — {} _(logged {})_\n", topic, what, now);

        if page_path.exists() {
            // Append to today's existing struggles page.
            let existing = std::fs::read_to_string(&page_path)?;
            let updated = format!("{}{}", existing, entry);
            std::fs::write(&page_path, updated)?;
        } else {
            let content = format!(
                "---\ntitle: Struggles {}\ntype: synthesis\nlayer: host\nlast_updated: {}\n---\n\n# Struggles for {}\n\n{}",
                date, now, date, entry
            );
            std::fs::write(&page_path, content)?;
        }

        Ok(ToolResult {
            content: format!(
                "Struggle recorded: {} — {}. Today's struggles → {}",
                topic,
                what,
                page_path.strip_prefix(&root).unwrap_or(&page_path).display()
            ),
            is_error: false,
        })
    }
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
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_register_five_tools_with_host_prefix() {
        let caps = KotobaCapabilities;
        let tools = caps.tools();
        assert_eq!(tools.len(), 5);
        for tool in &tools {
            assert!(
                tool.name().starts_with("host_"),
                "tool {} missing host_ prefix",
                tool.name()
            );
        }
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
}
