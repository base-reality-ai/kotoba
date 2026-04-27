use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use std::sync::OnceLock;

static SYNTAX_SET: OnceLock<syntect::parsing::SyntaxSet> = OnceLock::new();
static THEME_SET: OnceLock<syntect::highlighting::ThemeSet> = OnceLock::new();

fn syntax_set() -> &'static syntect::parsing::SyntaxSet {
    SYNTAX_SET.get_or_init(syntect::parsing::SyntaxSet::load_defaults_newlines)
}

fn theme_set() -> &'static syntect::highlighting::ThemeSet {
    THEME_SET.get_or_init(syntect::highlighting::ThemeSet::load_defaults)
}

/// Syntax-highlight a code block and return ratatui Lines.
fn highlight_code(code: &str, lang: &str) -> Vec<Line<'static>> {
    use syntect::easy::HighlightLines;
    use syntect::util::LinesWithEndings;

    let ss = syntax_set();
    let ts = theme_set();

    let syntax = ss
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let theme = &ts.themes["base16-ocean.dark"];
    let mut h = HighlightLines::new(syntax, theme);

    let mut lines: Vec<Line<'static>> = Vec::new();
    for line in LinesWithEndings::from(code) {
        let Ok(ranges) = h.highlight_line(line, ss) else {
            lines.push(Line::from(Span::raw(
                line.trim_end_matches('\n').to_string(),
            )));
            continue;
        };
        let spans: Vec<Span<'static>> = ranges
            .into_iter()
            .map(|(style, text)| {
                let text = text.trim_end_matches('\n').to_string();
                let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
                Span::styled(text, Style::default().fg(fg))
            })
            .collect();
        lines.push(Line::from(spans));
    }
    lines
}

/// Convert a markdown string into ratatui Lines for rendering.
pub fn render_markdown(md: &str) -> Vec<Line<'static>> {
    let parser = Parser::new_ext(md, Options::all());
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_spans: Vec<Span<'static>> = Vec::new();
    let mut style_stack: Vec<Style> = vec![Style::default()];

    // Code block accumulation state
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_content = String::new();

    // Link URL stack — parallel to style_stack entries pushed for Tag::Link
    let mut link_url_stack: Vec<String> = Vec::new();

    // Table state
    let mut in_table = false;
    let mut table_row_cells: Vec<String> = Vec::new();
    let mut table_cell_text = String::new();

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    let lvl = level as u8;
                    let s = if lvl == 1 {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Cyan)
                    };
                    style_stack.push(s);
                }
                Tag::Strong => {
                    style_stack.push(Style::default().add_modifier(Modifier::BOLD));
                }
                Tag::Emphasis => {
                    style_stack.push(Style::default().add_modifier(Modifier::ITALIC));
                }
                Tag::CodeBlock(kind) => {
                    in_code_block = true;
                    code_lang = match kind {
                        CodeBlockKind::Fenced(lang) => lang.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                    code_content.clear();
                }
                Tag::Item => {
                    current_spans.push(Span::raw("  • "));
                }
                Tag::Link { dest_url, .. } => {
                    let base = *style_stack.last().unwrap_or(&Style::default());
                    style_stack.push(base.fg(Color::Blue).add_modifier(Modifier::UNDERLINED));
                    link_url_stack.push(dest_url.to_string());
                }
                Tag::Strikethrough => {
                    let base = *style_stack.last().unwrap_or(&Style::default());
                    style_stack.push(base.add_modifier(Modifier::CROSSED_OUT));
                }
                Tag::Table(_) => {
                    in_table = true;
                }
                Tag::TableHead => {
                    table_row_cells.clear();
                }
                Tag::TableRow => {
                    table_row_cells.clear();
                }
                Tag::TableCell => {
                    table_cell_text.clear();
                }
                _ => {}
            },

            Event::End(tag) => match tag {
                TagEnd::Heading(_) | TagEnd::Paragraph => {
                    let spans = std::mem::take(&mut current_spans);
                    lines.push(Line::from(spans));
                    lines.push(Line::from(""));
                    style_stack.pop();
                }
                TagEnd::CodeBlock => {
                    // Render accumulated code with syntax highlighting
                    let highlighted = highlight_code(&code_content, &code_lang);
                    lines.extend(highlighted);
                    lines.push(Line::from(""));
                    in_code_block = false;
                    code_lang.clear();
                    code_content.clear();
                }
                TagEnd::Item => {
                    let spans = std::mem::take(&mut current_spans);
                    lines.push(Line::from(spans));
                }
                TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {
                    style_stack.pop();
                }
                TagEnd::Link => {
                    style_stack.pop();
                    if let Some(url) = link_url_stack.pop() {
                        if !url.is_empty() && !url.starts_with('#') {
                            current_spans.push(Span::styled(
                                format!(" ({url})"),
                                Style::default().fg(Color::DarkGray),
                            ));
                        }
                    }
                }
                TagEnd::TableCell => {
                    table_row_cells.push(table_cell_text.trim().to_string());
                    table_cell_text.clear();
                }
                TagEnd::TableHead => {
                    let row_text = table_row_cells.join(" | ");
                    lines.push(Line::from(Span::styled(
                        row_text,
                        Style::default().add_modifier(Modifier::BOLD),
                    )));
                    let sep = vec!["---"; table_row_cells.len()].join(" | ");
                    lines.push(Line::from(Span::raw(sep)));
                    table_row_cells.clear();
                }
                TagEnd::TableRow => {
                    let row_text = table_row_cells.join(" | ");
                    lines.push(Line::from(Span::raw(row_text)));
                    table_row_cells.clear();
                }
                TagEnd::Table => {
                    lines.push(Line::from(""));
                    in_table = false;
                }
                _ => {}
            },

            Event::Text(text) => {
                if in_code_block {
                    code_content.push_str(&text);
                } else if in_table {
                    table_cell_text.push_str(&text);
                } else {
                    let current_style = *style_stack.last().unwrap_or(&Style::default());
                    current_spans.push(Span::styled(text.to_string(), current_style));
                }
            }

            // Inline code — yellow, distinct from code blocks
            Event::Code(text) => {
                current_spans.push(Span::styled(
                    text.to_string(),
                    Style::default().fg(Color::Yellow),
                ));
            }

            Event::SoftBreak | Event::HardBreak => {
                let spans = std::mem::take(&mut current_spans);
                lines.push(Line::from(spans));
            }

            _ => {}
        }
    }

    // Flush any remaining spans
    if !current_spans.is_empty() {
        lines.push(Line::from(current_spans));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::raw(md.to_string())));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_to_string(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_link_renders_text_and_url() {
        let lines = render_markdown("[foo](https://example.com)");
        let text = lines_to_string(&lines);
        assert!(text.contains("foo"), "link text missing: {text}");
        assert!(
            text.contains("(https://example.com)"),
            "link url missing: {text}"
        );
    }

    #[test]
    fn test_anchor_link_no_url_suffix() {
        let lines = render_markdown("[foo](#section)");
        let text = lines_to_string(&lines);
        assert!(text.contains("foo"), "link text missing: {text}");
        assert!(
            !text.contains("(#section)"),
            "anchor url should be suppressed: {text}"
        );
    }

    #[test]
    fn test_strikethrough_renders_text() {
        let lines = render_markdown("~~deleted~~");
        let text = lines_to_string(&lines);
        assert!(
            text.contains("deleted"),
            "strikethrough text missing: {text}"
        );
    }

    #[test]
    fn test_plain_text_unchanged() {
        let lines = render_markdown("hello world");
        let text = lines_to_string(&lines);
        assert!(text.contains("hello world"), "plain text broken: {text}");
    }

    #[test]
    fn test_inline_code_renders_text() {
        let lines = render_markdown("Use `cargo build` to compile.");
        let text = lines_to_string(&lines);
        assert!(
            text.contains("cargo build"),
            "inline code text missing: {text}"
        );
    }

    #[test]
    fn test_heading_renders_text() {
        let lines = render_markdown("# My Heading");
        let text = lines_to_string(&lines);
        assert!(text.contains("My Heading"), "heading text missing: {text}");
    }

    #[test]
    fn test_bold_renders_text() {
        let lines = render_markdown("**important**");
        let text = lines_to_string(&lines);
        assert!(text.contains("important"), "bold text missing: {text}");
    }

    #[test]
    fn test_list_item_renders_bullet_and_text() {
        let lines = render_markdown("- item one");
        let text = lines_to_string(&lines);
        assert!(text.contains("item one"), "list item text missing: {text}");
        assert!(text.contains('•'), "bullet point missing: {text}");
    }

    #[test]
    fn test_empty_string_returns_nonempty_lines() {
        // Empty markdown falls back to pushing the raw string as a line
        let lines = render_markdown("");
        assert!(
            !lines.is_empty(),
            "render_markdown('') should never return empty"
        );
    }

    #[test]
    fn test_code_block_renders_content() {
        let md = "```rust\nfn main() {}\n```";
        let lines = render_markdown(md);
        let text = lines_to_string(&lines);
        assert!(
            text.contains("fn main()"),
            "code block content missing: {text}"
        );
    }

    #[test]
    fn test_multiple_paragraphs_separated() {
        let md = "First paragraph.\n\nSecond paragraph.";
        let lines = render_markdown(md);
        let text = lines_to_string(&lines);
        assert!(
            text.contains("First paragraph."),
            "first para missing: {text}"
        );
        assert!(
            text.contains("Second paragraph."),
            "second para missing: {text}"
        );
    }

    #[test]
    fn test_numbered_list_renders_items() {
        let md = "1. First item\n2. Second item\n3. Third item";
        let lines = render_markdown(md);
        let text = lines_to_string(&lines);
        assert!(
            text.contains("First item"),
            "numbered list item missing: {text}"
        );
        assert!(text.contains("Second item"));
    }

    #[test]
    fn render_markdown_table_visible() {
        let md = "| Name | Value |\n|------|-------|\n| foo  | 42    |\n| bar  | 99    |";
        let lines = render_markdown(md);
        let text = lines_to_string(&lines);
        assert!(text.contains("Name"), "header cell missing: {text}");
        assert!(text.contains("foo"), "body cell missing: {text}");
        assert!(text.contains("42"), "body value missing: {text}");
        assert!(text.contains("bar"), "second row missing: {text}");
    }

    #[test]
    fn render_markdown_table_separator() {
        let md = "| A | B |\n|---|---|\n| x | y |";
        let lines = render_markdown(md);
        let text = lines_to_string(&lines);
        assert!(text.contains("---"), "separator line missing: {text}");
    }
}
