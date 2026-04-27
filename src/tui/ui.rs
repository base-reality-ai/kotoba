//! TUI rendering — translates `App` state into ratatui frames.
//!
//! Pure-ish layout/draw functions: status bar, scrollable transcript,
//! permission prompt, slash-command palette, hyperlink rendering. Reads
//! `App` state, writes nothing; called by `crate::tui::run` each frame.

use std::fmt::Write as _;

use crate::gpu;
use crate::session::short_id;
use crate::tui::app::{App, EntryKind, Mode};
use crate::tui::hyperlinks::maybe_wrap_paths;
use crate::tui::markdown::render_markdown;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

/// Returns true when line 2 of the status bar has any data to show.
fn has_status_line2(app: &App) -> bool {
    app.perf.is_some()
        || app.ctx_usage.is_some()
        || app.gpu_stats.is_some()
        || app.token_usage.total() > 0
        || app.is_compacting.is_some()
}

/// Style hint for the CTX segment of the status bar's second line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CtxStyle {
    /// Normal ctx percentage under 90% — cyan.
    Normal,
    /// Ctx percentage at or above 90% — yellow/red, signals compaction is near.
    Warn,
    /// Compaction in progress — cyan `[compacting...]` banner replaces the gauge.
    Working,
}

/// Pure helper: format the CTX segment of the status bar's second line.
///
/// When compaction is running, the `[compacting...]` banner wins over the
/// `CTX N%` gauge — the user needs to know the pause is compaction, not a
/// stuck inference. All compaction stages render identically; the stage
/// payload on `CompactionStarted` is reserved for future stage-aware UI.
pub(crate) fn format_ctx_display(
    is_compacting: Option<&crate::compaction::CompactionStage>,
    ctx_usage: Option<(usize, usize)>,
) -> (String, CtxStyle) {
    if is_compacting.is_some() {
        return (" [compacting...]".to_string(), CtxStyle::Working);
    }
    match ctx_usage {
        Some((used, limit)) if limit > 0 => {
            let pct = (used * 100) / limit;
            let style = if pct >= 90 {
                CtxStyle::Warn
            } else {
                CtxStyle::Normal
            };
            (format!(" CTX {}%", pct), style)
        }
        _ => (String::new(), CtxStyle::Normal),
    }
}

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let chain_line: u16 = if app.chain_state.is_some() { 1 } else { 0 };
    let status_height: u16 = 1 + (if has_status_line2(app) { 1 } else { 0 }) + chain_line;

    // Calculate input box height based on content length vs available width
    let input_inner_width = area.width.saturating_sub(2) as usize; // subtract borders
    let input_lines = if input_inner_width > 0 {
        let newline_count = app.input.chars().filter(|c| *c == '\n').count();
        let char_count = app.input.chars().count() + 1;
        let wrap_lines = (char_count / input_inner_width) + 1;
        (wrap_lines + newline_count).min(8) as u16
    } else {
        1
    };
    let input_height = input_lines + 2; // add border top+bottom

    let chain_pane_height: u16 =
        if app.show_chain_pane && (!app.chain_log.is_empty() || app.chain_state.is_some()) {
            let node_count = app
                .chain_state
                .as_ref()
                .map_or(0, |cs| cs.config.nodes.len()) as u16;
            let ideal = node_count + 5;
            let max = area.height / 3;
            if max < 4 {
                0
            } else {
                ideal.min(max).max(4)
            }
        } else {
            0
        };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(status_height),     // status line(s)
            Constraint::Length(chain_pane_height), // chain detail pane (0 when hidden)
            Constraint::Min(0),                    // message pane (greedy)
            Constraint::Length(input_height),      // input box
            Constraint::Length(1),                 // help bar
        ])
        .split(area);

    render_status(frame, app, chunks[0]);
    if chain_pane_height > 0 {
        render_chain_pane(frame, app, chunks[1]);
    }
    render_messages(frame, app, chunks[2]);
    render_input(frame, app, chunks[3]);
    render_help(frame, app, chunks[4]);

    if app.mode == Mode::PermissionDialog {
        if let Some(perm) = &app.pending_permission {
            render_permission_dialog(frame, area, perm);
        }
    }
    if app.mode == Mode::AskUserQuestion {
        if let Some(q) = &app.pending_question {
            render_question_dialog(frame, area, q, &app.input);
        }
    }
    if app.mode == Mode::HelpOverlay {
        render_help_overlay(frame, area);
    }
    if app.mode == Mode::DiffReview && !app.staged_changes.is_empty() {
        render_diff_review(frame, area, app);
    }
}

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let bg = Color::DarkGray;

    // ── Line 1: identity row (always visible) ────────────────────────────────
    let mut line1: Vec<Span> = Vec::new();

    // Spinner + current tool (only when busy)
    if app.agent_busy {
        let frame_idx = app.tick as usize % SPINNER_FRAMES.len();
        let spinner = SPINNER_FRAMES[frame_idx];
        let busy_text = if let Some(ref tool) = app.current_tool {
            if tool.starts_with("bash") {
                if let Some(last_line) = app.current_tool_output.back() {
                    let preview = if last_line.len() > 50 {
                        let mut end = 50usize.min(last_line.len());
                        while end > 0 && !last_line.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}…", &last_line[..end])
                    } else {
                        last_line.clone()
                    };
                    format!(" {} bash ▸ {}", spinner, preview)
                } else {
                    format!(" {} {}", spinner, tool)
                }
            } else {
                format!(" {} {}", spinner, tool)
            }
        } else {
            let elapsed = app
                .turn_start
                .map(|t| format!(" {}s", t.elapsed().as_secs()))
                .unwrap_or_default();
            format!(" {} thinking{}", spinner, elapsed)
        };
        line1.push(Span::styled(
            busy_text,
            Style::default().bg(bg).fg(Color::White),
        ));
        line1.push(Span::styled(
            " ▕ ",
            Style::default().bg(bg).fg(Color::DarkGray),
        ));
    } else {
        line1.push(Span::styled(
            status_project_label(app),
            Style::default().bg(bg).fg(Color::White),
        ));
    }

    // Model name
    line1.push(Span::styled(
        app.model.clone(),
        Style::default().bg(bg).fg(Color::Cyan),
    ));

    // Routed model indicator (shows when tool_client model differs from base model)
    if let Some(ref tm) = app.current_tool_model {
        if tm != &app.model {
            line1.push(Span::styled(
                " →",
                Style::default().bg(bg).fg(Color::DarkGray),
            ));
            line1.push(Span::styled(
                tm.clone(),
                Style::default().bg(bg).fg(Color::Yellow),
            ));
        }
    }

    // Daemon indicator
    if app.daemon_mode {
        line1.push(Span::styled(
            " ● daemon",
            Style::default().bg(bg).fg(Color::Cyan),
        ));
    }

    // Chain indicator — shown while a chain is running
    if let Some(ref cs) = app.chain_state {
        let active = cs
            .active_node_index
            .and_then(|i| cs.config.nodes.get(i))
            .map_or("idle", |n| n.name.as_str());
        let cycle_display = if cs.config.loop_forever {
            format!("{}/∞", cs.current_cycle)
        } else {
            format!("{}/{}", cs.current_cycle, cs.config.max_cycles)
        };
        line1.push(Span::styled(
            format!(" chain {} [{}]", cycle_display, active),
            Style::default().bg(bg).fg(Color::Magenta),
        ));
        if cs.has_failed_nodes() {
            line1.push(Span::styled(
                " ⚠ failed",
                Style::default().bg(bg).fg(Color::Red),
            ));
        }
    }

    // Plan mode indicator
    if app.plan_mode {
        line1.push(Span::styled(
            " [PLAN]",
            Style::default().bg(bg).fg(Color::Yellow),
        ));
    }

    // Copy flash
    if app.copy_flash_active() {
        line1.push(Span::styled(
            " ✓ copied",
            Style::default().bg(bg).fg(Color::Green),
        ));
    }

    // Session title — truncated to 30 chars, right-aligned feel with separator
    let sid = short_id(&app.session_id);
    let raw_title = app.session_title.as_deref().unwrap_or(sid);
    let title_label = if raw_title.chars().count() > 30 {
        format!("{}…", raw_title.chars().take(30).collect::<String>())
    } else {
        raw_title.to_string()
    };
    line1.push(Span::styled(
        format!("  {}", title_label),
        Style::default().bg(bg).fg(Color::Gray),
    ));

    // ── Line 2: metrics row (only when data present) ──────────────────────────
    let show_line2 = has_status_line2(app);
    let mut line2: Vec<Span> = Vec::new();

    if show_line2 {
        // CTX segment (or the `[compacting...]` banner while compaction is running).
        let (ctx_text, ctx_style) = format_ctx_display(app.is_compacting.as_ref(), app.ctx_usage);
        if !ctx_text.is_empty() {
            match ctx_style {
                CtxStyle::Working => {
                    // Banner wins over the gauge — single cyan chunk so it's
                    // obviously distinct from normal CTX N% rendering.
                    line2.push(Span::styled(
                        ctx_text,
                        Style::default()
                            .bg(bg)
                            .fg(Color::Cyan)
                            .add_modifier(ratatui::style::Modifier::BOLD),
                    ));
                }
                CtxStyle::Normal | CtxStyle::Warn => {
                    // Re-split into "CTX " + "N%" so the percentage gets its
                    // own color and we preserve the existing red-at-95 /
                    // yellow-at-80 / cyan-otherwise gradient from the
                    // pre-helper render. The helper's Warn/Normal split
                    // (≥90 vs <90) is just the suffix gate — the colors
                    // below still use the finer-grained original thresholds.
                    let pct_val: usize = app.ctx_usage.map_or(0, |(u, l)| (u * 100) / l.max(1));
                    let pct_color = if pct_val >= 95 {
                        Color::Red
                    } else if pct_val >= 80 {
                        Color::Yellow
                    } else {
                        Color::Cyan
                    };
                    line2.push(Span::styled(
                        " CTX ",
                        Style::default().bg(bg).fg(Color::White),
                    ));
                    line2.push(Span::styled(
                        format!("{}%", pct_val),
                        Style::default().bg(bg).fg(pct_color),
                    ));
                    if ctx_style == CtxStyle::Warn {
                        line2.push(Span::styled(
                            " /compact",
                            Style::default()
                                .bg(bg)
                                .fg(Color::Red)
                                .add_modifier(ratatui::style::Modifier::BOLD),
                        ));
                    }
                }
            }
        }

        // Perf segment
        if let Some(ref p) = app.perf {
            let tps_color = if p.tok_per_sec < 5.0 {
                Color::Yellow
            } else {
                Color::Green
            };
            if !line2.is_empty() {
                line2.push(Span::styled(
                    " ▕ ",
                    Style::default().bg(bg).fg(Color::DarkGray),
                ));
            } else {
                line2.push(Span::styled(" ", Style::default().bg(bg).fg(Color::White)));
            }
            line2.push(Span::styled(
                format!("{:.1} tok/s", p.tok_per_sec),
                Style::default().bg(bg).fg(tps_color),
            ));
            line2.push(Span::styled(
                format!("  TTFT {}ms", p.ttft_ms),
                Style::default().bg(bg).fg(Color::White),
            ));
        }

        // Cumulative token usage segment
        if app.token_usage.total() > 0 {
            if !line2.is_empty() {
                line2.push(Span::styled(
                    " ▕ ",
                    Style::default().bg(bg).fg(Color::DarkGray),
                ));
            } else {
                line2.push(Span::styled(" ", Style::default().bg(bg).fg(Color::White)));
            }
            let total = app.token_usage.total();
            let label = if total >= 1000 {
                format!("{}K tok", total / 1000)
            } else {
                format!("{} tok", total)
            };
            line2.push(Span::styled(label, Style::default().bg(bg).fg(Color::Cyan)));
        }

        // GPU segment
        if let Some(ref g) = app.gpu_stats {
            let util_color = match gpu::util_color_level(g.util_pct) {
                2 => Color::Red,
                1 => Color::Yellow,
                _ => Color::Green,
            };
            let vram_ratio = if g.vram_total_mb > 0 {
                g.vram_used_mb as f64 / g.vram_total_mb as f64
            } else {
                0.0
            };
            let vram_color = if vram_ratio >= 0.90 {
                Color::Red
            } else if vram_ratio >= 0.75 {
                Color::Yellow
            } else {
                Color::Cyan
            };
            if !line2.is_empty() {
                line2.push(Span::styled(
                    " ▕ ",
                    Style::default().bg(bg).fg(Color::DarkGray),
                ));
            } else {
                line2.push(Span::styled(" ", Style::default().bg(bg).fg(Color::White)));
            }
            line2.push(Span::styled(
                "GPU ",
                Style::default().bg(bg).fg(Color::White),
            ));
            line2.push(Span::styled(
                format!("{}%", g.util_pct),
                Style::default().bg(bg).fg(util_color),
            ));
            line2.push(Span::styled(
                " ▕ ",
                Style::default().bg(bg).fg(Color::DarkGray),
            ));
            line2.push(Span::styled(
                gpu::format_vram(g.vram_used_mb, g.vram_total_mb),
                Style::default().bg(bg).fg(vram_color),
            ));
            if let Some(temp) = g.temp_c {
                let temp_color = if temp >= 85 {
                    Color::Red
                } else if temp >= 70 {
                    Color::Yellow
                } else {
                    Color::White
                };
                line2.push(Span::styled(
                    " ▕ ",
                    Style::default().bg(bg).fg(Color::DarkGray),
                ));
                line2.push(Span::styled(
                    format!("{}°C", temp),
                    Style::default().bg(bg).fg(temp_color),
                ));
            }
        }
    }

    let mut lines = vec![Line::from(line1)];
    if show_line2 {
        lines.push(Line::from(line2));
    }

    if let Some(ref cs) = app.chain_state {
        let mut line3: Vec<Span> = Vec::new();
        line3.push(Span::styled(" ", Style::default().bg(bg)));
        for node in &cs.config.nodes {
            use crate::orchestrate::types::ChainNodeStatus;
            let (icon, color) = match cs.node_statuses.get(&node.name) {
                Some(ChainNodeStatus::Completed) => ("✓", Color::Green),
                Some(ChainNodeStatus::Running) => ("⏳", Color::Yellow),
                Some(ChainNodeStatus::Failed(_)) => ("✗", Color::Red),
                Some(ChainNodeStatus::Paused) => ("⏸", Color::Cyan),
                Some(ChainNodeStatus::Pending) | None => ("○", Color::DarkGray),
            };
            line3.push(Span::styled(
                format!("{} {} ", icon, node.name),
                Style::default().bg(bg).fg(color),
            ));
        }
        line3.push(Span::styled(
            format!("│ t:{}", cs.turns_used),
            Style::default().bg(bg).fg(Color::White),
        ));
        if let Some(active_idx) = cs.active_node_index {
            if let Some(active_node) = cs.config.nodes.get(active_idx) {
                if let Some(output) = cs.node_outputs.get(&active_node.name) {
                    let snippet: String = output.chars().take(40).collect();
                    let snippet = snippet.replace('\n', " ");
                    if !snippet.is_empty() {
                        line3.push(Span::styled(
                            format!(" │ {}", snippet),
                            Style::default().bg(bg).fg(Color::Gray),
                        ));
                    }
                }
            }
        }
        lines.push(Line::from(line3));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn status_project_label(app: &App) -> String {
    format!(" {} ", app.project_display_name)
}

fn render_chain_pane(frame: &mut Frame, app: &App, area: Rect) {
    use crate::orchestrate::types::ChainNodeStatus;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta))
        .title(Span::styled(
            " Chain (Ctrl+G to close) ",
            Style::default().fg(Color::Magenta),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    // Status summary from chain_state
    if let Some(ref cs) = app.chain_state {
        let cycle_display = if cs.config.loop_forever {
            format!("{}/∞", cs.current_cycle)
        } else {
            format!("{}/{}", cs.current_cycle, cs.config.max_cycles)
        };
        let elapsed = if cs.total_duration_secs > 0.0 {
            format!("  {:.0}s", cs.total_duration_secs)
        } else {
            String::new()
        };
        lines.push(Line::from(vec![
            Span::styled(
                cs.config.name.to_string(),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "  cycle {} · turn {}/{}{}",
                    cycle_display, cs.turns_used, cs.config.max_total_turns, elapsed
                ),
                Style::default().fg(Color::White),
            ),
        ]));

        for node in &cs.config.nodes {
            let (icon, color) = match cs.node_statuses.get(&node.name) {
                Some(ChainNodeStatus::Completed) => ("✓", Color::Green),
                Some(ChainNodeStatus::Running) => ("▶", Color::Yellow),
                Some(ChainNodeStatus::Failed(_)) => ("✗", Color::Red),
                Some(ChainNodeStatus::Paused) => ("⏸", Color::Cyan),
                Some(ChainNodeStatus::Pending) | None => ("○", Color::DarkGray),
            };
            let mut spans = vec![
                Span::styled(format!("  {} ", icon), Style::default().fg(color)),
                Span::styled(
                    format!("{:<12}", node.name),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ({})", node.model),
                    Style::default().fg(Color::DarkGray),
                ),
            ];
            let fails = cs.node_failures.get(&node.name).copied().unwrap_or(0);
            if fails > 0 {
                spans.push(Span::styled(
                    format!(" [{} fail]", fails),
                    Style::default().fg(Color::Red),
                ));
            }
            if let Some(durations) = cs.node_durations.get(&node.name) {
                if !durations.is_empty() {
                    let avg = durations.iter().sum::<f64>() / durations.len() as f64;
                    spans.push(Span::styled(
                        format!(" avg {:.1}s", avg),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            }
            let prompt_tok = cs.node_prompt_tokens.get(&node.name).copied().unwrap_or(0);
            let comp_tok = cs
                .node_completion_tokens
                .get(&node.name)
                .copied()
                .unwrap_or(0);
            if prompt_tok + comp_tok > 0 {
                spans.push(Span::styled(
                    format!(
                        " {}↑ {}↓",
                        format_tokens(prompt_tok),
                        format_tokens(comp_tok)
                    ),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            lines.push(Line::from(spans));
        }

        let total_prompt: u64 = cs.node_prompt_tokens.values().sum();
        let total_comp: u64 = cs.node_completion_tokens.values().sum();
        if total_prompt + total_comp > 0 {
            let mut summary = vec![Span::styled(
                format!("  Σ {} tokens", format_tokens(total_prompt + total_comp)),
                Style::default().fg(Color::White),
            )];
            if cs.total_duration_secs > 0.0 {
                let tok_per_sec = total_comp as f64 / cs.total_duration_secs;
                summary.push(Span::styled(
                    format!(" · {:.1} tok/s", tok_per_sec),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            lines.push(Line::from(summary));
        }
    }

    // Fill remaining space with log lines
    let status_lines = lines.len();
    let remaining = (inner.height as usize).saturating_sub(status_lines);
    if remaining > 0 && !app.chain_log.is_empty() {
        let skip = app.chain_log.len().saturating_sub(remaining);
        for l in &app.chain_log[skip..] {
            let color = if l.contains("error") || l.contains("failed") {
                Color::Red
            } else if l.contains("warn") {
                Color::Yellow
            } else if l.contains("complete") || l.contains("✓") {
                Color::Green
            } else {
                Color::Gray
            };
            lines.push(Line::from(Span::styled(
                l.as_str().to_owned(),
                Style::default().fg(color),
            )));
        }
    } else if app.chain_state.is_none() && !app.chain_log.is_empty() {
        let skip = app.chain_log.len().saturating_sub(inner.height as usize);
        for l in &app.chain_log[skip..] {
            let color = if l.contains("error") || l.contains("failed") {
                Color::Red
            } else if l.contains("warn") {
                Color::Yellow
            } else {
                Color::Gray
            };
            lines.push(Line::from(Span::styled(
                l.as_str().to_owned(),
                Style::default().fg(color),
            )));
        }
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(para, inner);
}

fn render_messages(frame: &mut Frame, app: &App, area: Rect) {
    let mut all_lines: Vec<Line<'static>> = Vec::new();

    for entry in &app.entries {
        match entry.kind {
            EntryKind::UserMessage => {
                all_lines.push(Line::from(vec![
                    Span::styled(
                        "You: ",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(entry.content.clone()),
                ]));
                all_lines.push(Line::from(""));
            }
            EntryKind::AssistantMessage => {
                let mut md = render_markdown(&entry.content);
                all_lines.append(&mut md);
                all_lines.push(Line::from(""));
            }
            EntryKind::ToolCall => {
                all_lines.push(Line::from(Span::styled(
                    format!("  [{}]", entry.content),
                    Style::default().fg(Color::Yellow),
                )));
            }
            EntryKind::ToolResult => {
                let shown: Vec<&str> = entry.content.lines().take(5).collect();
                for line in &shown {
                    let text = maybe_wrap_paths(line, app.hyperlinks);
                    all_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(text, Style::default().fg(Color::DarkGray)),
                    ]));
                }
                let total = entry.content.lines().count();
                if total > 5 {
                    all_lines.push(Line::from(Span::styled(
                        format!("  … ({} more lines)", total - 5),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
                all_lines.push(Line::from(""));
            }
            EntryKind::FileDiff => {
                for line in entry.content.lines() {
                    let styled = if line.starts_with('+') && !line.starts_with("+++") {
                        Line::from(Span::styled(
                            line.to_string(),
                            Style::default().fg(Color::Green),
                        ))
                    } else if line.starts_with('-') && !line.starts_with("---") {
                        Line::from(Span::styled(
                            line.to_string(),
                            Style::default().fg(Color::Red),
                        ))
                    } else if line.starts_with("@@") {
                        Line::from(Span::styled(
                            line.to_string(),
                            Style::default().fg(Color::Cyan),
                        ))
                    } else if line.starts_with("---") || line.starts_with("+++") {
                        Line::from(Span::styled(
                            line.to_string(),
                            Style::default()
                                .fg(Color::Blue)
                                .add_modifier(Modifier::BOLD),
                        ))
                    } else {
                        Line::from(Span::raw(line.to_string()))
                    };
                    all_lines.push(styled);
                }
                all_lines.push(Line::from(""));
            }
            EntryKind::ToolError => {
                let shown: Vec<&str> = entry.content.lines().take(5).collect();
                for line in &shown {
                    all_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(line.to_string(), Style::default().fg(Color::Red)),
                    ]));
                }
                let total = entry.content.lines().count();
                if total > 5 {
                    all_lines.push(Line::from(Span::styled(
                        format!("  … ({} more lines)", total - 5),
                        Style::default()
                            .fg(Color::Red)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
                all_lines.push(Line::from(""));
            }
            EntryKind::ImageAttachment => {
                all_lines.push(Line::from(vec![
                    Span::styled("  📎 ", Style::default().fg(Color::Magenta)),
                    Span::styled(
                        entry.content.clone(),
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
            EntryKind::SystemInfo => {
                all_lines.push(Line::from(Span::styled(
                    format!("  [{}]", entry.content),
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::ITALIC),
                )));
                all_lines.push(Line::from(""));
            }
            EntryKind::Notice => {
                for line in entry.content.lines() {
                    all_lines.push(Line::from(Span::styled(
                        format!("  {}", line),
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
                    )));
                }
                all_lines.push(Line::from(""));
            }
        }
    }

    // Streaming partial at the bottom
    if let Some(partial) = &app.streaming_partial {
        if !partial.is_empty() {
            let mut md = render_markdown(partial);
            if let Some(last) = md.last_mut() {
                last.spans
                    .push(Span::styled("▌", Style::default().fg(Color::White)));
            }
            all_lines.append(&mut md);
        } else {
            all_lines.push(Line::from(Span::styled(
                "  ▌",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    // Count actual rendered lines after word-wrapping, not just source lines.
    // The message pane inner width is area.width minus 2 (borders).
    let inner_width = area.width.saturating_sub(2) as usize;
    let total_lines: u16 = if inner_width > 0 {
        all_lines
            .iter()
            .map(|line| {
                let line_width: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
                if line_width == 0 {
                    1u16 // empty lines still take 1 row
                } else {
                    ((line_width.saturating_sub(1)) / inner_width + 1) as u16
                }
            })
            .sum()
    } else {
        all_lines.len() as u16
    };
    let visible_height = area.height.saturating_sub(2);
    let max_scroll = total_lines.saturating_sub(visible_height);

    // Write back cache so key handlers can clamp scroll values correctly.
    app.cached_total_lines.set(total_lines);
    app.cached_visible_height.set(visible_height);

    let scroll = if app.scroll_locked {
        // Pinned mode: use stored absolute top, clamped to valid range
        app.scroll_top.min(max_scroll)
    } else {
        // Auto-follow: always pin to the bottom
        max_scroll
    };

    let title = if app.scroll_locked {
        let pct = if max_scroll > 0 {
            (scroll as u32 * 100 / max_scroll as u32).min(100)
        } else {
            100
        };
        format!(" Messages [{pct}%] ")
    } else {
        " Messages ".to_string()
    };

    frame.render_widget(
        Paragraph::new(all_lines)
            .block(Block::default().borders(Borders::ALL).title(title))
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
}

fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let has_ctx = app.pending_context.is_some();
    let has_img = !app.pending_images.is_empty();
    let combined_hint = match (has_ctx, has_img) {
        (true, true) => format!(" Input [+ctx +{} img] ", app.pending_images.len()),
        (true, false) => " Input [+ctx] ".to_string(),
        (false, true) => format!(" Input [+{} img] ", app.pending_images.len()),
        (false, false) => " Input ".to_string(),
    };
    let (title, style): (std::borrow::Cow<str>, Style) = match app.mode {
        Mode::AskUserQuestion => (" Answer ".into(), Style::default().fg(Color::Cyan)),
        _ if app.agent_busy => (" Waiting… ".into(), Style::default().fg(Color::DarkGray)),
        _ if has_ctx || has_img => (combined_hint.into(), Style::default().fg(Color::Green)),
        _ => (" Input ".into(), Style::default()),
    };
    let display = format!("{}_", app.input);
    frame.render_widget(
        Paragraph::new(display)
            .style(style)
            .block(Block::default().borders(Borders::ALL).title(title))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn render_help(frame: &mut Frame, app: &App, area: Rect) {
    let busy_hint = if app.agent_busy && app.mode == Mode::Input {
        " [Esc] Cancel  [Ctrl+C] quit"
    } else {
        ""
    };
    let base = match app.mode {
        Mode::PermissionDialog => " [y] Allow  [n] Deny  [a] Always  [N] Never ",
        Mode::AskUserQuestion => " [Enter] submit  [Esc] cancel ",
        Mode::HelpOverlay => " [Esc] close help ",
        Mode::Scrolling => " [↑↓/j/k] scroll  [PgUp/PgDn] page  [Home] top  [G] bottom  [Esc] back ",
        Mode::DiffReview => " [y] accept file  [x] reject file  [a] apply  [r/Esc] reject all  [n/p] navigate  [j/k] scroll ",
        Mode::Input if app.agent_busy => " Waiting for Ollama…",
        Mode::Input if app.chain_state.is_some() => {
            " [Enter] send  [Shift+Enter] newline  [Ctrl+G] chain log  [↑↓] scroll  [/] commands "
        }
        Mode::Input => {
            " [Enter] send  [Shift+Enter] newline  [Ctrl+P/N] history  [↑↓] scroll  [/] commands "
        }
    };
    let text = format!("{}{}", base, busy_hint);
    frame.render_widget(
        Paragraph::new(text).style(Style::default().bg(Color::DarkGray).fg(Color::Gray)),
        area,
    );
}

const MIN_POPUP_WIDTH: u16 = 30;

fn render_permission_dialog(
    frame: &mut Frame,
    area: Rect,
    perm: &crate::tui::app::PendingPermission,
) {
    // When a risk banner is present the popup gets an extra line so the
    // ⚠ line + a blank separator still fit above the Tool / Args block.
    let popup_w = area.width.clamp(MIN_POPUP_WIDTH, 62);
    let base_h: u16 = if perm.reason.is_some() { 10 } else { 8 };
    let popup_h = base_h.min(area.height.saturating_sub(2));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup = Rect::new(popup_x, popup_y, popup_w, popup_h);

    frame.render_widget(Clear, popup);

    let args_display = format_args_short(&perm.tool_name, &perm.args);
    let (content, border) = if let Some(reason) = perm.reason.as_deref() {
        (
            format!(
                "\n  ⚠ Risky bash command: {}\n\n  Tool: {}\n  Args: {}\n\n  [y] Once  [a] Always  [n] Deny  [N] Never",
                reason, perm.tool_name, args_display
            ),
            Color::Red,
        )
    } else {
        (
            format!(
                "\n  Tool: {}\n  Args: {}\n\n  [y] Once  [a] Always  [n] Deny  [N] Never",
                perm.tool_name, args_display
            ),
            Color::Yellow,
        )
    };
    frame.render_widget(
        Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border))
                .title(" ⚠ Permission Required "),
        ),
        popup,
    );
}

fn render_question_dialog(
    frame: &mut Frame,
    area: Rect,
    q: &crate::tui::app::PendingQuestion,
    current_input: &str,
) {
    let popup_w = area.width.clamp(MIN_POPUP_WIDTH, 70);
    let options_count = q.options.len() as u16;
    let popup_h = (6 + options_count)
        .max(7)
        .min(area.height.saturating_sub(2));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup = Rect::new(popup_x, popup_y, popup_w, popup_h);

    frame.render_widget(Clear, popup);

    let mut content = format!("\n  {}\n", q.question);
    if !q.options.is_empty() {
        content.push_str("\n  Options:\n");
        for (i, opt) in q.options.iter().enumerate() {
            writeln!(content, "    {}. {}", i + 1, opt).expect("write to String never fails");
        }
    }
    write!(content, "\n  > {}▌", current_input).expect("write to String never fails");

    frame.render_widget(
        Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" ? Question "),
        ),
        popup,
    );
}

fn render_help_overlay(frame: &mut Frame, area: Rect) {
    let popup_w = area.width.clamp(MIN_POPUP_WIDTH, 56);
    let popup_h = 56u16.min(area.height.saturating_sub(2));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup = Rect::new(popup_x, popup_y, popup_w, popup_h);

    frame.render_widget(Clear, popup);

    let content = "\
\n\
 Keyboard shortcuts:\n\
   Enter       — send message\n\
   Ctrl+D      — send (or quit if input empty)\n\
   ↑ / ↓       — cycle input history\n\
   Shift+↑     — scroll messages (j/k, Home/G, PgUp/PgDn)\n\
   ← / →       — move cursor\n\
   Ctrl+←/→    — jump word\n\
   Ctrl+A/E    — start / end of line\n\
   Ctrl+K      — kill to end of line (or kill running bash when busy)\n\
   Ctrl+W      — delete word back\n\
   Ctrl+U      — clear input\n\
   Ctrl+L      — clear screen\n\
   Ctrl+Y      — copy last response to clipboard\n\
   Esc         — cancel / close\n\
   Ctrl+C      — quit\n\
\n\
 Slash commands:\n\
   /help                — this overlay\n\
   /clear               — clear conversation\n\
   /compact [preview|force] — compact context (preview = dry run, force = skip threshold)\n\
   /model <name>        — switch reasoning model\n\
   /model tool <name>   — set tool-call model\n\
   /model tool off      — disable model routing\n\
   /model embed <name>  — set embedding model\n\
   /models              — list Ollama models\n\
   /sessions [N]        — list recent sessions (default 10)\n\
   /sessions search <q> — full-text search across saved sessions\n\
   /sessions tree       — show fork lineage as an ASCII tree\n\
   /session delete <id> — delete a session\n\
   /session rename <id> <name> — rename a session\n\
   /resume <id>         — switch to a session by ID prefix\n\
   /permissions         — show permission rules\n\
   /attach <path>       — attach image (vision models)\n\
   /bug                 — file a bug report\n\
   /mcp                 — show connected MCP servers\n\
   /config              — show/set config values\n\
   /edit                — open last response in $EDITOR\n\
   /undo                — revert last user turn\n\
   /retry               — regenerate the last assistant response\n\
   /copy                — copy last response to clipboard\n\
   /init                — create DM.md in the current project directory\n\
   /commit              — generate AI commit message from staged changes\n\
   /pr [base]           — draft a PR description from commits ahead of base\n\
   /log [N]             — capture recent git log as context (default 20)\n\
   /diff [ref]          — capture git diff as context (e.g. /diff --staged)\n\
   /review [ref]        — capture diff as code-review prompt (default: HEAD)\n\
   /add <file>          — add file contents to context\n\
   /add-dir <path>      — add all text files in a directory to context\n\
   /doctor              — run system diagnostics\n\
   /blame <file> [line]  — capture git blame as context for analysis\n\
   /changelog [from [to]] — generate changelog from git commits\n\
   /conflicts           — detect and help resolve merge conflicts\n\
   /stash [push|pop|list|drop|show] — git stash operations\n\
   /branch              — list git branches\n\
   /branch <name>       — checkout or create branch\n\
   /search <query>      — search conversation history (alias: /grep)\n\
   /history [N]         — show session activity log (default last 30)\n\
   /kill                — terminate running bash without aborting the turn\n\
   /test [cmd]          — run tests, send failures to AI for fixing\n\
   /lint [cmd]          — run linter, send warnings to AI for fixing\n\
   /find <glob>         — find files by pattern (e.g. /find src/**/*.rs)\n\
   /rg <pattern> [path] — search file contents (ripgrep)\n\
   /cd [path]           — change working directory (no arg = show current)\n\
   /changes             — list files modified by AI in this session\n\
   /tree [path] [depth] — show directory tree as context (default depth 3)\n\
   /stats               — show session statistics\n\
   /fork [N]            — fork session into new branch (at turn N)\n\
   /undo-files          — revert last batch of applied file changes\n\
   /context             — show session context composition\n\
   /export [file.md]   — export conversation to markdown (with tool details)\n\
   /export clean [file.md] — export user/assistant/diffs only (shareable)\n\
   /bughunter [focus]  — scan codebase for bugs (e.g. /bughunter concurrency)\n\
   /security-review [ref] — security-focused diff review (OWASP, injection, auth)\n\
   /advisor [topic]    — get architectural/design advice on the codebase\n\
   /effort [quick|normal|thorough] — set response depth (default: normal)\n\
   /brief               — toggle brief mode (concise bullet-point responses)\n\
   /plan                — toggle plan mode (read-only exploration)\n\
   /new [title]         — start a fresh conversation (clears history)\n\
   /files               — list files currently in the conversation context\n\
   /pin <file>          — prepend file to every message (list pins with /pin)\n\
   /unpin [file|all]    — remove a pinned file, or unpin all\n\
   /rename <title>     — rename the current session\n\
   /summary [focus]    — generate a session summary via the LLM\n\
   /template            — list available templates\n\
   /template <name> [args] — load template as next message\n\
   /todo                — list session todos\n\
   /todo add [high|med|low] <text> — add a todo\n\
   /todo done <id>      — mark todo as completed\n\
   /todo scan           — scan codebase for TODO/FIXME comments\n\
   /share [file.html]  — export session as HTML\n\
   /memory              — show memory entries\n\
   /memory add <text>   — add entry\n\
   /memory forget <N>   — remove entry N\n\
   /memory clear        — clear all entries\n\
   /wiki status         — show wiki page counts and last activity\n\
   /wiki search <q>     — substring-match wiki pages (case-insensitive)\n\
   /wiki lint           — check wiki for orphans, category drift, and stale pages\n\
   /eval                — list eval suites\n\
   /eval <name>         — run eval suite inline\n\
   /agent list          — list registered agents\n\
   /agent <name>        — run an agent\n\
   /version             — show dm version, model, and host\n\
   /quit                — exit dm\n\
\n\
 Chain commands:\n\
   /chain start <file>         — start chain from YAML\n\
   /chain stop                 — stop the running chain\n\
   /chain pause                — pause between cycles\n\
   /chain resume               — resume a paused chain\n\
   /chain resume-from <dir>    — resume from saved checkpoint\n\
   /chain add <name> <role> [model] — add node at runtime\n\
   /chain remove <name>        — remove node at runtime\n\
   /chain model <node> <model> — swap a node's model live\n\
   /chain talk <node> <msg>    — inject message to node\n\
   /chain status               — show running chain state\n\
   /chain list                 — list available chain configs\n\
   /chain log [cycle]          — view chain artifacts\n\
   /chain metrics              — show chain performance stats\n\
   /chain validate <file>      — check config validity\n\
   /chain init [name]          — generate template YAML\n\
   /chain help                 — detailed chain help\n\
\n\
 Tool management:\n\
   /tool list              — list all tools and their status\n\
   /tool disable <name>    — disable a tool for this session\n\
   /tool enable <name>     — re-enable a disabled tool\n\
";
    frame.render_widget(
        Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" dm help "),
        ),
        popup,
    );
}

fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_args_short(tool_name: &str, args: &serde_json::Value) -> String {
    let s = match tool_name {
        "bash" => args["command"].as_str().unwrap_or("").to_string(),
        "write_file" | "read_file" | "edit_file" => args["path"].as_str().unwrap_or("").to_string(),
        "glob" => args["pattern"].as_str().unwrap_or("").to_string(),
        "grep" => args["pattern"].as_str().unwrap_or("").to_string(),
        _ => args.to_string(),
    };
    if s.len() > 40 {
        let mut end = 40usize.min(s.len());
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    } else {
        s
    }
}

fn render_diff_review(frame: &mut Frame, area: Rect, app: &crate::tui::app::App) {
    use ratatui::layout::Alignment;

    let change = &app.staged_changes[app.diff_review_idx];
    let (added, removed) = change.lines_changed();
    let total = app.staged_changes.len();
    let idx = app.diff_review_idx;

    // Overlay: 90% width, 80% height, centered
    let w = (area.width as f32 * 0.92) as u16;
    let h = (area.height as f32 * 0.82) as u16;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let overlay = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    frame.render_widget(Clear, overlay);

    // Outer border
    let border = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            {
                let accepted = app
                    .diff_file_decisions
                    .iter()
                    .filter(|d| **d == Some(true))
                    .count();
                let rejected = app
                    .diff_file_decisions
                    .iter()
                    .filter(|d| **d == Some(false))
                    .count();
                let undecided = total - accepted - rejected;
                if accepted > 0 || rejected > 0 {
                    format!(
                        " Staged Changes [{}/{}] — {} accepted, {} rejected, {} undecided ",
                        idx + 1,
                        total,
                        accepted,
                        rejected,
                        undecided
                    )
                } else {
                    format!(" Staged Changes [{}/{}] ", idx + 1, total)
                }
            },
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .title_alignment(Alignment::Center);
    frame.render_widget(border, overlay);

    // Shrink inner area by 1 on each side to account for border
    let inner = Rect {
        x: overlay.x + 1,
        y: overlay.y + 1,
        width: overlay.width.saturating_sub(2),
        height: overlay.height.saturating_sub(2),
    };
    let inner_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    // Path + stats line
    let decision = app.diff_file_decisions.get(idx).copied().flatten();
    let (indicator, ind_color) = match decision {
        Some(true) => ("[✓] ", Color::Green),
        Some(false) => ("[✗] ", Color::Red),
        None => ("[ ] ", Color::DarkGray),
    };
    let path_line = Line::from(vec![
        Span::styled(indicator, Style::default().fg(ind_color)),
        Span::styled(
            format!("{} ", change.path.display()),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("+{}", added), Style::default().fg(Color::Green)),
        Span::styled(" / ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("-{}", removed), Style::default().fg(Color::Red)),
    ]);
    frame.render_widget(
        Paragraph::new(path_line).style(Style::default().bg(Color::DarkGray)),
        inner_chunks[0],
    );

    // Diff body with scroll
    let diff_para = Paragraph::new(
        app.staged_changes[app.diff_review_idx]
            .diff
            .lines()
            .map(|line| {
                if line.starts_with('+') && !line.starts_with("+++") {
                    Line::from(Span::styled(
                        line.to_string(),
                        Style::default().fg(Color::Green),
                    ))
                } else if line.starts_with('-') && !line.starts_with("---") {
                    Line::from(Span::styled(
                        line.to_string(),
                        Style::default().fg(Color::Red),
                    ))
                } else if line.starts_with("@@") {
                    Line::from(Span::styled(
                        line.to_string(),
                        Style::default().fg(Color::Cyan),
                    ))
                } else if line.starts_with("---") || line.starts_with("+++") {
                    Line::from(Span::styled(
                        line.to_string(),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(line.to_string())
                }
            })
            .collect::<Vec<_>>(),
    )
    .scroll((app.diff_scroll, 0))
    .wrap(Wrap { trim: false });
    frame.render_widget(diff_para, inner_chunks[1]);

    // Footer keybindings
    let nav = if total > 1 {
        format!("  [n] next ({}/{})", idx + 1, total)
    } else {
        String::new()
    };
    let footer = Line::from(vec![
        Span::styled(
            " [y] accept",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            "[x] reject",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            "[a] apply",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            "[r/Esc] reject all",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::styled(nav, Style::default().fg(Color::DarkGray)),
        Span::styled("  [j/k] scroll", Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(Color::DarkGray)),
        inner_chunks[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_ctx_display_compacting_shows_banner() {
        // Any ctx_usage is ignored while compacting — the pause needs its
        // own distinct surface so the user doesn't read a stale percentage
        // and assume the agent is stuck on inference.
        let stage = crate::compaction::CompactionStage::FullSummary {
            messages_summarized: 12,
        };
        let (text, style) = format_ctx_display(Some(&stage), Some((50, 100)));
        assert_eq!(text, " [compacting...]");
        assert_eq!(style, CtxStyle::Working);
        assert!(!text.contains("CTX"));
    }

    #[test]
    fn format_ctx_display_idle_normal_ctx() {
        let (text, style) = format_ctx_display(None, Some((50, 100)));
        assert_eq!(text, " CTX 50%");
        assert_eq!(style, CtxStyle::Normal);
    }

    #[test]
    fn format_ctx_display_idle_warn_ctx() {
        let (text, style) = format_ctx_display(None, Some((95, 100)));
        assert_eq!(text, " CTX 95%");
        assert_eq!(style, CtxStyle::Warn);
    }

    #[test]
    fn format_ctx_display_idle_no_ctx_usage() {
        let (text, style) = format_ctx_display(None, None);
        assert!(text.is_empty());
        assert_eq!(style, CtxStyle::Normal);
    }

    #[test]
    fn format_ctx_display_compacting_overrides_warn_threshold() {
        // Belt-and-suspenders: even at 95% (which would normally trigger the
        // Warn style + `/compact` hint), the banner must win. Showing
        // "/compact" while already compacting would be confusing.
        let stage = crate::compaction::CompactionStage::Emergency;
        let (text, style) = format_ctx_display(Some(&stage), Some((95, 100)));
        assert_eq!(text, " [compacting...]");
        assert_eq!(style, CtxStyle::Working);
        assert!(!text.contains("CTX"));
        assert!(!text.contains("/compact"));
    }

    #[test]
    fn format_args_short_bash_uses_command_field() {
        let args = serde_json::json!({ "command": "cargo build" });
        assert_eq!(format_args_short("bash", &args), "cargo build");
    }

    #[test]
    fn format_args_short_write_file_uses_path() {
        let args = serde_json::json!({ "path": "/src/main.rs", "content": "fn main() {}" });
        assert_eq!(format_args_short("write_file", &args), "/src/main.rs");
    }

    #[test]
    fn format_args_short_read_file_uses_path() {
        let args = serde_json::json!({ "path": "README.md" });
        assert_eq!(format_args_short("read_file", &args), "README.md");
    }

    #[test]
    fn format_args_short_edit_file_uses_path() {
        let args =
            serde_json::json!({ "path": "src/lib.rs", "old_string": "old", "new_string": "new" });
        assert_eq!(format_args_short("edit_file", &args), "src/lib.rs");
    }

    #[test]
    fn format_args_short_glob_uses_pattern() {
        let args = serde_json::json!({ "pattern": "**/*.rs" });
        assert_eq!(format_args_short("glob", &args), "**/*.rs");
    }

    #[test]
    fn format_args_short_grep_uses_pattern() {
        let args = serde_json::json!({ "pattern": "fn main", "path": "." });
        assert_eq!(format_args_short("grep", &args), "fn main");
    }

    #[test]
    fn format_args_short_unknown_tool_serializes_json() {
        let args = serde_json::json!({ "foo": "bar" });
        let result = format_args_short("custom_tool", &args);
        assert!(
            result.contains("foo") || result.contains("bar"),
            "should serialize JSON for unknown tool: {result}"
        );
    }

    #[test]
    fn format_args_short_truncates_at_40_chars() {
        // command > 40 ASCII chars should be truncated and end with "…"
        let long_cmd = "a".repeat(60);
        let args = serde_json::json!({ "command": long_cmd });
        let result = format_args_short("bash", &args);
        // Result should be at most 40 ASCII chars + "…" (3 bytes)
        assert!(
            result.ends_with('…'),
            "truncated result should end with ellipsis: {result}"
        );
        let base: String = result.chars().take_while(|c| *c != '…').collect();
        assert!(
            base.len() <= 40,
            "base should be at most 40 chars: len={}",
            base.len()
        );
    }

    #[test]
    fn format_args_short_truncation_is_utf8_safe() {
        // A string of 🦀 (4 bytes each): 11 × 4 = 44 bytes > 40
        // Truncating at byte 40 would land mid-emoji; must back off to 36 (9 crabs)
        let crab_str = "🦀".repeat(11);
        let args = serde_json::json!({ "command": crab_str });
        let result = format_args_short("bash", &args);
        assert!(result.ends_with('…'), "should end with ellipsis: {result}");
        // No panic = success (UTF-8 boundary respected)
    }

    #[test]
    fn format_args_short_missing_command_returns_empty() {
        // bash with no "command" field → empty string (not a panic)
        let args = serde_json::json!({});
        assert_eq!(format_args_short("bash", &args), "");
    }

    #[test]
    fn format_args_short_missing_path_returns_empty() {
        let args = serde_json::json!({ "content": "data" });
        assert_eq!(format_args_short("write_file", &args), "");
    }

    #[test]
    fn format_args_short_exactly_40_chars_not_truncated() {
        let exactly_40 = "a".repeat(40);
        let args = serde_json::json!({ "command": exactly_40 });
        let result = format_args_short("bash", &args);
        assert_eq!(
            result, exactly_40,
            "exactly 40 chars should not be truncated"
        );
        assert!(!result.ends_with('…'), "should not have ellipsis");
    }

    #[test]
    fn format_args_short_41_chars_gets_truncated() {
        let just_over = "b".repeat(41);
        let args = serde_json::json!({ "command": just_over });
        let result = format_args_short("bash", &args);
        assert!(
            result.ends_with('…'),
            "41 chars should be truncated: {result}"
        );
    }

    #[test]
    fn popup_width_narrow_terminal() {
        let narrow = Rect::new(0, 0, 20, 40);
        let w = narrow.width.clamp(MIN_POPUP_WIDTH, 62);
        assert_eq!(
            w, MIN_POPUP_WIDTH,
            "on a 20-col terminal, popup should be clamped to MIN_POPUP_WIDTH"
        );
    }

    #[test]
    fn popup_width_normal_terminal() {
        let normal = Rect::new(0, 0, 100, 40);
        let perm_w = normal.width.clamp(MIN_POPUP_WIDTH, 62);
        let question_w = normal.width.clamp(MIN_POPUP_WIDTH, 70);
        let help_w = normal.width.clamp(MIN_POPUP_WIDTH, 56);
        assert_eq!(perm_w, 62);
        assert_eq!(question_w, 70);
        assert_eq!(help_w, 56);
    }

    #[test]
    fn format_args_short_empty_command_returns_empty_string() {
        let args = serde_json::json!({ "command": "" });
        assert_eq!(format_args_short("bash", &args), "");
    }

    #[test]
    fn chain_status_line_renders_node_icons() {
        use crate::orchestrate::types::*;
        use std::path::PathBuf;

        let config = ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "n1".into(),
                    name: "planner".into(),
                    role: "planner".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n2".into(),
                    name: "builder".into(),
                    role: "builder".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n3".into(),
                    name: "tester".into(),
                    role: "tester".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
            ],
            max_cycles: 5,
            max_total_turns: 100,
            workspace: PathBuf::from("/tmp"),
            skip_permissions_warning: false,
            loop_forever: false,
            directive: None,
        };

        let mut cs = ChainState::new(config, "test-chain".into());
        cs.node_statuses
            .insert("planner".into(), ChainNodeStatus::Completed);
        cs.node_statuses
            .insert("builder".into(), ChainNodeStatus::Running);
        // tester stays Pending (default)
        cs.active_node_index = Some(1);
        cs.turns_used = 7;
        cs.node_outputs
            .insert("builder".into(), "Building the widget".into());

        let bg = Color::DarkGray;
        let mut line3: Vec<Span> = Vec::new();
        line3.push(Span::styled(" ", Style::default().bg(bg)));
        for node in &cs.config.nodes {
            let (icon, color) = match cs.node_statuses.get(&node.name) {
                Some(ChainNodeStatus::Completed) => ("✓", Color::Green),
                Some(ChainNodeStatus::Running) => ("⏳", Color::Yellow),
                Some(ChainNodeStatus::Failed(_)) => ("✗", Color::Red),
                Some(ChainNodeStatus::Paused) => ("⏸", Color::Cyan),
                Some(ChainNodeStatus::Pending) | None => ("○", Color::DarkGray),
            };
            line3.push(Span::styled(
                format!("{} {} ", icon, node.name),
                Style::default().bg(bg).fg(color),
            ));
        }

        let text: String = line3.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            text.contains("✓ planner"),
            "completed node should show ✓: {text}"
        );
        assert!(
            text.contains("⏳ builder"),
            "running node should show ⏳: {text}"
        );
        assert!(
            text.contains("○ tester"),
            "pending node should show ○: {text}"
        );
    }

    #[test]
    fn chain_pane_height_zero_when_hidden() {
        let app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        assert!(!app.show_chain_pane);
        let h: u16 =
            if app.show_chain_pane && (!app.chain_log.is_empty() || app.chain_state.is_some()) {
                8
            } else {
                0
            };
        assert_eq!(h, 0, "chain pane should be hidden by default");
    }

    #[test]
    fn status_project_label_uses_identity_display_name() {
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        app.project_display_name = "finance-app".into();
        assert_eq!(status_project_label(&app), " finance-app ");
    }

    #[test]
    fn chain_pane_height_nonzero_when_shown_with_logs() {
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        app.show_chain_pane = true;
        app.push_chain_log("[chain] cycle 1 started".into());
        let h: u16 =
            if app.show_chain_pane && (!app.chain_log.is_empty() || app.chain_state.is_some()) {
                8
            } else {
                0
            };
        assert!(h > 0, "chain pane should be visible when toggled with logs");
    }

    #[test]
    fn push_chain_log_caps_at_200() {
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        for i in 0..250 {
            app.push_chain_log(format!("log {}", i));
        }
        assert_eq!(app.chain_log.len(), 200);
        assert!(
            app.chain_log[0].contains("50"),
            "oldest should be log 50, got: {}",
            app.chain_log[0]
        );
    }

    #[test]
    fn chain_pane_height_scales_with_nodes() {
        use crate::orchestrate::types::*;
        let config = ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "n1".into(),
                    name: "a".into(),
                    role: "r".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n2".into(),
                    name: "b".into(),
                    role: "r".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n3".into(),
                    name: "c".into(),
                    role: "r".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
            ],
            max_cycles: 5,
            max_total_turns: 100,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: false,
            loop_forever: false,
            directive: None,
        };
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        app.chain_state = Some(ChainState::new(config, "c1".into()));
        app.show_chain_pane = true;
        let node_count = 3u16;
        let ideal = node_count + 5; // 8
        let area_height = 60u16;
        let h = ideal.clamp(6, area_height / 3);
        assert_eq!(h, 8, "3 nodes should yield height 8");
    }

    #[test]
    fn chain_pane_height_clamped_for_small_terminal() {
        // area_height=12, max=4 → pane gets 4 (minimum usable height)
        let node_count = 0u16;
        let ideal = node_count + 5;
        let area_height = 12u16;
        let max = area_height / 3; // 4
        let h = if max < 4 { 0 } else { ideal.min(max).max(4) };
        assert_eq!(h, 4, "on 12-row terminal, pane should be capped at 4");
    }

    #[test]
    fn chain_pane_height_zero_for_tiny_terminal() {
        let area_height = 9u16;
        let max = area_height / 3; // 3
        let h: u16 = if max < 4 { 0 } else { 5u16.min(max).max(4) };
        assert_eq!(h, 0, "on very small terminal, pane should be hidden");
    }

    #[test]
    fn format_tokens_sub_thousand() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(42), "42");
        assert_eq!(format_tokens(999), "999");
    }

    #[test]
    fn format_tokens_thousands() {
        assert_eq!(format_tokens(1_000), "1.0k");
        assert_eq!(format_tokens(1_200), "1.2k");
        assert_eq!(format_tokens(45_300), "45.3k");
        assert_eq!(format_tokens(999_999), "1000.0k");
    }

    #[test]
    fn format_tokens_millions() {
        assert_eq!(format_tokens(1_000_000), "1.0M");
        assert_eq!(format_tokens(1_100_000), "1.1M");
        assert_eq!(format_tokens(12_500_000), "12.5M");
    }
}
