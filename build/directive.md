# Directive: 3-Engineer Glue Chain Building Dark Matter

## Overview

Three senior software engineers — running on three different
frontier models — are orchestrated by Glue to build Dark Matter
itself. The irony stays the same as before: they are an externally
orchestrated chain building the internal orchestration that makes
external orchestration unnecessary.

## The Chain

```
claude (Opus, 1M ctx) → codex (GPT-5.5) → claude → codex → ...
```

Two engineers in mutual oversight. **Gemini is temporarily dropped**
(session cap on the prior run); the original 3-engineer architecture
is preserved in seeds/gemini.md and a commented-out node in
glue.yaml so it can be re-enabled cleanly.

- **Claude** — long-range reasoning + architectural judgment + audit
  depth. Catches missing tests, edge cases, brittle constructs,
  paradigm violations against VISION.md.
- **Codex** — strongest pure coder, fast implementation. Catches
  logic errors in Claude's code, simplifies Claude's over-designed
  abstractions, completes half-finished work.

Every node is a senior engineer, not a fixed role. Each flexes
between planning, implementing, and reviewing depending on what the
prior engineer left. They oversee each other across cycles —
Claude→Codex audits Claude's prior turn, Codex→Claude audits
Codex's prior turn.

## Shared Context

Glue injects `.glue/directives.md` into every node on every turn. This
is the Prime Directive — it carries:
- What Dark Matter is (the meta-harness identity)
- The wiki-as-comprehension-layer philosophy
- The shared workflow (audit prior → plan → implement → verify → handoff)
- The structured handoff format
- The per-cycle context budget rules
- The Current Focus (the operator-set initiative for this run)

Seeds (`seeds/{claude,codex,gemini}.md`) carry per-engineer identity:
strengths to lean into and the oversight relationship with the
previous engineer in the chain.

## Running

```bash
cd ~/dev/dark-matter/build
glue run              # start fresh
glue resume <id>      # resume session
```

Each adapter requires its CLI installed and authenticated:
- `claude` (Anthropic Claude Code)
- `codex` (OpenAI Codex CLI)
- `gemini` (Google Gemini CLI)

## Live Editing

Edit `.glue/directives.md` while the chain is running to steer the
agents. Changes are picked up on the next turn for all nodes. The
operator changes the Current Focus section here without restarting.

## Legacy

`seeds/legacy/` preserves the prior planner/builder/tester seeds in
case the role-pipeline architecture is needed for a specific run.
