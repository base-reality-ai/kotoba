# kotoba (言葉)

Kotoba is a Japanese-learning host project built on the **dark-matter spine**.
Dark Matter isn't a tool acting on this project — it's the engine kotoba grows
around. The wiki at `.dm/wiki/` is the learner's second brain; the chain
orchestration drives planner → conversation → recorder loops; the LLM
agents take on personas (Yuki, Tanaka-sensei, …) that accumulate lore over
time.

For the dm paradigm, see `VISION.md` at repo root. **Read it first** if you
encounter a paradigm decision and aren't sure which way to go.

## Identity

This project runs in **host mode**. `.dm/identity.toml` carries:

- `mode = "host"`
- `host_project = "kotoba"`
- `canonical_dm_revision = <sha at spawn>`
- `canonical_dm_repo = "https://github.com/base-reality-ai/Dark-Matter"`

Every dm subsystem (TUI title, doctor, wiki snippet header, system prompt)
consults this and reflects "kotoba" rather than "dark-matter".

## What kotoba is

A conversational Japanese tutor that learns about you the same way you
learn the language. Three things make it different from Anki, Duolingo,
or chatting with raw ChatGPT:

1. **The wiki accumulates everything you encounter.** Every kanji you
   meet, every grammar point you stumble on, every word you keep
   forgetting — they live as wiki entities at
   `.dm/wiki/entities/Vocabulary/`, `Kanji/`, `GrammarPoint/`. You own
   the second brain.
2. **Personas are real characters with continuity.** Yuki the tutor
   remembers what you covered yesterday, what you struggled with last
   week, what topics light you up. She's a wiki entity that accumulates
   session-by-session lore — not a fresh prompt every time.
3. **Planner → conversation → recorder.** Before each session a
   planner agent reads your wiki and proposes today's focus. After
   each session a recorder agent logs what was new, what was
   practiced, what got confused. The conversation in between is just
   you and Yuki.

## Architecture

```
┌──────────────────────────────────────────────┐
│  kotoba (host project)                       │
│                                              │
│  src/host_main.rs   — entry, registers caps  │
│  src/host_caps.rs   — 5 host tools           │
│  src/domain.rs      — Vocabulary/Kanji/...   │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  dark-matter spine (kernel library)    │  │
│  │  TUI │ chain │ wiki │ tools │ daemon   │  │
│  │  (untouched, version-pinned at spawn)  │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  .dm/                                        │
│    identity.toml    — mode=host, kotoba      │
│    wiki/                                     │
│      entities/Vocabulary/<word>.md           │
│      entities/Kanji/<character>.md           │
│      entities/GrammarPoint/<concept>.md      │
│      entities/Persona/<name>.md              │
│      synthesis/struggles-<date>.md           │
└──────────────────────────────────────────────┘
```

## Host capabilities

When `kotoba` boots, it installs `KotobaCapabilities` into the dm spine
via `dark_matter::host::install_host_capabilities`. The kernel registry
then exposes these tools to every agent on every turn:

| Tool | Purpose |
|---|---|
| `host_invoke_persona` | Switch the active conversational persona (loads the persona's wiki entity as system prompt) |
| `host_log_vocabulary` | Record a Japanese word with kanji/kana/romaji/meaning/JLPT/example |
| `host_log_kanji` | Record a kanji with on'yomi/kun'yomi/radicals/mnemonic |
| `host_record_struggle` | Flag something the learner stumbled on (planner reads this next session) |
| `host_quiz_me` | Pull due-for-review vocabulary from the wiki |

See `.dm/wiki/concepts/host-capabilities.md` (kernel layer) and the
docstrings in `src/host_caps.rs` for the contract.

## Personas (v0.1)

**Yuki (ゆうき)** — patient tutor, です/ます register, scaffolded
immersion. Speaks Japanese but provides English glosses for new words,
accepts romaji from learners. Wiki entity at
`.dm/wiki/entities/Persona/Yuki.md`.

Future personas (v0.2+):
- **Tanaka-sensei (田中先生)** — formal classroom tutor, structured
  lessons
- **Hiro (ひろ)** — casual conversation partner, タメ口, Tokyo dialect
- **Watanabe-buchou (渡辺部長)** — business mentor, full 敬語

Each persona is a wiki entity. They accumulate lore (sessions count,
topics covered, your interactions with them) so you have a continuing
relationship rather than a fresh prompt every time.

## Running

```bash
cargo build --release
./target/release/kotoba                # launch the TUI in host mode
./target/release/kotoba help           # see CLI options
./target/release/kotoba dm doctor      # run dm's doctor in host mode
```

## Backend models

Kotoba reads role-specific model settings from environment variables
first, then `.dm/kotoba.toml`, then built-in defaults. The v0.2
planner and recorder are rule-based host capabilities, so their model
values are loaded and documented for the v0.3 agent-driven versions;
`KOTOBA_PERSONA_MODEL` is active now and is passed to the dm TUI as
`--model` by `kotoba session`.

| Role | Env var | TOML key | Default |
|---|---|---|---|
| Conversation persona | `KOTOBA_PERSONA_MODEL` | `persona_model` | `gemini-3.1-pro-preview` |
| Pre-session planner | `KOTOBA_PLANNER_MODEL` | `planner_model` | `claude-opus` |
| Post-session recorder | `KOTOBA_RECORDER_MODEL` | `recorder_model` | `claude-opus` |

Example `.dm/kotoba.toml`:

```toml
[models]
persona_model = "gemini-3.1-pro-preview"
planner_model = "claude-opus"
recorder_model = "claude-opus"
```

Environment variables override the TOML file for local experiments:

```bash
KOTOBA_PERSONA_MODEL=gemini-3.1-pro-preview kotoba session
```

## v0.1 scope

What's in this cut:
- Domain types (Vocabulary, Kanji, GrammarPoint, Persona, Session,
  Mastery, ImmersionMode)
- 5 host capabilities, registered at startup
- Yuki seeded as a wiki entity
- Integration test verifying caps register
- Documentation (this file, README, wiki concept pages)
- TUI delegation via `kotoba` → `dm` subprocess pattern

What's deferred to v0.2:
- Planner / recorder agent chains (currently you'd run them by hand)
- Spaced-repetition timing on `host_quiz_me` (currently order-of-disk
  list, not due-date weighted)
- Multiple personas
- Voice / audio input
- Web UI
- JLPT progression tracking
- `dm sync` flow when canonical dm ships kernel improvements
