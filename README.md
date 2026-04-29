# kotoba (言葉)

> **A Japanese-learning second brain, grown around the dark-matter spine.**

Kotoba isn't an app you point at your studies — it's an environment your
studies live inside. The wiki at `.dm/wiki/` accumulates everything you've
encountered: vocabulary, kanji, grammar points, the personas you've
conversed with, the things you keep stumbling on. Conversational AI tutors
(Yuki for v0.1) talk to you in Japanese while quietly logging what was new,
what was practiced, and what got confused, so future sessions pick up where
this one left off.

This is what makes it different from Anki, Duolingo, or chatting with raw
ChatGPT: **comprehension compounds**. The system isn't just generating
responses — it's maintaining a living model of what you've actually learned
and using that to plan what comes next.

## Status

**v0.1 — scaffolding shipped.** Domain types, 5 host capabilities, Yuki
seeded as a wiki entity, integration test green. The conversation TUI
launches via `kotoba` but persona-driven planner/recorder agent chains
arrive in v0.2.

See `DM.md` for the design doc; see `VISION.md` for the dark-matter spine
paradigm this is built on.

## Quick start

```bash
cargo build --release

# Launch the conversation TUI in host mode.
./target/release/kotoba

# Or pass through to dm directly:
./target/release/kotoba dm doctor    # verify identity = host, kotoba

# Help:
./target/release/kotoba help
```

The first time you run `kotoba`, dm's TUI opens with kotoba's host
capabilities registered. Yuki's persona definition is at
`.dm/wiki/entities/Persona/Yuki.md` and is loaded via
`host_invoke_persona`.

## Architecture in one paragraph

`src/host_main.rs` is the kotoba binary's entry point. It registers
`KotobaCapabilities` (5 tools defined in `src/host_caps.rs`) into the
dm spine via `dark_matter::host::install_host_capabilities`, then
delegates to the `dm` kernel binary. The kernel — TUI, chain
orchestration, wiki, sessions, daemon — is unchanged from canonical
dark-matter and lives at `src/main.rs` and the `dark_matter` library
crate. Kotoba's domain (Vocabulary, Kanji, Persona, Session) is in
`src/domain.rs` and the durable storage is the host-layer wiki under
`.dm/wiki/entities/`.

## Backend models

Local Ollama (the default) won't be Japanese-fluent. Recommended:

- **Conversation persona →** `gemini-3.1-pro-preview` (strongest in Japanese)
- **Planner / recorder →** `claude-opus`

Configure kotoba role models with environment variables or `.dm/kotoba.toml`:
`KOTOBA_PERSONA_MODEL`, `KOTOBA_PLANNER_MODEL`, and
`KOTOBA_RECORDER_MODEL`. The planner and recorder are rule-based by default;
set `KOTOBA_PLANNER_USE_LLM=1` or `KOTOBA_RECORDER_USE_LLM=1` to run that
role through dm's current Ollama-compatible capture path.

## License

MIT. Inherits from canonical dark-matter.
