# Dark Matter — Vision

> *This document captures the foundational paradigm of Dark Matter.
> Everything else — architecture, roadmap, the chain that builds dm —
> follows from what is written here. Treat it as authoritative.*

## The Flip

Every other AI coding tool today — Claude Code, Cursor, Copilot,
Aider — sits **outside** the project. You install them, point them
at code, and they perform operations on the project. The project is
the subject; the tool is external.

Dark Matter inverts this.

A project does not *use* dm. A project **is grown around dm**.
Dm is the engine; the project is the car. Dm is the kernel; the
project is the OS. Dm is the spine; the project's domain code is
the rest of the body.

The boundary between "the tool" and "the work" is dissolved. Dm's
chain orchestration, comprehension wiki, multi-agent reasoning,
TUI, daemon, MCP, and tool surface are all available to the host
project as its native runtime — not as a separate process the host
shells out to. The host project's developers don't think of "using
an AI assistant"; they think of working **inside** an environment
that is continuously aware of what they're building.

## Two Modes of Existence

Dark Matter exists in exactly two modes, and a running dm always
knows which it is:

### `kernel` mode (canonical)

This is dark-matter as it is developed. The github repo at
`base-reality-ai/dark-matter` is the canonical source. The
3-engineer glue chain (claude → codex → gemini) operates here.
The wiki tracks dm's own subsystems. This mode is what every
maintainer of dark-matter sees.

There is exactly one canonical dm. It never changes its identity.

### `host` mode (spawned)

This is what every other instance of dm becomes. An operator runs
`dm spawn <project-name>` (or pulls dm into a fresh dir). The
spawned project carries dm as its kernel. Identity is captured in
`.dm/identity.toml`:

```toml
mode = "host"
host_project = "finance-tracker"
canonical_dm_revision = "<git sha at spawn time>"
```

In host mode:
- The wiki tracks the **host project's** concepts, entities, decisions,
  and drift. Not dm's. Dm's intrinsic concepts (chain, wiki, agents)
  are a stable substrate the host project rarely cares about.
- The TUI title, session metadata, and chain seeds all reference the
  host project. Dm's own subsystems are invisible plumbing.
- The host project's domain code lives alongside dm's source — they
  ship as one binary, one dependency graph, one identity. The user
  doesn't "run dm in a finance app project"; they run **the finance
  app**, which happens to have dm as its core.

## Why This Matters

A tool you point at a project carries no memory of why you made
yesterday's decision, no model of what the project is becoming, no
ability to act between your sessions. It is a powerful pen.

A spine that grows with the project carries everything. The wiki
holds why that decision was made. The chain can run continuously,
maintaining the project while the operator sleeps. Capabilities
accumulate. Knowledge compounds. The system is not "AI-assisted
software" — it **is** the software, with intelligence woven into
the substrate.

The host project's developer does not pick up and put down a tool.
They live inside an environment that is continuously, materially
aware of what they are building.

## Historical Analogues

This pattern has been built before, several times, by people who
saw the same flip:

- **Smalltalk image.** The IDE *is* the runtime *is* the application.
  You cannot separate them. Code and tools and program state share
  one persistent world.
- **Lisp machine.** The system you write programs in is the system
  that runs your program. There is no boundary between development
  and execution.
- **Emacs.** An editor that grew into a Lisp environment that grew
  into an OS-shaped thing. The boundary between "the editor" and
  "your work" is intentionally porous.
- **Plan 9.** The OS-as-application philosophy: every component is
  the system, not separate from it.

None of these are mainstream today. Each, for the people who lived
inside them, was uniquely powerful. Dark Matter takes the same
flip and applies it to the era of multi-agent intelligence: the
spine you grow your project around carries the agents, the
orchestration, and the comprehension as native organs — not bolted-
on services.

## What Dark Matter Provides as Spine

A host project that carries dm as its kernel inherits, by default:

- **TUI runtime** — `dm` launches a ratatui-based interactive
  terminal that the host project can extend with its own slash
  commands and tool surfaces.
- **Multi-agent chain orchestration** — the host project can spawn
  chains of agents (using local Ollama, Claude, GPT, Gemini, or
  any combination) to perform autonomous work in its domain.
- **Comprehension wiki** — the host project's `.dm/wiki/` becomes
  its own living understanding of itself: entities, concepts,
  synthesis, drift detection, lint.
- **Tool registry** — file ops, bash, grep, MCP, etc., already
  there. The host project adds its own domain tools alongside.
- **Daemon and web modes** — the host project can run dm as a
  background service, expose an HTTP API, or front a web UI.
- **Session persistence** — every interaction with the host project
  is recorded, replayable, resumable.

The host project supplies what only it can: its **domain**. The
tools that operate on its problem space, the entities its wiki
should track, the agents its chain should orchestrate, the surface
its users actually see.

## What This Forces Us to Decide

Several architectural questions become first-order once this
paradigm is taken seriously:

### Identity protocol

Every dm subsystem (wiki, chain, sessions, TUI title, daemon name)
must check `mode` and `host_project` and behave accordingly. A
single `.dm/identity.toml` is the source of truth. Subsystems that
make assumptions about being-canonical-dm are bugs in the spawn
paradigm.

### Capability extension

A finance app needs `categorize_transaction`. A game needs
`simulate_tick`. A scientific compute project needs `run_experiment`.
Dm's tool registry accepts additions from the host project on two axes:

- Compile-time host capabilities: a spawned host crate implements
  `HostCapabilities`, returns `host_`-prefixed tools, and calls
  `install_host_capabilities` once at startup. Every dm registry
  constructor then merges those host tools alongside kernel tools.
- Runtime MCP capabilities: host projects can add external MCP servers
  without recompiling by declaring them in `.dm/mcp_servers.json`.
  Project-local entries override same-name global `~/.dm/mcp_servers.json`
  entries.

The `host_` prefix is enforced at registration time because tool-call
schemas allow `[A-Za-z0-9_-]` names but reject `host:`. In host mode,
prompt surfaces group host capabilities ahead of the kernel substrate so
agents can distinguish project-domain organs from dm's inherited spine.

### Knowledge isolation

In a host project, dm's intrinsic concepts ("chain", "wiki",
"agent", "orchestrator") and the host project's domain concepts
both live in the wiki. They must not blur. Two layers — a stable
"kernel layer" the host inherits, and an evolving "host layer" the
host owns — is the most likely model. Details to be designed.

### Update flow

Each spawned project forks from canonical dm at a point in time.
When canonical dm ships improvements, do spawned projects rebase?
Patch? Stay frozen? Recommended default: **frozen**. An
explicit `dm sync` operation lets operators pull kernel
improvements when they want them, with full visibility into what's
changing. Otherwise downstream forks drift and breakages compound
silently.

### Visibility of kernel source

Should the host project see dm's source sitting under it, or
should dm be opaque (a library it links against)? The visible-source
model enables hacking and learning at the cost of drift risk. The
opaque model enforces a clean abstraction at the cost of
inflexibility. Likely answer: **visible by default** — host project
developers should be able to read every line of the engine they
live inside, and the spawn process makes the kernel/host boundary
clear without hiding either side.

## The Bootstrap Recursion

The 3-engineer glue chain (claude → codex → gemini) currently
develops dm in `kernel` mode. Its initial focus is making dm
**spawnable**: identity protocol, `dm spawn` command, library form
factor, end-to-end smoke. Once those land, the chain itself becomes
re-usable in any spawned project, where it will materialize host-
project code instead of dm code.

This is the meta-aware recursion: dm-the-spine is being built by
the very chain that will, after this initiative, become available
in every project that grows around dm.

## What Stays Constant

- The github repo at `base-reality-ai/dark-matter` is canonical and
  is never modified by spawned projects' work.
- Dm's intrinsic identity — meta-aware harness, comprehension
  through the wiki, chain orchestration as a first-class primitive,
  flux state — does not change between modes.
- The wiki always reflects what dm is incubating: in kernel mode,
  that is dm itself; in host mode, that is the host project.

## What This Document Is For

This document is the source of truth for the paradigm. The glue
chain references it on every cycle. The DM.md system prompt
references it. New contributors to dark-matter read it first.
When the paradigm is unclear in a future decision, return here.

The document does not change without operator review. The chain
may propose changes via diff in a handoff but does not modify
`VISION.md` in place.
