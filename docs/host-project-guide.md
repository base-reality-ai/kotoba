# Host-project guide

A walkthrough for developers working inside a project spawned via
`dm spawn`. Companion to the wiki concept pages — this guide is the
operator workflow, those pages are the architectural contracts.

If you maintain canonical dm itself, you want `CONTRIBUTING.md`
instead. This document is for the spawned project's day-to-day.

## What you have after `dm spawn <name>`

`dm spawn finance-app` produces a directory that *is* a host project.
Conceptually:

- **Your code** lives in your own modules — typically `src/host_main.rs`,
  `src/domain.rs`, plus any modules you add. dm's kernel sources live
  alongside but are not yours to edit casually.
- **dm's kernel** ships as the same source tree you spawned from.
  You can read every file. Sync mechanics keep these in step with
  canonical dm without you tracking it manually.
- **Identity** is sealed in `.dm/identity.toml`. Mode is `host`,
  `host_project` is the name you spawned with, and
  `canonical_dm_revision` plus `canonical_dm_repo` together record
  exactly what kernel snapshot you started from. Don't hand-edit this
  unless the file's contract page (`spawn-identity.md`) says you may.
- **Wiki** is initialized at `.dm/wiki/` with the canonical layered
  layout: `entities/`, `concepts/`, `summaries/`, `synthesis/`,
  plus `index.md` and `schema.md`. Empty for your domain content,
  ready to fill.

## First run

```
cargo run --bin <name>          # exercise your host binary
cargo run --bin dm              # drop into dm's TUI on this project
cargo run --bin dm doctor       # confirm host mode is detected
```

`dm doctor` should report `Mode: host` and your `host_project`. If
it doesn't, your `.dm/identity.toml` was not picked up — check you
ran `dm` from the project root.

## Adding a host capability

Host capabilities are dm tools that live in your project's binary,
prefixed `host_` so kernel namespaces stay clean. The full contract
is in `.dm/wiki/concepts/host-capabilities.md`; the install-once
constraint that shapes how you compose them is in
`.dm/wiki/concepts/host-capabilities-install-once.md`.

Minimum viable shape:

```rust
use async_trait::async_trait;
use dark_matter::host::{install_host_capabilities, HostCapabilities};
use dark_matter::tools::{Tool, ToolResult};
use serde_json::{json, Value};

struct MyHostCaps;

impl HostCapabilities for MyHostCaps {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostAddTaskTool)]
    }
}

struct HostAddTaskTool;

#[async_trait]
impl Tool for HostAddTaskTool {
    fn name(&self) -> &'static str { "host_add_task" }
    fn description(&self) -> &'static str { "Append a task." }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {"title": {"type": "string"}}})
    }
    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let title = args.get("title").and_then(Value::as_str).unwrap_or("");
        Ok(ToolResult { content: format!("added: {title}"), is_error: false })
    }
}

fn main() -> anyhow::Result<()> {
    install_host_capabilities(Box::new(MyHostCaps))?;
    // ... your normal startup ...
    Ok(())
}
```

Important rules:

- **Install once.** `install_host_capabilities` writes a process-global
  `OnceLock`. A second call returns an error. If your project has
  multiple capability sets, build one aggregate `HostCapabilities`
  impl that delegates internally.
- **`host_` prefix.** Every tool name you expose must start with
  `host_`. The registry rejects others to keep the kernel namespace
  clean.
- **Tools coexist.** Kernel tools (`bash`, `read_file`, etc.) stay
  available alongside your host tools. The TUI / chain / sub-agents
  see both.

## Adding a host wiki concept

When you describe your domain in the wiki — data models, internal
patterns, decisions — those pages must carry `layer: host` so dm's
sync layer skips them and host-mode search ranks them ahead of
inherited kernel concepts. The reference walkthrough is in
`.dm/wiki/concepts/host-tracker-data-model.md`.

Frontmatter shape:

```
---
title: Task data model
type: concept
layer: host
sources:
  - src/domain.rs
last_updated: 2026-04-26 00:00:00
---
# Task data model

...your content...
```

Two rules:

- **`layer: host` is the protection.** Without it, the page defaults
  to kernel and `dm sync` will treat it as kernel-scope.
- **`sources:` points at your code.** Don't reference canonical dm
  paths from your host concept pages — that drifts when you sync.

## Pulling kernel updates: `dm sync`

When canonical dm advances, your project stays frozen until you opt
in. The full contract is in `.dm/wiki/concepts/dm-sync.md`. Day-to-day
workflow:

```
dm sync --status     # is canonical ahead of you?
dm sync --dry-run    # what would change, with kernel/host scope tags
dm sync              # apply (cargo check gate, atomic rename, audit log)
dm sync --abort      # discard a half-applied sync
```

`dm sync` is read-only until you run it without flags:

- `--status` queries `canonical_dm_repo` for HEAD and reports drift.
  No clone, no writes.
- `--dry-run` clones canonical to `.dm/sync-staging/`, computes the
  per-file delta, classifies entries as kernel or host, surfaces
  conflicts for review. Performs no writes to your host tree.
- The default `dm sync` (no flags) runs the dry-run flow, gates on
  `cargo check` of the merged staged tree, then atomically swaps
  clean kernel files into your tree, bumps
  `canonical_dm_revision`, appends an audit entry to
  `.dm/wiki/log.md`, and removes the staging directory.
- `--abort` discards `.dm/sync-staging/` so a fresh `dm sync` can
  restart. Idempotent — running it when nothing is staged just
  prints an informational note.

If `cargo check` fails inside the staged tree, sync preserves
`.dm/sync-staging/` so you can inspect the merged tree and resolve
conflicts. Then re-run `dm sync` (or run `--abort` to discard).

## What sync will not touch

Per `.dm/wiki/concepts/dm-sync.md`, sync's kernel scope is a
deliberate whitelist. Host-owned files are never modified by sync:

- `.dm/identity.toml` (except `canonical_dm_revision` on success).
- `src/host_main.rs`, `src/domain.rs`, anything else you write.
- Any file under `.dm/wiki/` whose `layer:` frontmatter is `host`.
- `tests/host_*.rs` files (host-shaped test convention).
- Host-added entries in `Cargo.toml` (your `[[bin]]` blocks,
  host-only `[dependencies]`).

If sync wants to update a file you've changed, it surfaces it as
conflict and refuses to apply automatically. Review, decide whether
your edit or the canonical edit wins, resolve manually.

## Troubleshooting

- **`dm doctor` says kernel mode**: your `.dm/identity.toml` is missing
  or has `mode = "kernel"`. Check the file is at `<project>/.dm/identity.toml`.
- **`dm sync --status` errors with "canonical_dm_repo not set"**:
  legacy host identity from before Tier 2. Re-run `dm spawn` against
  the same canonical, or hand-edit identity.toml to add the
  `canonical_dm_repo` field.
- **Your host_-prefixed tool doesn't show up**: confirm you call
  `install_host_capabilities` exactly once at startup. Multiple
  installs error; missing install silently leaves your tools out.
- **`dm sync --dry-run` shows your host page as kernel-scope**:
  the page is missing `layer: host` in frontmatter. Add it; the
  default is kernel.

## Where the contracts live

This guide intentionally stays light. The architectural contracts
that constrain spawned-project behavior are in the wiki under
`.dm/wiki/concepts/`:

- `wiki-layering.md` — kernel/host layer split.
- `dm-sync.md` — sync semantics and CLI surface.
- `spawn-identity.md` — identity protocol fields.
- `host-capabilities.md` — capability install hook contract.
- `host-capabilities-install-once.md` — composition rule for
  multi-domain hosts.
- `host-tracker-data-model.md` — reference for host-layer concept
  pages.
- `kernel-substrate-stability.md` — what stays stable across kernel
  versions.

Read those when this guide is ambiguous; they are the source of
truth.
