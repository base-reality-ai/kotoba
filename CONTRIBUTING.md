# Contributing to Dark Matter

Dark Matter is developed in **kernel mode** in this repository. A
spawned project is host mode; it grows domain code around this kernel.
If you are working inside a spawned project, read
`docs/host-project-guide.md` instead.

`VISION.md` is the architectural source of truth. Read it before making
paradigm-level changes. Do not edit it casually; changes to `VISION.md`
are propose-only unless the operator explicitly applies them.

## Development model

This repo is worked by a chain of senior coding agents and the human
operator. The current chain alternates:

```
claude -> codex -> claude -> codex -> ...
```

Each turn should produce one concrete improvement: a bug fix, test,
small feature slice, wiki update, or deliberate tier transition. Avoid
large speculative rewrites. The next engineer should be able to audit
the diff quickly and continue.

## Contribution rules

- **Read before writing.** Check current code and wiki state before
  planning against memory.
- **Fix bugs before features.** Compile breaks, failing tests, panics,
  and regressions are priority zero.
- **Tests only go up.** Do not delete tests unless the production code
  they cover is provably dead.
- **No stubs in user-facing paths.** Anything exposed by CLI, TUI,
  daemon, MCP, or tools should work end to end for its shipped scope.
- **Keep increments small.** One concrete change per cycle is easier
  to review and safer to sync into host projects later.
- **Error messages include next steps.** Prefer
  `Error: X. Try: Y.` over dead-end diagnostics.
- **Wiki keeps learning.** If code behavior moves, update the relevant
  `.dm/wiki/` concept or entity page.

## Verification

For normal Rust changes, run:

```
cargo check
cargo test --lib
cargo clippy --all-targets -- -D warnings
```

Run targeted integration tests for the touched behavior. Examples:

```
cargo test --test spawn_smoke
cargo test --test sync_smoke
cargo test --test tier3_host_project
cargo test --test wiki_layering
```

Keep output concise in handoffs. For clean runs, summarize as
`cargo test --lib: PASS 3630` or similar. For failures, include only
the failing test names and actionable error lines.

## Scope guards

These require explicit operator approval:

- Editing `VISION.md` in place. Draft proposed wording in a handoff
  instead.
- Pushing branches, force-pushing, or changing remote state.
- Editing `~/dev/glue`. Adapter robustness work belongs there later,
  but the operator may want separate commits.
- Adding external dependencies to `Cargo.toml` or other package
  manifests.
- Reverting changes you did not make. The worktree may be dirty from
  another engineer or generated output; work with it.

Never use destructive commands such as `git reset --hard` unless the
operator explicitly asks.

## Kernel vs host responsibilities

Kernel-mode code should remain clean by default when
`.dm/identity.toml` is absent or says `mode = "kernel"`.

Host-mode behavior should be explicit and identity-aware:

- Host capabilities enter through `HostCapabilities` and
  `install_host_capabilities`.
- Host tools use the `host_` prefix.
- Host wiki pages use `layer: host`.
- Host projects update kernel code through `dm sync`, not silent
  auto-update.

When adding identity-aware behavior, update the relevant tests and
consider whether the identity-consumer documentation needs a wiki
refresh.

## Handoff format

Every chain turn ends with:

```
## Handoff to <next-engineer>

### Audit of <prior-engineer's last turn>
- Reviewed <prior>'s diff — no issues.

### What I did this turn
- <file:line-range> — <one-line summary>

### Verification
- cargo check: PASS
- cargo test --lib: PASS <count>
- clippy: clean

### What's next (priority order)
1. <specific actionable item>
2. <specific actionable item>
```

If there are blockers or operator decisions pending, include them
briefly. Otherwise omit that section.

## Where to look

- `VISION.md` — authoritative paradigm.
- `DM.md` — short project operating frame.
- `.dm/wiki/concepts/wiki-layering.md` — kernel/host wiki split.
- `.dm/wiki/concepts/dm-sync.md` — explicit host update flow.
- `.dm/wiki/concepts/kernel-substrate-stability.md` — host-facing
  stable surfaces.
- `docs/host-project-guide.md` — workflow for spawned projects.
