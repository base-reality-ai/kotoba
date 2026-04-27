# Host Skeleton

This example shows the minimal shape of a spawned host project: domain code
beside the dm kernel, a host wiki, and one compile-time host capability.

The capability lives in `src/host_caps.rs`:

```rust
pub struct HostSkeletonCapabilities;

impl HostCapabilities for HostSkeletonCapabilities {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostEchoTool)]
    }
}
```

`main.rs` installs that provider once at startup:

```rust
host_caps::install()?;
```

After that, every dm registry built through `default_registry`,
`default_registry_with_events`, or `sub_agent_registry` receives the host
tool alongside kernel tools. The example proves the path by dispatching
`host_echo` through a normal `default_registry`.

Run it:

```sh
cargo run --manifest-path examples/host-skeleton/Cargo.toml
```

The `dm` delegation branch installs the same capabilities before handing
control to the kernel path, so uncommenting the TUI launch keeps host tools
available there too.

See `.dm/wiki/concepts/host-capabilities.md` for the full contract:
`HostCapabilities`, `install_host_capabilities`, the `host_` namespace rule,
and runtime MCP extension through `.dm/mcp_servers.json`.
