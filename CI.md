# Development and CI

The workspace pins Rust 1.95.0 and Cargo.lock. GitHub Actions runs formatting, strict all-target/all-feature Clippy and all-target/all-feature tests. No deployment or broker/cloud account action is configured. Registry/native build access is needed for an initial install; test bodies are offline/synthetic.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
```

Resource-constrained local builds can set CARGO_BUILD_JOBS=2, CARGO_PROFILE_DEV_DEBUG=0 and CARGO_PROFILE_TEST_DEBUG=0. These affect build resources, not fill semantics. Current suite: four tests (two engine/return and two cache-containment cases). Zero-test example targets do not imply those examples were exercised against real data. CLI --help is safe; ingest downloads and show reads existing cache. [Design](docs/design.md) records source/model gaps; passing CI does not prove QC/Nautilus parity, comprehensive accounting correctness or investment performance.
