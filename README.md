# lbr — local event-driven backtesting

Workspace version **0.1.1**, Rust 1.95.0 / edition 2024. The current engine is event-driven; the former VectorEngine/target_weights README examples were superseded and do not match the exported API.

| Crate | Implemented scope |
|---|---|
| lbr | Bar/Parquet cache, aligned ndarray panels, indicators, event engine, portfolio and metrics |
| lbr-cli | Data driver: Yahoo daily ingest and cached panel inspection |
| lbr-nautilus | Nautilus bridge, simulated venue/instrument conversion, examples and a mean-reversion implementation |

Private tuned strategy configurations remain outside the public workspace, but the bridge and public examples do contain strategy code. Alpaca/live brokerage integration is not implemented here.

## Build and offline checks

```bash
cargo test --workspace --all-targets --all-features --locked
cargo run -p lbr-cli -- --help
```

Initial dependency builds need registry access and a C toolchain. The public test suite has four tests: two event/return checks and two cache-path regressions. [CI](CI.md) gives lint/test commands. No market request is made by --help or these synthetic tests.

## Data CLI

`ingest` contacts Yahoo and writes per-symbol Parquet files; it is not an offline smoke test. `show` reads existing cache files. Both accept --symbols (comma-separated) and --cache (default data/lake). Ingest also requires --start (inclusive) and --end (exclusive), ISO dates. Use `cargo run -p lbr-cli -- ingest --help` before choosing an authorized acquisition scope.

The cache is keyed by lowercased symbol, not content/date range. New ingest overwrites that symbol's cache. A cache is neither an immutable delivery nor provider-origin proof. Preserve a separate dated manifest/hash if a study needs reproducible inputs.

## Strategy API

Implement `Strategy::name` and mutable `on_bar(&Slice, &mut Context)`; `on_fill` is optional. Use Context.set_holdings/liquidate, EventEngine::new, explicit FillMode, Arc<Panel> and boxed Strategy with run. The complete synthetic example is [engine_smoke.rs](crates/lbr/tests/engine_smoke.rs); it compiles in CI. Old `VectorEngine` and `target_weights` interfaces are absent.

[Design and data contracts](docs/design.md) describes timing, cost units and limitations. [Documentation audit](docs/DOCUMENTATION_AUDIT.md) records reviewed coverage and known gaps. No exact QC parity, native margin model, production readiness or economic alpha is claimed. License: MIT.
