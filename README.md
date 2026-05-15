# lbr — Local Backtest in Rust

A local, vectorized, parallel backtesting platform. **Platform only — no strategy code shipped.**

## Layout

Public workspace (this repository):

| Crate | Purpose |
|---|---|
| `lbr` | The platform. Data, panels, indicators, vector engine, metrics, `Strategy` trait. Knows nothing about specific strategies. |
| `lbr-cli` | The `lbr` command-line driver. Strategy-agnostic — `ingest` data, `show` panels. |

Strategy implementations are developed **outside** this repo (in a private
workspace or your own repo) as crates that depend on `lbr`. The platform
intentionally has zero knowledge of any strategy.

## Goals

- **Vectorized research** — fast parameter sweeps via Rayon over independent jobs.
- **Multi-source data** — Yahoo Finance for v0.1; Alpaca planned for live.
- **Local-first** — CSV cache on disk, no cloud dependency.
- **Hard strategy ↔ platform decoupling** — concrete strategies never live in this repo.

## Quickstart

```bash
# Ingest a few ETFs from Yahoo Finance.
cargo run -p lbr-cli --release -- ingest \
    --symbols SPY,QQQ,IWM,GLD,TLT \
    --start 2009-08-01 --end 2026-05-13

# Inspect a cached panel.
cargo run -p lbr-cli --release -- show \
    --symbols SPY,QQQ,IWM,GLD,TLT
```

## Plugging in a strategy

Create a separate crate (private or your own) with:

```toml
[dependencies]
lbr = { path = "<path-to-this-repo>/crates/lbr" }
ndarray = "0.16"
```

Then implement the `Strategy` trait:

```rust
use lbr::{Strategy, Panel};
use ndarray::Array2;

pub struct MyStrategy { /* ... */ }

impl Strategy for MyStrategy {
    fn name(&self) -> &str { "MyStrategy" }
    fn target_weights(&self, panel: &Panel) -> Array2<f64> {
        // produce (T, N) target weights
        Array2::from_elem((panel.t(), panel.n()), 0.0)
    }
}
```

Run it through the engine:

```rust
use lbr::{VectorEngine, engine::run};
use lbr::config::CostConfig;

let engine = VectorEngine::new(70_000.0, CostConfig::default());
let result = run(&engine, &panel, &MyStrategy { /* ... */ });
println!("{}", lbr::metrics::pretty(&result.metrics));
```

## Stack

- Rust 1.95 stable (edition 2024)
- `ndarray` for panels & numeric kernels
- `rayon` for parallel sweeps
- `tokio` for venue I/O
- `yahoo_finance_api` for daily bars
- `clap` for the CLI

## Design

See [`docs/design.md`](docs/design.md) for the architecture, data flow, and module map.

## License

MIT.
