# lbr — Local Backtest in Rust

A local, vectorized, parallel backtesting platform.

## Layout

This is a Cargo workspace with three crates:

| Crate | Purpose |
|---|---|
| `lbr` | The platform. Data, panels, indicators, vector engine, metrics, `Strategy` trait. Knows nothing about specific strategies. |
| `lbr-strategies` | Reusable, *generic* strategy frameworks built on top of `lbr` (e.g. `MeanRevStrategy`). Tuned configs are loaded at runtime, not committed. |
| `lbr-cli` | The `lbr` command-line driver: ingest data + run backtests from YAML. |

## Goals

- **Vectorized research** — fast parameter sweeps via Rayon over independent jobs.
- **Multi-source data** — Yahoo Finance for v0.1; Alpaca planned for live.
- **Local-first** — CSV cache on disk, no cloud dependency.
- **Strategy ↔ platform decoupling** — strategy code is a downstream crate; the platform never imports a concrete strategy.

## Quickstart

```bash
# Ingest a few ETFs from Yahoo Finance.
cargo run -p lbr-cli --release -- ingest \
    --symbols SPY,QQQ,IWM,GLD,TLT \
    --start 2009-08-01 --end 2026-05-13

# Run a strategy from a YAML config.
cargo run -p lbr-cli --release -- run --cfg configs/example.yaml
```

Or run the bundled mean-reversion demo:

```bash
cargo run -p lbr-strategies --example mean_rev_demo --release
```

## Stack

- Rust 1.95 stable (edition 2024)
- `ndarray` for panels & numeric kernels
- `rayon` for parallel sweeps
- `tokio` for venue I/O
- `yahoo_finance_api` for daily bars
- `serde` + YAML for configs
- `clap` for the CLI

## Design

See [`docs/design.md`](docs/design.md) for the architecture, data flow, and module map.

## License

MIT.
