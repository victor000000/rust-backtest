# rust-backtest

A local, vectorized, parallel backtesting platform written in Rust.

## Goals

- **Vectorized research** — fast parameter sweeps across historical panels.
- **Event-driven engine** — same code path for backtest, paper, and live trading.
- **Multi-source data** — Alpaca for trading-active data + live execution; Yahoo Finance for deep historical coverage.
- **Local-first** — Parquet lake on disk, no cloud dependency for research.

## Status

Early design phase. No code yet.

## Stack (planned)

- Rust (stable)
- Polars + ndarray for data & numeric kernels
- Rayon for parallel sweeps
- Tokio for venue I/O
- Parquet for storage
- Alpaca + Yahoo Finance adapters

## License

TBD.
