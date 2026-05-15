//! `lbr` — Local Backtest in Rust.
//!
//! Event-driven backtesting platform. The library provides:
//! data loading, panels, indicators, the event engine, metrics, and the
//! `Strategy` trait. Concrete strategies live in downstream crates.
//!
//! See `docs/design.md` for the architecture.

pub mod config;
pub mod data;
pub mod engine;
pub mod indicators;
pub mod metrics;

pub use data::{Bar, Panel};
pub use engine::{EventBacktestResult, EventEngine, FillMode, Strategy, run, run_parallel};
