//! `lbr` — Local Backtest in Rust.
//!
//! Vectorized parallel backtesting platform. The library provides:
//! data loading, panels, indicators, the vector engine, metrics, and the
//! `Strategy` trait. Concrete strategies live in downstream crates.
//!
//! See `docs/design.md` for the architecture.

pub mod config;
pub mod data;
pub mod engine;
pub mod indicators;
pub mod metrics;
pub mod strategy;

pub use data::{Bar, Panel};
pub use engine::{BacktestResult, VectorEngine};
pub use strategy::Strategy;
