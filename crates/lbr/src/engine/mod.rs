//! Vector backtest engine + parallel runner.

mod vector;

pub use vector::{BacktestResult, VectorEngine, run, run_parallel};
