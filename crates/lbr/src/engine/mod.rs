//! Backtest engines: vector mode + event-driven mode.
//!
//! Both engines consume the same `Panel` data and the same `Strategy`/`EventStrategy`
//! abstractions live in their respective traits. Vector mode is fast and parallel
//! (run many strategies via Rayon); event mode mirrors S002/Lean/Nautilus semantics
//! with per-bar order generation, next-open fills, and an explicit portfolio.

mod event;
mod vector;

pub use event::{
    Bar as EventBar, Context as EventContext, EventBacktestResult, EventEngine, EventStrategy,
    Fill, Order, OrderKind, OrderState, Portfolio, Position, Slice, run_event,
};
pub use vector::{BacktestResult, VectorEngine, run, run_parallel};
