//! Event-driven backtest engine.
//!
//! A single engine consumes `Strategy::on_bar(slice, ctx)` and `on_fill(fill)`,
//! produces an equity curve + full order/fill history. Per-bar order generation,
//! configurable fill mode (next-open, next-close, same-close), proper Portfolio
//! with cash + shares + mark-to-market.
//!
//! Parallelism comes from running many engines concurrently (one per parameter
//! set / fold) via Rayon — see `run_parallel`.

mod event;

pub use event::{
    Bar as EventBar, Context, EventBacktestResult, EventEngine, Fill, FillMode, Order, OrderKind,
    OrderState, Portfolio, Position, Slice, Strategy, run, run_parallel,
};
