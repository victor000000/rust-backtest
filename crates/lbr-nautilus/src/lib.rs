//! `lbr-nautilus` — bridge between lbr's data layer and Nautilus Trader's
//! `BacktestEngine`.
//!
//! Provides:
//! - `to_nautilus_bars(...)` — convert a Vec<lbr::Bar> + symbol → Nautilus Bar `Data`.
//! - `build_equity(...)` — build a Nautilus `Equity` instrument for a US stock/ETF.
//! - `make_engine(...)` — set up a `BacktestEngine` with a simulated USD venue.
//!
//! Strategies are written in Nautilus's `Strategy` trait (via the
//! `nautilus_trading::strategy` module). See the `nautilus_ema_cross` example.

pub mod bridge;
pub mod mean_rev_strategy;

pub use bridge::{
    NAUTILUS_VENUE_ID, build_equity, daily_bar_type, make_engine, to_nautilus_bars,
    to_nautilus_quotes,
};
pub use mean_rev_strategy::{InstrumentTuning, NautilusMeanRev};
