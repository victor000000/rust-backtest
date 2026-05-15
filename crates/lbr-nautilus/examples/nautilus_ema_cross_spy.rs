//! End-to-end Nautilus backtest using OUR cached Parquet data.
//!
//! Loads SPY daily bars from `data/lake/`, converts them to Nautilus `Bar`s,
//! sets up a `BacktestEngine` with a simulated US equity venue, and runs
//! Nautilus's bundled `EmaCross` strategy. Proves the platform's data layer
//! plugs into Nautilus's production engine.

use anyhow::Result;
use std::path::PathBuf;

use lbr::data;
use lbr_nautilus::{build_equity, make_engine, to_nautilus_quotes};
use nautilus_model::{instruments::Instrument, types::Quantity};
use nautilus_trading::examples::strategies::EmaCross;

fn main() -> Result<()> {
    let cache = PathBuf::from("/home/txy/lbr/data/lake");
    let symbol = "SPY";

    let bars = data::load_cached(&cache, symbol)?;
    println!("Loaded {} bars for {symbol} from cache", bars.len());

    let mut engine = make_engine(70_000.0)?;
    let instrument = build_equity(symbol);
    let instrument_id = instrument.id();
    engine.add_instrument(&instrument)?;

    // EmaCross subscribes to quotes; convert our bars to synthetic close-quotes.
    let nautilus_quotes = to_nautilus_quotes(instrument_id, &bars, 4, 0);
    println!("Converted to {} Nautilus QuoteTicks", nautilus_quotes.len());
    engine.add_data(nautilus_quotes, None, true, true)?;

    // Nautilus's bundled EMA cross strategy.
    let strategy = EmaCross::new(instrument_id, Quantity::from("10"), 10, 30);
    engine.add_strategy(strategy)?;

    println!("Running Nautilus BacktestEngine ...");
    let start = std::time::Instant::now();
    engine.run(None, None, None, false)?;
    let dt = start.elapsed();
    println!("Backtest finished in {:.3} s", dt.as_secs_f64());

    let result = engine.get_result();
    println!("\n--- Nautilus BacktestResult ---");
    println!("{result:#?}");

    Ok(())
}
