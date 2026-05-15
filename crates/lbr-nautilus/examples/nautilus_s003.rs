//! S003-style mean-reversion across a small ETF universe running through
//! NautilusTrader's BacktestEngine. Demonstrates: bridge → engine → multi-
//! instrument strategy with per-instrument tuning + market orders.

use anyhow::Result;
use std::path::PathBuf;

use lbr::data;
use lbr_nautilus::{
    InstrumentTuning, NautilusMeanRev, build_equity, make_engine, to_nautilus_bars,
};
use nautilus_model::{instruments::Instrument, types::Quantity};

fn main() -> Result<()> {
    let cache = PathBuf::from("/home/txy/lbr/data/lake");
    let universe = [
        "SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "XLF", "XLK", "XLV", "XLI",
    ];

    let mut engine = make_engine(70_000.0)?;
    let mut tunings = Vec::new();
    let mut total_bars = 0usize;

    for sym in &universe {
        let bars = data::load_cached(&cache, sym)?;
        if bars.is_empty() {
            eprintln!("warn: no bars for {sym}");
            continue;
        }
        let instrument = build_equity(sym);
        let instrument_id = instrument.id();
        engine.add_instrument(&instrument)?;
        let nbars = to_nautilus_bars(instrument_id, &bars, 4, 0);
        total_bars += nbars.len();
        engine.add_data(nbars, None, true, true)?;
        tunings.push(InstrumentTuning::default_for(
            instrument_id,
            Quantity::from("10"),
        ));
    }
    println!(
        "Universe: {} instruments, {} total Nautilus bars",
        tunings.len(),
        total_bars
    );

    let strategy = NautilusMeanRev::new(tunings);
    engine.add_strategy(strategy)?;

    println!("Running NautilusTrader BacktestEngine ...");
    let start = std::time::Instant::now();
    engine.run(None, None, None, false)?;
    let dt = start.elapsed();
    println!("Backtest finished in {:.3} s", dt.as_secs_f64());

    let result = engine.get_result();
    println!("\n--- Nautilus BacktestResult ---");
    println!("iterations: {}", result.iterations);
    println!("total_events: {}", result.total_events);
    println!("total_orders: {}", result.total_orders);
    println!("total_positions: {}", result.total_positions);
    println!("stats_pnls: {:#?}", result.stats_pnls);
    println!("stats_returns: {:#?}", result.stats_returns);

    Ok(())
}
