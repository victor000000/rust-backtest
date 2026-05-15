//! Demo: a small mean-reversion backtest on a handful of liquid ETFs.
//!
//! Run with:
//!   cargo run --example mean_rev_demo --release
//!
//! Caches data under `data/lake/`. First run hits Yahoo; subsequent runs are
//! offline.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use time::macros::date;

use lbr::config::CostConfig;
use lbr::data::{self, Panel};
use lbr::engine::{VectorEngine, run};
use lbr::metrics::pretty;
use lbr_strategies::{MeanRevCfg, MeanRevStrategy};

#[tokio::main]
async fn main() -> Result<()> {
    let cache = PathBuf::from("data/lake");
    let symbols = vec![
        "SPY".to_string(),
        "QQQ".to_string(),
        "IWM".to_string(),
        "GLD".to_string(),
        "TLT".to_string(),
    ];

    // Fetch from Yahoo if not cached.
    for sym in &symbols {
        let cached = data::load_cached(&cache, sym)?;
        if cached.is_empty() {
            println!("Fetching {sym} from Yahoo …");
            let bars = data::fetch_bars(sym, date!(2009 - 08 - 01), date!(2026 - 05 - 13)).await?;
            data::save_cached(&cache, sym, &bars)?;
        }
    }

    let mut series = Vec::with_capacity(symbols.len());
    for sym in &symbols {
        let bars = data::load_cached(&cache, sym)?;
        series.push((sym.clone(), bars));
    }
    let panel = Arc::new(Panel::from_series(series));
    println!("Panel: T={} bars, N={} symbols", panel.t(), panel.n());

    let engine = VectorEngine::new(
        70_000.0,
        CostConfig {
            commission_bps: 1.0,
            slippage_bps: 2.0,
        },
    );
    let cfg = MeanRevCfg::uniform("MeanRev_demo", &["SPY", "QQQ", "IWM", "GLD", "TLT"]);
    let strategy = MeanRevStrategy::new(cfg);
    let result = run(&engine, panel.as_ref(), &strategy);

    println!("\nStrategy: {}", result.name);
    println!("{}", pretty(&result.metrics));

    Ok(())
}
