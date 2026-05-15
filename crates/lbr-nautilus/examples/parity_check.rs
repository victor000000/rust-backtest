//! Parity check: same WWMA mean-reversion strategy on SAME SPY data through
//! both engines (lbr::engine and NautilusTrader's BacktestEngine).
//!
//! Strategy params:
//! - WWMA period 14
//! - Enter when price < WWMA × 0.95
//! - Exit when price > WWMA × 1.05 OR pnl <= -4% OR pnl >= 11%
//! - Trade size: 10 shares
//!
//! Reports total trade count + win rate from each engine; should match
//! closely (up to first ~14 bars where WWMA seeds differ: ours uses SMA(14)
//! seed, Nautilus's WilderMovingAverage uses first-price seed).

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

use lbr::config::CostConfig;
use lbr::data::{self, Panel};
use lbr::engine::{Context, EventEngine, FillMode, Slice, Strategy, run};
use lbr_nautilus::{
    InstrumentTuning, NautilusMeanRev, build_equity, make_engine, to_nautilus_bars,
};
use nautilus_model::{instruments::Instrument, types::Quantity};

// ── Our engine: mirror NautilusMeanRev's logic exactly ──

struct SimpleMeanRev {
    period: usize,
    ef: f64,
    xf: f64,
    sl: f64,
    tp: f64,
    qty_shares: f64,
    // pre-computed WWMA series (Nautilus-style: seed with first price).
    wwma_nautilus_seed: ndarray::Array1<f64>,
    target_panel_col: usize,
    invested: bool,
    entry_px: f64,
    n_buys: usize,
    n_sells: usize,
}

impl SimpleMeanRev {
    fn new(period: usize, panel: &Panel) -> Self {
        let alpha = 1.0 / period as f64;
        let prices: Vec<f64> = (0..panel.t())
            .map(|i| {
                let v = panel.close[(i, 0)];
                if v.is_finite() { v } else { f64::NAN }
            })
            .collect();
        let mut ma = ndarray::Array1::from_elem(panel.t(), f64::NAN);
        let mut value = f64::NAN;
        let mut count: usize = 0;
        for i in 0..panel.t() {
            if !prices[i].is_finite() {
                continue;
            }
            if !value.is_finite() {
                value = prices[i];
                count = 1;
            } else {
                value = alpha * prices[i] + (1.0 - alpha) * value;
                count += 1;
            }
            if count >= period {
                ma[i] = value;
            }
        }
        Self {
            period,
            ef: 0.95,
            xf: 1.05,
            sl: 0.04,
            tp: 0.11,
            qty_shares: 10.0,
            wwma_nautilus_seed: ma,
            target_panel_col: 0,
            invested: false,
            entry_px: 0.0,
            n_buys: 0,
            n_sells: 0,
        }
    }
}

impl Strategy for SimpleMeanRev {
    fn name(&self) -> &str {
        "SimpleMeanRev"
    }
    fn on_bar(&mut self, slice: &Slice<'_>, ctx: &mut Context<'_>) {
        let i = slice.bar_idx;
        let bar = &slice.bars[self.target_panel_col];
        let p = bar.close;
        if !p.is_finite() {
            return;
        }
        let ma = self.wwma_nautilus_seed[i];
        if !ma.is_finite() {
            return;
        }
        // Match Nautilus's per-share sizing: target_weight = qty_shares × price / equity.
        let target_wt = self.qty_shares * p / ctx.portfolio.equity.max(1.0);
        if self.invested {
            let pnl = if self.entry_px > 0.0 {
                p / self.entry_px - 1.0
            } else {
                0.0
            };
            let mut exit = false;
            if pnl <= -self.sl {
                exit = true;
            }
            if !exit && pnl >= self.tp {
                exit = true;
            }
            if !exit && p > ma * self.xf {
                exit = true;
            }
            if exit {
                ctx.liquidate(self.target_panel_col);
                self.invested = false;
                self.entry_px = 0.0;
                self.n_sells += 1;
            }
        } else if p < ma * self.ef {
            ctx.set_holdings(self.target_panel_col, target_wt);
            self.invested = true;
            self.entry_px = p;
            self.n_buys += 1;
        }
        let _ = self.period;
    }
}

fn main() -> Result<()> {
    let cache = PathBuf::from("/home/txy/lbr/data/lake");
    let bars = data::load_cached(&cache, "SPY")?;
    println!("Loaded {} bars for SPY", bars.len());

    // ── lbr::engine run ──
    let panel = Arc::new(Panel::from_series(vec![("SPY".into(), bars.clone())]));
    let cost = CostConfig::default(); // zero cost for parity check
    let engine = EventEngine::new(70_000.0, cost).with_fill_mode(FillMode::SameClose);
    let mut strat = SimpleMeanRev::new(14, panel.as_ref());
    let target_n_buys = std::ptr::addr_of_mut!(strat.n_buys);
    let target_n_sells = std::ptr::addr_of_mut!(strat.n_sells);

    let start = std::time::Instant::now();
    let r = run(&engine, panel.clone(), Box::new(strat));
    let dt_lbr = start.elapsed();
    // SAFETY: strat moved into run(); these are stale. Re-extract from result counts.
    let _ = (target_n_buys, target_n_sells);

    println!("\n--- lbr::engine ---");
    println!("time: {:.3} s", dt_lbr.as_secs_f64());
    println!("equity: {:.0}", r.equity[r.equity.len() - 1]);
    println!("Sharpe: {:.3}", r.metrics.sharpe);
    println!("MaxDD: {:.2}%", r.metrics.max_drawdown * 100.0);
    println!("orders: {}, fills: {}", r.n_orders, r.n_fills);

    // ── Nautilus run ──
    let mut nautilus_engine = make_engine(70_000.0)?;
    let instrument = build_equity("SPY");
    let instrument_id = instrument.id();
    nautilus_engine.add_instrument(&instrument)?;
    let nbars = to_nautilus_bars(instrument_id, &bars, 4, 0);
    nautilus_engine.add_data(nbars, None, true, true)?;

    let tuning = InstrumentTuning {
        instrument_id,
        trade_size: Quantity::from("10"),
        period: 14,
        ef: 0.95,
        xf: 1.05,
        sl: 0.04,
        tp: 0.11,
    };
    let strategy = NautilusMeanRev::new(vec![tuning]);
    nautilus_engine.add_strategy(strategy)?;

    let start = std::time::Instant::now();
    nautilus_engine.run(None, None, None, false)?;
    let dt_nautilus = start.elapsed();
    let result = nautilus_engine.get_result();

    println!("\n--- NautilusTrader ---");
    println!("time: {:.3} s", dt_nautilus.as_secs_f64());
    println!("iterations: {}", result.iterations);
    println!("total_orders: {}", result.total_orders);
    println!("total_events: {}", result.total_events);
    if let Some(usd) = result.stats_pnls.get("USD") {
        if let Some(wr) = usd.get("Win Rate") {
            println!("Win Rate: {:.4}", wr);
        }
        if let Some(exp) = usd.get("Expectancy") {
            println!("Expectancy/trade: ${:.2}", exp);
        }
    }

    Ok(())
}
