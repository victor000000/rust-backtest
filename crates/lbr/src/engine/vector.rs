//! Vector mode backtest engine.
//!
//! Given target weights W (T, N) and prices, simulate the equity curve:
//!   r_t       = price_t / price_{t-1} - 1
//!   pnl_t     = sum_j W_{t-1, j} * r_{t, j}              ← lag-1 weight
//!   turnover  = sum_j | W_{t, j} - W_{t-1, j} |
//!   cost_t    = turnover * (commission_bps + slippage_bps)
//!   equity_t  = equity_{t-1} * (1 + pnl_t - cost_t)
//!
//! Operates entirely on ndarray — no per-bar loops in user code.

use anyhow::Result;
use ndarray::Array1;
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::CostConfig;
use crate::data::Panel;
use crate::indicators;
use crate::metrics::{Metrics, compute};
use crate::strategy::Strategy;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub name: String,
    pub equity: Array1<f64>,
    pub metrics: Metrics,
    pub starting_cash: f64,
    pub n_bars: usize,
    pub n_symbols: usize,
}

pub struct VectorEngine {
    pub starting_cash: f64,
    pub cost: CostConfig,
}

impl VectorEngine {
    pub fn new(starting_cash: f64, cost: CostConfig) -> Self {
        Self {
            starting_cash,
            cost,
        }
    }
}

pub fn run(engine: &VectorEngine, panel: &Panel, strategy: &dyn Strategy) -> BacktestResult {
    let weights = strategy.target_weights(panel);
    let prices = if panel.adj_close.iter().any(|v| v.is_finite()) {
        &panel.adj_close
    } else {
        &panel.close
    };
    let returns = indicators::returns_axis0(prices);
    let (t, n) = weights.dim();

    let mut equity = Array1::from_elem(t, engine.starting_cash);
    let mut prev_w = Array1::from_elem(n, 0.0);
    let cost_rate = (engine.cost.commission_bps + engine.cost.slippage_bps) / 10_000.0;

    for i in 1..t {
        // PnL is yesterday's weight × today's return
        let mut pnl = 0.0;
        for j in 0..n {
            let w = prev_w[j];
            if w == 0.0 {
                continue;
            }
            let r = returns[(i, j)];
            if r.is_finite() {
                pnl += w * r;
            }
        }
        // Apply today's rebalance and accrue cost
        let mut turnover = 0.0;
        for j in 0..n {
            let w = weights[(i, j)];
            let w = if w.is_finite() { w } else { 0.0 };
            turnover += (w - prev_w[j]).abs();
            prev_w[j] = w;
        }
        let cost = turnover * cost_rate;
        equity[i] = equity[i - 1] * (1.0 + pnl - cost);
    }

    let m = compute(&equity);
    BacktestResult {
        name: strategy.name().to_string(),
        equity,
        metrics: m,
        starting_cash: engine.starting_cash,
        n_bars: t,
        n_symbols: n,
    }
}

/// Run many strategies in parallel against the same panel via Rayon.
pub fn run_parallel(
    engine: &VectorEngine,
    panel: Arc<Panel>,
    strategies: Vec<Box<dyn Strategy>>,
) -> Result<Vec<BacktestResult>> {
    let panel_ref = panel.as_ref();
    let results: Vec<_> = strategies
        .into_par_iter()
        .map(|s| run(engine, panel_ref, s.as_ref()))
        .collect();
    Ok(results)
}
