//! Vector mode backtest engine.
//!
//! Given target weights W (T, N) and prices, simulate the equity curve:
//!   r_t       = price_t / price_{t-1} - 1
//!   pnl_t     = W_{t-1, :} · r_{t, :}                    ← dot product
//!   turnover  = || W_{t, :} - W_{t-1, :} ||_1
//!   cost_t    = turnover · (commission_bps + slippage_bps) / 10_000
//!   equity_t  = equity_{t-1} · (1 + pnl_t - cost_t)
//!
//! The hot path is a row-wise dot product + L1 norm via ndarray — no per-symbol
//! scalar loop, so auto-vectorization (SSE/AVX) kicks in on release builds.

use anyhow::Result;
use ndarray::{Array1, Array2};
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

/// Replace NaN values with 0.0 so they're neutral in dot products.
fn nan_to_zero(mut a: Array2<f64>) -> Array2<f64> {
    a.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    a
}

pub fn run(engine: &VectorEngine, panel: &Panel, strategy: &dyn Strategy) -> BacktestResult {
    let mut weights = strategy.target_weights(panel);
    weights.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });

    let prices = if panel.adj_close.iter().any(|v| v.is_finite()) {
        &panel.adj_close
    } else {
        &panel.close
    };
    let returns = nan_to_zero(indicators::returns_axis0(prices));

    let (t, n) = weights.dim();
    let mut equity = Array1::from_elem(t, engine.starting_cash);
    let mut prev_w = Array1::zeros(n);
    let cost_rate = (engine.cost.commission_bps + engine.cost.slippage_bps) / 10_000.0;

    for i in 1..t {
        // Vectorized PnL: yesterday's weights · today's returns.
        let r_row = returns.row(i);
        let pnl = prev_w.dot(&r_row);

        // Vectorized turnover: L1 norm of weight delta.
        let w_row = weights.row(i);
        let mut diff = &w_row - &prev_w;
        diff.mapv_inplace(f64::abs);
        let turnover = diff.sum();

        equity[i] = equity[i - 1] * (1.0 + pnl - turnover * cost_rate);
        prev_w.assign(&w_row);
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
