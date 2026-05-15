//! End-to-end smoke test of the vector engine using a synthetic panel and a
//! "buy and hold" strategy. Asserts that the equity curve is positive and the
//! engine accounts for lag-1 weights correctly.

use ndarray::{Array2, array};
use std::sync::Arc;

use lbr::config::CostConfig;
use lbr::data::{Bar, Panel};
use lbr::engine::{VectorEngine, run};
use lbr::strategy::Strategy;

struct BuyAndHold;

impl Strategy for BuyAndHold {
    fn name(&self) -> &str {
        "BuyAndHold"
    }
    fn target_weights(&self, panel: &lbr::Panel) -> Array2<f64> {
        Array2::from_elem((panel.t(), panel.n()), 1.0)
    }
}

fn make_bars(prices: &[f64]) -> Vec<Bar> {
    prices
        .iter()
        .enumerate()
        .map(|(i, &p)| Bar {
            date: 19000 + i as i32,
            open: p,
            high: p,
            low: p,
            close: p,
            adj_close: p,
            volume: 1_000.0,
        })
        .collect()
}

#[test]
fn buy_and_hold_recovers_underlying_return() {
    let series = vec![(
        "AAA".to_string(),
        make_bars(&[100.0, 101.0, 102.0, 103.0, 104.0]),
    )];
    let panel = Arc::new(Panel::from_series(series));
    let engine = VectorEngine::new(10_000.0, CostConfig::default());
    let r = run(&engine, panel.as_ref(), &BuyAndHold);

    // Engine applies lag-1 weights, so buy-and-hold on 5 days captures only 3
    // of 4 daily returns. Compound: 1.0099 × 1.0098 × 1.0097 ≈ 1.0297.
    let total = r.metrics.total_return;
    assert!(
        (0.025..0.035).contains(&total),
        "total_return={total} not in [0.025, 0.035]"
    );

    // Equity curve must be monotonically non-decreasing in this all-up scenario.
    for i in 1..r.equity.len() {
        assert!(r.equity[i] >= r.equity[i - 1] - 1e-9);
    }
}

#[test]
fn returns_match_simple_ratio() {
    let ret = lbr::indicators::returns_axis0(&array![[100.0], [110.0], [99.0]]);
    assert!((ret[(1, 0)] - 0.10).abs() < 1e-9);
    assert!((ret[(2, 0)] - (-11.0 / 110.0)).abs() < 1e-9);
    assert!(ret[(0, 0)].is_nan());
}
