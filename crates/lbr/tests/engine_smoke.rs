//! End-to-end smoke test of the event engine using a synthetic panel and a
//! "buy and hold" strategy. Asserts the equity curve rises monotonically when
//! the underlying climbs.

use ndarray::array;
use std::sync::Arc;

use lbr::config::CostConfig;
use lbr::data::{Bar, Panel};
use lbr::engine::{Context, EventEngine, FillMode, Slice, Strategy, run};

struct BuyAndHold {
    bought: bool,
}

impl Strategy for BuyAndHold {
    fn name(&self) -> &str {
        "BuyAndHold"
    }
    fn on_bar(&mut self, _slice: &Slice<'_>, ctx: &mut Context<'_>) {
        if !self.bought {
            ctx.set_holdings(0, 1.0);
            self.bought = true;
        }
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
    let engine =
        EventEngine::new(10_000.0, CostConfig::default()).with_fill_mode(FillMode::SameClose);
    let r = run(
        &engine,
        panel.clone(),
        Box::new(BuyAndHold { bought: false }),
    );

    // Buy at close of bar 0 (price 100), hold to bar 4 (price 104). Total
    // return = 104/100 - 1 = 4%.
    let total = r.metrics.total_return;
    assert!(
        (0.035..0.045).contains(&total),
        "total_return={total} not in [0.035, 0.045]"
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
