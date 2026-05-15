//! Event-driven backtest engine.
//!
//! Mirrors the Nautilus/Lean shape: a Clock, a DataFeed that emits bars
//! into a typed channel, a Strategy actor that consumes bars and produces
//! Orders, an ExecutionEngine that fills orders against the next bar, and a
//! Portfolio that tracks holdings and equity. The engine is intentionally
//! sequential — concurrency comes from running many event-engine instances
//! in parallel (one per parameter set / fold).
//!
//! For backtest:
//!   - DataFeed replays a `Panel` bar-by-bar.
//!   - Signals generated at bar `i`'s close fill at bar `i+1`'s open
//!     (S002 semantics).
//!   - Equity is marked at each bar's close.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array1;

use crate::config::CostConfig;
use crate::data::Panel;
use crate::metrics::{Metrics, compute};

// ─── Bar / events ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub date: i32,
    pub symbol_idx: usize,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub adj_close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Slice<'a> {
    pub date: i32,
    pub bar_idx: usize,
    /// One bar per symbol in panel order. NaN-padded for missing symbols.
    pub bars: &'a [Bar],
}

// ─── Orders ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    New,
    Submitted,
    Filled,
    Canceled,
    Rejected,
}

#[derive(Debug, Clone, Copy)]
pub enum OrderKind {
    /// Set holdings to `weight` fraction of equity (matches S002's set_holdings).
    SetHoldings { weight: f64 },
    /// Close the position.
    Liquidate,
}

#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub id: u64,
    pub symbol_idx: usize,
    pub kind: OrderKind,
    pub state: OrderState,
    /// Bar at which the order was placed (last close observed).
    pub placed_bar: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Fill {
    pub order_id: u64,
    pub symbol_idx: usize,
    pub fill_bar: usize,
    pub fill_price: f64,
    /// Target weight (fraction of equity).
    pub weight: f64,
    /// Resulting position size in shares (signed).
    pub shares: f64,
}

// ─── Portfolio / Account ──────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Position {
    pub shares: f64,
    pub avg_cost: f64,
}

pub struct Portfolio {
    pub cash: f64,
    pub equity: f64,
    pub positions: HashMap<usize, Position>,
    pub mark_prices: Vec<f64>,
}

impl Portfolio {
    pub fn new(starting_cash: f64, n_symbols: usize) -> Self {
        Self {
            cash: starting_cash,
            equity: starting_cash,
            positions: HashMap::new(),
            mark_prices: vec![f64::NAN; n_symbols],
        }
    }

    /// Apply a fill: adjust cash, position shares, and avg cost. Linear costs.
    pub fn apply_fill(&mut self, fill: &Fill, cost_rate: f64) {
        let pos = self.positions.entry(fill.symbol_idx).or_default();
        let trade_shares = fill.shares - pos.shares;
        let trade_notional = trade_shares.abs() * fill.fill_price;
        let cost = trade_notional * cost_rate;

        self.cash -= trade_shares * fill.fill_price + cost;
        if (pos.shares + trade_shares).abs() < 1e-12 {
            pos.shares = 0.0;
            pos.avg_cost = 0.0;
        } else if trade_shares.signum() == pos.shares.signum() || pos.shares == 0.0 {
            // Adding to position — update avg cost.
            let total_cost = pos.avg_cost * pos.shares + fill.fill_price * trade_shares;
            pos.shares += trade_shares;
            if pos.shares != 0.0 {
                pos.avg_cost = total_cost / pos.shares;
            }
        } else {
            // Reducing/closing — realized PnL goes through cash already.
            pos.shares += trade_shares;
        }
    }

    /// Mark-to-market equity at end of bar.
    pub fn mark_to_market(&mut self) {
        let mut holdings_value = 0.0;
        for (&sym_idx, pos) in &self.positions {
            let price = self.mark_prices[sym_idx];
            if price.is_finite() && pos.shares.abs() > 1e-12 {
                holdings_value += pos.shares * price;
            }
        }
        self.equity = self.cash + holdings_value;
    }
}

// ─── Strategy trait (event-driven) ────────────────────────────────────────

/// Context exposed to event strategies. Provides read-only access to
/// portfolio and a place to enqueue orders.
pub struct Context<'a> {
    pub portfolio: &'a Portfolio,
    pub bar_idx: usize,
    pub n_symbols: usize,
    /// Pending orders accumulated this bar.
    pub pending: &'a mut Vec<Order>,
    /// Internal counter for order IDs.
    pub next_order_id: &'a mut u64,
}

impl Context<'_> {
    pub fn set_holdings(&mut self, symbol_idx: usize, weight: f64) {
        let id = *self.next_order_id;
        *self.next_order_id += 1;
        self.pending.push(Order {
            id,
            symbol_idx,
            kind: OrderKind::SetHoldings { weight },
            state: OrderState::New,
            placed_bar: self.bar_idx,
        });
    }
    pub fn liquidate(&mut self, symbol_idx: usize) {
        let id = *self.next_order_id;
        *self.next_order_id += 1;
        self.pending.push(Order {
            id,
            symbol_idx,
            kind: OrderKind::Liquidate,
            state: OrderState::New,
            placed_bar: self.bar_idx,
        });
    }
}

/// Event-driven strategy. Called once per bar with a Slice of all market data.
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn on_bar(&mut self, slice: &Slice<'_>, ctx: &mut Context<'_>);
    fn on_fill(&mut self, _fill: &Fill) {}
}

// ─── Event engine ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EventBacktestResult {
    pub name: String,
    pub equity: Array1<f64>,
    pub metrics: Metrics,
    pub starting_cash: f64,
    pub n_bars: usize,
    pub n_symbols: usize,
    pub n_fills: u64,
    pub n_orders: u64,
}

/// Order-fill timing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    /// Orders queued at bar i fill at open of bar i+1 (S002 / Lean default).
    NextOpen,
    /// Orders queued at bar i fill at close of bar i+1 (1-bar realistic lag).
    NextClose,
    /// Orders queued at bar i fill IMMEDIATELY at close of bar i (no lag).
    /// Equivalent to vector-mode lag-1 timing; useful as a parity check.
    SameClose,
}

pub struct EventEngine {
    pub starting_cash: f64,
    pub cost: CostConfig,
    pub fill_mode: FillMode,
    /// Deprecated: keep for backwards-compat. If `true`, sets `fill_mode = NextOpen`.
    pub fill_at_next_open: bool,
}

impl EventEngine {
    pub fn new(starting_cash: f64, cost: CostConfig) -> Self {
        Self {
            starting_cash,
            cost,
            fill_mode: FillMode::NextOpen,
            fill_at_next_open: true,
        }
    }
    pub fn with_fill_mode(mut self, mode: FillMode) -> Self {
        self.fill_mode = mode;
        self.fill_at_next_open = matches!(mode, FillMode::NextOpen);
        self
    }
}

pub fn run(
    engine: &EventEngine,
    panel: Arc<Panel>,
    mut strategy: Box<dyn Strategy>,
) -> EventBacktestResult {
    let panel = panel.as_ref();
    let t = panel.t();
    let n = panel.n();
    let cost_rate = (engine.cost.commission_bps + engine.cost.slippage_bps) / 10_000.0;
    let mut portfolio = Portfolio::new(engine.starting_cash, n);
    let mut equity_curve = Array1::from_elem(t, engine.starting_cash);

    let mut pending: Vec<Order> = Vec::new();
    let mut next_order_id: u64 = 1;
    let mut n_fills = 0u64;
    let mut n_orders = 0u64;

    // Scratch buffer for current bar's Slice.
    let mut current_bars = vec![
        Bar {
            date: 0,
            symbol_idx: 0,
            open: f64::NAN,
            high: f64::NAN,
            low: f64::NAN,
            close: f64::NAN,
            adj_close: f64::NAN,
            volume: f64::NAN,
        };
        n
    ];

    // Helper: compute fill price given the current bar i and a fill mode.
    // Returns None if the price is unavailable for this bar/symbol.
    let fill_price_at = |i_fill: usize, j: usize, mode: FillMode| -> Option<f64> {
        let raw_open = panel.open[(i_fill, j)];
        let raw_close = panel.close[(i_fill, j)];
        let adj = panel.adj_close[(i_fill, j)];
        match mode {
            FillMode::NextOpen => {
                if raw_open.is_finite()
                    && raw_close.is_finite()
                    && raw_close > 0.0
                    && adj.is_finite()
                {
                    Some(raw_open * adj / raw_close)
                } else if raw_open.is_finite() {
                    Some(raw_open)
                } else if adj.is_finite() {
                    Some(adj)
                } else {
                    None
                }
            }
            FillMode::NextClose | FillMode::SameClose => {
                if adj.is_finite() {
                    Some(adj)
                } else {
                    None
                }
            }
        }
    };

    let fill_orders = |i_fill: usize,
                       portfolio: &mut Portfolio,
                       pending: &mut Vec<Order>,
                       n_fills: &mut u64,
                       strategy: &mut Box<dyn Strategy>| {
        if pending.is_empty() {
            return;
        }
        let mut filled: Vec<Fill> = Vec::with_capacity(pending.len());
        for order in pending.drain(..) {
            let j = order.symbol_idx;
            let Some(fill_price) = fill_price_at(i_fill, j, engine.fill_mode) else {
                continue;
            };
            if !fill_price.is_finite() || fill_price <= 0.0 {
                continue;
            }
            let target_shares = match order.kind {
                OrderKind::SetHoldings { weight } => portfolio.equity * weight / fill_price,
                OrderKind::Liquidate => 0.0,
            };
            let fill = Fill {
                order_id: order.id,
                symbol_idx: j,
                fill_bar: i_fill,
                fill_price,
                weight: match order.kind {
                    OrderKind::SetHoldings { weight } => weight,
                    OrderKind::Liquidate => 0.0,
                },
                shares: target_shares,
            };
            portfolio.apply_fill(&fill, cost_rate);
            filled.push(fill);
            *n_fills += 1;
        }
        for f in &filled {
            strategy.on_fill(f);
        }
    };

    for i in 0..t {
        // 1. Build the slice (read-only view of bar i).
        for (j, bar) in current_bars.iter_mut().enumerate().take(n) {
            *bar = Bar {
                date: panel.dates[i],
                symbol_idx: j,
                open: panel.open[(i, j)],
                high: panel.high[(i, j)],
                low: panel.low[(i, j)],
                close: panel.close[(i, j)],
                adj_close: panel.adj_close[(i, j)],
                volume: panel.volume[(i, j)],
            };
            let mark = if panel.adj_close[(i, j)].is_finite() {
                panel.adj_close[(i, j)]
            } else {
                panel.close[(i, j)]
            };
            portfolio.mark_prices[j] = mark;
        }

        // 2. Execute any pending orders queued at bar i-1 against bar i.
        //    (NextOpen/NextClose semantics — same-close fills are handled later.)
        if matches!(engine.fill_mode, FillMode::NextOpen | FillMode::NextClose) {
            fill_orders(i, &mut portfolio, &mut pending, &mut n_fills, &mut strategy);
        }

        // 3. Strategy decides at this bar's close — emits pending orders.
        let slice = Slice {
            date: panel.dates[i],
            bar_idx: i,
            bars: &current_bars,
        };
        let mut ctx = Context {
            portfolio: &portfolio,
            bar_idx: i,
            n_symbols: n,
            pending: &mut pending,
            next_order_id: &mut next_order_id,
        };
        strategy.on_bar(&slice, &mut ctx);
        n_orders += pending.len() as u64;

        // 4. Same-close fills: execute IMMEDIATELY at this bar's close (parity
        //    with vector mode's lag-1 close-to-close timing).
        if engine.fill_mode == FillMode::SameClose {
            fill_orders(i, &mut portfolio, &mut pending, &mut n_fills, &mut strategy);
        }

        // 5. Mark to market at end of bar.
        portfolio.mark_to_market();
        equity_curve[i] = portfolio.equity;
    }

    let metrics = compute(&equity_curve);
    EventBacktestResult {
        name: strategy.name().to_string(),
        equity: equity_curve,
        metrics,
        starting_cash: engine.starting_cash,
        n_bars: t,
        n_symbols: n,
        n_fills,
        n_orders,
    }
}

/// Run many strategies in parallel against the same panel via Rayon.
/// Each strategy gets its own engine instance and runs independently.
pub fn run_parallel(
    engine_cfg: impl Fn() -> EventEngine + Send + Sync,
    panel: Arc<Panel>,
    strategies: Vec<Box<dyn Strategy>>,
) -> Vec<EventBacktestResult> {
    use rayon::prelude::*;
    strategies
        .into_par_iter()
        .map(|s| {
            let engine = engine_cfg();
            run(&engine, panel.clone(), s)
        })
        .collect()
}
