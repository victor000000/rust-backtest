//! NautilusMeanRev — S002/S003-style per-instrument mean-reversion strategy
//! implemented as a real Nautilus `Strategy`.
//!
//! Each instrument has its own config (WWMA/RMA period, entry/exit factors,
//! SL, TP, trade size). On every received Bar:
//!
//! 1. Update the per-instrument Wilder MA (RMA).
//! 2. If in position: check SL, TP, indicator exit (price > MA × xf). Liquidate
//!    via market order on signal.
//! 3. If flat: if `price < MA × ef`, open a long market order of `trade_size`.
//!
//! No cross-instrument budget coordination (each instrument decides
//! independently). Combined exposure naturally caps when fewer signals fire.

use std::collections::HashMap;
use std::fmt::Debug;

use nautilus_common::actor::DataActor;
use nautilus_indicators::{
    average::rma::WilderMovingAverage,
    indicator::{Indicator, MovingAverage},
};
use nautilus_model::{
    data::Bar as NBar,
    enums::{OrderSide, PriceType, TimeInForce},
    identifiers::{InstrumentId, StrategyId},
    types::Quantity,
};

use nautilus_trading::{
    nautilus_strategy,
    strategy::{Strategy, StrategyConfig, StrategyCore},
};

use crate::bridge::daily_bar_type;

/// Per-instrument tuning.
#[derive(Debug, Clone)]
pub struct InstrumentTuning {
    pub instrument_id: InstrumentId,
    pub trade_size: Quantity,
    pub period: usize,
    /// Entry factor: enter when price < MA × ef. Default 0.95.
    pub ef: f64,
    /// Exit factor: exit when price > MA × xf. Default 1.05.
    pub xf: f64,
    /// Stop-loss (fraction of entry price).
    pub sl: f64,
    /// Take-profit (fraction of entry price).
    pub tp: f64,
}

impl InstrumentTuning {
    pub fn default_for(instrument_id: InstrumentId, trade_size: Quantity) -> Self {
        Self {
            instrument_id,
            trade_size,
            period: 14,
            ef: 0.95,
            xf: 1.05,
            sl: 0.04,
            tp: 0.11,
        }
    }
}

#[derive(Default, Clone, Copy)]
struct PosState {
    invested: bool,
    entry_px: f64,
}

pub struct NautilusMeanRev {
    pub(crate) core: StrategyCore,
    cfgs: HashMap<InstrumentId, InstrumentTuning>,
    indicators: HashMap<InstrumentId, WilderMovingAverage>,
    positions: HashMap<InstrumentId, PosState>,
}

impl NautilusMeanRev {
    pub fn new(tunings: Vec<InstrumentTuning>) -> Self {
        let cfg = StrategyConfig {
            strategy_id: Some(StrategyId::from("NAUTILUS_MEAN_REV-001")),
            order_id_tag: Some("001".into()),
            ..Default::default()
        };
        let mut indicators = HashMap::new();
        let mut positions = HashMap::new();
        let mut cfgs = HashMap::new();
        for t in tunings {
            indicators.insert(
                t.instrument_id,
                WilderMovingAverage::new(t.period, Some(PriceType::Last)),
            );
            positions.insert(t.instrument_id, PosState::default());
            cfgs.insert(t.instrument_id, t);
        }
        Self {
            core: StrategyCore::new(cfg),
            cfgs,
            indicators,
            positions,
        }
    }

    fn submit_market(
        &mut self,
        instrument_id: InstrumentId,
        side: OrderSide,
        qty: Quantity,
    ) -> anyhow::Result<()> {
        let order = self.core.order_factory().market(
            instrument_id,
            side,
            qty,
            Some(TimeInForce::Gtc),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        self.submit_order(order, None, None)
    }
}

nautilus_strategy!(NautilusMeanRev);

impl Debug for NautilusMeanRev {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NautilusMeanRev")
            .field("instruments", &self.cfgs.len())
            .finish()
    }
}

impl DataActor for NautilusMeanRev {
    fn on_start(&mut self) -> anyhow::Result<()> {
        let ids: Vec<InstrumentId> = self.cfgs.keys().copied().collect();
        for id in ids {
            let bar_type = daily_bar_type(id);
            self.subscribe_bars(bar_type, None, None);
        }
        Ok(())
    }

    fn on_bar(&mut self, bar: &NBar) -> anyhow::Result<()> {
        let id = bar.instrument_id();
        let cfg = match self.cfgs.get(&id) {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        let ind = match self.indicators.get_mut(&id) {
            Some(i) => i,
            None => return Ok(()),
        };
        ind.handle_bar(bar);
        if !ind.initialized() {
            return Ok(());
        }
        let ma = ind.value();
        let price: f64 = (&bar.close).into();

        let mut pos = self.positions.get(&id).copied().unwrap_or_default();

        if pos.invested {
            let pnl = if pos.entry_px > 0.0 {
                price / pos.entry_px - 1.0
            } else {
                0.0
            };
            let mut exit = false;
            if cfg.sl > 0.0 && pnl <= -cfg.sl {
                exit = true;
            }
            if !exit && cfg.tp > 0.0 && pnl >= cfg.tp {
                exit = true;
            }
            if !exit && price > ma * cfg.xf {
                exit = true;
            }
            if exit {
                self.submit_market(id, OrderSide::Sell, cfg.trade_size)?;
                pos = PosState::default();
            }
        } else if price < ma * cfg.ef {
            self.submit_market(id, OrderSide::Buy, cfg.trade_size)?;
            pos = PosState {
                invested: true,
                entry_px: price,
            };
        }

        self.positions.insert(id, pos);
        Ok(())
    }
}
