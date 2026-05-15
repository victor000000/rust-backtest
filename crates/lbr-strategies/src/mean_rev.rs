//! Per-symbol mean-reversion strategy.
//!
//! Generic, vectorized framework — concrete tuned configurations are loaded
//! externally (YAML/JSON) and are intentionally not committed to the library.
//!
//! Per-symbol config carries:
//! - Indicator family + period
//! - Target weight when invested
//! - Entry/exit thresholds (for oscillators) or factors (for MA-like)
//! - Stop-loss and take-profit (% of entry)
//! - A trend filter (SMA period + below/above factors)
//!
//! For "MA-like" indicators (`wwma`, `sma`, `ema`):
//!     enter when `price < val * ef`,  exit when `price > val * xf`.
//! For oscillators (`rsi`, `zscore`):
//!     enter when `val <= e`,          exit when `val >= x`.
//! SL/TP applied at the next-bar close based on running entry price.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use lbr::Strategy;
use lbr::data::Panel;
use lbr::indicators;

/// Indicator family.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Ind {
    Wwma,
    Sma,
    Ema,
    Rsi,
    Zscore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCfg {
    pub symbol: String,
    pub ind: Ind,
    pub p: usize,
    pub wt: f64,
    #[serde(default)]
    pub e: f64,
    #[serde(default)]
    pub x: f64,
    #[serde(default = "default_ef")]
    pub ef: f64,
    #[serde(default = "default_xf")]
    pub xf: f64,
    #[serde(default)]
    pub sl: f64,
    #[serde(default)]
    pub tp: f64,
    #[serde(default = "default_sma_p")]
    pub sma_p: usize,
    #[serde(default = "default_cf")]
    pub cf: f64,
    #[serde(default = "default_tf_t")]
    pub tf_t: f64,
}

fn default_ef() -> f64 {
    0.95
}
fn default_xf() -> f64 {
    1.05
}
fn default_sma_p() -> usize {
    20
}
fn default_cf() -> f64 {
    0.85
}
fn default_tf_t() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanRevCfg {
    pub name: String,
    #[serde(default = "default_gross_cap")]
    pub gross_cap: f64,
    pub symbols: Vec<SymbolCfg>,
}

fn default_gross_cap() -> f64 {
    1.0
}

impl MeanRevCfg {
    /// Convenience: build a uniform default for a list of symbols.
    pub fn uniform(name: &str, symbols: &[&str]) -> Self {
        let cfgs = symbols
            .iter()
            .map(|s| SymbolCfg {
                symbol: s.to_string(),
                ind: Ind::Wwma,
                p: 14,
                wt: 0.065,
                e: 0.0,
                x: 0.0,
                ef: 0.95,
                xf: 1.05,
                sl: 0.04,
                tp: 0.11,
                sma_p: 20,
                cf: 0.85,
                tf_t: 1.0,
            })
            .collect();
        Self {
            name: name.into(),
            gross_cap: 1.0,
            symbols: cfgs,
        }
    }
}

pub struct MeanRevStrategy {
    cfg: MeanRevCfg,
}

impl MeanRevStrategy {
    pub fn new(cfg: MeanRevCfg) -> Self {
        Self { cfg }
    }
}

impl Strategy for MeanRevStrategy {
    fn name(&self) -> &str {
        &self.cfg.name
    }

    fn target_weights(&self, panel: &Panel) -> Array2<f64> {
        let (t, n) = panel.close.dim();
        let mut weights = Array2::from_elem((t, n), 0.0);

        let mut col_for: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (j, sym) in panel.symbols.iter().enumerate() {
            col_for.insert(sym.to_uppercase(), j);
        }

        let price = if panel.adj_close.iter().any(|v| v.is_finite()) {
            &panel.adj_close
        } else {
            &panel.close
        };

        for sc in &self.cfg.symbols {
            let Some(&j) = col_for.get(&sc.symbol.to_uppercase()) else {
                continue;
            };

            let mut col = Array2::from_elem((t, 1), f64::NAN);
            for i in 0..t {
                col[(i, 0)] = price[(i, j)];
            }

            let ind = match sc.ind {
                Ind::Wwma => indicators::wwma(&col, sc.p),
                Ind::Sma => indicators::sma(&col, sc.p),
                Ind::Ema => indicators::ema(&col, sc.p),
                Ind::Rsi => indicators::rsi(&col, sc.p),
                Ind::Zscore => indicators::zscore(&col, sc.p),
            };
            let tf = indicators::sma(&col, sc.sma_p);

            let mut invested = false;
            let mut entry_px = 0.0f64;
            for i in 0..t {
                let p = col[(i, 0)];
                let v = ind[(i, 0)];
                let tf_v = tf[(i, 0)];
                if !p.is_finite() {
                    continue;
                }

                if invested {
                    let pnl = if entry_px > 0.0 {
                        p / entry_px - 1.0
                    } else {
                        0.0
                    };
                    let mut exit = false;
                    if sc.sl > 0.0 && pnl <= -sc.sl {
                        exit = true;
                    }
                    if !exit && sc.tp > 0.0 && pnl >= sc.tp {
                        exit = true;
                    }
                    if !exit && v.is_finite() {
                        exit = match sc.ind {
                            Ind::Wwma | Ind::Sma | Ind::Ema => p > v * sc.xf,
                            Ind::Rsi | Ind::Zscore => sc.x != 0.0 && v >= sc.x,
                        };
                    }
                    if exit {
                        invested = false;
                        entry_px = 0.0;
                        weights[(i, j)] = 0.0;
                    } else {
                        weights[(i, j)] = sc.wt;
                    }
                } else {
                    if !v.is_finite() || !tf_v.is_finite() {
                        continue;
                    }
                    let below_cf = p < tf_v * sc.cf;
                    let above_floor = sc.tf_t == 0.0 || p > tf_v * sc.tf_t;
                    if !(below_cf && above_floor) {
                        continue;
                    }
                    let entry = match sc.ind {
                        Ind::Wwma | Ind::Sma | Ind::Ema => p < v * sc.ef,
                        Ind::Rsi | Ind::Zscore => sc.e != 0.0 && v <= sc.e,
                    };
                    if entry {
                        invested = true;
                        entry_px = p;
                        weights[(i, j)] = sc.wt;
                    }
                }
            }
        }

        for i in 0..t {
            let gross: f64 = (0..n).map(|j| weights[(i, j)].abs()).sum();
            if gross > self.cfg.gross_cap && gross > 0.0 {
                let scale = self.cfg.gross_cap / gross;
                for j in 0..n {
                    weights[(i, j)] *= scale;
                }
            }
        }

        weights
    }
}
