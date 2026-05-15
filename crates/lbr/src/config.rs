use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Directory where cached CSV bars live.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    /// Universe symbols.
    pub symbols: Vec<String>,
    /// Inclusive start date (YYYY-MM-DD).
    pub start: String,
    /// Exclusive end date (YYYY-MM-DD).
    pub end: String,
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from("data/lake")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Per-side commission in fractional dollars per trade (e.g. 0.0005 = 5 bps).
    #[serde(default)]
    pub commission_bps: f64,
    /// Slippage in fractional price per trade (e.g. 0.0005 = 5 bps).
    #[serde(default)]
    pub slippage_bps: f64,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            commission_bps: 0.0,
            slippage_bps: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub data: DataConfig,
    #[serde(default)]
    pub cost: CostConfig,
    #[serde(default = "default_cash")]
    pub starting_cash: f64,
}

fn default_cash() -> f64 {
    70_000.0
}
