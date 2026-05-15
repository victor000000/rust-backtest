//! `lbr-strategies` — reusable, generic strategy frameworks built on `lbr`.
//!
//! Currently exposes:
//! - `MeanRev` — per-symbol mean-reversion (indicator + thresholds + SL/TP).
//!
//! Concrete *tuned* strategies (specific symbol configs) are intentionally
//! NOT shipped here. They are loaded from YAML/JSON at runtime by users.

pub mod mean_rev;

pub use mean_rev::{Ind, MeanRevCfg, MeanRevStrategy, SymbolCfg};
