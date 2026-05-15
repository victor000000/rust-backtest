//! Strategy trait — the platform's single extension point.
//!
//! Concrete strategies live in downstream crates (e.g. `lbr-strategies`) and
//! private user crates. The platform knows nothing about specific strategies.

use ndarray::Array2;

use crate::data::Panel;

/// A vectorized strategy produces target weights `(T, N)` over the panel.
///
/// Weight at row `i` is the target for the close of bar `i`; the engine
/// applies it lagged by one bar so `pnl[i] = W[i-1] · r[i]`.
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn target_weights(&self, panel: &Panel) -> Array2<f64>;
}
