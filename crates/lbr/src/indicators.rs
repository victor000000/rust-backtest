//! Vectorized technical indicators.
//!
//! All indicators produce a `(T, N)` output and operate per-column. Columns
//! are processed in parallel via Rayon so wide universes scale across cores.
//! Within each column, recursive/stateful indicators (EMA, RSI, WWMA) are
//! sequential by construction.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use rayon::prelude::*;

/// Apply a per-column kernel in parallel across the `N` columns.
fn par_columns<F>(prices: &Array2<f64>, kernel: F) -> Array2<f64>
where
    F: Fn(ArrayView1<f64>, ArrayViewMut1<f64>) + Send + Sync,
{
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    out.axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(prices.axis_iter(Axis(1)).into_par_iter())
        .for_each(|(out_col, in_col)| kernel(in_col, out_col));
    out
}

/// Simple moving average — vectorized rolling sum.
pub fn sma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..col.len() {
            let v = col[i];
            if v.is_finite() {
                sum += v;
                count += 1;
            }
            if i >= period {
                let old = col[i - period];
                if old.is_finite() {
                    sum -= old;
                    count -= 1;
                }
            }
            if count == period {
                out[i] = sum / period as f64;
            }
        }
    })
}

/// Wilder's smoothed moving average (RMA), alpha = 1/period. Seeded by SMA.
pub fn wwma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let alpha = 1.0 / period as f64;
    par_columns(prices, move |col, mut out| {
        let mut seed_sum = 0.0;
        let mut seed_count = 0usize;
        let mut last = f64::NAN;
        for i in 0..col.len() {
            let v = col[i];
            if !v.is_finite() {
                continue;
            }
            if seed_count < period {
                seed_sum += v;
                seed_count += 1;
                if seed_count == period {
                    last = seed_sum / period as f64;
                    out[i] = last;
                }
            } else {
                last += alpha * (v - last);
                out[i] = last;
            }
        }
    })
}

/// Exponential moving average, alpha = 2 / (period + 1). Seeded by SMA.
pub fn ema(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    par_columns(prices, move |col, mut out| {
        let mut seed_sum = 0.0;
        let mut seed_count = 0usize;
        let mut last = f64::NAN;
        for i in 0..col.len() {
            let v = col[i];
            if !v.is_finite() {
                continue;
            }
            if seed_count < period {
                seed_sum += v;
                seed_count += 1;
                if seed_count == period {
                    last = seed_sum / period as f64;
                    out[i] = last;
                }
            } else {
                last += alpha * (v - last);
                out[i] = last;
            }
        }
    })
}

/// Relative Strength Index, Wilder smoothing. Range [0, 100].
pub fn rsi(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let alpha = 1.0 / period as f64;
    par_columns(prices, move |col, mut out| {
        let mut gains = Vec::with_capacity(period);
        let mut losses = Vec::with_capacity(period);
        let mut prev = f64::NAN;
        let mut avg_g = f64::NAN;
        let mut avg_l = f64::NAN;
        for i in 0..col.len() {
            let v = col[i];
            if !v.is_finite() {
                prev = v;
                continue;
            }
            if prev.is_finite() {
                let chg = v - prev;
                let g = chg.max(0.0);
                let l = (-chg).max(0.0);
                if !avg_g.is_finite() {
                    gains.push(g);
                    losses.push(l);
                    if gains.len() == period {
                        avg_g = gains.iter().sum::<f64>() / period as f64;
                        avg_l = losses.iter().sum::<f64>() / period as f64;
                    }
                } else {
                    avg_g += alpha * (g - avg_g);
                    avg_l += alpha * (l - avg_l);
                }
                if avg_g.is_finite() && avg_l.is_finite() {
                    if avg_l == 0.0 {
                        out[i] = 100.0;
                    } else {
                        let rs = avg_g / avg_l;
                        out[i] = 100.0 - 100.0 / (1.0 + rs);
                    }
                }
            }
            prev = v;
        }
    })
}

/// Z-score of price against a trailing window mean & std.
pub fn zscore(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in (period - 1)..col.len() {
            let win = col.slice(ndarray::s![i + 1 - period..=i]);
            let mut sum = 0.0;
            let mut count = 0usize;
            for &v in win.iter() {
                if v.is_finite() {
                    sum += v;
                    count += 1;
                }
            }
            if count < period {
                continue;
            }
            let mean = sum / count as f64;
            let mut var = 0.0;
            for &v in win.iter() {
                if v.is_finite() {
                    let d = v - mean;
                    var += d * d;
                }
            }
            var /= count as f64;
            let sd = var.sqrt();
            if sd > 0.0 {
                out[i] = (col[i] - mean) / sd;
            }
        }
    })
}

/// Daily simple returns. Length T with NaN at i=0.
pub fn returns_axis0(prices: &Array2<f64>) -> Array2<f64> {
    let (t, _) = prices.dim();
    if t < 2 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, |col, mut out| {
        for i in 1..col.len() {
            let prev = col[i - 1];
            let cur = col[i];
            if prev.is_finite() && cur.is_finite() && prev > 0.0 {
                out[i] = cur / prev - 1.0;
            }
        }
    })
}

/// Trailing standard deviation of the cross-sectional mean (utility).
pub fn rolling_std(prices: &Array2<f64>, period: usize) -> Array1<f64> {
    let (t, _) = prices.dim();
    let mut out = Array1::from_elem(t, f64::NAN);
    let mean = prices.mean_axis(Axis(1)).unwrap();
    for i in (period - 1)..t {
        let mut var = 0.0;
        let mut count = 0usize;
        for k in (i + 1 - period)..=i {
            let v = mean[k];
            if v.is_finite() {
                let d = v - mean[i];
                var += d * d;
                count += 1;
            }
        }
        if count > 1 {
            out[i] = (var / count as f64).sqrt();
        }
    }
    out
}
