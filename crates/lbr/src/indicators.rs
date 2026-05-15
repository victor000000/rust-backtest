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

/// Kaufman's Adaptive Moving Average (KAMA).
/// SC = (ER * (fast - slow) + slow)^2; fast = 2/3, slow = 2/31.
pub fn kama(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let fast = 2.0 / 3.0;
    let slow = 2.0 / 31.0;
    par_columns(prices, move |col, mut out| {
        let len = col.len();
        if len < period + 1 {
            return;
        }
        // Seed at index `period` with the price (Wilder-style); KAMA then evolves.
        let mut last = f64::NAN;
        for i in 0..len {
            let v = col[i];
            if !v.is_finite() {
                continue;
            }
            if !last.is_finite() && i >= period {
                last = v;
                out[i] = last;
                continue;
            }
            if i < period {
                continue;
            }
            let change = (v - col[i - period]).abs();
            let mut vol = 0.0_f64;
            for k in (i - period + 1)..=i {
                let a = col[k];
                let b = col[k - 1];
                if a.is_finite() && b.is_finite() {
                    vol += (a - b).abs();
                }
            }
            let er = if vol > 0.0 { change / vol } else { 0.0 };
            let sc = (er * (fast - slow) + slow).powi(2);
            last += sc * (v - last);
            out[i] = last;
        }
    })
}

/// Commodity Channel Index (CCI), close-only proxy. Period default 14.
/// Range commonly -100…+100 (oversold below -100).
pub fn cci(prices: &Array2<f64>, period: usize) -> Array2<f64> {
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
            let mut mad = 0.0;
            for &v in win.iter() {
                if v.is_finite() {
                    mad += (v - mean).abs();
                }
            }
            mad /= count as f64;
            if mad > 0.0 {
                out[i] = (col[i] - mean) / (0.015 * mad);
            }
        }
    })
}

/// Chande Momentum Oscillator (CMO). Range -100…+100.
pub fn cmo(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in period..col.len() {
            let mut up = 0.0;
            let mut dn = 0.0;
            let mut count = 0usize;
            for k in (i - period + 1)..=i {
                let a = col[k];
                let b = col[k - 1];
                if a.is_finite() && b.is_finite() {
                    let d = a - b;
                    if d > 0.0 {
                        up += d;
                    } else {
                        dn += -d;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let denom = up + dn;
                if denom > 0.0 {
                    out[i] = 100.0 * (up - dn) / denom;
                } else {
                    out[i] = 0.0;
                }
            }
        }
    })
}

/// Money Flow Index proxy — uses RSI on close. Real MFI requires high/low/
/// volume per bar; this close-only proxy preserves the threshold semantics.
pub fn mfi(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    rsi(prices, period)
}

/// Arnaud Legoux Moving Average. Gaussian-weighted MA with offset 0.85, sigma 6.
pub fn alma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let offset = 0.85_f64;
    let sigma = 6.0_f64;
    let m = offset * (period as f64 - 1.0);
    let s = period as f64 / sigma;
    let mut weights: Vec<f64> = (0..period)
        .map(|i| (-((i as f64 - m).powi(2)) / (2.0 * s * s)).exp())
        .collect();
    let norm: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= norm;
    }
    let weights = std::sync::Arc::new(weights);
    par_columns(prices, move |col, mut out| {
        let len = col.len();
        for i in (period - 1)..len {
            let mut sum = 0.0;
            let mut ok = true;
            for k in 0..period {
                let v = col[i - (period - 1 - k)];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                sum += v * weights[k];
            }
            if ok {
                out[i] = sum;
            }
        }
    })
}

/// Least Squares Moving Average. Linear regression over `period` bars
/// evaluated at the end of the window.
pub fn lsma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let n = period as f64;
    // x = 0..n; sum_x = n(n-1)/2; sum_x2 = (n-1)n(2n-1)/6
    let sum_x = n * (n - 1.0) / 2.0;
    let sum_x2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    let denom = n * sum_x2 - sum_x * sum_x;
    par_columns(prices, move |col, mut out| {
        let len = col.len();
        for i in (period - 1)..len {
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut count = 0_usize;
            for k in 0..period {
                let x = k as f64;
                let y = col[i - (period - 1) + k];
                if y.is_finite() {
                    sum_y += y;
                    sum_xy += x * y;
                    count += 1;
                }
            }
            if count == period {
                let slope = (n * sum_xy - sum_x * sum_y) / denom;
                let intercept = (sum_y - slope * sum_x) / n;
                out[i] = intercept + slope * (n - 1.0);
            }
        }
    })
}

/// Triple Exponential Moving Average: 3·EMA − 3·EMA(EMA) + EMA(EMA(EMA)).
pub fn tema(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let e1 = ema(prices, period);
    let e2 = ema(&e1, period);
    let e3 = ema(&e2, period);
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = e1[(i, j)];
            let b = e2[(i, j)];
            let c = e3[(i, j)];
            if a.is_finite() && b.is_finite() && c.is_finite() {
                out[(i, j)] = 3.0 * a - 3.0 * b + c;
            }
        }
    }
    out
}

/// Double Exponential Moving Average: 2·EMA − EMA(EMA).
pub fn dema(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let e1 = ema(prices, period);
    let e2 = ema(&e1, period);
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = e1[(i, j)];
            let b = e2[(i, j)];
            if a.is_finite() && b.is_finite() {
                out[(i, j)] = 2.0 * a - b;
            }
        }
    }
    out
}

/// Stochastic RSI: stochastic oscillator applied to a 14-period RSI series.
/// Returns values in [0, 100].
pub fn srsi(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let rsi_vals = rsi(prices, 14);
    let (t, _) = prices.dim();
    if period < 2 || t < period + 14 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(&rsi_vals, move |col, mut out| {
        for i in (period - 1)..col.len() {
            let win = col.slice(ndarray::s![i + 1 - period..=i]);
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut ok = true;
            for &v in win.iter() {
                if v.is_finite() {
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if ok && max_v > min_v {
                out[i] = 100.0 * (col[i] - min_v) / (max_v - min_v);
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
