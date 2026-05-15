//! Vectorized technical indicators. All produce a column of length T per symbol,
//! NaN-padded at the start until enough history exists.
//!
//! Designed to be cheap to call per (panel, period) — they walk the column once.

use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Simple moving average over `period`. Operates column-wise over a (T, N) matrix.
pub fn sma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if period == 0 || t < period {
        return out;
    }
    for j in 0..n {
        let mut sum = 0.0;
        let mut count: usize = 0;
        for i in 0..t {
            let v = prices[(i, j)];
            if v.is_finite() {
                sum += v;
                count += 1;
            }
            if i >= period {
                let old = prices[(i - period, j)];
                if old.is_finite() {
                    sum -= old;
                    count -= 1;
                }
            }
            if count == period {
                out[(i, j)] = sum / period as f64;
            }
        }
    }
    out
}

/// Wilder's smoothed moving average (a.k.a. WWMA / RMA): exponential with alpha = 1/period.
/// Seeds with the SMA of the first `period` values.
pub fn wwma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if period == 0 || t < period {
        return out;
    }
    let alpha = 1.0 / period as f64;
    for j in 0..n {
        let col = prices.column(j);
        let mut seed_sum = 0.0;
        let mut seed_count = 0usize;
        let mut last = f64::NAN;
        for i in 0..t {
            let v = col[i];
            if !v.is_finite() {
                continue;
            }
            if seed_count < period {
                seed_sum += v;
                seed_count += 1;
                if seed_count == period {
                    last = seed_sum / period as f64;
                    out[(i, j)] = last;
                }
            } else {
                last = last + alpha * (v - last);
                out[(i, j)] = last;
            }
        }
    }
    out
}

/// Exponential moving average with alpha = 2 / (period + 1).
pub fn ema(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if period == 0 || t < period {
        return out;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    for j in 0..n {
        let col = prices.column(j);
        let mut seed_sum = 0.0;
        let mut seed_count = 0usize;
        let mut last = f64::NAN;
        for i in 0..t {
            let v = col[i];
            if !v.is_finite() {
                continue;
            }
            if seed_count < period {
                seed_sum += v;
                seed_count += 1;
                if seed_count == period {
                    last = seed_sum / period as f64;
                    out[(i, j)] = last;
                }
            } else {
                last = last + alpha * (v - last);
                out[(i, j)] = last;
            }
        }
    }
    out
}

/// Relative Strength Index, Wilder smoothing. Returns values in [0, 100].
pub fn rsi(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if period == 0 || t < period + 1 {
        return out;
    }
    let alpha = 1.0 / period as f64;
    for j in 0..n {
        let col = prices.column(j);
        let mut gains = Vec::with_capacity(period);
        let mut losses = Vec::with_capacity(period);
        let mut prev = f64::NAN;
        let mut avg_g = f64::NAN;
        let mut avg_l = f64::NAN;
        for i in 0..t {
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
                    avg_g = avg_g + alpha * (g - avg_g);
                    avg_l = avg_l + alpha * (l - avg_l);
                }
                if avg_g.is_finite() && avg_l.is_finite() {
                    if avg_l == 0.0 {
                        out[(i, j)] = 100.0;
                    } else {
                        let rs = avg_g / avg_l;
                        out[(i, j)] = 100.0 - 100.0 / (1.0 + rs);
                    }
                }
            }
            prev = v;
        }
    }
    out
}

/// Z-score of price against an SMA, in std units of trailing returns (period bars).
/// Useful as a normalized "distance from mean" signal — proxy for several S002 oscillators.
pub fn zscore(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if period < 2 || t < period {
        return out;
    }
    for j in 0..n {
        let col = prices.column(j);
        for i in (period - 1)..t {
            let win: ArrayView1<f64> = col.slice(ndarray::s![i + 1 - period..=i]);
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
                out[(i, j)] = (col[i] - mean) / sd;
            }
        }
    }
    out
}

/// Daily simple returns from a price column. Length T, with NaN at i=0.
pub fn returns_axis0(prices: &Array2<f64>) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in 1..t {
            let prev = prices[(i - 1, j)];
            let cur = prices[(i, j)];
            if prev.is_finite() && cur.is_finite() && prev > 0.0 {
                out[(i, j)] = cur / prev - 1.0;
            }
        }
    }
    out
}

/// Trailing standard deviation (population) over period bars.
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
