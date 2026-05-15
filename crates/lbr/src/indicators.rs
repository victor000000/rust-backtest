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

/// Weighted Moving Average: linear weights 1..n.
pub fn wma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let norm = (period * (period + 1)) as f64 / 2.0;
    par_columns(prices, move |col, mut out| {
        for i in (period - 1)..col.len() {
            let mut sum = 0.0;
            let mut ok = true;
            for k in 0..period {
                let v = col[i - (period - 1) + k];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                sum += v * (k + 1) as f64;
            }
            if ok {
                out[i] = sum / norm;
            }
        }
    })
}

/// Hull Moving Average: WMA(2·WMA(n/2) − WMA(n), √n).
pub fn hma(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    if period < 2 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let half = (period / 2).max(2);
    let sqrt_n = (period as f64).sqrt().round() as usize;
    let sqrt_n = sqrt_n.max(2);

    let wma_full = wma(prices, period);
    let wma_half = wma(prices, half);
    let (t, n) = prices.dim();
    let mut intermediate = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = wma_half[(i, j)];
            let b = wma_full[(i, j)];
            if a.is_finite() && b.is_finite() {
                intermediate[(i, j)] = 2.0 * a - b;
            }
        }
    }
    wma(&intermediate, sqrt_n)
}

/// MACD histogram = (EMA(fast) − EMA(slow)) − EMA(MACD line, signal).
/// Returns the histogram component.
pub fn macd(prices: &Array2<f64>, fast: usize, slow: usize, signal: usize) -> Array2<f64> {
    let e_fast = ema(prices, fast);
    let e_slow = ema(prices, slow);
    let (t, n) = prices.dim();
    let mut macd_line = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = e_fast[(i, j)];
            let b = e_slow[(i, j)];
            if a.is_finite() && b.is_finite() {
                macd_line[(i, j)] = a - b;
            }
        }
    }
    let sig = ema(&macd_line, signal);
    let mut hist = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = macd_line[(i, j)];
            let b = sig[(i, j)];
            if a.is_finite() && b.is_finite() {
                hist[(i, j)] = a - b;
            }
        }
    }
    hist
}

/// True Strength Index: 100·EMA(EMA(Δp, r), s) / EMA(EMA(|Δp|, r), s).
/// Returns values in roughly [-100, 100].
pub fn tsi(prices: &Array2<f64>, r: usize, s: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let mut mom = Array2::from_elem((t, n), f64::NAN);
    let mut abs_mom = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in 1..t {
            let cur = prices[(i, j)];
            let prev = prices[(i - 1, j)];
            if cur.is_finite() && prev.is_finite() {
                let d = cur - prev;
                mom[(i, j)] = d;
                abs_mom[(i, j)] = d.abs();
            }
        }
    }
    let num = ema(&ema(&mom, r), s);
    let den = ema(&ema(&abs_mom, r), s);
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = num[(i, j)];
            let b = den[(i, j)];
            if a.is_finite() && b.is_finite() && b > 0.0 {
                out[(i, j)] = 100.0 * a / b;
            }
        }
    }
    out
}

/// Detrended Price Oscillator: price[t − (n/2 + 1)] − SMA(close, n).
pub fn dpo(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    let shift = period / 2 + 1;
    let start = period.saturating_sub(1).max(shift);
    par_columns(prices, move |col, mut out| {
        let len = col.len();
        for i in start..len {
            let p = col[i - shift];
            let mut sum = 0.0;
            let mut count = 0_usize;
            for k in 0..period {
                let idx = i + 1 - period + k;
                let v = col[idx];
                if v.is_finite() {
                    sum += v;
                    count += 1;
                }
            }
            if count == period && p.is_finite() {
                out[i] = p - sum / period as f64;
            }
        }
    })
}

/// Absolute Price Oscillator: EMA(fast) − EMA(slow).
pub fn apo(prices: &Array2<f64>, fast: usize, slow: usize) -> Array2<f64> {
    let f = ema(prices, fast);
    let s = ema(prices, slow);
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = f[(i, j)];
            let b = s[(i, j)];
            if a.is_finite() && b.is_finite() {
                out[(i, j)] = a - b;
            }
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
// High/Low/Close indicators — these need the full bar, not just close.
// ────────────────────────────────────────────────────────────────────────

/// Internal Bar Strength: (close − low) / (high − low). Per-bar, [0, 1].
pub fn ibs(high: &Array2<f64>, low: &Array2<f64>, close: &Array2<f64>) -> Array2<f64> {
    let (t, n) = close.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let h = high[(i, j)];
            let l = low[(i, j)];
            let c = close[(i, j)];
            if h.is_finite() && l.is_finite() && c.is_finite() && h > l {
                out[(i, j)] = (c - l) / (h - l);
            }
        }
    }
    out
}

/// Stochastic %K: 100·(close − min(low,n)) / (max(high,n) − min(low,n)).
pub fn stoch(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let (t, n) = close.dim();
    if period < 2 || t < period {
        return Array2::from_elem(close.dim(), f64::NAN);
    }
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in (period - 1)..t {
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            let mut ok = true;
            for k in (i + 1 - period)..=i {
                let h = high[(k, j)];
                let l = low[(k, j)];
                if !h.is_finite() || !l.is_finite() {
                    ok = false;
                    break;
                }
                if h > hi {
                    hi = h;
                }
                if l < lo {
                    lo = l;
                }
            }
            let c = close[(i, j)];
            if ok && c.is_finite() && hi > lo {
                out[(i, j)] = 100.0 * (c - lo) / (hi - lo);
            }
        }
    }
    out
}

/// True Range per bar: max(high-low, |high-prev_close|, |low-prev_close|).
fn true_range(high: &Array2<f64>, low: &Array2<f64>, close: &Array2<f64>) -> Array2<f64> {
    let (t, n) = close.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in 0..t {
            let h = high[(i, j)];
            let l = low[(i, j)];
            if !h.is_finite() || !l.is_finite() {
                continue;
            }
            let mut tr = h - l;
            if i > 0 {
                let pc = close[(i - 1, j)];
                if pc.is_finite() {
                    tr = tr.max((h - pc).abs()).max((l - pc).abs());
                }
            }
            out[(i, j)] = tr;
        }
    }
    out
}

/// Average True Range, Wilder smoothing.
pub fn atr(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let tr = true_range(high, low, close);
    wwma(&tr, period)
}

/// Normalized ATR: 100 · ATR / close. Range typically a few percent.
pub fn natr(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let a = atr(high, low, close, period);
    let (t, n) = close.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let av = a[(i, j)];
            let c = close[(i, j)];
            if av.is_finite() && c.is_finite() && c > 0.0 {
                out[(i, j)] = 100.0 * av / c;
            }
        }
    }
    out
}

/// DeMarker indicator: SMA(DeMax,n) / (SMA(DeMax,n) + SMA(DeMin,n)).
/// Range [0, 1]; oversold below 0.3, overbought above 0.7.
pub fn dem(high: &Array2<f64>, low: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = high.dim();
    let mut demax = Array2::from_elem((t, n), 0.0);
    let mut demin = Array2::from_elem((t, n), 0.0);
    for j in 0..n {
        for i in 1..t {
            let h = high[(i, j)];
            let ph = high[(i - 1, j)];
            let l = low[(i, j)];
            let pl = low[(i - 1, j)];
            if h.is_finite() && ph.is_finite() {
                demax[(i, j)] = (h - ph).max(0.0);
            }
            if l.is_finite() && pl.is_finite() {
                demin[(i, j)] = (pl - l).max(0.0);
            }
        }
    }
    let s_max = sma(&demax, period);
    let s_min = sma(&demin, period);
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = s_max[(i, j)];
            let b = s_min[(i, j)];
            if a.is_finite() && b.is_finite() && (a + b) > 0.0 {
                out[(i, j)] = a / (a + b);
            }
        }
    }
    out
}

/// Awesome Oscillator: SMA(median_price, 5) − SMA(median_price, 34).
pub fn ao(high: &Array2<f64>, low: &Array2<f64>) -> Array2<f64> {
    let (t, n) = high.dim();
    let mut median = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let h = high[(i, j)];
            let l = low[(i, j)];
            if h.is_finite() && l.is_finite() {
                median[(i, j)] = 0.5 * (h + l);
            }
        }
    }
    let s5 = sma(&median, 5);
    let s34 = sma(&median, 34);
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = s5[(i, j)];
            let b = s34[(i, j)];
            if a.is_finite() && b.is_finite() {
                out[(i, j)] = a - b;
            }
        }
    }
    out
}

/// Choppiness Index: 100 · log10(ΣTR / (max(high,n) − min(low,n))) / log10(n).
/// Range 0..100; > 60 = choppy, < 40 = trending.
pub fn chop(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let (t, n) = high.dim();
    if period < 2 || t < period {
        return Array2::from_elem((t, n), f64::NAN);
    }
    let tr = true_range(high, low, close);
    let log_n = (period as f64).log10();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in (period - 1)..t {
            let mut sum_tr = 0.0;
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            let mut ok = true;
            for k in (i + 1 - period)..=i {
                let h = high[(k, j)];
                let l = low[(k, j)];
                let t = tr[(k, j)];
                if !h.is_finite() || !l.is_finite() || !t.is_finite() {
                    ok = false;
                    break;
                }
                sum_tr += t;
                if h > hi {
                    hi = h;
                }
                if l < lo {
                    lo = l;
                }
            }
            if ok && hi > lo && sum_tr > 0.0 {
                out[(i, j)] = 100.0 * (sum_tr / (hi - lo)).log10() / log_n;
            }
        }
    }
    out
}

/// Bollinger Bands lower band: SMA(n) − k·σ.
pub fn bb_lower(prices: &Array2<f64>, period: usize, k: f64) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in (period - 1)..col.len() {
            let mut sum = 0.0;
            let mut count = 0usize;
            for k_ in 0..period {
                let v = col[i - (period - 1) + k_];
                if v.is_finite() {
                    sum += v;
                    count += 1;
                }
            }
            if count != period {
                continue;
            }
            let m = sum / period as f64;
            let mut var = 0.0;
            for k_ in 0..period {
                let v = col[i - (period - 1) + k_];
                let d = v - m;
                var += d * d;
            }
            let sd = (var / period as f64).sqrt();
            out[i] = m - k * sd;
        }
    })
}

/// Rate of Change: 100·(price − price[n]) / price[n]. Range -∞..+∞.
pub fn roc(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in period..col.len() {
            let cur = col[i];
            let prev = col[i - period];
            if cur.is_finite() && prev.is_finite() && prev != 0.0 {
                out[i] = 100.0 * (cur - prev) / prev;
            }
        }
    })
}

/// Momentum (raw difference): price − price[n].
pub fn momentum(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period == 0 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in period..col.len() {
            let cur = col[i];
            let prev = col[i - period];
            if cur.is_finite() && prev.is_finite() {
                out[i] = cur - prev;
            }
        }
    })
}

/// Fisher Transform on price-mapped value in [-1, 1].
/// Maps to normal-ish distribution; range typically [-3, +3].
pub fn fisher(high: &Array2<f64>, low: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, n) = high.dim();
    if period < 2 || t < period {
        return Array2::from_elem((t, n), f64::NAN);
    }
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        let mut prev_v = 0.0_f64;
        let mut prev_fish = 0.0_f64;
        for i in (period - 1)..t {
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            let mut ok = true;
            for k in (i + 1 - period)..=i {
                let h = high[(k, j)];
                let l = low[(k, j)];
                if !h.is_finite() || !l.is_finite() {
                    ok = false;
                    break;
                }
                let med = 0.5 * (h + l);
                if med > hi {
                    hi = med;
                }
                if med < lo {
                    lo = med;
                }
            }
            if !ok || hi == lo {
                continue;
            }
            let h = high[(i, j)];
            let l = low[(i, j)];
            let med = 0.5 * (h + l);
            let raw = 2.0 * ((med - lo) / (hi - lo) - 0.5);
            let v = 0.33 * raw + 0.67 * prev_v;
            let v = v.clamp(-0.999, 0.999);
            let fish = 0.5 * ((1.0 + v) / (1.0 - v)).ln() + 0.5 * prev_fish;
            out[(i, j)] = fish;
            prev_v = v;
            prev_fish = fish;
        }
    }
    out
}

/// Aroon Oscillator: 100·(periods_since_max − periods_since_min) / period.
/// Range [-100, 100].
pub fn aroon(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period + 1 {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in period..col.len() {
            let mut max_v = f64::NEG_INFINITY;
            let mut min_v = f64::INFINITY;
            let mut max_idx = 0;
            let mut min_idx = 0;
            for k in 0..=period {
                let v = col[i - period + k];
                if !v.is_finite() {
                    continue;
                }
                if v > max_v {
                    max_v = v;
                    max_idx = k;
                }
                if v < min_v {
                    min_v = v;
                    min_idx = k;
                }
            }
            let up = 100.0 * max_idx as f64 / period as f64;
            let down = 100.0 * min_idx as f64 / period as f64;
            out[i] = up - down;
        }
    })
}

/// Ultimate Oscillator with periods 7, 14, 28. Range [0, 100].
pub fn ultosc(high: &Array2<f64>, low: &Array2<f64>, close: &Array2<f64>) -> Array2<f64> {
    let (t, n) = close.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if t < 30 {
        return out;
    }
    for j in 0..n {
        let mut bp = vec![0.0_f64; t];
        let mut tr = vec![0.0_f64; t];
        for i in 1..t {
            let c = close[(i, j)];
            let h = high[(i, j)];
            let l = low[(i, j)];
            let pc = close[(i - 1, j)];
            if !c.is_finite() || !h.is_finite() || !l.is_finite() || !pc.is_finite() {
                continue;
            }
            let min_lp = l.min(pc);
            let max_hp = h.max(pc);
            bp[i] = c - min_lp;
            tr[i] = max_hp - min_lp;
        }
        for i in 28..t {
            let (mut sb7, mut st7) = (0.0_f64, 0.0_f64);
            let (mut sb14, mut st14) = (0.0_f64, 0.0_f64);
            let (mut sb28, mut st28) = (0.0_f64, 0.0_f64);
            for k in 0..7 {
                sb7 += bp[i - k];
                st7 += tr[i - k];
            }
            for k in 0..14 {
                sb14 += bp[i - k];
                st14 += tr[i - k];
            }
            for k in 0..28 {
                sb28 += bp[i - k];
                st28 += tr[i - k];
            }
            if st7 > 0.0 && st14 > 0.0 && st28 > 0.0 {
                let a7 = sb7 / st7;
                let a14 = sb14 / st14;
                let a28 = sb28 / st28;
                out[(i, j)] = 100.0 * (4.0 * a7 + 2.0 * a14 + a28) / 7.0;
            }
        }
    }
    out
}

/// Williams %R: -100·(max(high,n) − close) / (max(high,n) − min(low,n)).
/// Range [-100, 0]; oversold below -80.
pub fn wilr(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let (t, n) = close.dim();
    if period < 2 || t < period {
        return Array2::from_elem((t, n), f64::NAN);
    }
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in (period - 1)..t {
            let mut hi = f64::NEG_INFINITY;
            let mut lo = f64::INFINITY;
            for k in (i + 1 - period)..=i {
                let h = high[(k, j)];
                let l = low[(k, j)];
                if h.is_finite() && h > hi {
                    hi = h;
                }
                if l.is_finite() && l < lo {
                    lo = l;
                }
            }
            let c = close[(i, j)];
            if c.is_finite() && hi > lo {
                out[(i, j)] = -100.0 * (hi - c) / (hi - lo);
            }
        }
    }
    out
}

/// Connors RSI: (RSI(3) + StreakRSI(2) + PctRank(100)) / 3.
pub fn crsi(prices: &Array2<f64>, _: usize) -> Array2<f64> {
    let (t, n) = prices.dim();
    let rsi3 = rsi(prices, 3);
    let mut streak = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        let mut s: i32 = 0;
        let mut prev = f64::NAN;
        for i in 0..t {
            let v = prices[(i, j)];
            if v.is_finite() && prev.is_finite() {
                if v > prev {
                    s = if s > 0 { s + 1 } else { 1 };
                } else if v < prev {
                    s = if s < 0 { s - 1 } else { -1 };
                } else {
                    s = 0;
                }
            }
            streak[(i, j)] = s as f64;
            if v.is_finite() {
                prev = v;
            }
        }
    }
    let streak_rsi2 = rsi(&streak, 2);
    let mut pct_rank = Array2::from_elem((t, n), f64::NAN);
    let win = 100;
    for j in 0..n {
        for i in 1..t {
            let cur = prices[(i, j)];
            let prev = prices[(i - 1, j)];
            if !cur.is_finite() || !prev.is_finite() || prev <= 0.0 {
                continue;
            }
            let r = cur / prev - 1.0;
            if i < win + 1 {
                continue;
            }
            let mut count = 0;
            let mut below = 0;
            for k in (i - win)..i {
                let a = prices[(k, j)];
                let b = prices[(k - 1, j)];
                if a.is_finite() && b.is_finite() && b > 0.0 {
                    let rk = a / b - 1.0;
                    count += 1;
                    if rk < r {
                        below += 1;
                    }
                }
            }
            if count > 0 {
                pct_rank[(i, j)] = 100.0 * below as f64 / count as f64;
            }
        }
    }
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = rsi3[(i, j)];
            let b = streak_rsi2[(i, j)];
            let c = pct_rank[(i, j)];
            if a.is_finite() && b.is_finite() && c.is_finite() {
                out[(i, j)] = (a + b + c) / 3.0;
            }
        }
    }
    out
}

/// Tillson T3 — generalized DEMA with volume factor v=0.7.
pub fn t3(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let v = 0.7_f64;
    let e1 = ema(prices, period);
    let e2 = ema(&e1, period);
    let e3 = ema(&e2, period);
    let e4 = ema(&e3, period);
    let e5 = ema(&e4, period);
    let e6 = ema(&e5, period);
    let c1 = -v * v * v;
    let c2 = 3.0 * v * v + 3.0 * v * v * v;
    let c3 = -6.0 * v * v - 3.0 * v - 3.0 * v * v * v;
    let c4 = 1.0 + 3.0 * v + v * v * v + 3.0 * v * v;
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for i in 0..t {
        for j in 0..n {
            let a = e3[(i, j)];
            let b = e4[(i, j)];
            let c = e5[(i, j)];
            let d = e6[(i, j)];
            if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                out[(i, j)] = c1 * d + c2 * c + c3 * b + c4 * a;
            }
        }
    }
    out
}

/// FRAMA (Fractal Adaptive MA) with default settings.
pub fn frama(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let p = (period / 2) * 2; // make even
    let p = p.max(4);
    let (t, n) = prices.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        let mut last = f64::NAN;
        for i in (p - 1)..t {
            // Highs / lows for the two halves
            let half = p / 2;
            let mut h1 = f64::NEG_INFINITY;
            let mut l1 = f64::INFINITY;
            let mut h2 = f64::NEG_INFINITY;
            let mut l2 = f64::INFINITY;
            let mut h = f64::NEG_INFINITY;
            let mut l = f64::INFINITY;
            let mut ok = true;
            for k in 0..p {
                let v = prices[(i - (p - 1) + k, j)];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                if v > h {
                    h = v;
                }
                if v < l {
                    l = v;
                }
                if k < half {
                    if v > h1 {
                        h1 = v;
                    }
                    if v < l1 {
                        l1 = v;
                    }
                } else {
                    if v > h2 {
                        h2 = v;
                    }
                    if v < l2 {
                        l2 = v;
                    }
                }
            }
            if !ok {
                continue;
            }
            let n1 = (h1 - l1) / half as f64;
            let n2 = (h2 - l2) / half as f64;
            let n3 = (h - l) / p as f64;
            if n1 + n2 <= 0.0 || n3 <= 0.0 {
                continue;
            }
            let d = ((n1 + n2).ln() - n3.ln()) / (2.0_f64).ln();
            let alpha = ((-4.6 * (d - 1.0)).exp()).clamp(0.01, 1.0);
            let v = prices[(i, j)];
            if !last.is_finite() {
                last = v;
            } else {
                last = alpha * v + (1.0 - alpha) * last;
            }
            out[(i, j)] = last;
        }
    }
    out
}

/// Average Directional Index, Wilder smoothing. Range [0, 100].
/// Standard trend-strength indicator: ADX < 20 = weak trend, > 25 = strong.
pub fn adx(
    high: &Array2<f64>,
    low: &Array2<f64>,
    close: &Array2<f64>,
    period: usize,
) -> Array2<f64> {
    let (t, n) = close.dim();
    if period < 2 || t < period * 2 + 1 {
        return Array2::from_elem((t, n), f64::NAN);
    }
    let tr = true_range(high, low, close);
    // Directional moves.
    let mut pdm = Array2::from_elem((t, n), 0.0);
    let mut ndm = Array2::from_elem((t, n), 0.0);
    for j in 0..n {
        for i in 1..t {
            let h = high[(i, j)];
            let ph = high[(i - 1, j)];
            let l = low[(i, j)];
            let pl = low[(i - 1, j)];
            if h.is_finite() && ph.is_finite() && l.is_finite() && pl.is_finite() {
                let up = h - ph;
                let down = pl - l;
                pdm[(i, j)] = if up > down && up > 0.0 { up } else { 0.0 };
                ndm[(i, j)] = if down > up && down > 0.0 { down } else { 0.0 };
            }
        }
    }
    let smooth_tr = wwma(&tr, period);
    let smooth_pdm = wwma(&pdm, period);
    let smooth_ndm = wwma(&ndm, period);
    let mut dx = Array2::from_elem((t, n), f64::NAN);
    for j in 0..n {
        for i in 0..t {
            let st = smooth_tr[(i, j)];
            let sp = smooth_pdm[(i, j)];
            let sn = smooth_ndm[(i, j)];
            if st.is_finite() && st > 0.0 && sp.is_finite() && sn.is_finite() {
                let pdi = 100.0 * sp / st;
                let ndi = 100.0 * sn / st;
                let s = pdi + ndi;
                if s > 0.0 {
                    dx[(i, j)] = 100.0 * (pdi - ndi).abs() / s;
                }
            }
        }
    }
    wwma(&dx, period)
}

/// Bollinger Band width relative to the middle band:
/// `(upper − lower) / middle = 2·k·σ / SMA(n)`. Useful as a vol-regime gate.
pub fn bb_width(prices: &Array2<f64>, period: usize, k: f64) -> Array2<f64> {
    let (t, _) = prices.dim();
    if period < 2 || t < period {
        return Array2::from_elem(prices.dim(), f64::NAN);
    }
    par_columns(prices, move |col, mut out| {
        for i in (period - 1)..col.len() {
            let mut sum = 0.0;
            let mut count = 0usize;
            for k_ in 0..period {
                let v = col[i - (period - 1) + k_];
                if v.is_finite() {
                    sum += v;
                    count += 1;
                }
            }
            if count != period {
                continue;
            }
            let m = sum / period as f64;
            let mut var = 0.0;
            for k_ in 0..period {
                let v = col[i - (period - 1) + k_];
                let d = v - m;
                var += d * d;
            }
            let sd = (var / period as f64).sqrt();
            if m > 0.0 {
                out[i] = (2.0 * k * sd) / m;
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
