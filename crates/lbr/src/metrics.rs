//! Backtest performance metrics. All operate on an equity curve (T points).

use ndarray::Array1;

/// Daily-to-annual factor. 252 trading days.
pub const TRADING_DAYS: f64 = 252.0;

#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    pub total_return: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown: f64,
    pub calmar: f64,
    pub annual_vol: f64,
    pub win_rate: f64,
    pub n_bars: usize,
}

pub fn compute(equity: &Array1<f64>) -> Metrics {
    let n = equity.len();
    if n < 2 {
        return Metrics {
            total_return: 0.0,
            cagr: 0.0,
            sharpe: 0.0,
            sortino: 0.0,
            max_drawdown: 0.0,
            calmar: 0.0,
            annual_vol: 0.0,
            win_rate: 0.0,
            n_bars: n,
        };
    }
    let start = equity[0];
    let end = equity[n - 1];
    let total_return = end / start - 1.0;

    let mut rets = Vec::with_capacity(n - 1);
    let mut wins = 0usize;
    let mut nonzero = 0usize;
    for i in 1..n {
        let p = equity[i - 1];
        let c = equity[i];
        if p.is_finite() && c.is_finite() && p > 0.0 {
            let r = c / p - 1.0;
            rets.push(r);
            if r != 0.0 {
                nonzero += 1;
                if r > 0.0 {
                    wins += 1;
                }
            }
        }
    }

    let mean: f64 = rets.iter().sum::<f64>() / rets.len().max(1) as f64;
    let var: f64 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len().max(1) as f64;
    let sd = var.sqrt();
    let down: Vec<f64> = rets.iter().filter(|r| **r < 0.0).copied().collect();
    let dvar: f64 = down.iter().map(|r| r * r).sum::<f64>() / down.len().max(1) as f64;
    let dsd = dvar.sqrt();

    let annual_ret = mean * TRADING_DAYS;
    let annual_vol = sd * TRADING_DAYS.sqrt();
    let sharpe = if annual_vol > 0.0 {
        annual_ret / annual_vol
    } else {
        0.0
    };
    let sortino = if dsd > 0.0 {
        annual_ret / (dsd * TRADING_DAYS.sqrt())
    } else {
        0.0
    };

    let years = (n as f64 - 1.0) / TRADING_DAYS;
    let cagr = if years > 0.0 && start > 0.0 && end > 0.0 {
        (end / start).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    let mut peak = start;
    let mut mdd = 0.0;
    for &e in equity.iter() {
        if e > peak {
            peak = e;
        }
        if peak > 0.0 {
            let dd = (peak - e) / peak;
            if dd > mdd {
                mdd = dd;
            }
        }
    }

    let calmar = if mdd > 0.0 { cagr / mdd } else { 0.0 };
    let win_rate = if nonzero > 0 {
        wins as f64 / nonzero as f64
    } else {
        0.0
    };

    Metrics {
        total_return,
        cagr,
        sharpe,
        sortino,
        max_drawdown: mdd,
        calmar,
        annual_vol,
        win_rate,
        n_bars: n,
    }
}

pub fn pretty(m: &Metrics) -> String {
    format!(
        "bars: {}\n\
         total_return : {:>8.2}%\n\
         CAGR         : {:>8.2}%\n\
         Sharpe       : {:>8.3}\n\
         Sortino      : {:>8.3}\n\
         Max DD       : {:>8.2}%\n\
         Calmar       : {:>8.3}\n\
         Annual Vol   : {:>8.2}%\n\
         Win Rate     : {:>8.2}%",
        m.n_bars,
        m.total_return * 100.0,
        m.cagr * 100.0,
        m.sharpe,
        m.sortino,
        m.max_drawdown * 100.0,
        m.calmar,
        m.annual_vol * 100.0,
        m.win_rate * 100.0,
    )
}
