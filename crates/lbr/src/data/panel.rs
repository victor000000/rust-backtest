//! Aligned panel of bars: shape (T, N) — T trading dates × N symbols.
//!
//! All vectorized strategy code consumes panels. Dates are the union of dates
//! across symbols; symbols missing on a date carry `NaN`.

use ndarray::Array2;
use std::collections::BTreeSet;
use std::sync::Arc;

use super::Bar;

#[derive(Debug, Clone)]
pub struct Panel {
    pub symbols: Arc<Vec<String>>,
    /// Trading dates as days-since-epoch, length T, monotonically increasing.
    pub dates: Arc<Vec<i32>>,
    pub open: Array2<f64>,
    pub high: Array2<f64>,
    pub low: Array2<f64>,
    pub close: Array2<f64>,
    pub adj_close: Array2<f64>,
    pub volume: Array2<f64>,
}

impl Panel {
    pub fn t(&self) -> usize {
        self.dates.len()
    }
    pub fn n(&self) -> usize {
        self.symbols.len()
    }

    /// Build a panel from per-symbol bar series. Dates are the sorted union;
    /// missing rows are NaN. The series do NOT need to be pre-sorted.
    pub fn from_series(series: Vec<(String, Vec<Bar>)>) -> Self {
        let symbols: Vec<String> = series.iter().map(|(s, _)| s.clone()).collect();
        let mut date_set: BTreeSet<i32> = BTreeSet::new();
        for (_, bars) in &series {
            for b in bars {
                date_set.insert(b.date);
            }
        }
        let dates: Vec<i32> = date_set.into_iter().collect();
        let t = dates.len();
        let n = symbols.len();

        let mut open = Array2::from_elem((t, n), f64::NAN);
        let mut high = Array2::from_elem((t, n), f64::NAN);
        let mut low = Array2::from_elem((t, n), f64::NAN);
        let mut close = Array2::from_elem((t, n), f64::NAN);
        let mut adj = Array2::from_elem((t, n), f64::NAN);
        let mut vol = Array2::from_elem((t, n), f64::NAN);

        let date_index: std::collections::HashMap<i32, usize> =
            dates.iter().enumerate().map(|(i, d)| (*d, i)).collect();

        for (j, (_, bars)) in series.iter().enumerate() {
            for b in bars {
                if let Some(&i) = date_index.get(&b.date) {
                    open[(i, j)] = b.open;
                    high[(i, j)] = b.high;
                    low[(i, j)] = b.low;
                    close[(i, j)] = b.close;
                    adj[(i, j)] = b.adj_close;
                    vol[(i, j)] = b.volume;
                }
            }
        }

        Panel {
            symbols: Arc::new(symbols),
            dates: Arc::new(dates),
            open,
            high,
            low,
            close,
            adj_close: adj,
            volume: vol,
        }
    }
}
