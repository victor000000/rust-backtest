//! Data layer: canonical bar schema, Yahoo loader, on-disk cache, aligned panels.

mod cache;
mod panel;
mod yahoo;

pub use cache::{load_cached, save_cached};
pub use panel::Panel;
pub use yahoo::{fetch_bars, fetch_universe};

use serde::{Deserialize, Serialize};

/// Canonical daily bar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bar {
    /// Date as days since UNIX epoch.
    pub date: i32,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Split-and-dividend-adjusted close. Yahoo provides this directly.
    pub adj_close: f64,
    pub volume: f64,
}

pub fn date_to_i32(d: time::Date) -> i32 {
    let epoch = time::Date::from_calendar_date(1970, time::Month::January, 1).unwrap();
    (d - epoch).whole_days() as i32
}

pub fn i32_to_date(d: i32) -> time::Date {
    let epoch = time::Date::from_calendar_date(1970, time::Month::January, 1).unwrap();
    epoch + time::Duration::days(d as i64)
}
