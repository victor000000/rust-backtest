//! Yahoo Finance data client. Wraps the `yahoo_finance_api` crate behind a
//! uniform `DataClient`-style fetch function.

use anyhow::{Context, Result};
use time::{Date, OffsetDateTime, Time};

use super::Bar;

/// Fetch daily bars for a single symbol over `[start, end)` (UTC).
pub async fn fetch_bars(symbol: &str, start: Date, end: Date) -> Result<Vec<Bar>> {
    let provider = yahoo_finance_api::YahooConnector::new().context("creating Yahoo connector")?;

    // yahoo_finance_api expects OffsetDateTime, inclusive on both ends.
    let start_dt = OffsetDateTime::new_utc(start, Time::MIDNIGHT);
    let end_dt = OffsetDateTime::new_utc(end, Time::MIDNIGHT);

    let resp = provider
        .get_quote_history(symbol, start_dt, end_dt)
        .await
        .with_context(|| format!("Yahoo fetch failed for {symbol}"))?;
    let quotes = resp
        .quotes()
        .with_context(|| format!("Yahoo decode failed for {symbol}"))?;

    let bars = quotes
        .into_iter()
        .map(|q| {
            let dt = OffsetDateTime::from_unix_timestamp(q.timestamp as i64)
                .expect("yahoo timestamp in range");
            let date = dt.date();
            Bar {
                date: super::date_to_i32(date),
                open: q.open,
                high: q.high,
                low: q.low,
                close: q.close,
                adj_close: q.adjclose,
                volume: q.volume as f64,
            }
        })
        .collect();

    Ok(bars)
}

/// Concurrently fetch a universe of symbols.
pub async fn fetch_universe(
    symbols: &[String],
    start: Date,
    end: Date,
) -> Vec<(String, Result<Vec<Bar>>)> {
    let mut handles = Vec::with_capacity(symbols.len());
    for sym in symbols {
        let sym = sym.clone();
        handles.push(tokio::spawn(async move {
            let bars = fetch_bars(&sym, start, end).await;
            (sym, bars)
        }));
    }
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok(v) => out.push(v),
            Err(e) => out.push(("?".into(), Err(anyhow::anyhow!(e)))),
        }
    }
    out
}
