//! Conversions between lbr Bar series and Nautilus types.

use anyhow::Result;
use rust_decimal::Decimal;

use nautilus_backtest::{
    config::{BacktestEngineConfig, SimulatedVenueConfig},
    engine::BacktestEngine,
};
use nautilus_core::UnixNanos;
use nautilus_model::{
    data::{Bar as NBar, BarSpecification, BarType, Data, QuoteTick},
    enums::{AccountType, AggregationSource, BarAggregation, BookType, OmsType, PriceType},
    identifiers::{InstrumentId, Symbol, Venue},
    instruments::{Equity, InstrumentAny},
    types::{Currency, Money, Price, Quantity},
};
use ustr::Ustr;

use lbr::Bar as LBar;

/// Simulated venue identifier for our backtests.
pub const NAUTILUS_VENUE_ID: &str = "SIM";

/// Days-since-1970 to UnixNanos.
fn date_to_ns(date_days: i32) -> UnixNanos {
    let secs = (date_days as i64) * 86_400;
    UnixNanos::from((secs as u64) * 1_000_000_000)
}

/// Build a Nautilus `Equity` instrument for a US stock/ETF.
pub fn build_equity(symbol: &str) -> InstrumentAny {
    let venue = Venue::from(NAUTILUS_VENUE_ID);
    let sym = Symbol::from(symbol);
    let id = InstrumentId::new(sym, venue);
    let raw_symbol = Symbol::from(symbol);
    let currency = Currency::USD();
    let price_precision: u8 = 4;
    let price_increment = Price::new(0.0001, price_precision);
    let equity = Equity::new(
        id,
        raw_symbol,
        None, // isin
        currency,
        price_precision,
        price_increment,
        None,                 // lot_size
        None,                 // max_qty
        None,                 // min_qty
        None,                 // max_price
        None,                 // min_price
        Some(Decimal::ZERO),  // margin_init
        Some(Decimal::ZERO),  // margin_maint
        Some(Decimal::ZERO),  // maker_fee
        Some(Decimal::ZERO),  // taker_fee
        None,                 // info
        UnixNanos::default(), // ts_event
        UnixNanos::default(), // ts_init
    );
    InstrumentAny::Equity(equity)
}

/// Build a daily-bar `BarType` for an instrument.
pub fn daily_bar_type(instrument_id: InstrumentId) -> BarType {
    let spec = BarSpecification::new(1, BarAggregation::Day, PriceType::Last);
    BarType::new(instrument_id, spec, AggregationSource::External)
}

/// Convert lbr Bar series to Nautilus Bar `Data` vec.
///
/// Uses `close` (not `adj_close`) since Nautilus prefers raw market prices;
/// adjust externally if you want adjusted-price simulations.
pub fn to_nautilus_bars(
    instrument_id: InstrumentId,
    bars: &[LBar],
    price_precision: u8,
    size_precision: u8,
) -> Vec<Data> {
    let bar_type = daily_bar_type(instrument_id);
    let mut out = Vec::with_capacity(bars.len());
    for b in bars {
        if !b.close.is_finite() || b.close <= 0.0 {
            continue;
        }
        let ts = date_to_ns(b.date);
        let open = Price::new(b.open.max(0.0001), price_precision);
        let high = Price::new(b.high.max(0.0001), price_precision);
        let low = Price::new(b.low.max(0.0001), price_precision);
        let close = Price::new(b.close, price_precision);
        let vol = if b.volume.is_finite() && b.volume >= 0.0 {
            b.volume
        } else {
            0.0
        };
        let volume = Quantity::new(vol, size_precision);
        let nbar = NBar::new(bar_type, open, high, low, close, volume, ts, ts);
        out.push(Data::Bar(nbar));
    }
    out
}

/// Convert lbr Bar series → Nautilus `QuoteTick` `Data`, synthesizing tight
/// bid/ask quotes (bid = close, ask = close + tick) from each bar's close.
/// Useful for running Nautilus strategies that subscribe to quotes (like
/// the bundled `EmaCross`).
pub fn to_nautilus_quotes(
    instrument_id: InstrumentId,
    bars: &[LBar],
    price_precision: u8,
    size_precision: u8,
) -> Vec<Data> {
    let tick = 10f64.powi(-(price_precision as i32));
    let mut out = Vec::with_capacity(bars.len());
    for b in bars {
        if !b.close.is_finite() || b.close <= 0.0 {
            continue;
        }
        let ts = date_to_ns(b.date);
        let bid = Price::new(b.close, price_precision);
        let ask = Price::new(b.close + tick, price_precision);
        let size = Quantity::new(100.0, size_precision);
        let q = QuoteTick::new(instrument_id, bid, ask, size, size, ts, ts);
        out.push(Data::Quote(q));
    }
    out
}

/// Build a fresh `BacktestEngine` with a simulated USD venue + given starting
/// cash, configured for cash-account equity trading.
pub fn make_engine(starting_cash_usd: f64) -> Result<BacktestEngine> {
    let mut engine = BacktestEngine::new(BacktestEngineConfig::default())?;
    let venue_cfg = SimulatedVenueConfig::builder()
        .venue(Venue::from(NAUTILUS_VENUE_ID))
        .oms_type(OmsType::Netting)
        .account_type(AccountType::Cash)
        .book_type(BookType::L1_MBP)
        .starting_balances(vec![Money::new(starting_cash_usd, Currency::USD())])
        .build();
    engine.add_venue(venue_cfg)?;
    let _ = Ustr::from("USD");
    Ok(engine)
}
