//! `lbr` CLI — strategy-agnostic data driver for the platform.
//!
//! The public CLI handles data ingest and panel inspection. Running a
//! strategy requires a separate binary that depends on `lbr` and a private
//! strategy crate of your own — the platform stays strategy-free.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use time::Date;
use time::macros::format_description;

use lbr::data::{self, Panel};

#[derive(Parser, Debug)]
#[command(
    name = "lbr",
    about = "Local vectorized backtest in Rust — data driver"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Pull daily bars from Yahoo Finance and cache them locally.
    Ingest {
        /// Symbols, comma-separated (e.g. SPY,QQQ,IWM).
        #[arg(long)]
        symbols: String,
        /// Inclusive start date (YYYY-MM-DD).
        #[arg(long)]
        start: String,
        /// Exclusive end date (YYYY-MM-DD).
        #[arg(long)]
        end: String,
        /// Cache directory.
        #[arg(long, default_value = "data/lake")]
        cache: PathBuf,
    },
    /// Inspect a cached panel: rows, columns, date range.
    Show {
        /// Symbols, comma-separated (e.g. SPY,QQQ,IWM).
        #[arg(long)]
        symbols: String,
        /// Cache directory.
        #[arg(long, default_value = "data/lake")]
        cache: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest {
            symbols,
            start,
            end,
            cache,
        } => {
            let fmt = format_description!("[year]-[month]-[day]");
            let start = Date::parse(&start, &fmt).context("parsing --start")?;
            let end = Date::parse(&end, &fmt).context("parsing --end")?;
            let syms: Vec<String> = symbols
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            println!(
                "Fetching {} symbols from Yahoo: {start} → {end}",
                syms.len()
            );
            let results = data::fetch_universe(&syms, start, end).await;
            for (sym, res) in results {
                match res {
                    Ok(bars) => {
                        data::save_cached(&cache, &sym, &bars)?;
                        println!(
                            "  {:>6}  {:>5} bars  → {}/{}.csv",
                            sym,
                            bars.len(),
                            cache.display(),
                            sym.to_lowercase()
                        );
                    }
                    Err(e) => println!("  {:>6}  ERROR: {e:#}", sym),
                }
            }
        }
        Cmd::Show { symbols, cache } => {
            let syms: Vec<String> = symbols
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let mut series = Vec::with_capacity(syms.len());
            for sym in &syms {
                let bars = data::load_cached(&cache, sym)?;
                if bars.is_empty() {
                    eprintln!("warning: no cached bars for {sym} — skipping");
                    continue;
                }
                series.push((sym.clone(), bars));
            }
            if series.is_empty() {
                anyhow::bail!("no symbols with cached data; run `lbr ingest` first");
            }
            let panel = Arc::new(Panel::from_series(series));
            let first = data::i32_to_date(panel.dates[0]);
            let last = data::i32_to_date(panel.dates[panel.t() - 1]);
            println!("Panel: T={} bars  N={} symbols", panel.t(), panel.n());
            println!("Dates: {first} → {last}");
            println!("Symbols: {:?}", panel.symbols);
        }
    }

    Ok(())
}
