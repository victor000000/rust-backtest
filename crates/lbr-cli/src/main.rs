//! `lbr` CLI — ingest, list, and run vector backtests.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use time::Date;
use time::macros::format_description;

use lbr::config::CostConfig;
use lbr::data::{self, Panel};
use lbr::engine::{VectorEngine, run};
use lbr::metrics::pretty;
use lbr_strategies::{MeanRevCfg, MeanRevStrategy};

#[derive(Parser, Debug)]
#[command(name = "lbr", about = "Local vectorized backtest in Rust")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Pull daily bars from Yahoo and write them to the local cache.
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
    /// Run a backtest from a YAML strategy config.
    Run {
        /// Path to a strategy YAML config.
        #[arg(long)]
        cfg: PathBuf,
        /// Cache directory.
        #[arg(long, default_value = "data/lake")]
        cache: PathBuf,
        /// Starting cash.
        #[arg(long, default_value_t = 70_000.0)]
        cash: f64,
        /// Commission in basis points per side.
        #[arg(long, default_value_t = 0.0)]
        commission_bps: f64,
        /// Slippage in basis points per turnover unit.
        #[arg(long, default_value_t = 0.0)]
        slippage_bps: f64,
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
        Cmd::Run {
            cfg,
            cache,
            cash,
            commission_bps,
            slippage_bps,
        } => {
            let raw =
                std::fs::read_to_string(&cfg).with_context(|| format!("read {}", cfg.display()))?;
            let mr_cfg: MeanRevCfg = serde_yaml::from_str(&raw).context("parsing strategy YAML")?;

            // Load each symbol from cache and align into a panel.
            let mut series = Vec::with_capacity(mr_cfg.symbols.len());
            for sc in &mr_cfg.symbols {
                let bars = data::load_cached(&cache, &sc.symbol)?;
                if bars.is_empty() {
                    eprintln!("warning: no cached bars for {} — skipping", sc.symbol);
                    continue;
                }
                series.push((sc.symbol.clone(), bars));
            }
            if series.is_empty() {
                anyhow::bail!("no symbols with cached data; run `lbr ingest` first");
            }
            let panel = Arc::new(Panel::from_series(series));
            println!(
                "Loaded panel: T={} bars, N={} symbols",
                panel.t(),
                panel.n()
            );

            let engine = VectorEngine::new(
                cash,
                CostConfig {
                    commission_bps,
                    slippage_bps,
                },
            );
            let strategy = MeanRevStrategy::new(mr_cfg);
            let result = run(&engine, panel.as_ref(), &strategy);

            println!("\nStrategy: {}", result.name);
            println!("{}", pretty(&result.metrics));
        }
    }

    Ok(())
}
