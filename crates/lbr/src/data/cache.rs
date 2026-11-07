//! On-disk bar cache, written as Parquet via polars.
//!
//! Each symbol gets one file: `<cache_dir>/<symbol>.parquet`. Columns match
//! the canonical `Bar` shape (date, open, high, low, close, adj_close,
//! volume). Polars' Parquet reader/writer is fast and gives us LazyFrame
//! access for downstream pipelines.

use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::Bar;

fn path_for(cache_dir: &Path, symbol: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        !symbol.trim().is_empty()
            && symbol != "."
            && symbol != ".."
            && !symbol
                .chars()
                .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':')),
        "symbol must be a nonempty filename without path separators or control characters"
    );
    Ok(cache_dir.join(format!("{}.parquet", symbol.to_lowercase())))
}

fn bars_to_df(bars: &[Bar]) -> Result<DataFrame> {
    let date: Vec<i32> = bars.iter().map(|b| b.date).collect();
    let open: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let adj: Vec<f64> = bars.iter().map(|b| b.adj_close).collect();
    let vol: Vec<f64> = bars.iter().map(|b| b.volume).collect();

    let df = df! {
        "date" => &date,
        "open" => &open,
        "high" => &high,
        "low" => &low,
        "close" => &close,
        "adj_close" => &adj,
        "volume" => &vol,
    }
    .context("building DataFrame from bars")?;
    Ok(df)
}

fn df_to_bars(df: &DataFrame) -> Result<Vec<Bar>> {
    let n = df.height();
    let date = df.column("date")?.i32()?;
    let open = df.column("open")?.f64()?;
    let high = df.column("high")?.f64()?;
    let low = df.column("low")?.f64()?;
    let close = df.column("close")?.f64()?;
    let adj = df.column("adj_close")?.f64()?;
    let vol = df.column("volume")?.f64()?;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(Bar {
            date: date.get(i).unwrap_or(0),
            open: open.get(i).unwrap_or(f64::NAN),
            high: high.get(i).unwrap_or(f64::NAN),
            low: low.get(i).unwrap_or(f64::NAN),
            close: close.get(i).unwrap_or(f64::NAN),
            adj_close: adj.get(i).unwrap_or(f64::NAN),
            volume: vol.get(i).unwrap_or(f64::NAN),
        });
    }
    Ok(out)
}

pub fn save_cached(cache_dir: &Path, symbol: &str, bars: &[Bar]) -> Result<()> {
    let path = path_for(cache_dir, symbol)?;
    std::fs::create_dir_all(cache_dir).with_context(|| format!("mkdir {}", cache_dir.display()))?;
    let mut df = bars_to_df(bars)?;
    let mut f = File::create(&path).with_context(|| format!("create {}", path.display()))?;
    ParquetWriter::new(&mut f)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df)
        .with_context(|| format!("writing Parquet {}", path.display()))?;
    Ok(())
}

pub fn load_cached(cache_dir: &Path, symbol: &str) -> Result<Vec<Bar>> {
    let path = path_for(cache_dir, symbol)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let df = ParquetReader::new(f)
        .finish()
        .with_context(|| format!("reading Parquet {}", path.display()))?;
    df_to_bars(&df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_paths_stay_in_cache() {
        let root = Path::new("cache");
        for symbol in ["SPY", "^GSPC", "BRK.B", "BRK-B", "EURUSD=X"] {
            assert_eq!(path_for(root, symbol).unwrap().parent(), Some(root));
        }
        for symbol in [
            "",
            " ",
            ".",
            "..",
            "../escape",
            "/tmp/escape",
            "a/b",
            "a\\b",
            "C:escape",
            "bad\0",
        ] {
            assert!(path_for(root, symbol).is_err(), "{symbol:?}");
        }
    }

    #[test]
    fn invalid_symbol_fails_before_io() {
        let root = std::env::temp_dir().join(format!("lbr-invalid-symbol-{}", std::process::id()));
        assert!(!root.exists());
        assert!(save_cached(&root, "../escape", &[]).is_err());
        assert!(load_cached(&root, "../escape").is_err());
        assert!(!root.exists());
    }
}
