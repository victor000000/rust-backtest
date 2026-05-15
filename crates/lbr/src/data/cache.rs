//! CSV-based on-disk bar cache. One file per `(symbol)`, stored under
//! `<cache_dir>/<symbol>.csv`. Simple and inspectable.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use super::Bar;

fn path_for(cache_dir: &Path, symbol: &str) -> PathBuf {
    cache_dir.join(format!("{}.csv", symbol.to_lowercase()))
}

pub fn save_cached(cache_dir: &Path, symbol: &str, bars: &[Bar]) -> Result<()> {
    fs::create_dir_all(cache_dir).with_context(|| format!("mkdir {}", cache_dir.display()))?;
    let path = path_for(cache_dir, symbol);
    let mut w =
        csv::Writer::from_path(&path).with_context(|| format!("open {}", path.display()))?;
    w.write_record([
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ])?;
    for b in bars {
        w.write_record(&[
            b.date.to_string(),
            b.open.to_string(),
            b.high.to_string(),
            b.low.to_string(),
            b.close.to_string(),
            b.adj_close.to_string(),
            b.volume.to_string(),
        ])?;
    }
    w.flush()?;
    Ok(())
}

pub fn load_cached(cache_dir: &Path, symbol: &str) -> Result<Vec<Bar>> {
    let path = path_for(cache_dir, symbol);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rdr =
        csv::Reader::from_path(&path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        out.push(Bar {
            date: rec[0].parse()?,
            open: rec[1].parse()?,
            high: rec[2].parse()?,
            low: rec[3].parse()?,
            close: rec[4].parse()?,
            adj_close: rec[5].parse()?,
            volume: rec[6].parse()?,
        });
    }
    Ok(out)
}
