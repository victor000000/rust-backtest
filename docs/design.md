# lbr — System Design

A local, vectorized, parallel backtesting platform in Rust.

## Goals

- **Vectorized research** — fast parameter sweeps across historical panels.
- **Event-driven engine (later)** — same code path for backtest, paper, and live.
- **Multi-source data** — Yahoo Finance for v0.1; Alpaca planned for live.
- **Local-first** — CSV/Parquet on disk, no cloud dependency for research.
- **Validation target** — reproduce a known QuantConnect strategy ("S002")
  performance shape (CAGR, Sharpe, MaxDD) within reasonable accuracy.

## Non-goals (v0.1)

- Tick / sub-daily data (daily only).
- Live execution (Alpaca adapter is v0.2).
- Multi-currency, options, futures.
- Survivorship-bias correction (delisted symbols).
- Pipeline DAG, Strategy Tree, OrderTicket state machine
  (these are documented in `design-extended.md` as v2+ ideas).

## Architecture

```
═══════════════════════════════════════════════════════════════════════════
                          EXTERNAL VENUES
═══════════════════════════════════════════════════════════════════════════
                       ┌───────────────────┐
                       │  Yahoo Finance    │     (Alpaca planned in v0.2)
                       │  data only        │
                       └─────────┬─────────┘
                                 │
═══════════════════════════════════════════════════════════════════════════
                          ADAPTER LAYER
═══════════════════════════════════════════════════════════════════════════
                       ┌─────────────────┐
                       │  Yahoo Adapter  │   (DataClient — uniform fetch API)
                       └────────┬────────┘
                                ▼
═══════════════════════════════════════════════════════════════════════════
                          DATA LAYER
═══════════════════════════════════════════════════════════════════════════
                       ┌──────────────────┐
                       │   Bar Cache      │  data/lake/<symbol>.csv
                       │   (CSV / Parquet)│
                       └────────┬─────────┘
                                ▼
                       ┌──────────────────┐
                       │   Panel (T × N)  │  ndarray, Arc-shared
                       │   open/high/low/ │
                       │   close/adj/vol  │
                       └────────┬─────────┘
                                │
═══════════════════════════════════════════════════════════════════════════
                          ENGINE (vector mode)
═══════════════════════════════════════════════════════════════════════════
   ┌──────────────────────────────────────────────────────────────────┐
   │  Sweep Runner (Rayon)                                            │
   │    ├── strategy 1 ┐                                              │
   │    ├── strategy 2 ┼─► panel ──► weights ──► returns ──► equity   │
   │    └── strategy N ┘                                              │
   │                                                                  │
   │  Per-job pipeline:                                               │
   │    Strategy::target_weights(panel) ──► W (T,N)                   │
   │    returns_axis0(prices)             ──► R (T,N)                 │
   │    pnl[i]    = Σ_j W[i-1,j] · R[i,j]                             │
   │    turnover  = Σ_j |W[i,j] − W[i-1,j]|                           │
   │    cost      = turnover · (commission_bps + slippage_bps)/10_000 │
   │    equity[i] = equity[i-1] · (1 + pnl[i] − cost[i])              │
   └────────────────────────┬─────────────────────────────────────────┘
                            ▼
═══════════════════════════════════════════════════════════════════════════
                          SHARED CORE
═══════════════════════════════════════════════════════════════════════════
   ┌──────────────────────────────────────────────────────────────────┐
   │  Strategy trait │ Cost Config │ Metrics                          │
   │  Indicators (sma · ema · wwma · rsi · zscore · returns · std)    │
   └────────────────────────┬─────────────────────────────────────────┘
                            ▼
═══════════════════════════════════════════════════════════════════════════
                          OUTPUT
═══════════════════════════════════════════════════════════════════════════
   ┌──────────────────────────────────────────────────────────────────┐
   │  Equity curve · Metrics (Sharpe, Sortino, CAGR, MaxDD, Calmar)   │
   │  Pretty CLI report                                               │
   └──────────────────────────────────────────────────────────────────┘
```

## Module map

| Module | Role |
|---|---|
| `data::yahoo` | Wraps `yahoo_finance_api`; uniform async `fetch_bars`. |
| `data::cache` | CSV cache under `data/lake/`. Trivially inspectable. |
| `data::panel` | Aligned `(T, N)` panel; symbol union, date intersection. |
| `indicators` | Vectorized SMA, EMA, WWMA, RSI, Z-score, returns, rolling std. |
| `strategy` | `Strategy` trait; reusable `MeanRevStrategy` framework. |
| `engine::vector` | Lag-1 weight × returns simulator; Rayon parallel runner. |
| `metrics` | CAGR · Sharpe · Sortino · MaxDD · Calmar · Win-Rate. |
| `config` | Serde-loadable backtest, data, and cost configs. |

## Strategy interface

```rust
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn target_weights(&self, panel: &Panel) -> Array2<f64>;
}
```

One trait, one method. The engine handles the lag, returns, costs, and the
equity curve. The strategy is responsible only for producing target weights
from a panel.

## Mean-reversion framework (`MeanRevStrategy`)

A generic, per-symbol mean-reversion engine. Symbol config carries:

| Field | Meaning |
|---|---|
| `ind`, `p` | Indicator family (`wwma`/`sma`/`ema`/`rsi`/`zscore`) and period. |
| `wt` | Target weight when invested. |
| `e`, `x` | Entry/exit thresholds for oscillators. |
| `ef`, `xf` | Entry/exit factors for MA-like (price < val·ef → enter). |
| `sl`, `tp` | Stop-loss / take-profit (% of entry price). |
| `sma_p`, `cf`, `tf_t` | Trend filter: SMA period + below/above factors. |

Gross exposure is capped per-row. SL/TP are applied at the *next* bar close
based on the running entry price.

Tuned configurations (e.g. S002's per-ETF table) are loaded externally and
are deliberately **not** part of this open-source library.

## Parallelism

- **Outer parallelism (primary):** Rayon over independent strategy/parameter
  jobs. The panel is shared as `Arc<Panel>` — loaded once, read-only.
- **Inner parallelism (later):** ndarray ops can be parallelized per-column
  when universes grow.

## Determinism

- All strategy decisions are pure functions of the panel and the config.
- No RNG in the engine.
- Caches are content-addressable by symbol+date range; deleting the cache
  reproduces the run.

## Validation plan

1. Ingest the S002 ETF universe from Yahoo for the strategy's date range.
2. Load the (private) tuned per-ETF config.
3. Run the vector engine.
4. Compare: CAGR, Sharpe, MaxDD, Win-Rate against the QC reference.

A perfect match is not expected for v0.1 because:
- Yahoo daily data differs from QC's adjusted feeds in ways that affect
  thinly traded ETFs.
- The vector model applies fills at close; QC fills at the open of the next
  session via a scheduled pre-market routine.
- Some indicators in S002 (e.g. KAMA, FRAMA, TRIX, NATR) are not yet
  implemented — v0.1 covers WWMA/SMA/EMA/RSI/Z-score.

The bar is: directional agreement on Sharpe/CAGR, with documented gaps.

## CI

GitHub Actions: `check`, `clippy -D warnings`, `fmt --check`, `test` on
stable. PRs and pushes to `main` trigger the workflow.

## Repo discipline

- Public: framework code, public example, CI, README, this design doc.
- Private (gitignored): tuned strategy configs (`configs/s002*.yaml`),
  any private example referencing them, the Parquet/CSV data lake, results.

## Future (v0.2+)

- Alpaca data + execution adapters; event-driven engine; live mode.
- Parquet cache (replacing CSV).
- Pipeline DAG for factor research.
- Walk-forward / CV folds as first-class objects.
- Strategy Tree composability.
- OrderTicket state machine + brokerage capability flags.
