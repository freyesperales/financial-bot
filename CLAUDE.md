# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Launch the GUI app
python investment_report_gui.py

# Download historical data for a symbol (interactive CLI)
python data_downloader.py [SYMBOL]

# Run backtesting optimization (interactive CLI)
python optimize_backtest.py

# Generate a text report (non-GUI)
python generate_investment_report.py

# Consolidated multi-factor screener from the last analysis (non-GUI)
# Default horizon is "6m" (6-month returns: medium-term momentum dominates);
# use --horizon 20d for the short-term timing profile.
python screener.py --cache

# Full headless pipeline: retrain ML (--train) + analyze + screener.
# --horizon 6m trains the model on a 126-trading-day > +8% target with
# 130-day purge; --horizon 20d uses the original 20d > +3% target.
python run_analysis_headless.py --train --horizon 6m

# Bi-weekly report with prediction tracking (PDF via Edge headless + optional email)
python weekly_report.py [--no-ai] [--no-email] [--open]

# Portfolio construction from the screener top (MV Ledoit-Wolf + ERC, VaR, stress)
python build_portfolio.py --horizon 6m

# Batch-download history for new symbols / refresh existing (non-interactive)
python update_universe_data.py --refresh

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (pytest, etc.)
pip install -r requirements-dev.txt

# Run the test suite
pytest -q
```

No build step. The test suite (added in stage 0 of `PROGRESS.md`) lives in `tests/` and uses fixture CSVs in `tests/fixtures/` extracted from `trading_bot.db` on first run.

**Plan en curso:** ver `PROGRESS.md` en la raíz. Etapas 0-5 con casillas `[ ]/[x]`. Antes de tocar nada relevante, leer el estado actual ahí.

## Architecture

This is a Windows desktop app (tkinter) for technical analysis of ~228 NYSE/NASDAQ stocks and ETFs compatible with the N26 brokerage. The universe includes ~40 European companies via US-listed ADRs and Europe-exposure ETFs; `stock_universe.get_region(sym)` returns "Europa" / "USA" / "Canadá".

### Module map

| File | Role |
|---|---|
| `investment_report_gui.py` | Main entry point. Multi-tab dark-theme GUI. Orchestrates all other modules via background threads and a `queue.Queue` for thread-safe UI updates. |
| `stock_analyzer.py` (`StockAnalyzer`) | Core analysis engine. Downloads data, computes weighted technical scores (trend, momentum, volatility, volume, price action, S/R) per symbol. ETFs and bond ETFs use separate weight sets. |
| `stock_universe.py` | Static registry of ~160 symbols split into `STOCK_UNIVERSE` (by sector) and `ETF_UNIVERSE`. Helpers: `get_all_symbols()`, `get_sector(sym)`, `is_etf(sym)`. |
| `indicators_advanced.py` (`AdvancedIndicators`) | Stateless static methods: Fibonacci retracements, Ichimoku Cloud, ADX, Volume Profile, candlestick pattern detection. Used by both `StockAnalyzer` and `Backtester`. |
| `database.py` (`TradingDatabase`) | SQLite wrapper (`trading_bot.db`). Tables: `precios` (OHLCV + indicators), `señales`, `operaciones`, `journal_trades`, `signal_snapshots`, `config`. Instantiated with `check_same_thread=False` for GUI thread safety. |
| `data_downloader.py` (`DataDownloader`) | Downloads via `yfinance`. Stores in DB via `TradingDatabase`. `append_precios()` for incremental updates; `bulk_insert_precios()` replaces all rows for a symbol. |
| `claude_analyzer.py` | Enriches technical analysis with qualitative commentary by calling the Claude Code CLI as a subprocess (`claude -p`). Locates the `claude` executable via `shutil.which` and several known Windows install paths. Runs in a background thread. |
| `backtester.py` (`Backtester`) | Simulates the voting-based entry strategy on historical data from the DB. Requires data to be downloaded first. |
| `optimize_backtest.py` | Interactive CLI that grid-searches strategy parameters using `Backtester`. |
| `generate_investment_report.py` | Non-GUI Markdown report generator. Uses `StockAnalyzer` output. |
| `screener.py` | Consolidated multi-factor screener with horizon profiles (`HORIZON_WEIGHTS`): "6m" (default — medium-term momentum 25%, risk 20%, fundamentals 15%) and "20d" (short-term technical 35%). Merges total_score, momentum_6m, confidence, calibrated ml_prob, fundamentals, RS and risk into one 0-100 `screener_score`, ranked by region (USA/Europa) and type. CLI: `python screener.py --cache [--horizon 6m|20d]`. |
| `update_universe_data.py` | Non-interactive batch downloader: full history for symbols missing from the DB, `--refresh` to top up existing ones. |
| `weekly_report.py` | Bi-weekly PDF report with prediction tracking: snapshots each run's top-20 to `tracking/predictions_log.csv`, evaluates past snapshots vs SPY ("¿se cumplen las predicciones?"), rule-based interpretation + optional AI commentary via `claude -p`, HTML→PDF via Edge headless into `informes/`. Optional SMTP email if `email_config.json` exists (template: `email_config.example.json`). Scheduled via Windows Task Scheduler task **FinancialBot-Informe-6m** (Mon & Fri 12:00, runs `run_informe.bat`: refresh data → analyze → report). |
| `build_portfolio.py` | Converts the screener top-N into portfolio weights: correlation dedup (>0.92), MV (Ledoit-Wolf, score tilt) + ERC with name/sector caps, VaR-95 and stress tests (COVID-2020, Bear-2022). |

### Key data flow

```
yfinance → DataDownloader → TradingDatabase (SQLite)
                                    ↓
                            StockAnalyzer
                           /     |      \
              AdvancedIndicators  |   DataDownloader (incremental fetch)
                                  ↓
                        investment_report_gui.py
                       (queue → UI thread updates)
                                  |
                         ClaudeAnalyzer (subprocess)
```

### Scoring system

`StockAnalyzer` returns a `total_score` (0–100) per symbol. Six sub-scores are computed and merged with configurable weights. The GUI exposes the top results as BUY / NEUTRAL / SELL signals with a confidence label (`ALTA` / `MEDIA` / `BAJA`).

### Threading model

The GUI runs analysis in `threading.Thread` workers. Results are posted to a `queue.Queue` and consumed by a polling loop (`root.after(100, ...)`) on the main tkinter thread. Never call tkinter widgets directly from a worker thread.

### Claude CLI integration

`claude_analyzer.py` shells out to `claude -p "<prompt>"` (the locally installed Claude Code CLI). It does **not** use the Anthropic Python SDK or an API key. If the `claude` executable is not found, the module logs a warning and the GUI continues without AI enrichment.
