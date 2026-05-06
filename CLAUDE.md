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

This is a Windows desktop app (tkinter) for technical analysis of ~160 NYSE/NASDAQ stocks and ETFs compatible with the N26 brokerage.

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
