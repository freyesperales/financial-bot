# Plan de mejora — estado de progreso

> **Para reanudar tras desconexión:** lee este archivo desde arriba. La sección "Estado actual" indica exactamente dónde retomar. Las casillas `[x]` están hechas, `[ ]` pendientes, `[~]` en curso.

---

## Estado actual

- **Etapa activa:** Etapa 5 — Modelo predictivo ML
- **Etapa activa:** —  TODAS LAS ETAPAS COMPLETADAS (0-5)
- **Última subtarea completada:** 5.5 Pestaña Modelo ML (166 tests verdes)
- **Próximo paso al reanudar:** Mejoras opcionales o nueva etapa

---

## Visión global de etapas

| # | Etapa | Estado | Estimación |
|---|---|---|---|
| 0 | Estabilización y red de seguridad | [x] hecho | 1 semana |
| 1 | Refactor del núcleo analítico | [x] hecho | 2 semanas |
| 2 | Modelado financiero más serio | [x] hecho | 2-3 semanas |
| 3 | Optimización de portafolio | [x] hecho | 2 semanas |
| 4 | Modernización del GUI | [x] hecho | 2 semanas |
| 5 | Modelo predictivo ML (opcional) | [x] hecho | 3-4 semanas |

---

## Etapa 0 — Estabilización (HECHO)

**Objetivo:** poder cambiar código sin romperlo.

- [x] **0.1** Estructura `tests/` con pytest + `conftest.py` con fixture CSV de AAPL, SPY y TLT (extraídos de la DB local).
- [x] **0.2** Tests de regresión sobre `calculate_technical_score`, `calculate_volume_ma_signal`, `calculate_adx_signal`, `calculate_risk_metrics`, `calculate_timeframe_alignment`, `calculate_backtest`, `calculate_signal_confidence`. Tests basados en propiedades, no snapshots ciegos.
- [x] **0.3** Módulo `logging_setup.py` con `RotatingFileHandler`, nivel configurable por env var (`FINBOT_LOG_LEVEL`), archivo en `logs/finbot.log`. Integrado en `database.py`.
- [x] **0.4** En `database.py`:
  - `PRAGMA journal_mode=WAL`
  - `PRAGMA synchronous=NORMAL`
  - `cleanup_old_snapshots(keep=20)` invocado automáticamente al iniciar.
- [x] **0.5** Añadir `pytest`, `pytest-cov` a `requirements-dev.txt` + `pytest.ini`.
- [x] **0.6** Documentado en `CLAUDE.md` el comando `pytest -q`.

**Criterio de éxito:** ✅ `pytest` pasa con 30 tests (mín. requerido: 10). Cambiar un peso del score altera el total y los tests de scoring lo detectan.

---

## Etapa 1 — Refactor del núcleo analítico

- [x] **1.1** `indicators_core.py` con funciones puras (`compute_emas`, `compute_rsi`, `compute_macd`, `compute_bbands`, `compute_atr`, `compute_adx`, `compute_volume_ma`).
- [x] **1.2** Refactor `stock_analyzer.py` y `backtester.py` para consumirlo. Eliminar duplicados.
- [x] **1.3** `scoring_service.py` con `compute_subscores`, `compute_total_score`, `compute_confidence`.
- [x] **1.4** Mover `_compute_buy_score`, `_compute_dip_score`, `_compute_bigdip_score`, `_compute_bond_score` desde el GUI a `scoring_service.py`.
- [x] **1.5** Suavizar cliffs del score con función continua (`_piecewise` + `_lerp`).
- [x] **1.6** Caché en disco para `yfinance.Ticker(...).info` con TTL 24 h en `cache/fundamentals/`.
- [x] **1.7** Arreglar backtest interno (`stock_analyzer.calculate_backtest`):
  - `in_position` state machine, sin solapamiento.
  - Cierre SL/TP intra-vela usando high/low. ATR-based SL/TP, fallback 2%/3%.
  - Métricas adicionales: Sortino, Calmar, exposure %, avg_hold_days.
- [x] **1.8** Migrar `last_analysis.pkl` → parquet. Backward compat: carga pkl legado y lo migra. `pyarrow` añadido a requirements.txt.

---

## Etapa 2 — Modelado financiero

- [x] **2.1** `market_regime.py` — régimen risk_on/transition/risk_off/panic. SMA200, slope, ADX, VIX. Caché 1h. `compute_regime(spy_df, vix)` sin I/O para tests.
- [x] **2.2** Pesos adaptativos por régimen. `REGIME_WEIGHTS` + `REGIME_WEIGHTS_ETF` en `scoring_service.py`. `get_weights(regime, etf)`. `StockAnalyzer._current_regime` actualizado en `analyze_all_stocks`.
- [x] **2.3** `walk_forward_backtest()` en `StockAnalyzer` — IS 3 años, OOS 6 meses rolling. Devuelve folds, IS/OOS win_rate, expectancy y ratio de degradación.
- [x] **2.4** Métricas adicionales en `calculate_risk_metrics`: Sortino, Calmar, downside deviation, Beta, Correlación 60d, Information Ratio (estos tres requieren `spy_rets`). ⚠️ Alineación posicional (no por fecha).
- [x] **2.5** Kelly Half-Kelly via block bootstrap (100 iter, block 10d) en `_kelly_block_bootstrap`. Devuelve mediana + IC P5/P95. Cap al 25%.
- [x] **2.6** Cap combinado RS + multi-TF en `compute_confidence` (máximo ±2). Evita inflar el score cuando dos señales correladas coinciden.
- [x] **2.7** `annotate_sector_vol_zscores(results)` — z-score sectorial del vol_ratio. Llamado al final de `analyze_all_stocks()`.

---

## Etapa 3 — Optimización de portafolio

- [x] **3.1** `portfolio_optimizer.py` (Mean-Variance Ledoit-Wolf + ERC) con restricciones (max % nombre, sector, beta).
- [x] **3.2** Rebalance simulado semanal vs mensual con costes 5 bps. (`simulate_rebalance`)
- [x] **3.3** Risk budgeting: VaR-95 1d objetivo 1.5 %. (`portfolio_var`, `suggest_var_budget`)
- [x] **3.4** Trade Journal extendido: `fees_eur`, `slippage_bps`, `take_profit`, `pnl_net_eur`, `mae_pct`. `journal_update_mae`, `journal_get_stats`. Migración idempotente.
- [x] **3.5** Stress tests (Mar-2020, Oct-2022). (`stress_test`)
- [x] **3.6** Alertas inteligentes: `_detect_smart_alerts` — stop técnico (SL próximo/alcanzado), tendencia semanal bajista, deterioro fundamental, caída de score > 15 pts. Popup unificado con severidad.

---

## Etapa 4 — Modernización GUI

- [x] **4.1** Partir `investment_report_gui.py` por pestaña en `gui/tabs/*.py`. Paquete `gui/` creado. Mixins: `JournalTabMixin`, `AlertsMixin`, `PortfolioRiskTabMixin`. Monolito: 6229 → 5411 líneas (−818).
- [x] **4.2** `gui/data_controller.py` — `DataController` con métodos tipados: `get_top_buys`, `get_risk_summary`, `get_portfolio_weights_df`, `get_sector_summary`, `summary_stats`. Integrado en `_run_analysis`.
- [x] **4.3** Reemplazar polling `root.after(50)` por `event_generate('<<QueueUpdate>>')` desde workers + fallback 500ms. `_q_notify()` en todos los `q.put` finales de workers.
- [x] **4.4** Vista "Riesgo del portafolio" con gráficos (equity curve, drawdown, beta rolling). Tab `PortfolioRiskTabMixin` en `gui/tabs/portfolio_risk_tab.py`.

---

## Etapa 5 — Modelo predictivo ML (opcional)

- [x] **5.1** Dataset (sub-scores + macro + sector relativo + fundamentos lagged) y target retorno 20d > +3 %.
- [x] **5.2** LightGBM con early_stopping y purged CV (López de Prado).
- [x] **5.3** Comparar Sharpe OOS modelo vs reglas.
- [x] **5.4** Integrar como señal extra (peso máx +1).
- [x] **5.5** Pestaña SHAP / feature importance.

---

## Bitácora de cambios

> Anotar fecha (YYYY-MM-DD) y resumen en una línea por cambio aplicado.

- 2026-05-06 — Documento `PROGRESS.md` creado. Plan completo registrado.
- 2026-05-06 — Etapa 5 COMPLETA (5.5):
  - `gui/tabs/ml_tab.py`: `MLTabMixin` con pestaña "Modelo ML".
  - Header con status del modelo (AUC, iter, features).
  - Botones: "Entrenar Modelo" (bg thread → build_dataset + train_model + save) y "Comparar Sharpe OOS".
  - Gráfico de feature importance (matplotlib, top 20, gain normalizado).
  - Tabla CV por fold (Fold / Train / Test / AUC / Iter).
  - Panel comparación: ML vs Reglas vs Buy&Hold (Sharpe, win_rate, avg_return, max_dd).
  - Integrado en InvestmentReportGUI: import, herencia, frame, tab, _build_ml_tab().
  - 166 tests verdes, import OK.
- 2026-05-06 — Etapa 5.4 completa:
  - `ml_signal.py`: `_result_to_features`, `_annotate_rs_sector`, `enrich_ml_signals`, `invalidate_model_cache`.
  - `scoring_service.compute_confidence`: sección ML lee `ml_prob` del result row → ±1 pt con labels en aligned/against.
  - `stock_analyzer.analyze_all_stocks`: llama `enrich_ml_signals` post `annotate_sector_vol_zscores`; re-calcula confidence con señal ML.
  - 20 tests nuevos. 166 tests verdes en total.
- 2026-05-06 — Etapa 5.3 completa:
  - `ml_compare.py`: `_portfolio_metrics`, `_simulate_strategy`, `compare_strategies`, `find_optimal_threshold`, `print_comparison`.
  - Simula estrategia ML vs reglas (total_score) vs buy-and-hold en el mismo período OOS.
  - Métricas: Sharpe, Sortino, win_rate, avg_return, ann_return, ann_vol, max_drawdown.
  - `find_optimal_threshold` barre umbrales ML para calibrar punto de entrada.
  - 25 tests nuevos. 146 tests verdes en total.
- 2026-05-06 — Etapa 5.2 completa:
  - `ml_model.py`: `PurgedWalkForwardCV` (purge + embargo), `train_model` con early stopping, `evaluate_oos`, `predict_proba`, `feature_importance`, `save_model`, `load_model`.
  - LightGBM 4.6, sklearn 1.6. Añadidos a requirements.txt.
  - 24 tests nuevos. 121 tests verdes en total.
- 2026-05-06 — Etapa 5.1 completa:
  - `ml_dataset.py`: `build_dataset`, `load_or_build_dataset`, `_features_at_row`, `_build_spy_regime_map`, `_add_sector_features`, `_load_fund_features`.
  - Features: 6 subscores + total_score + 8 indicadores raw + 4 macro SPY + 2 sector-relativas + 5 fundamentales lagged.
  - Target: retorno forward 20d > 3% (binario).
  - Caché parquet en `cache/ml_dataset.parquet`.
  - 20 tests nuevos. 97 tests verdes en total.
- 2026-05-06 — Etapa 4 completa (4.1-4.4):
  - `gui/` paquete + `gui/tabs/`: `JournalTabMixin`, `AlertsMixin`, `PortfolioRiskTabMixin`. Monolito: 6229 → 5411 líneas.
  - `gui/data_controller.py`: DataController con métodos tipados, actualizado en cada análisis.
  - Threading: `_q_notify()` + `event_generate('<<QueueUpdate>>')` sustituye polling 50ms; fallback 500ms.
  - 77 tests verdes, import OK.
- 2026-05-06 — Etapa 3 completa (3.4, 3.6):
  - `database.py`: journal extendido (fees, take_profit, pnl_net, MAE), migración idempotente, `journal_update_mae`, `journal_get_stats`, fix numpy.int64 binding.
  - `investment_report_gui.py`: formulario con TP y comisión de entrada/salida, tabla abiertas/cerradas con MAE y P&L neto, stats de historial, `_detect_smart_alerts`, popup unificado con severidad.
  - 77 tests verdes (12 nuevos en `test_journal_extended.py`).
- 2026-05-06 — Etapa 3 parcial (3.1, 3.2, 3.3, 3.5):
  - `portfolio_optimizer.py` nuevo modulo con optimize_mv (Ledoit-Wolf), optimize_erc, simulate_rebalance, portfolio_var, suggest_var_budget, stress_test.
  - 33 tests nuevos + 65 verdes en total. Pendiente: 3.4 (Trade Journal), 3.6 (alertas).
- 2026-05-06 — Etapa 2 completa (2.1-2.7):
  - `walk_forward_backtest()` IS/OOS rolling.
  - Kelly block bootstrap con IC P5/P95.
  - `annotate_sector_vol_zscores()` — vol z-score por sector.
  - 32 tests verdes.
- 2026-05-06 — Etapa 2 parcial (2.1, 2.2, 2.4, 2.6):
  - `market_regime.py` (detector con caché 1h, testable sin red).
  - `scoring_service.get_weights(regime, etf)` + integración en `StockAnalyzer`.
  - `calculate_risk_metrics`: Sortino, Calmar, Beta, Corr60d, IR.
  - Cap RS+TF en `compute_confidence` (doble-conteo eliminado).
  - Test `test_weight_change_affects_total_score` actualizado para régimen.
  - 31 tests verdes.
- 2026-05-06 — Etapa 1 completa al 100%:
  - `indicators_core.py` (fuente única de indicadores).
  - `scoring_service.py` (scoring desacoplado del GUI, `_piecewise` para suavizar cliffs).
  - `fundamentals_cache.py` (TTL 24h, fallback a caché expirada).
  - `stock_analyzer.calculate_backtest` reescrito con state machine (sin solapamiento), SL/TP intra-vela ATR-based, Sortino, Calmar, exposure %, avg_hold_days.
  - `investment_report_gui.py`: pkl → parquet, nuevas claves bt_* expuestas.
  - 31 tests verdes.
- 2026-05-06 — Etapa 0 completa:
  - `logging_setup.py` (logger central con rotación).
  - `database.py`: WAL + synchronous=NORMAL + cleanup automático de snapshots.
  - `tests/` con 30 tests verdes (DB, indicadores, scoring, logging).
  - `requirements-dev.txt`, `pytest.ini`, fixtures CSV extraídos de la DB.
  - `CLAUDE.md` actualizado con instrucciones de tests y referencia a `PROGRESS.md`.
