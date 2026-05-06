# Financial BOT

> Analizador técnico de acciones para la cartera N26 — escritorio Windows con IA integrada

Desktop app en Python/Tkinter para análisis técnico de ~160 acciones y ETFs de NYSE/NASDAQ compatibles con la corredora N26. Combina un sistema de scoring multifactorial, optimización de portafolio, backtesting y un modelo predictivo LightGBM.

---

## Características principales

### Análisis técnico
- **Score 0–100** por símbolo: tendencia, momentum, volatilidad, volumen, price action y soporte/resistencia
- **Confianza combinada** (MUY ALTA → MUY BAJA): integra 10+ señales incluyendo ADX, RSI, patrones de velas, fundamentales, multi-timeframe y Fuerza Relativa vs S&P500
- **Régimen de mercado** automático (risk-on / transition / risk-off / panic) que ajusta los pesos del scoring
- **Fundamentos lagged** via yfinance con caché 24h (P/E, PEG, márgenes, deuda)

### Portafolio y riesgo
- **Optimización Mean-Variance** (Ledoit-Wolf) y **Equal Risk Contribution (ERC)**
- **Kelly Half-Kelly** via block bootstrap con IC P5/P95
- **VaR-95**, CVaR, stress tests (COVID Mar-2020, Bear Oct-2022)
- **Walk-forward backtest** IS/OOS rolling (3 años IS, 6 meses OOS)
- **Trade Journal** con MAE, fees, slippage, P&L neto y estadísticas de historial

### Modelo ML (LightGBM)
- Dataset con 27 features: subscores técnicos + macro SPY + sector-relativas + fundamentales lagged
- **Purged Walk-Forward CV** (López de Prado, Cap. 7) — sin lookahead en datos financieros de panel
- Señal ML integrada como ±1 punto en el scoring de confianza
- Comparación Sharpe OOS: modelo vs reglas vs buy-and-hold
- Pestaña dedicada con gráfico de feature importance y métricas CV por fold

### GUI
- Dark theme multi-tab (tkinter)
- Pestañas: Dashboard · ¿Qué Comprar? · Comprar en Dip · Predicción · Portfolio · Mis Inversiones · Riesgo · Modelo ML · Análisis Técnico · Gráfico · Análisis IA
- Alertas inteligentes: stop técnico, deterioro fundamental, caída de score >15 pts
- Enriquecimiento con Claude AI (requiere Claude Code CLI instalado)
- Threading con `event_generate('<<QueueUpdate>>')` — sin polling agresivo

---

## Stack técnico

| Capa | Tecnología |
|---|---|
| GUI | Python 3.13 · Tkinter · Matplotlib |
| Datos | yfinance · SQLite (WAL) · pyarrow/parquet |
| Indicadores | pandas-ta · NumPy |
| Portafolio | scikit-learn (Ledoit-Wolf) · scipy (SLSQP) |
| ML | LightGBM 4.x · scikit-learn |
| Testing | pytest · 166 tests |

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/freyesperales/financial-bot.git
cd financial-bot

# 2. Entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Dependencias
pip install -r requirements.txt

# 4. Dependencias de desarrollo (tests)
pip install -r requirements-dev.txt
```

### Requisito previo: datos históricos

La app lee de una base SQLite local (`trading_bot.db`) que **no se incluye** en el repo.
Para generarla, descarga datos con yfinance antes del primer análisis:

```bash
python data_downloader.py AAPL    # descarga un símbolo
# — o desde el GUI: botón "Actualizar Datos"
```

---

## Uso

```bash
# Lanzar la app
python investment_report_gui.py

# Ejecutar tests
pytest -q

# Generar reporte de texto (sin GUI)
python generate_investment_report.py
```

### Flujo recomendado

1. **Actualizar datos** — botón en GUI o `data_downloader.py`
2. **Analizar** — escanea los ~160 símbolos y rellena todas las pestañas
3. **Revisar** — Dashboard → ¿Qué Comprar? → Riesgo del Portafolio
4. **Modelo ML** (opcional) — pestaña "Modelo ML" → `Entrenar Modelo` → `Comparar Sharpe OOS`

---

## Arquitectura

```
yfinance → DataDownloader → trading_bot.db (SQLite)
                                   ↓
                           StockAnalyzer
                          /      |       \
             indicators_core  scoring_service  market_regime
                                   ↓
                       investment_report_gui.py
                      (queue → UI thread · mixins por pestaña)
                                   |
                    ┌──────────────┼──────────────┐
              portfolio_optimizer  ml_model    claude_analyzer
```

### Módulos principales

| Módulo | Rol |
|---|---|
| `stock_analyzer.py` | Motor de análisis: descarga, indicadores, scoring, backtest |
| `scoring_service.py` | Subscores técnicos, pesos adaptativos por régimen, confianza |
| `indicators_core.py` | Indicadores puros (EMA, RSI, MACD, BB, ATR, ADX, Vol MA) |
| `portfolio_optimizer.py` | MV, ERC, simulate_rebalance, VaR, stress_test |
| `ml_dataset.py` | Dataset builder para el modelo ML con features lagged |
| `ml_model.py` | PurgedWalkForwardCV + LightGBM + persistencia |
| `ml_compare.py` | Comparación Sharpe OOS modelo vs reglas |
| `ml_signal.py` | Enriquece result rows con ml_prob en batch |
| `market_regime.py` | Detector de régimen con caché 1h |
| `database.py` | SQLite WAL · journal de trades · snapshots de señales |
| `gui/tabs/` | Mixins por pestaña: Journal, Alerts, PortfolioRisk, ML |

---

## Tests

```bash
pytest -q                     # suite completa (166 tests)
pytest tests/test_ml_model.py # solo ML
pytest --cov=. --cov-report=term-missing
```

Los fixtures CSV (AAPL, SPY, TLT) se generan automáticamente desde la DB en el primer `pytest` si ésta existe, o se usan los incluidos en `tests/fixtures/`.

---

## Roadmap completado

| Etapa | Descripción |
|---|---|
| 0 | Estabilización: tests, logging, WAL, fixtures |
| 1 | Refactor núcleo: indicators_core, scoring_service, caché fundamentales |
| 2 | Modelado financiero: walk-forward, Kelly bootstrap, régimen, métricas riesgo |
| 3 | Optimización portafolio: MV/ERC, VaR, journal extendido, alertas inteligentes |
| 4 | GUI modernización: mixins, DataController, event-driven threading |
| 5 | Modelo predictivo ML: LightGBM + purged CV + integración en scoring |

---

## Notas

- Compatible con las acciones y ETFs disponibles en **N26** (NYSE/NASDAQ)
- La integración con **Claude AI** requiere el [Claude Code CLI](https://claude.ai/code) instalado localmente
- La base de datos (`trading_bot.db`) no se versiona — contiene datos personales del journal de trades
- Los modelos ML entrenados se guardan en `cache/ml/` (excluido del repo)

---

*Proyecto personal de análisis financiero — no constituye asesoramiento de inversión.*
