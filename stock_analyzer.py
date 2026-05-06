"""
Analizador Multi-Acción
Analiza 50 acciones, calcula indicadores técnicos ponderados,
correlaciones y genera recomendaciones de inversión
"""
import pandas as pd
import numpy as np
from database import TradingDatabase
from data_downloader import DataDownloader
from stock_universe import STOCK_UNIVERSE, ETF_UNIVERSE, get_all_symbols, get_sector, is_etf
import pandas_ta as ta
from datetime import datetime, timedelta
from indicators_advanced import AdvancedIndicators
import indicators_core as ic
from fundamentals_cache import get_info as _get_fund_info
from logging_setup import get_logger
import market_regime as _mr
from scoring_service import get_weights as _get_regime_weights

log = get_logger(__name__)


def _kelly_block_bootstrap(
    close_vals: np.ndarray,
    n_hold: int = 20,
    block_size: int = 10,
    n_iter: int = 100,
) -> tuple:
    """
    Estima Half-Kelly mediante block bootstrap, preservando autocorrelación.

    Devuelve (median, p5, p95, median_win_rate_pct).
    Fallback conservador (0.05, 0.01, 0.15, None) si no hay suficientes datos.
    """
    rets = np.diff(close_vals) / np.where(close_vals[:-1] != 0, close_vals[:-1], 1)
    n = len(rets)
    if n < n_hold + 5:
        return 0.05, 0.01, 0.15, None

    max_start  = max(1, n - block_size + 1)
    n_blocks   = (n // block_size) + 2
    rng        = np.random.default_rng(42)
    all_starts = rng.integers(0, max_start, size=(n_iter, n_blocks))

    kelly_samples: list = []
    wr_samples:    list = []

    for i in range(n_iter):
        blocks  = [rets[s: s + block_size] for s in all_starts[i]]
        sampled = np.concatenate(blocks)[:n]

        # Precios a partir de retornos resampleados
        prices      = np.empty(n + 1)
        prices[0]   = close_vals[0]
        prices[1:]  = close_vals[0] * np.cumprod(1.0 + sampled)

        # Retornos de los "trades" (entry cada `step` días, hold `n_hold` días)
        idx   = np.arange(0, n + 1 - n_hold, 5)
        if len(idx) < 3:
            continue
        t_rets = (prices[idx + n_hold] / np.where(prices[idx] != 0, prices[idx], 1) - 1) * 100

        wins   = t_rets[t_rets > 0]
        losses = np.abs(t_rets[t_rets <= 0])
        total  = len(wins) + len(losses)
        if total < 3 or len(losses) == 0:
            continue

        wr    = len(wins) / total
        aw    = float(np.mean(wins))    if len(wins)   > 0 else 0.001
        al    = float(np.mean(losses))  if len(losses) > 0 else 0.001
        raw_k = wr - (1 - wr) / (aw / al)
        kelly_samples.append(max(0.0, min(0.25, raw_k / 2)))
        wr_samples.append(wr * 100)

    if not kelly_samples:
        return 0.05, 0.01, 0.15, None

    ka = np.array(kelly_samples)
    wr_med = round(float(np.median(wr_samples)), 1) if wr_samples else None
    return (float(np.median(ka)),
            float(np.percentile(ka, 5)),
            float(np.percentile(ka, 95)),
            wr_med)


class StockAnalyzer:
    """Analizador técnico multi-acción"""

    def __init__(self, db):
        self.db = db
        self.dl = DataDownloader(db)

        # ── Pesos para acciones individuales (default) ──────────────────────
        self.weights = {
            'trend_strength':    0.25,
            'momentum':          0.20,
            'volatility':        0.15,
            'volume':            0.15,
            'price_action':      0.15,
            'support_resistance':0.10,
        }

        # ── Pesos para ETFs de renta variable (SPY, QQQ, XLK, ARKK…) ──────
        # Tendencia y momentum son más limpias en índices; S/R menos relevante.
        self._weights_etf_equity = {
            'trend_strength':    0.30,
            'momentum':          0.25,
            'volatility':        0.15,
            'volume':            0.15,
            'price_action':      0.10,
            'support_resistance':0.05,
        }

        # ── Pesos para ETFs de renta fija (TLT, AGG, HYG, TIP) ─────────────
        # Los bonos se mueven por tipos de interés: la tendencia sigue siendo
        # útil (dirección de tipos), momentum también, volumen menos crítico,
        # S/R muy irrelevante.
        self._weights_etf_bond = {
            'trend_strength':    0.30,
            'momentum':          0.25,
            'volatility':        0.20,
            'volume':            0.10,
            'price_action':      0.10,
            'support_resistance':0.05,
        }

        # Símbolos de ETFs de renta fija
        self._BOND_ETFS = {'TLT', 'AGG', 'HYG', 'TIP'}

        # Régimen de mercado activo (actualizado en analyze_all_stocks)
        self._current_regime = 'transition'

    def _get_weights(self, symbol: str) -> dict:
        """Devuelve pesos según tipo de instrumento y régimen de mercado activo."""
        if symbol in self._BOND_ETFS:
            return self._weights_etf_bond  # bonos: pesos fijos (anti-cíclicos)
        return _get_regime_weights(self._current_regime, etf=is_etf(symbol))

    def download_all_data(self, interval="1d", force=False):
        """Descarga datos históricos de todas las acciones"""
        symbols = get_all_symbols()
        total = len(symbols)

        print("=" * 70)
        print(f"  📥 Descargando datos de {total} acciones")
        print("=" * 70)
        print()

        success = 0
        failed = []

        for i, symbol in enumerate(symbols, 1):
            sector = get_sector(symbol)
            print(f"[{i:2}/{total}] {symbol:6} ({sector:15})", end=" ")

            try:
                df = self.dl.get_max_historical_data(symbol, interval=interval, force_download=force)
                if df is not None and len(df) > 0:
                    print(f"✅ {len(df):5} registros")
                    success += 1
                else:
                    print("❌ Sin datos")
                    failed.append(symbol)
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
                failed.append(symbol)

        print()
        print("=" * 70)
        print(f"✅ Descargados: {success}/{total}")
        if failed:
            print(f"❌ Fallidos: {', '.join(failed)}")
        print("=" * 70)
        print()

        return success, failed

    def calculate_technical_score(self, df, symbol):
        """
        Calcula score técnico (0-100) basado en indicadores ponderados

        Returns:
            dict con scores parciales y total
        """
        if len(df) < 100:
            return None

        try:
            # Asegurar orden cronológico ascendente (más antiguo primero)
            df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

            # Calcular indicadores
            df = self._calculate_indicators(df)

            # Último precio
            current = df.iloc[-1]
            prev = df.iloc[-2]

            scores = {}

            # 1. TREND STRENGTH (25%)
            # EMA 9 vs 21, EMA 21 vs 50
            ema_9  = current['ema_9']
            ema_21 = current['ema_21']
            ema_50 = current['ema_50']  # ya calculado en indicators_core.compute_emas

            trend_score = 0
            if ema_9 > ema_21 > ema_50:
                trend_score = 100  # Tendencia alcista fuerte
            elif ema_9 > ema_21:
                trend_score = 75   # Tendencia alcista moderada
            elif ema_21 > ema_50:
                trend_score = 50   # Neutral alcista
            elif ema_9 < ema_21 < ema_50:
                trend_score = 0    # Tendencia bajista fuerte
            elif ema_9 < ema_21:
                trend_score = 25   # Tendencia bajista moderada
            else:
                trend_score = 50   # Neutral

            scores['trend_strength'] = trend_score

            # 2. MOMENTUM (20%)
            # RSI, MACD
            rsi = current['rsi']

            # MACD
            macd_diff = current['macd'] - current['macd_signal']
            macd_prev_diff = prev['macd'] - prev['macd_signal']

            momentum_score = 0

            # RSI contribution (50%)
            # Para mediano plazo: zona oversold recovery (30-40) es mejor entrada
            # que zona overbought (60-70) — menor riesgo de reversión
            if 40 <= rsi <= 60:
                rsi_score = 100  # Zona saludable — momentum equilibrado
            elif 30 <= rsi < 40:
                rsi_score = 85   # Oversold recovery — buena entrada mediano plazo
            elif 60 < rsi <= 70:
                rsi_score = 65   # Momentum fuerte pero riesgo de pullback
            elif rsi < 30:
                rsi_score = 45   # Extremo oversold — puede ser trampa bajista
            else:  # rsi > 70
                rsi_score = 25   # Muy sobrecomprado — alta probabilidad de corrección

            # MACD contribution (50%)
            if macd_diff > 0 and macd_diff > macd_prev_diff:
                macd_score = 100  # MACD positivo y creciendo
            elif macd_diff > 0:
                macd_score = 75   # MACD positivo
            elif macd_diff < 0 and macd_diff > macd_prev_diff:
                macd_score = 50   # MACD negativo pero mejorando
            else:
                macd_score = 25   # MACD negativo y empeorando

            momentum_score = (rsi_score + macd_score) / 2
            scores['momentum'] = momentum_score

            # 3. VOLATILITY (15%)
            # Bollinger Bands, ATR
            bb_upper = current['bb_upper']
            bb_lower = current['bb_lower']
            bb_middle = current['bb_middle']
            price = current['close']

            # Posición en BB
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100

            # ATR como % del precio
            atr_pct = (current['atr'] / price) * 100

            volatility_score = 0

            # BB position (60%)
            if 40 <= bb_position <= 60:
                bb_score = 100  # En el medio
            elif 20 <= bb_position < 40 or 60 < bb_position <= 80:
                bb_score = 75   # Cerca del medio
            elif bb_position < 20:
                bb_score = 50   # Cerca del lower (oversold)
            else:  # bb_position > 80
                bb_score = 50   # Cerca del upper (overbought)

            # ATR (40%) - menos volatilidad es mejor para inversión
            if atr_pct < 2:
                atr_score = 100  # Baja volatilidad
            elif atr_pct < 4:
                atr_score = 75   # Volatilidad moderada
            elif atr_pct < 6:
                atr_score = 50   # Alta volatilidad
            else:
                atr_score = 25   # Muy alta volatilidad

            volatility_score = (bb_score * 0.6 + atr_score * 0.4)
            scores['volatility'] = volatility_score

            # 4. VOLUME (15%)
            # Volumen actual vs promedio
            vol_20 = df['volume'].rolling(20).mean().iloc[-1]
            vol_current = current['volume']

            vol_ratio = vol_current / vol_20

            if vol_ratio > 1.5:
                volume_score = 100  # Volumen muy alto (interés fuerte)
            elif vol_ratio > 1.2:
                volume_score = 75   # Volumen alto
            elif vol_ratio > 0.8:
                volume_score = 50   # Volumen normal
            else:
                volume_score = 25   # Volumen bajo

            scores['volume'] = volume_score

            # 5. PRICE ACTION (15%)
            # Cambio últimos 5, 20 días
            change_5d = ((current['close'] / df.iloc[-6]['close']) - 1) * 100
            change_20d = ((current['close'] / df.iloc[-21]['close']) - 1) * 100

            # Score basado en momentum positivo
            if change_5d > 5 and change_20d > 10:
                price_action_score = 100  # Momentum fuerte
            elif change_5d > 2 and change_20d > 5:
                price_action_score = 75   # Momentum moderado
            elif change_5d > 0 and change_20d > 0:
                price_action_score = 50   # Momentum positivo
            elif change_5d > 0 or change_20d > 0:
                price_action_score = 40   # Momentum mixto
            else:
                price_action_score = 20   # Momentum negativo

            scores['price_action'] = price_action_score

            # 6. SUPPORT/RESISTANCE (10%)
            # Distancia a máximos/mínimos de 52 semanas
            high_52w = df['high'].rolling(252).max().iloc[-1]
            low_52w = df['low'].rolling(252).min().iloc[-1]

            distance_to_high = ((high_52w - price) / price) * 100
            distance_to_low = ((price - low_52w) / price) * 100

            # Mejor estar cerca de mínimos que de máximos
            if distance_to_high > 20:
                sr_score = 100  # Lejos del máximo (espacio para crecer)
            elif distance_to_high > 10:
                sr_score = 75
            elif distance_to_high > 5:
                sr_score = 50
            else:
                sr_score = 25   # Muy cerca del máximo

            scores['support_resistance'] = sr_score

            # SCORE TOTAL PONDERADO (pesos diferenciados por tipo)
            w = self._get_weights(symbol)
            total_score = sum(scores[key] * w[key] for key in scores.keys())

            # Explicación del score
            score_explanation = self._generate_score_explanation(scores, w)

            return {
                'scores': scores,
                'total_score': total_score,
                'score_explanation': score_explanation,
                'price': price,
                'change_5d': change_5d,
                'change_20d': change_20d,
                'rsi': rsi,
                'atr_pct': atr_pct,
                'vol_ratio': vol_ratio,
                'bb_position': bb_position,
                'bb_lower': bb_lower,
                'bb_upper': bb_upper,
                'support_52w': low_52w,
                'resistance_52w': high_52w,
                'ema_9': ema_9,
                'ema_21': ema_21,
                'ema_50': ema_50
            }

        except Exception as e:
            print(f"⚠️  Error calculando score para {symbol}: {e}")
            return None

    def _generate_score_explanation(self, scores, weights):
        """Genera explicación textual del score"""
        explanations = []

        # Trend Strength
        trend_score = scores['trend_strength']
        if trend_score >= 75:
            explanations.append(f"✅ Tendencia alcista fuerte ({trend_score:.0f}/100)")
        elif trend_score >= 50:
            explanations.append(f"↗️ Tendencia alcista moderada ({trend_score:.0f}/100)")
        else:
            explanations.append(f"↘️ Tendencia débil o bajista ({trend_score:.0f}/100)")

        # Momentum
        mom_score = scores['momentum']
        if mom_score >= 75:
            explanations.append(f"✅ Momentum positivo ({mom_score:.0f}/100)")
        elif mom_score >= 50:
            explanations.append(f"→ Momentum neutral ({mom_score:.0f}/100)")
        else:
            explanations.append(f"⚠️ Momentum negativo ({mom_score:.0f}/100)")

        # Volatilidad
        vol_score = scores['volatility']
        if vol_score >= 75:
            explanations.append(f"✅ Volatilidad controlada ({vol_score:.0f}/100)")
        else:
            explanations.append(f"⚠️ Alta volatilidad ({vol_score:.0f}/100)")

        # Identificar factor más fuerte
        max_component = max(scores.items(), key=lambda x: x[1] * weights.get(x[0], 0))
        explanations.append(f"🎯 Factor principal: {max_component[0]}")

        return " | ".join(explanations)

    def _calculate_indicators(self, df):
        """Calcula indicadores técnicos usando indicators_core como fuente única."""
        return ic.compute_all(df)

    def check_data_freshness(self, df):
        """Verifica qué tan actualizados están los datos"""
        if df.empty:
            return None, "Sin datos"

        from datetime import datetime, timedelta
        import pandas as pd

        last_date = pd.to_datetime(df.iloc[0]['timestamp'])
        now = datetime.now()

        # Calcular días desde última actualización
        days_old = (now - last_date).days

        if days_old == 0:
            status = "✅ Actualizado (hoy)"
        elif days_old == 1:
            status = "✅ Reciente (ayer)"
        elif days_old <= 3:
            status = f"⚠️ {days_old} días antiguo"
        elif days_old <= 7:
            status = f"⚠️ {days_old} días antiguo - Actualizar recomendado"
        else:
            status = f"❌ {days_old} días antiguo - ACTUALIZAR URGENTE"

        return last_date, status

    def analyze_all_stocks(self):
        """Analiza todas las acciones y genera ranking"""
        symbols = get_all_symbols()
        results = []

        print("=" * 70)
        print("  📊 Analizando Indicadores Técnicos")
        print("=" * 70)
        print()

        # Régimen de mercado (una vez al inicio — condiciona los pesos)
        print("Detectando régimen de mercado...", end=" ")
        try:
            regime_data = _mr.get_regime()
            self._current_regime = regime_data['regime']
            fallback_note = " (fallback)" if regime_data.get('fallback') else ""
            print(f"{self._current_regime.upper()} (score {regime_data['score']}/6){fallback_note}")
        except Exception:
            self._current_regime = 'transition'
            print("No disponible — usando 'transition'")

        # VIX una vez al inicio (contexto global del mercado)
        print("Obteniendo VIX...", end=" ")
        vix_data = self.get_vix_level()
        if vix_data:
            print(f"VIX {vix_data['vix']} — {vix_data['level']} ({vix_data['signal']})")
        else:
            print("No disponible")

        # SPY una vez al inicio (referencia de fuerza relativa)
        print("Obteniendo SPY (referencia RS)...", end=" ")
        spy_data = self._get_spy_returns()
        if spy_data:
            print(f"SPY 60d: {spy_data['ret_60d']:+.1f}% | 252d: {spy_data['ret_252d']:+.1f}%")
        else:
            print("No disponible")

        # Retornos diarios SPY para Beta / IR / Correlación
        spy_daily_rets = None
        if spy_data and spy_data.get('close') is not None:
            try:
                spy_daily_rets = spy_data['close'].pct_change(fill_method=None).dropna()
            except Exception:
                pass
        print()

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i:2}/{len(symbols)}] Analizando {symbol:6}", end=" ")

            # Cargar datos
            df = self.db.get_precios(symbol)

            if df.empty or len(df) < 100:
                print("❌ Datos insuficientes")
                continue

            # Verificar frescura de datos
            last_date, freshness_status = self.check_data_freshness(df)

            # Calcular score
            analysis = self.calculate_technical_score(df, symbol)

            if analysis is None:
                print("❌ Error en análisis")
                continue

            # Calcular señal de volumen MA (estrategia monto transado)
            vol_signal    = self.calculate_volume_ma_signal(df)
            candle_signal = self.calculate_candlestick_signal(df)
            adx_signal    = self.calculate_adx_signal(df)
            fund_data     = self.calculate_fundamentals(symbol)
            tf_data       = self.calculate_timeframe_alignment(df)
            rs_data       = self.calculate_relative_strength(df, spy_data)
            risk_data     = self.calculate_risk_metrics(df, spy_rets=spy_daily_rets)
            bt_data       = self.calculate_backtest(df)

            result = {
                'symbol': symbol,
                'sector': get_sector(symbol),
                'total_score': analysis['total_score'],
                'price': analysis['price'],
                'change_5d': analysis['change_5d'],
                'change_20d': analysis['change_20d'],
                'rsi': analysis['rsi'],
                'support_52w': analysis['support_52w'],
                'resistance_52w': analysis['resistance_52w'],
                'last_update': last_date.strftime('%Y-%m-%d') if last_date else 'N/A',
                'data_freshness': freshness_status,
                # Volumen MA
                'vol_signal':    vol_signal['signal']           if vol_signal else 'N/A',
                'vol_ratio':     vol_signal['ratio']            if vol_signal else None,
                'vol_strength':  vol_signal['signal_strength']  if vol_signal else None,
                'vol_ma7':       vol_signal['ma7']              if vol_signal else None,
                'vol_ma60':      vol_signal['ma60']             if vol_signal else None,
                'vol_ma7_trend': vol_signal['ma7_trend_pct']    if vol_signal else None,
                # Patrones velas
                'candle_pattern':  candle_signal['pattern']  if candle_signal else 'Sin patrón',
                'candle_signal':   candle_signal['signal']   if candle_signal else 'NEUTRO',
                'candle_strength': candle_signal['strength'] if candle_signal else '—',
                # ADX
                'adx_value':     adx_signal['adx']       if adx_signal else None,
                'adx_strength':  adx_signal['strength']  if adx_signal else '—',
                'adx_direction': adx_signal['direction'] if adx_signal else '—',
                'adx_plus_di':   adx_signal['plus_di']   if adx_signal else None,
                'adx_minus_di':  adx_signal['minus_di']  if adx_signal else None,
                # Fundamentales
                'fund_score':   fund_data['fund_score']  if fund_data else 50,
                'fund_signal':  fund_data['fund_signal'] if fund_data else 'N/A',
                'fund_pe':      fund_data['pe']          if fund_data else None,
                'fund_peg':     fund_data['peg']         if fund_data else None,
                'fund_debt':    fund_data['debt_equity'] if fund_data else None,
                'fund_growth':  fund_data['rev_growth']  if fund_data else None,
                'fund_margins': fund_data['margins']     if fund_data else None,
                # Multi-Timeframe
                'tf_alignment':    tf_data['alignment']    if tf_data else 'N/A',
                'tf_signal':       tf_data['signal']       if tf_data else 'NEUTRO',
                'tf_conf_pts':     tf_data['conf_pts']     if tf_data else 0,
                'tf_weekly_rsi':   tf_data['weekly_rsi']   if tf_data else None,
                'tf_weekly_trend': tf_data['weekly_trend'] if tf_data else '—',
                'tf_daily_trend':  tf_data['daily_trend']  if tf_data else '—',
                # Fuerza Relativa vs SPY
                'rs_score':  rs_data['rs_score']  if rs_data else None,
                'rs_signal': rs_data['rs_signal'] if rs_data else 'N/A',
                'rs_20d':    rs_data['rs_20d']    if rs_data else None,
                'rs_60d':    rs_data['rs_60d']    if rs_data else None,
                'rs_252d':   rs_data['rs_252d']   if rs_data else None,
                # Risk Management
                'sharpe':       risk_data['sharpe']      if risk_data else None,
                'max_dd':       risk_data['max_dd']      if risk_data else None,
                'kelly_pct':    risk_data['kelly_pct']   if risk_data else None,
                'win_rate':     risk_data['win_rate']    if risk_data else None,
                'ann_return':   risk_data['ann_return']  if risk_data else None,
                'ann_vol':      risk_data['ann_vol']     if risk_data else None,
                'risk_score':   risk_data['risk_score']  if risk_data else None,
                'risk_signal':  risk_data['risk_signal'] if risk_data else 'N/A',
                # Backtesting
                'bt_win_rate_tech':   bt_data['bt_win_rate_tech']   if bt_data else None,
                'bt_avg_ret_tech':    bt_data['bt_avg_ret_tech']    if bt_data else None,
                'bt_expectancy_tech': bt_data['bt_expectancy_tech'] if bt_data else None,
                'bt_n_tech':          bt_data['bt_n_tech']          if bt_data else 0,
                'bt_best_tech':       bt_data['bt_best_tech']       if bt_data else None,
                'bt_worst_tech':      bt_data['bt_worst_tech']      if bt_data else None,
                'bt_win_rate_vol':    bt_data['bt_win_rate_vol']    if bt_data else None,
                'bt_avg_ret_vol':     bt_data['bt_avg_ret_vol']     if bt_data else None,
                'bt_expectancy_vol':  bt_data['bt_expectancy_vol']  if bt_data else None,
                'bt_n_vol':           bt_data['bt_n_vol']           if bt_data else 0,
                'bt_buy_hold':        bt_data['bt_buy_hold']        if bt_data else None,
                **analysis['scores']
            }

            # Confianza combinada (necesita el result ya construido)
            conf = self.calculate_signal_confidence(result, vix_data)
            result['confidence']       = conf['confidence']
            result['confidence_score'] = conf['score']

            results.append(result)

            adx_label  = f"| ADX {adx_signal['adx']:.0f} ({adx_signal['strength']})" if adx_signal else ""
            vol_label  = f"| Vol: {vol_signal['signal']} ({vol_signal['ratio']:.2f})" if vol_signal else ""
            tf_label   = f"| TF: {tf_data['alignment']}" if tf_data else ""
            conf_label = f"| Conf: {conf['confidence']}"
            print(f"OK Score: {analysis['total_score']:.1f}/100 {adx_label} {vol_label} {tf_label} {conf_label} | {freshness_status}")

        print()
        print("=" * 70)
        print(f"✅ Análisis completado: {len(results)} acciones")
        print("=" * 70)
        print()

        self.annotate_sector_vol_zscores(results)

        # Señal ML (Etapa 5.4): añade ml_prob y actualiza confianza si el modelo existe
        try:
            from ml_signal import enrich_ml_signals
            _regime_arg = locals().get("regime_data")
            enrich_ml_signals(results, vix_data=vix_data, regime=_regime_arg)
        except Exception as _ml_err:
            log.warning("ML signal enrichment omitido: %s", _ml_err)

        return pd.DataFrame(results)

    def calculate_correlations(self, symbols=None):
        """Calcula matriz de correlación entre acciones"""
        if symbols is None:
            symbols = get_all_symbols()

        print("📊 Calculando correlaciones...")

        # Obtener precios de cierre para todas las acciones
        prices = {}

        for symbol in symbols:
            df = self.db.get_precios(symbol)
            if not df.empty and len(df) > 100:
                prices[symbol] = df.set_index('timestamp')['close']

        if not prices:
            return None

        # Crear DataFrame con precios
        price_df = pd.DataFrame(prices)

        # Calcular retornos diarios
        returns = price_df.pct_change(fill_method=None).dropna()

        # Matriz de correlación
        corr_matrix = returns.corr()

        print(f"✅ Correlaciones calculadas para {len(prices)} acciones")

        return corr_matrix

    def find_diversified_portfolio(self, df_ranked, n=10, max_corr=0.7):
        """
        Encuentra portfolio diversificado con baja correlación

        Args:
            df_ranked: DataFrame con acciones rankeadas
            n: Número de acciones a seleccionar
            max_corr: Correlación máxima permitida entre acciones

        Returns:
            Lista de símbolos seleccionados
        """
        print(f"🎯 Buscando portfolio diversificado ({n} acciones, max_corr={max_corr})...")

        # Calcular correlaciones
        symbols = df_ranked['symbol'].tolist()
        corr_matrix = self.calculate_correlations(symbols)

        if corr_matrix is None:
            return df_ranked.head(n)['symbol'].tolist()

        selected = []
        candidates = df_ranked.sort_values('total_score', ascending=False)['symbol'].tolist()

        for symbol in candidates:
            if len(selected) >= n:
                break

            if len(selected) == 0:
                selected.append(symbol)
                continue

            # Verificar correlación con acciones ya seleccionadas
            correlations = [abs(corr_matrix.loc[symbol, s]) for s in selected if s in corr_matrix.index and symbol in corr_matrix.index]

            if not correlations or max(correlations) <= max_corr:
                selected.append(symbol)

        print(f"✅ Portfolio seleccionado: {len(selected)} acciones")

        return selected

    def calculate_volume_ma_signal(self, df):
        """
        Estrategia de tu padre: MA7 vs MA60 del monto transado diario.

        Monto transado = precio_cierre × volumen (dinero que se movió ese día)

        Lógica:
        - Si MA7 > MA60 → COMPRAR (actividad creciente, dinero entrando)
        - Si MA7 < MA60 → VENDER  (actividad decreciente, dinero saliendo)

        Returns dict con señal, ratio y datos de las medias
        """
        if len(df) < 65:
            return None

        df_sorted = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

        # Monto transado = precio cierre × volumen
        df_sorted['monto_transado'] = df_sorted['close'] * df_sorted['volume']

        # Medias móviles del monto transado
        ma7  = df_sorted['monto_transado'].rolling(window=7).mean()
        ma60 = df_sorted['monto_transado'].rolling(window=60).mean()

        ma7_actual  = ma7.iloc[-1]
        ma60_actual = ma60.iloc[-1]

        if ma60_actual == 0 or pd.isna(ma7_actual) or pd.isna(ma60_actual):
            return None

        ratio = ma7_actual / ma60_actual  # >1 = MA7 mayor que MA60

        # Señal
        if ratio > 1.0:
            signal = "COMPRAR"
            signal_strength = min(100, (ratio - 1.0) * 200)  # 0-100
        else:
            signal = "VENDER"
            signal_strength = min(100, (1.0 - ratio) * 200)  # 0-100

        # Tendencia de los últimos 7 días de la MA7 (¿subiendo o bajando?)
        if len(ma7) >= 8:
            ma7_prev = ma7.iloc[-8]
            if not pd.isna(ma7_prev) and ma7_prev > 0:
                ma7_trend = ((ma7_actual - ma7_prev) / ma7_prev) * 100
            else:
                ma7_trend = 0
        else:
            ma7_trend = 0

        return {
            'signal': signal,
            'signal_strength': round(signal_strength, 1),
            'ratio': round(ratio, 3),
            'ma7': round(ma7_actual, 0),
            'ma60': round(ma60_actual, 0),
            'ma7_trend_pct': round(ma7_trend, 2),
            'monto_ultimo': round(df_sorted['monto_transado'].iloc[-1], 0),
        }

    @staticmethod
    def annotate_sector_vol_zscores(results: list) -> None:
        """Añade 'vol_zscore' a cada result dict comparando vol_ratio con el sector.

        Z-score = (ratio_símbolo - media_sector) / std_sector.
        Modifica results in-place. Llámalo antes de crear el DataFrame final.
        """
        from collections import defaultdict
        import math

        by_sector: dict = defaultdict(list)
        for r in results:
            v = r.get('vol_ratio')
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                by_sector[r.get('sector', '')].append(float(v))

        sector_stats: dict = {}
        for sec, vals in by_sector.items():
            if len(vals) < 2:
                sector_stats[sec] = (vals[0] if vals else 1.0, 0.0)
                continue
            mean = sum(vals) / len(vals)
            std  = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            sector_stats[sec] = (mean, std)

        for r in results:
            ratio  = r.get('vol_ratio')
            sector = r.get('sector', '')
            mean, std = sector_stats.get(sector, (1.0, 0.0))
            if ratio is not None and std > 0:
                r['vol_zscore'] = round((float(ratio) - mean) / std, 2)
            else:
                r['vol_zscore'] = None

    def calculate_candlestick_signal(self, df):
        """
        Detecta patrones de velas japonesas en las últimas 3 velas.

        Patrones detectados:
        - Hammer / Hanging Man         → reversión
        - Bullish / Bearish Engulfing  → reversión fuerte
        - Doji                         → indecisión
        - Morning Star / Evening Star  → reversión muy fuerte (3 velas)

        Returns dict con patrón detectado, señal y fuerza.
        Si hay múltiples patrones toma el de mayor fuerza.
        """
        if len(df) < 3:
            return None

        df_sorted = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        patterns = AdvancedIndicators.detect_candlestick_patterns(df_sorted)

        if not patterns:
            return {'pattern': 'Sin patrón', 'signal': 'NEUTRO', 'strength': '—'}

        # Ordenar por fuerza: muy fuerte > fuerte > débil
        strength_order = {'muy fuerte': 3, 'fuerte': 2, 'débil': 1}
        patterns_sorted = sorted(patterns,
                                 key=lambda p: strength_order.get(p['strength'], 0),
                                 reverse=True)
        best = patterns_sorted[0]

        # Si hay múltiples patrones con igual fuerza, concatenar nombres
        top_strength = best['strength']
        same_strength = [p for p in patterns_sorted if p['strength'] == top_strength]
        names = ' + '.join(p['pattern'] for p in same_strength)

        return {
            'pattern': names,
            'signal': best['signal'],
            'strength': top_strength,
            'all_patterns': patterns,
        }

    # ─────────────────────────────────────────────
    #  ETAPA 2: ALINEACIÓN MULTI-TEMPORALIDAD
    # ─────────────────────────────────────────────

    def _resample_to_weekly(self, df):
        """Convierte datos diarios a velas semanales (cierre viernes)."""
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_indexed = df_sorted.set_index('timestamp')

        weekly = df_indexed.resample('W-FRI').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna(subset=['close'])

        return weekly.reset_index()

    def calculate_timeframe_alignment(self, df):
        """
        Alineacion entre temporalidad diaria y semanal.

        Compara EMAs 9/21/50 en marco semanal con EMAs 9/21 diarias.
        Produce una etiqueta de alineacion + puntos de confianza.

        Returns dict con:
            alignment   : etiqueta (p.ej. 'ALINEADO ALCISTA')
            signal      : ALCISTA / BAJISTA / NEUTRO
            conf_pts    : -2 a +2
            weekly_rsi  : RSI en vela semanal
            weekly_trend: tendencia semanal
            daily_trend : tendencia diaria
        """
        if df.empty or len(df) < 60:
            return None

        try:
            # -- SEMANALES --
            df_w = self._resample_to_weekly(df)
            if len(df_w) < 20:
                return None

            df_w = df_w.copy()
            df_w['ema9']  = ta.ema(df_w['close'], length=9)
            df_w['ema21'] = ta.ema(df_w['close'], length=21)
            df_w['ema50'] = ta.ema(df_w['close'], length=50)
            df_w['rsi']   = ta.rsi(df_w['close'], length=14)
            df_w.dropna(subset=['ema9', 'ema21', 'ema50'], inplace=True)

            if df_w.empty:
                return None

            last_w  = df_w.iloc[-1]
            w_close = float(last_w['close'])
            w_ema9  = float(last_w['ema9'])
            w_ema21 = float(last_w['ema21'])
            w_ema50 = float(last_w['ema50'])
            w_rsi   = float(last_w['rsi']) if not pd.isna(last_w['rsi']) else None

            # -- DIARIAS --
            df_d = df.sort_values('timestamp').copy()
            df_d['ema9d']  = ta.ema(df_d['close'], length=9)
            df_d['ema21d'] = ta.ema(df_d['close'], length=21)
            df_d.dropna(subset=['ema9d', 'ema21d'], inplace=True)

            if df_d.empty:
                return None

            last_d  = df_d.iloc[-1]
            d_close = float(last_d['close'])
            d_ema9  = float(last_d['ema9d'])
            d_ema21 = float(last_d['ema21d'])

            # -- TENDENCIAS --
            w_bull = (w_close > w_ema9) and (w_ema9 > w_ema21) and (w_ema21 > w_ema50)
            w_bear = (w_close < w_ema9) and (w_ema9 < w_ema21) and (w_ema21 < w_ema50)
            d_bull = (d_close > d_ema9) and (d_ema9 > d_ema21)
            d_bear = (d_close < d_ema9) and (d_ema9 < d_ema21)

            # -- ALINEACION --
            if w_bull and d_bull:
                alignment, signal, conf_pts = 'ALINEADO ALCISTA',      'ALCISTA', +2
            elif w_bull and not d_bull and not d_bear:
                alignment, signal, conf_pts = 'CONSOLIDANDO ALCISTA',  'ALCISTA', +1
            elif w_bull and d_bear:
                alignment, signal, conf_pts = 'CORRECCION EN ALCISTA', 'ALCISTA',  0
            elif w_bear and d_bull:
                alignment, signal, conf_pts = 'REBOTE EN BAJISTA',     'BAJISTA', -1
            elif w_bear and not d_bull and not d_bear:
                alignment, signal, conf_pts = 'CONSOLIDANDO BAJISTA',  'BAJISTA', -1
            elif w_bear and d_bear:
                alignment, signal, conf_pts = 'ALINEADO BAJISTA',      'BAJISTA', -2
            elif not w_bull and not w_bear and d_bull:
                alignment, signal, conf_pts = 'REBOTE EN LATERAL',     'NEUTRO',   0
            elif not w_bull and not w_bear and d_bear:
                alignment, signal, conf_pts = 'CAIDA EN LATERAL',      'NEUTRO',  -1
            else:
                alignment, signal, conf_pts = 'LATERAL',               'NEUTRO',   0

            return {
                'alignment':    alignment,
                'signal':       signal,
                'conf_pts':     conf_pts,
                'weekly_rsi':   round(w_rsi, 1) if w_rsi is not None else None,
                'weekly_trend': 'ALCISTA' if w_bull else ('BAJISTA' if w_bear else 'LATERAL'),
                'daily_trend':  'ALCISTA' if d_bull else ('BAJISTA' if d_bear else 'LATERAL'),
            }

        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  ETAPA 5: RISK MANAGEMENT
    # ─────────────────────────────────────────────

    def calculate_risk_metrics(self, df, spy_rets=None):
        """
        Métricas de riesgo sobre los últimos 252 días:

          Sharpe, Sortino, Calmar, Max Drawdown, Downside Deviation,
          Kelly (Half-Kelly), + si spy_rets disponible: Beta, Correlación 60d,
          Information Ratio.

        spy_rets — pd.Series de retornos diarios SPY (opcional). Si se provee
                   se calculan Beta, Correlation y Information Ratio.
        """
        if df.empty or len(df) < 60:
            return None

        try:
            df_s  = df.sort_values('timestamp').copy()
            close = df_s['close'].reset_index(drop=True)

            window  = min(252, len(close))
            close_w = close.iloc[-window:]

            daily_ret = close_w.pct_change(fill_method=None).dropna()
            if len(daily_ret) < 20:
                return None

            # ── Sharpe ────────────────────────────────────────────
            rf_daily   = 0.045 / 252
            excess     = daily_ret - rf_daily
            ann_return = float(daily_ret.mean() * 252)
            ann_vol    = float(daily_ret.std() * 252 ** 0.5)
            sharpe     = float(excess.mean() / daily_ret.std() * 252 ** 0.5) if ann_vol > 0 else 0.0

            # ── Sortino ───────────────────────────────────────────
            downside   = daily_ret[daily_ret < 0]
            ds_dev_ann = float(downside.std() * 252 ** 0.5) if len(downside) >= 5 else None
            sortino    = float((ann_return - 0.045) / ds_dev_ann) if ds_dev_ann and ds_dev_ann > 0 else None

            # ── Max Drawdown ──────────────────────────────────────
            roll_max = close_w.cummax()
            drawdown = (close_w - roll_max) / roll_max * 100
            max_dd   = float(drawdown.min())

            # ── Calmar ────────────────────────────────────────────
            calmar = float(ann_return * 100 / abs(max_dd)) if max_dd < 0 else None

            # ── Kelly Half-Kelly (block bootstrap) ───────────────────────────
            half_k, k_p5, k_p95, win_rate_pct = _kelly_block_bootstrap(
                close_w.values.astype(float))
            kelly_pct     = round(half_k * 100, 1)
            kelly_pct_p5  = round(k_p5   * 100, 1)
            kelly_pct_p95 = round(k_p95  * 100, 1)

            # ── Beta, Correlación, Information Ratio (requieren SPY) ──
            beta    = None
            corr60  = None
            info_ratio = None

            if spy_rets is not None and len(spy_rets) >= 30:
                try:
                    # Alinear por posición (ambas son series de retornos diarios
                    # recientes; los índices son incompatibles — int vs DatetimeIndex)
                    n_asset = len(daily_ret)
                    spy_vals = spy_rets.values[-n_asset:] if len(spy_rets) >= n_asset \
                               else spy_rets.values
                    asset_vals = daily_ret.values[-len(spy_vals):]
                    min_n = min(len(asset_vals), len(spy_vals))
                    if min_n >= 30:
                        a_v = asset_vals[-min_n:]
                        s_v = spy_vals[-min_n:]
                        var_spy = float(np.var(s_v, ddof=1))
                        if var_spy > 0:
                            beta = float(np.cov(a_v, s_v, ddof=1)[0, 1] / var_spy)
                        # Correlación últimos 60d
                        n60 = min(60, min_n)
                        if n60 >= 20:
                            corr60 = float(np.corrcoef(a_v[-n60:], s_v[-n60:])[0, 1])
                        # Information Ratio
                        active = a_v - s_v
                        te = float(np.std(active, ddof=1) * 252 ** 0.5)
                        if te > 0:
                            info_ratio = round(float(np.mean(active) * 252) / te, 2)
                except Exception:
                    pass

            # ── Risk Score ────────────────────────────────────────
            s_sharpe = (100 if sharpe >= 1.5 else 80 if sharpe >= 1.0 else
                        60 if sharpe >= 0.5 else 40 if sharpe >= 0 else 10)
            s_dd     = (100 if max_dd >= -10 else 75 if max_dd >= -20 else
                        50 if max_dd >= -30 else 25 if max_dd >= -40 else 0)
            risk_score  = round(s_sharpe * 0.50 + s_dd * 0.50, 1)
            risk_signal = ('BAJO RIESGO' if risk_score >= 70 else
                           'RIESGO MEDIO' if risk_score >= 45 else 'ALTO RIESGO')

            return {
                'sharpe':      round(sharpe, 2),
                'sortino':     round(sortino, 2)    if sortino    is not None else None,
                'calmar':      round(calmar,  2)    if calmar     is not None else None,
                'max_dd':      round(max_dd, 1),
                'ds_dev_ann':  round(ds_dev_ann * 100, 2) if ds_dev_ann is not None else None,
                'kelly_pct':   kelly_pct,
                'win_rate':    win_rate_pct,
                'ann_return':  round(ann_return * 100, 1),
                'ann_vol':     round(ann_vol * 100, 1),
                'beta':        round(beta, 2)       if beta       is not None else None,
                'corr_60d':    round(corr60, 2)     if corr60     is not None else None,
                'info_ratio':   info_ratio,
                'kelly_pct_p5': kelly_pct_p5,
                'kelly_pct_p95':kelly_pct_p95,
                'risk_score':  risk_score,
                'risk_signal': risk_signal,
            }

        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  ETAPA 5b: WALK-FORWARD BACKTEST
    # ─────────────────────────────────────────────

    def walk_forward_backtest(self, df, is_years=3, oos_months=6, forward_days=20):
        """
        Walk-forward backtest para detectar overfitting IS vs OOS.

        Metodología:
          IS  (in-sample)  = is_years × 252 barras  → se ajusta la estrategia
          OOS (out-sample) = oos_months × 21 barras → se evalúa sin retoque

        El split se hace rolling: se desplaza oos_months cada iteración.
        En cada split se corre calculate_backtest sobre la ventana IS y OOS.

        Devuelve dict con:
          folds          — lista de dicts por fold (is_wr, oos_wr, is_exp, oos_exp)
          is_win_rate    — media win_rate IS (estrategia técnica + vol)
          oos_win_rate   — media win_rate OOS
          is_expectancy  — media expectancy IS
          oos_expectancy — media expectancy OOS
          degradation    — oos_win_rate / is_win_rate  (1.0 = sin degradación)
          n_folds        — número de folds procesados
        """
        if df.empty or len(df) < 120:
            return None

        try:
            df_s = df.sort_values('timestamp').reset_index(drop=True)
            n    = len(df_s)

            is_bars  = int(is_years   * 252)
            oos_bars = int(oos_months * 21)
            step     = oos_bars          # desplazamiento entre folds

            if n < is_bars + oos_bars:
                return None

            folds = []
            fold_start = 0

            while fold_start + is_bars + oos_bars <= n:
                is_end  = fold_start + is_bars
                oos_end = is_end + oos_bars

                df_is  = df_s.iloc[fold_start:is_end].reset_index(drop=True)
                df_oos = df_s.iloc[is_end:oos_end].reset_index(drop=True)

                bt_is  = self.calculate_backtest(df_is,  forward_days=forward_days)
                bt_oos = self.calculate_backtest(df_oos, forward_days=forward_days)

                if bt_is and bt_oos:
                    folds.append({
                        'fold':      len(folds) + 1,
                        'is_start':  str(df_is['timestamp'].iloc[0])[:10],
                        'is_end':    str(df_is['timestamp'].iloc[-1])[:10],
                        'oos_start': str(df_oos['timestamp'].iloc[0])[:10],
                        'oos_end':   str(df_oos['timestamp'].iloc[-1])[:10],
                        'is_wr':     bt_is['bt_win_rate_vol'],
                        'oos_wr':    bt_oos['bt_win_rate_vol'],
                        'is_exp':    bt_is['bt_expectancy_vol'],
                        'oos_exp':   bt_oos['bt_expectancy_vol'],
                        'is_n':      bt_is['bt_n_vol'],
                        'oos_n':     bt_oos['bt_n_vol'],
                    })

                fold_start += step

            if not folds:
                return None

            def _mean(key):
                vals = [f[key] for f in folds if f[key] is not None]
                return round(sum(vals) / len(vals), 1) if vals else None

            is_wr  = _mean('is_wr')
            oos_wr = _mean('oos_wr')
            degradation = round(oos_wr / is_wr, 3) if is_wr and is_wr > 0 else None

            return {
                'folds':          folds,
                'n_folds':        len(folds),
                'is_win_rate':    is_wr,
                'oos_win_rate':   oos_wr,
                'is_expectancy':  _mean('is_exp'),
                'oos_expectancy': _mean('oos_exp'),
                'degradation':    degradation,
            }

        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  ETAPA 6: BACKTESTING
    # ─────────────────────────────────────────────

    def calculate_backtest(self, df, forward_days=20):
        """
        Backtest histórico con state machine (sin solapamiento de trades).

        Estrategias:
          1. TÉCNICA SOLA  — entrada EMA9>EMA21>EMA50
          2. TÉCNICA + VOL — entrada además Vol_MA7>Vol_MA60

        SL = entry × (1 − 2 × ATR%),  TP = entry × (1 + 3 × ATR%).
        Fallback si ATR ausente: SL 2%, TP 3%.
        Exit intra-vela via high/low de cada barra. Máx. holding = forward_days.

        Claves devueltas (sufijo _tech y _vol):
          bt_win_rate, bt_avg_ret, bt_expectancy, bt_n, bt_best, bt_worst,
          bt_sortino, bt_calmar, bt_max_dd, bt_exposure_pct, bt_avg_hold_days.
        """
        if df.empty or len(df) < 120:
            return None

        try:
            df_s = df.sort_values('timestamp').reset_index(drop=True).copy()

            max_rows = 252 * 3
            if len(df_s) > max_rows:
                df_s = df_s.iloc[-max_rows:].reset_index(drop=True)

            df_s = ic.compute_emas(df_s, spans=(9, 21, 50))
            df_s = ic.compute_atr(df_s)
            df_s = ic.compute_volume_ma(df_s, spans=(7, 60))
            df_s = df_s.rename(columns={
                'ema_9': 'ema9', 'ema_21': 'ema21', 'ema_50': 'ema50',
                'vol_ma_7': 'ma7', 'vol_ma_60': 'ma60',
            })
            df_s.dropna(subset=['ema9', 'ema21', 'ema50', 'ma7', 'ma60'], inplace=True)
            df_s = df_s.reset_index(drop=True)

            n_bars   = len(df_s)
            has_ohlc = ('high' in df_s.columns and 'low' in df_s.columns)
            has_atr  = 'atr' in df_s.columns

            if n_bars < 30:
                return None

            def _run_strategy(require_vol: bool) -> list:
                trades: list = []
                in_pos  = False
                e_idx   = 0
                e_price = 0.0
                sl_lvl  = 0.0
                tp_lvl  = 0.0

                for i in range(n_bars):
                    row = df_s.iloc[i]

                    if not in_pos:
                        sig = (float(row['ema9']) > float(row['ema21']) and
                               float(row['ema21']) > float(row['ema50']))
                        if require_vol:
                            sig = sig and float(row['ma7']) > float(row['ma60'])
                        if not sig:
                            continue

                        in_pos  = True
                        e_idx   = i
                        e_price = float(row['close'])

                        atr_v = None
                        if has_atr:
                            raw = row['atr']
                            if not pd.isna(raw):
                                atr_v = float(raw)
                        if atr_v and atr_v > 0:
                            atr_pct = atr_v / e_price
                            sl_lvl  = e_price * (1 - 2 * atr_pct)
                            tp_lvl  = e_price * (1 + 3 * atr_pct)
                        else:
                            sl_lvl = e_price * 0.98
                            tp_lvl = e_price * 1.03
                        continue  # no exit check on entry bar

                    # in_pos — check exit via intra-candle OHLC
                    bar_lo = float(row['low'])  if has_ohlc else float(row['close'])
                    bar_hi = float(row['high']) if has_ohlc else float(row['close'])
                    hold   = i - e_idx

                    x_price = None
                    if bar_lo <= sl_lvl:
                        x_price = sl_lvl
                    elif bar_hi >= tp_lvl:
                        x_price = tp_lvl
                    elif hold >= forward_days:
                        x_price = float(row['close'])

                    if x_price is not None:
                        trades.append({'ret': (x_price / e_price - 1) * 100, 'hold': hold})
                        in_pos = False

                return trades

            def _stats(trades: list):
                if not trades:
                    return None
                rets  = [t['ret']  for t in trades]
                holds = [t['hold'] for t in trades]
                n     = len(rets)
                wins  = [r for r in rets if r > 0]
                loss  = [r for r in rets if r <= 0]
                wr    = len(wins) / n
                aw    = sum(wins) / len(wins) if wins else 0.0
                al    = sum(loss) / len(loss) if loss else 0.0
                avg   = sum(rets) / n

                avg_hold = sum(holds) / n if n > 0 else float(forward_days)
                tpy      = 252 / max(avg_hold, 1)

                # Sortino (annualized, downside deviation)
                ds_sq  = [r * r for r in rets if r < 0]
                ds_std = (sum(ds_sq) / len(ds_sq)) ** 0.5 if ds_sq else 0.0
                sortino = (avg * tpy ** 0.5) / ds_std if ds_std > 0 else None

                # Max drawdown (cumulative %)
                cum = 0.0; peak = 0.0; max_dd = 0.0
                for r in rets:
                    cum += r
                    if cum > peak: peak = cum
                    dd = peak - cum
                    if dd > max_dd: max_dd = dd

                # Calmar (annualized return / max drawdown)
                ann_ret = avg * tpy
                calmar  = ann_ret / max_dd if max_dd > 0 else None

                exposure = (sum(holds) / n_bars * 100) if n_bars > 0 else 0.0

                return {
                    'n':          n,
                    'win_rate':   round(wr * 100, 1),
                    'avg_ret':    round(avg, 2),
                    'avg_win':    round(aw, 2),
                    'avg_loss':   round(al, 2),
                    'best':       round(max(rets), 1),
                    'worst':      round(min(rets), 1),
                    'expectancy': round(wr * aw + (1 - wr) * al, 2),
                    'sortino':    round(sortino, 2) if sortino is not None else None,
                    'calmar':     round(calmar,  2) if calmar  is not None else None,
                    'max_dd':     round(max_dd, 2),
                    'exposure':   round(exposure, 1),
                    'avg_hold':   round(avg_hold, 1),
                }

            st_tech = _stats(_run_strategy(require_vol=False))
            st_vol  = _stats(_run_strategy(require_vol=True))

            buy_hold = round(
                (float(df_s['close'].iloc[-1]) / float(df_s['close'].iloc[0]) - 1) * 100, 1)

            def _g(s, k):
                return s[k] if s else None

            return {
                'bt_win_rate_tech':      _g(st_tech, 'win_rate'),
                'bt_avg_ret_tech':       _g(st_tech, 'avg_ret'),
                'bt_expectancy_tech':    _g(st_tech, 'expectancy'),
                'bt_n_tech':             st_tech['n'] if st_tech else 0,
                'bt_best_tech':          _g(st_tech, 'best'),
                'bt_worst_tech':         _g(st_tech, 'worst'),
                'bt_sortino_tech':       _g(st_tech, 'sortino'),
                'bt_calmar_tech':        _g(st_tech, 'calmar'),
                'bt_max_dd_tech':        _g(st_tech, 'max_dd'),
                'bt_exposure_pct_tech':  _g(st_tech, 'exposure'),
                'bt_avg_hold_days_tech': _g(st_tech, 'avg_hold'),
                'bt_win_rate_vol':       _g(st_vol,  'win_rate'),
                'bt_avg_ret_vol':        _g(st_vol,  'avg_ret'),
                'bt_expectancy_vol':     _g(st_vol,  'expectancy'),
                'bt_n_vol':              st_vol['n'] if st_vol else 0,
                'bt_sortino_vol':        _g(st_vol,  'sortino'),
                'bt_calmar_vol':         _g(st_vol,  'calmar'),
                'bt_max_dd_vol':         _g(st_vol,  'max_dd'),
                'bt_exposure_pct_vol':   _g(st_vol,  'exposure'),
                'bt_avg_hold_days_vol':  _g(st_vol,  'avg_hold'),
                'bt_buy_hold':           buy_hold,
                'bt_forward_days':       forward_days,
            }

        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  ETAPA 4: FUERZA RELATIVA vs S&P 500 (SPY)
    # ─────────────────────────────────────────────

    @staticmethod
    def _get_spy_returns():
        """
        Descarga SPY y calcula retornos acumulados a 20d, 60d y 252d.
        Se llama una sola vez antes del bucle principal.
        """
        import yfinance as yf
        try:
            df = yf.download('SPY', period='2y', interval='1d', progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_index()
            close = df['close']
            return {
                'close': close,
                'ret_20d':  float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21  else None,
                'ret_60d':  float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) >= 61  else None,
                'ret_252d': float((close.iloc[-1] / close.iloc[-253] - 1) * 100) if len(close) >= 253 else None,
            }
        except Exception:
            return None

    def calculate_relative_strength(self, df, spy_data):
        """
        Compara el rendimiento del activo contra SPY en 20d, 60d y 252d.

        RS = retorno_activo - retorno_SPY  (en puntos porcentuales)

        Ejemplos:
          Activo +30%, SPY +15%  → RS_60d = +15  (supera al mercado)
          Activo  -5%, SPY +10%  → RS_60d = -15  (pierde contra el mercado)

        Score RS (0-100):
          Promedio ponderado de los tres periodos
          20d:30%  60d:40%  252d:30%

        Señal:
          LIDER    si score >= 65
          MERCADO  si score >= 40
          REZAGADO si score <  40
        """
        if df.empty or spy_data is None:
            return None

        try:
            df_s = df.sort_values('timestamp').copy()
            df_s['timestamp'] = pd.to_datetime(df_s['timestamp'])
            close = df_s.set_index('timestamp')['close'].sort_index()

            def rs(n):
                if len(close) < n + 1:
                    return None
                asset_ret = float((close.iloc[-1] / close.iloc[-(n+1)] - 1) * 100)
                spy_ret   = spy_data.get(f'ret_{n}d')
                if spy_ret is None:
                    return None
                return round(asset_ret - spy_ret, 2)

            rs_20d  = rs(20)
            rs_60d  = rs(60)
            rs_252d = rs(252)

            def rs_to_subscore(v):
                """Convierte RS puntual a subscore 0-100"""
                if v is None:
                    return 50
                if v >= 20:   return 100
                if v >= 10:   return 80
                if v >= 5:    return 65
                if v >= 0:    return 55
                if v >= -5:   return 40
                if v >= -10:  return 25
                if v >= -20:  return 10
                return 0

            s20  = rs_to_subscore(rs_20d)
            s60  = rs_to_subscore(rs_60d)
            s252 = rs_to_subscore(rs_252d)

            # Ponderado: 20d=30%, 60d=40%, 252d=30%
            score = s20 * 0.30 + s60 * 0.40 + s252 * 0.30

            if score >= 65:
                rs_signal = 'LIDER'
            elif score >= 40:
                rs_signal = 'MERCADO'
            else:
                rs_signal = 'REZAGADO'

            return {
                'rs_score':   round(score, 1),
                'rs_signal':  rs_signal,
                'rs_20d':     rs_20d,
                'rs_60d':     rs_60d,
                'rs_252d':    rs_252d,
            }

        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  ETAPA 1: ADX · FUNDAMENTALES · VIX · CONFIANZA
    # ─────────────────────────────────────────────

    @staticmethod
    def get_vix_level():
        """
        Descarga el VIX (índice del miedo del mercado S&P 500).

        VIX > 35 → Pánico      → Históricamente, OPORTUNIDAD de compra
        VIX > 25 → Miedo       → Mercado nervioso, pero corregible
        VIX > 20 → Precaución  → Por encima de la media histórica
        VIX < 15 → Complacencia → Mercado caro, reducir exposición
        """
        import yfinance as yf
        try:
            df = yf.download("^VIX", period="5d", interval="1d", progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            vix = float(df['close'].iloc[-1])

            if vix > 35:
                level, signal, color = 'PÁNICO',       'COMPRAR AGRESIVO',   'buy'
            elif vix > 25:
                level, signal, color = 'MIEDO',         'COMPRAR',            'buy'
            elif vix > 20:
                level, signal, color = 'PRECAUCIÓN',    'NEUTRAL',            'warn'
            elif vix > 15:
                level, signal, color = 'NORMAL',        'NEUTRAL',            'neutral'
            else:
                level, signal, color = 'COMPLACENCIA',  'REDUCIR EXPOSICIÓN', 'sell'

            return {'vix': round(vix, 2), 'level': level, 'signal': signal, 'color': color}
        except Exception:
            return None

    def calculate_adx_signal(self, df):
        """
        ADX (Average Directional Index) — mide la FUERZA de la tendencia.
        No dice dirección, dice si la tendencia es real o no.

        ADX < 20 → Sin tendencia real (EMA/MACD poco fiables en este contexto)
        ADX 20-25 → Tendencia emergente
        ADX > 25 → Tendencia confirmada
        ADX > 40 → Tendencia muy fuerte

        +DI > -DI → dirección alcista
        -DI > +DI → dirección bajista
        """
        if len(df) < 20:
            return None
        try:
            df_s = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
            adx_data = AdvancedIndicators.adx(df_s, length=14)
            if not adx_data:
                return None

            adx = adx_data['adx']
            plus_di  = adx_data['plus_di']
            minus_di = adx_data['minus_di']

            if adx > 40:
                strength, conf_mult = 'muy fuerte', 1.20
            elif adx > 25:
                strength, conf_mult = 'fuerte',     1.00
            elif adx > 20:
                strength, conf_mult = 'moderada',   0.85
            else:
                strength, conf_mult = 'débil',      0.70

            direction = 'alcista' if plus_di > minus_di else 'bajista'

            return {
                'adx': round(adx, 1),
                'plus_di': round(plus_di, 1),
                'minus_di': round(minus_di, 1),
                'strength': strength,
                'direction': direction,
                'signal': adx_data['signal'],   # COMPRA / VENTA / None
                'conf_mult': conf_mult,
            }
        except Exception:
            return None

    def calculate_fundamentals(self, symbol):
        """
        Para ACCIONES: P/E, PEG, Deuda/Capital, Crecimiento ingresos, Márgenes.
        Para ETFs:     AUM, Yield, Retorno 3 años, Retorno 5 años — sin P/E.
        """
        import yfinance as yf

        if is_etf(symbol):
            return self._calculate_etf_metrics(symbol)

        try:
            info = _get_fund_info(symbol)
            pe   = info.get('trailingPE') or info.get('forwardPE')
            peg  = info.get('pegRatio')
            d_e  = info.get('debtToEquity')       # en % (100 = D/E de 1x)
            rev_g = info.get('revenueGrowth')     # decimal (0.15 = 15%)
            margins = info.get('profitMargins')   # decimal

            score = 50
            signals  = []
            warnings = []

            def safe(v):
                return v is not None and not (isinstance(v, float) and (v != v))

            # P/E
            if safe(pe):
                if pe < 0:
                    score -= 10; warnings.append(f"P/E negativo ({pe:.1f}) = pérdidas")
                elif pe < 15:
                    score += 15; signals.append(f"P/E {pe:.1f}x → barato")
                elif pe < 25:
                    score += 8
                elif pe < 40:
                    score -= 5
                else:
                    score -= 15; warnings.append(f"P/E {pe:.1f}x → muy caro")

            # PEG
            if safe(peg):
                if peg < 0:
                    score -= 5
                elif peg < 1:
                    score += 15; signals.append(f"PEG {peg:.2f} < 1 → subvalorado")
                elif peg < 2:
                    score += 5
                else:
                    score -= 10; warnings.append(f"PEG {peg:.1f} → crecimiento caro")

            # Deuda/Capital
            if safe(d_e):
                if d_e < 30:
                    score += 10; signals.append("Deuda baja")
                elif d_e < 100:
                    score += 3
                elif d_e < 200:
                    score -= 5
                else:
                    score -= 15; warnings.append(f"Deuda alta (D/E {d_e:.0f}%)")

            # Crecimiento ingresos
            if safe(rev_g):
                if rev_g > 0.20:
                    score += 15; signals.append(f"Creciendo {rev_g*100:.0f}% anual")
                elif rev_g > 0.10:
                    score += 8;  signals.append(f"Creciendo {rev_g*100:.0f}%")
                elif rev_g > 0:
                    score += 3
                else:
                    score -= 8;  warnings.append(f"Ingresos cayendo {rev_g*100:.0f}%")

            # Márgenes
            if safe(margins):
                if margins > 0.20:
                    score += 8;  signals.append(f"Margen {margins*100:.0f}%")
                elif margins > 0.10:
                    score += 4
                elif margins < 0:
                    score -= 10; warnings.append("Empresa en pérdidas")

            score = max(0, min(100, score))

            if score >= 65:
                fund_signal = 'FAVORABLE'
            elif score >= 40:
                fund_signal = 'NEUTRO'
            else:
                fund_signal = 'DESFAVORABLE'

            return {
                'fund_score':    round(score, 1),
                'fund_signal':   fund_signal,
                'pe':            round(pe, 1)        if safe(pe)      else None,
                'peg':           round(peg, 2)       if safe(peg)     else None,
                'debt_equity':   round(d_e, 1)       if safe(d_e)     else None,
                'rev_growth':    round(rev_g * 100, 1) if safe(rev_g) else None,
                'margins':       round(margins * 100, 1) if safe(margins) else None,
                'signals':       signals,
                'warnings':      warnings,
            }
        except Exception as e:
            return {
                'fund_score': 50, 'fund_signal': 'N/A',
                'pe': None, 'peg': None, 'debt_equity': None,
                'rev_growth': None, 'margins': None,
                'signals': [], 'warnings': [str(e)[:60]],
            }

    def _calculate_etf_metrics(self, symbol):
        """
        Métricas específicas para ETFs vía yfinance.
        Los ETFs no tienen P/E ni PEG — se evalúan por AUM, yield y rendimiento histórico.

        Score ETF:
        - AUM > $10B:       +15 (muy líquido, bajo riesgo de cierre)
        - Yield > 2%:       +10 (genera ingresos)
        - Retorno 3y > 10%: +15 (rendimiento sólido)
        - Retorno 3y > 5%:  +8
        - Retorno 3y < 0:   -15 (mal desempeño histórico)
        """
        try:
            info = _get_fund_info(symbol)

            aum       = info.get('totalAssets')
            yld       = info.get('yield') or info.get('dividendYield')
            ret_3y    = info.get('threeYearAverageReturn')
            ret_5y    = info.get('fiveYearAverageReturn')
            category  = info.get('category') or info.get('fundFamily') or ''

            def safe(v):
                return v is not None and not (isinstance(v, float) and v != v)

            score   = 50
            signals = []
            warnings= []

            # AUM (liquidez)
            if safe(aum):
                if aum > 10e9:
                    score += 15; signals.append(f"AUM ${aum/1e9:.0f}B — muy liquido")
                elif aum > 1e9:
                    score += 8;  signals.append(f"AUM ${aum/1e9:.1f}B")
                elif aum < 100e6:
                    score -= 10; warnings.append(f"AUM bajo ${aum/1e6:.0f}M — riesgo cierre")

            # Retorno 3 años
            if safe(ret_3y):
                r3 = ret_3y * 100
                if r3 > 15:
                    score += 15; signals.append(f"Retorno 3y: +{r3:.1f}%")
                elif r3 > 8:
                    score += 8;  signals.append(f"Retorno 3y: +{r3:.1f}%")
                elif r3 > 0:
                    score += 3
                else:
                    score -= 15; warnings.append(f"Retorno 3y negativo: {r3:.1f}%")

            # Yield
            if safe(yld) and yld > 0:
                y = yld * 100
                if y > 3:
                    score += 10; signals.append(f"Yield {y:.1f}%")
                elif y > 1:
                    score += 5

            score = max(0, min(100, score))

            if score >= 65:
                fund_signal = 'FAVORABLE'
            elif score >= 40:
                fund_signal = 'NEUTRO'
            else:
                fund_signal = 'DESFAVORABLE'

            return {
                'fund_score':   round(score, 1),
                'fund_signal':  fund_signal,
                'pe':           None,   # ETFs no tienen P/E
                'peg':          None,
                'debt_equity':  None,
                'rev_growth':   round(ret_3y * 100, 1) if safe(ret_3y) else None,
                'margins':      round(ret_5y * 100, 1) if safe(ret_5y) else None,
                'signals':      signals,
                'warnings':     warnings,
                '_is_etf':      True,
                '_aum':         aum,
                '_yield':       round(yld * 100, 2) if safe(yld) else None,
            }
        except Exception as e:
            return {
                'fund_score': 50, 'fund_signal': 'N/A',
                'pe': None, 'peg': None, 'debt_equity': None,
                'rev_growth': None, 'margins': None,
                'signals': [], 'warnings': [str(e)[:60]],
                '_is_etf': True,
            }

    def calculate_signal_confidence(self, result_row, vix_data=None):
        """
        Combina TODAS las señales disponibles en un único nivel de confianza.

        Señales contadas:
        - Score técnico (peso doble si >= 70)
        - ADX: tendencia real confirmada
        - Volumen MA: dinero entrando/saliendo
        - Patrón de vela (peso doble si muy fuerte)
        - Fundamentales
        - VIX: contexto macro del mercado

        Returns:
            confidence: MUY ALTA / ALTA / MEDIA / BAJA / MUY BAJA
            score: número de señales alineadas (puede ser negativo)
            aligned: lista de señales a favor
            against: lista de señales en contra
        """
        pts = 0
        aligned = []
        against = []

        # Score técnico
        tech = result_row.get('total_score', 0) or 0
        if tech >= 70:
            pts += 2; aligned.append(f"Score técnico {tech:.0f}/100")
        elif tech >= 55:
            pts += 1
        elif tech < 40:
            pts -= 1

        # ADX
        adx = result_row.get('adx_value')
        if adx:
            if adx > 30:
                pts += 1; aligned.append(f"Tendencia confirmada (ADX {adx:.0f})")
            elif adx < 18:
                pts -= 1; against.append(f"Sin tendencia real (ADX {adx:.0f})")

        # Volumen MA
        vs = result_row.get('vol_signal')
        if vs == 'COMPRAR':
            pts += 1; aligned.append("Dinero entrando (Vol MA)")
        elif vs == 'VENDER':
            pts -= 1; against.append("Dinero saliendo (Vol MA)")

        # Patrón vela
        cs = result_row.get('candle_signal')
        ck = result_row.get('candle_strength', '')
        cp = result_row.get('candle_pattern', '')
        if cs == 'COMPRA':
            if ck == 'muy fuerte':
                pts += 2; aligned.append(f"Vela {cp} (muy fuerte)")
            else:
                pts += 1; aligned.append(f"Vela {cp}")
        elif cs == 'VENTA':
            pts -= 1; against.append(f"Vela bajista: {cp}")

        # Fundamentales
        fs = result_row.get('fund_signal')
        if fs == 'FAVORABLE':
            pts += 1; aligned.append("Fundamentales sólidos")
        elif fs == 'DESFAVORABLE':
            pts -= 1; against.append("Fundamentales débiles")

        # VIX
        if vix_data:
            vix_sig = vix_data.get('signal', '')
            if vix_sig in ('COMPRAR AGRESIVO', 'COMPRAR'):
                pts += 1; aligned.append(f"Miedo en mercado — oportunidad (VIX {vix_data['vix']:.0f})")
            elif vix_sig == 'REDUCIR EXPOSICIÓN':
                pts -= 1; against.append(f"Mercado complaciente (VIX {vix_data['vix']:.0f})")

        # Backtest histórico (fiabilidad de la señal combinada)
        bt_wr = result_row.get('bt_win_rate_vol')
        bt_exp = result_row.get('bt_expectancy_vol')
        if bt_wr is not None:
            if bt_wr >= 65:
                pts += 1; aligned.append(f"Señal históricamente fiable ({bt_wr:.0f}% win rate)")
            elif bt_wr < 40:
                pts -= 1; against.append(f"Señal poco fiable históricamente ({bt_wr:.0f}% win rate)")

        # Riesgo (Sharpe + Max Drawdown)
        sharpe = result_row.get('sharpe')
        max_dd = result_row.get('max_dd')
        if sharpe is not None:
            if sharpe >= 1.5:
                pts += 1; aligned.append(f"Sharpe excelente ({sharpe:.2f})")
            elif sharpe < 0:
                pts -= 1; against.append(f"Sharpe negativo ({sharpe:.2f}) — pierde vs cash")
        if max_dd is not None and max_dd < -35:
            pts -= 1; against.append(f"Drawdown severo ({max_dd:.1f}%)")

        # Fuerza Relativa vs SPY
        rs_sig = result_row.get('rs_signal')
        rs_60d = result_row.get('rs_60d')
        if rs_sig == 'LIDER':
            pts += 1; aligned.append(f"Supera al S&P500 (RS60d {rs_60d:+.1f}%)" if rs_60d else "Lider vs S&P500")
        elif rs_sig == 'REZAGADO':
            pts -= 1; against.append(f"Rezagado vs S&P500 (RS60d {rs_60d:+.1f}%)" if rs_60d else "Rezagado vs S&P500")

        # Multi-Timeframe alignment
        tf_pts = result_row.get('tf_conf_pts')
        if tf_pts is not None:
            tf_align = result_row.get('tf_alignment', '')
            if tf_pts >= 2:
                pts += 2; aligned.append(f"Multi-TF alineado alcista")
            elif tf_pts == 1:
                pts += 1; aligned.append(f"Tendencia semanal alcista")
            elif tf_pts == -1:
                pts -= 1; against.append(f"Tendencia semanal bajista ({tf_align})")
            elif tf_pts <= -2:
                pts -= 2; against.append(f"Multi-TF alineado bajista")

        if pts >= 5:
            confidence = 'MUY ALTA'
        elif pts >= 3:
            confidence = 'ALTA'
        elif pts >= 1:
            confidence = 'MEDIA'
        elif pts >= -1:
            confidence = 'BAJA'
        else:
            confidence = 'MUY BAJA'

        return {
            'confidence': confidence,
            'score': pts,
            'aligned': aligned,
            'against': against,
        }

    def _target_time_range(self, gain_pct, atr_pct, adx_value=None):
        """
        Calcula rango realista de días para alcanzar un objetivo de precio.

        Lógica:
        - El rango diario efectivo = fracción del ATR (no todo el ATR es dirección neta)
        - Con tendencia fuerte (ADX alto), el precio avanza más directamente
        - Con baja tendencia (ADX < 20), el precio meandrea → más días necesarios

        Returns: (dias_minimo, dias_maximo)
        """
        if atr_pct <= 0 or gain_pct <= 0:
            return 10, 60

        # Factor de avance diario neto según fuerza de tendencia
        if adx_value and adx_value > 35:
            fast, slow = 0.85, 0.45   # Tendencia muy fuerte
        elif adx_value and adx_value > 25:
            fast, slow = 0.65, 0.32   # Tendencia buena
        elif adx_value and adx_value > 20:
            fast, slow = 0.48, 0.22   # Tendencia débil
        else:
            fast, slow = 0.32, 0.14   # Sin tendencia clara

        days_min = max(3,  int(gain_pct / (atr_pct * fast)))
        days_max = max(days_min + 5, int(gain_pct / (atr_pct * slow)))

        # Caps razonables
        days_min = min(days_min, 120)
        days_max = min(days_max, 365)

        return days_min, days_max

    def calculate_fibonacci_levels(self, df, lookback=60):
        """Calcula niveles de Fibonacci basados en swing reciente"""
        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low

        return {
            'fib_0': high,
            'fib_236': high - (diff * 0.236),
            'fib_382': high - (diff * 0.382),
            'fib_500': high - (diff * 0.500),
            'fib_618': high - (diff * 0.618),
            'fib_786': high - (diff * 0.786),
            'fib_100': low
        }

    def find_recent_support_resistance(self, df, window=20):
        """Encuentra niveles de soporte/resistencia recientes"""
        recent = df.tail(100)

        # Resistencias: máximos locales
        resistance_levels = []
        for i in range(window, len(recent) - window):
            if recent['high'].iloc[i] == recent['high'].iloc[i-window:i+window].max():
                resistance_levels.append(recent['high'].iloc[i])

        # Soportes: mínimos locales
        support_levels = []
        for i in range(window, len(recent) - window):
            if recent['low'].iloc[i] == recent['low'].iloc[i-window:i+window].min():
                support_levels.append(recent['low'].iloc[i])

        return {
            'supports': sorted(support_levels) if support_levels else [recent['low'].min()],
            'resistances': sorted(resistance_levels, reverse=True) if resistance_levels else [recent['high'].max()]
        }

    def generate_recommendation(self, symbol, analysis, df=None, adx_value=None):
        """Genera recomendación de compra/venta con niveles mejorados"""
        price = analysis['price']
        score = analysis['total_score']
        rsi = analysis['rsi']
        change_5d = analysis['change_5d']
        change_20d = analysis['change_20d']

        # Calcular niveles realistas si tenemos el DataFrame
        if df is not None:
            df_sorted = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
            fib_levels = self.calculate_fibonacci_levels(df_sorted)
            sr_levels = self.find_recent_support_resistance(df_sorted)
            bb_lower = analysis.get('bb_lower', price * 0.95)
            bb_upper = analysis.get('bb_upper', price * 1.05)
        else:
            # Fallback si no tenemos DataFrame
            fib_levels = None
            sr_levels = {'supports': [price * 0.90], 'resistances': [price * 1.15]}
            bb_lower = price * 0.95
            bb_upper = price * 1.05

        # === DETERMINAR TIMING ===
        timing = "NEUTRAL"
        timing_reason = ""

        # Evaluar momentum y posición
        # Detectar momentum excesivo (ya subió mucho) - EVITAR
        if change_5d > 10 and rsi > 70:
            timing = "ESPERAR RETROCESO"
            timing_reason = (f"⚠️ Stock YA subió {change_5d:.1f}% (RSI {rsi:.0f}). "
                           f"Movimiento ya ocurrió. Esperar corrección a niveles de entrada sugeridos.")

        # Momentum fuerte pero no extremo - ESPERAR PULLBACK
        elif change_5d > 8 and rsi > 65:
            timing = "COMPRAR EN PULLBACK"
            timing_reason = (f"↗️ Subida reciente de {change_5d:.1f}% (RSI {rsi:.0f}). "
                           f"Esperar retroceso menor (3-5%) para mejor punto de entrada.")

        # Subida moderada - PRECAUCIÓN
        elif change_5d > 5 and rsi > 60:
            timing = "ESPERAR PULLBACK"
            timing_reason = (f"📈 Subió {change_5d:.1f}% recientemente. "
                           f"Esperar corrección antes de entrar. No perseguir el precio.")

        # Caída fuerte + sobreventa extrema - OPORTUNIDAD FUERTE
        elif change_5d < -8 and rsi < 30:
            timing = "COMPRAR AHORA"
            timing_reason = (f"🎯 OPORTUNIDAD: Cayó {change_5d:.1f}% (RSI {rsi:.0f}). "
                           f"Sobreventa extrema en stock con score {score:.0f}/100. "
                           f"Corrección creó punto de entrada atractivo.")

        # Caída moderada + RSI bajo - OPORTUNIDAD BUENA
        elif change_5d < -5 and rsi < 40:
            timing = "COMPRAR GRADUAL"
            timing_reason = (f"✅ Corrección de {change_5d:.1f}% (RSI {rsi:.0f}) creó oportunidad. "
                           f"Score {score:.0f}/100 indica fundamentos sólidos. "
                           f"Comprar gradualmente en niveles sugeridos.")

        # Caída leve con score alto - OPORTUNIDAD
        elif change_5d < -2 and score >= 70:
            timing = "COMPRAR AHORA"
            timing_reason = (f"💡 Retroceso menor ({change_5d:.1f}%) en stock de calidad (score {score:.0f}/100). "
                           f"RSI {rsi:.0f} neutral. Corrección normal - buen punto de entrada.")

        # Precio estable, RSI neutral - MOMENTO IDEAL
        elif 40 <= rsi <= 60 and abs(change_5d) < 3:
            timing = "COMPRAR AHORA"
            timing_reason = (f"✅ Condiciones ideales: Precio estable ({change_5d:+.1f}%), RSI {rsi:.0f} neutral. "
                           f"No está ni sobrecomprado ni sobrevendido. Momento óptimo para entrar.")

        # Precio bajó pero score bajo - PRECAUCIÓN
        elif change_5d < -5 and score < 50:
            timing = "ESPERAR"
            timing_reason = (f"⚠️ Cayó {change_5d:.1f}% pero score bajo ({score:.0f}/100). "
                           f"Caída puede continuar. Esperar estabilización.")

        # Default para casos mixtos
        else:
            timing = "COMPRAR GRADUAL"
            timing_reason = (f"→ Cambio {change_5d:+.1f}%, RSI {rsi:.0f}, Score {score:.0f}/100. "
                           f"Condiciones mixtas. Entrar gradualmente usando niveles de entrada.")

        # === NIVELES DE ENTRADA REALISTAS ===
        # Todos los niveles deben estar DEBAJO del precio actual

        # Nivel 1: Pullback conservador (3%)
        entry_1 = price * 0.97

        # Nivel 2: Banda de Bollinger inferior o Fib 38.2% (lo que esté más bajo pero razonable)
        if fib_levels and fib_levels['fib_382'] < price:
            # Usar Fibonacci si está por debajo del precio
            fib_entry = min(fib_levels['fib_382'], fib_levels['fib_500'])
            entry_2 = max(min(fib_entry, price * 0.95), price * 0.90)  # Entre -5% y -10%
        else:
            entry_2 = price * 0.92

        # Nivel 3: Soporte reciente más cercano (entre -8% y -15%)
        if sr_levels and sr_levels['supports']:
            nearby_supports = [s for s in sr_levels['supports'] if price * 0.85 < s < price * 0.92]
            entry_3 = nearby_supports[0] if nearby_supports else price * 0.88
        else:
            entry_3 = price * 0.88

        # Asegurar que todos estén por debajo del precio y ordenados
        entry_levels = [e for e in [entry_1, entry_2, entry_3] if e < price * 0.99]
        if len(entry_levels) < 3:
            # Fallback: crear niveles simples
            entry_levels = [price * 0.97, price * 0.92, price * 0.88]

        entry_levels = sorted(entry_levels, reverse=True)  # De mayor a menor

        # === NIVELES DE SALIDA REALISTAS ===
        # Todos los niveles deben estar ENCIMA del precio actual

        # TP1: Objetivo conservador (+12%)
        exit_1 = price * 1.12

        # TP2: Resistencia cercana o +18%
        if sr_levels and sr_levels['resistances']:
            nearby_resistances = [r for r in sr_levels['resistances'] if price * 1.12 < r < price * 1.28]
            exit_2 = nearby_resistances[0] if nearby_resistances else price * 1.18
        else:
            exit_2 = price * 1.18

        # TP3: Resistencia mayor o +25-30%
        if fib_levels and fib_levels['fib_0'] > price * 1.10:
            exit_3 = min(fib_levels['fib_0'], price * 1.35)
        else:
            exit_3 = price * 1.25

        # Asegurar que todos estén por encima del precio (mínimo +10%)
        exit_levels = [e for e in [exit_1, exit_2, exit_3] if e > price * 1.09]
        if len(exit_levels) < 3:
            # Fallback: crear niveles simples
            exit_levels = [price * 1.12, price * 1.20, price * 1.28]

        exit_levels = sorted(exit_levels)  # De menor a mayor

        # === STOP LOSS REALISTA ===
        # Basado en soporte reciente, pero no más de -10%
        if sr_levels and sr_levels['supports']:
            nearby_supports = [s for s in sr_levels['supports'] if price * 0.88 < s < price * 0.95]
            if nearby_supports:
                nearest_support = max(nearby_supports)
                stop_loss = nearest_support * 0.98  # 2% bajo soporte
            else:
                stop_loss = price * 0.93  # -7% como fallback
        else:
            stop_loss = price * 0.93  # -7% como fallback

        # Asegurar que stop loss no esté demasiado lejos (máximo -10%)
        stop_loss = max(stop_loss, price * 0.90)

        # === RECOMENDACIÓN FINAL ===

        # Base recommendation on score
        if score >= 75:
            base_rec = "COMPRA FUERTE"
            position_size = "Grande (5-10%)"
        elif score >= 60:
            base_rec = "COMPRA"
            position_size = "Media (3-5%)"
        elif score >= 50:
            base_rec = "COMPRA MODERADA"
            position_size = "Pequeña (1-3%)"
        elif score >= 40:
            base_rec = "ESPERAR"
            position_size = "No invertir aún"
        else:
            base_rec = "EVITAR"
            position_size = "No invertir"

        # Ajustar por timing
        if timing == "ESPERAR RETROCESO" and score >= 60:
            recommendation = f"{base_rec} - ESPERAR RETROCESO"
        elif timing == "COMPRAR EN PULLBACK":
            recommendation = f"{base_rec} EN PULLBACK"
        elif timing == "COMPRAR AHORA" and score >= 50:
            recommendation = f"{base_rec} AHORA"
        elif timing == "COMPRAR GRADUAL":
            recommendation = f"{base_rec} GRADUAL"
        else:
            recommendation = base_rec

        # Risk-reward ratio
        avg_target = sum(exit_levels) / len(exit_levels)
        risk = price - stop_loss
        reward = avg_target - price
        rr_ratio = reward / risk if risk > 0 else 0

        # Calcular ganancias y rango temporal para cada objetivo
        atr_pct = analysis.get('atr_pct', 2.0)
        exit_targets = []
        for i, exit_price in enumerate(exit_levels, 1):
            gain_pct = ((exit_price - price) / price) * 100

            # Rango de días usando ATR + ADX
            days_min, days_max = self._target_time_range(
                abs(gain_pct), atr_pct, adx_value)

            exit_targets.append({
                'level': i,
                'price': exit_price,
                'gain_pct': gain_pct,
                'days_min': days_min,
                'days_max': days_max,
                'estimated_days': (days_min + days_max) // 2,  # compatibilidad
            })

        # Explicación del cambio para claridad
        if change_5d > 5:
            change_explanation = f"⬆️ Subió {change_5d:.1f}% en últimos 5 días (PASADO)"
        elif change_5d < -5:
            change_explanation = f"⬇️ Bajó {abs(change_5d):.1f}% en últimos 5 días (PASADO, posible oportunidad)"
        elif change_5d > 0:
            change_explanation = f"↗️ Subió {change_5d:.1f}% en últimos 5 días"
        elif change_5d < 0:
            change_explanation = f"↘️ Bajó {abs(change_5d):.1f}% en últimos 5 días (oportunidad)"
        else:
            change_explanation = "→ Precio estable últimos 5 días"

        return {
            'recommendation': recommendation,
            'position_size': position_size,
            'timing': timing,
            'timing_reason': timing_reason,
            'change_explanation': change_explanation,
            'entry_levels': entry_levels,
            'exit_levels': exit_levels,
            'exit_targets': exit_targets,
            'stop_loss': stop_loss,
            'rr_ratio': rr_ratio,
            'risk_pct': (risk / price) * 100,
            'reward_pct': (reward / price) * 100
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  🎯 Analizador Multi-Acción - 50 Acciones Diversificadas")
    print("=" * 70)
    print()

    db = TradingDatabase()
    analyzer = StockAnalyzer(db)

    print("Este script analiza 50 acciones de diferentes sectores")
    print("y genera recomendaciones de inversión basadas en análisis técnico.")
    print()

    # Descargar datos
    response = input("¿Descargar/actualizar datos históricos? (S/n): ").strip().upper()

    if response != 'N':
        analyzer.download_all_data(interval="1d", force=True)

    # Analizar
    print()
    df_results = analyzer.analyze_all_stocks()

    if df_results.empty:
        print("❌ No se pudieron analizar acciones")
        db.close()
        exit(1)

    # Mostrar top 10
    print()
    print("=" * 70)
    print("  🏆 TOP 10 ACCIONES POR SCORE TÉCNICO")
    print("=" * 70)
    print()

    top_10 = df_results.sort_values('total_score', ascending=False).head(10)
    print(top_10[['symbol', 'sector', 'total_score', 'price', 'change_5d', 'rsi']].to_string(index=False))

    print()
    db.close()
