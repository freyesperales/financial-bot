"""
Backtester - Motor de Backtesting para Trading Bot
Simula estrategia con datos históricos y calcula métricas
"""
import pandas as pd
import numpy as np
from datetime import datetime
from database import TradingDatabase
from indicators_advanced import AdvancedIndicators
import indicators_core as ic


class Backtester:
    def __init__(self, symbol, strategy_params=None):
        """
        Inicializar backtester

        Args:
            symbol: Ticker de la acción
            strategy_params: Parámetros de estrategia
        """
        self.symbol = symbol
        self.db = TradingDatabase()
        self.indicators = AdvancedIndicators()

        # Parámetros por defecto
        self.params = strategy_params or {
            'votos_min': 4,
            'stop_loss_pct': 1.5,
            'rr_ratio': 2.0,
            'rsi_oversold': 35,
            'rsi_overbought': 65
        }

        # Resultados
        self.trades = []
        self.equity_curve = []
        self.metrics = {}

    def load_historical_data(self, start_date=None, end_date=None):
        """Cargar datos históricos de la base de datos"""
        print(f"📥 Cargando datos históricos de {self.symbol}...")

        df = self.db.get_precios(self.symbol, start_date, end_date)

        if df.empty:
            print("❌ No hay datos en la base de datos")
            print("   Ejecuta primero: python data_downloader.py")
            return None

        # Ordenar por fecha
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)

        print(f"✅ Cargados {len(df)} registros")
        print(f"   Desde: {df['timestamp'].min()}")
        print(f"   Hasta: {df['timestamp'].max()}")

        return df

    def calculate_indicators(self, df):
        """Calcular indicadores técnicos usando indicators_core como fuente única."""
        print("📊 Calculando indicadores...")
        try:
            result = ic.compute_all(df)
            print("✅ Indicadores calculados")
            return result
        except Exception as e:
            print(f"⚠️  Error calculando indicadores: {e}")
            return df.copy()

    def generate_signals(self, df):
        """Generar señales de trading basadas en estrategia de votación"""
        print("🔮 Generando señales...")

        signals = []

        for i in range(26, len(df)):  # Empezar desde 26 para tener datos suficientes
            row = df.iloc[i]
            prev = df.iloc[i-1]

            votos_compra = 0
            votos_venta = 0
            detalle = {}

            # 1. RSI
            if not pd.isna(row['rsi']):
                if row['rsi'] < self.params['rsi_oversold']:
                    votos_compra += 1
                    detalle['RSI'] = f"{row['rsi']:.1f} sobrevendido"
                elif row['rsi'] > self.params['rsi_overbought']:
                    votos_venta += 1
                    detalle['RSI'] = f"{row['rsi']:.1f} sobrecomprado"

            # 2. MACD
            if not pd.isna(row['macd']) and not pd.isna(prev['macd']):
                if prev['macd'] < prev['macd_signal'] and row['macd'] > row['macd_signal']:
                    votos_compra += 1
                    detalle['MACD'] = "cruce alcista"
                elif prev['macd'] > prev['macd_signal'] and row['macd'] < row['macd_signal']:
                    votos_venta += 1
                    detalle['MACD'] = "cruce bajista"

            # 3. Bollinger Bands
            if not pd.isna(row['bb_lower']) and not pd.isna(row['bb_upper']):
                if row['close'] <= row['bb_lower']:
                    votos_compra += 1
                    detalle['BB'] = "bajo banda inferior"
                elif row['close'] >= row['bb_upper']:
                    votos_venta += 1
                    detalle['BB'] = "sobre banda superior"

            # 4. EMA Crossover
            if not pd.isna(row['ema_9']) and not pd.isna(row['ema_21']):
                if prev['ema_9'] < prev['ema_21'] and row['ema_9'] > row['ema_21']:
                    votos_compra += 1
                    detalle['EMA'] = "cruce alcista"
                elif prev['ema_9'] > prev['ema_21'] and row['ema_9'] < row['ema_21']:
                    votos_venta += 1
                    detalle['EMA'] = "cruce bajista"

            # 5. ADX (indicador avanzado)
            window = df.iloc[i-14:i+1]
            adx = self.indicators.adx(window)
            if adx and adx['signal']:
                if adx['signal'] == "COMPRA" and adx['trend_strength'] == "fuerte":
                    votos_compra += 1
                    detalle['ADX'] = f"tendencia alcista fuerte ({adx['adx']:.1f})"
                elif adx['signal'] == "VENTA" and adx['trend_strength'] == "fuerte":
                    votos_venta += 1
                    detalle['ADX'] = f"tendencia bajista fuerte ({adx['adx']:.1f})"

            # 6. Patrones de velas
            patterns = self.indicators.detect_candlestick_patterns(window)
            for pattern in patterns:
                if pattern['strength'] in ['fuerte', 'muy fuerte']:
                    if pattern['signal'] == "COMPRA":
                        votos_compra += 1
                        detalle['Patrón'] = pattern['pattern']
                    elif pattern['signal'] == "VENTA":
                        votos_venta += 1
                        detalle['Patrón'] = pattern['pattern']

            # Determinar señal final
            signal = None
            if votos_compra >= self.params['votos_min']:
                signal = "COMPRA"
            elif votos_venta >= self.params['votos_min']:
                signal = "VENTA"

            if signal:
                signals.append({
                    'index': i,
                    'timestamp': row['timestamp'],
                    'signal': signal,
                    'price': row['close'],
                    'votos_compra': votos_compra,
                    'votos_venta': votos_venta,
                    'detalle': detalle,
                    'atr': row['atr'] if not pd.isna(row['atr']) else row['close'] * 0.015
                })

        print(f"✅ {len(signals)} señales generadas")
        return signals

    def simulate_trades(self, df, signals):
        """Simular operaciones basadas en señales"""
        print("💰 Simulando operaciones...")

        in_position = False
        entry_price = None
        entry_index = None
        entry_signal = None
        stop_loss = None
        take_profit = None

        for signal in signals:
            if not in_position and signal['signal'] == "COMPRA":
                # ENTRADA
                in_position = True
                entry_price = signal['price']
                entry_index = signal['index']
                entry_signal = signal

                # Calcular SL y TP
                atr = signal['atr']
                stop_loss = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.params['rr_ratio'])

            elif in_position:
                # Verificar si se alcanza SL o TP
                current_price = signal['price']
                current_index = signal['index']

                # Revisar cada vela desde entrada hasta señal actual
                for i in range(entry_index, current_index + 1):
                    row = df.iloc[i]

                    # Stop Loss
                    if row['low'] <= stop_loss:
                        self.trades.append({
                            'entry_date': entry_signal['timestamp'],
                            'entry_price': entry_price,
                            'exit_date': row['timestamp'],
                            'exit_price': stop_loss,
                            'pnl': stop_loss - entry_price,
                            'pnl_pct': ((stop_loss - entry_price) / entry_price) * 100,
                            'reason': 'STOP_LOSS',
                            'votos': entry_signal['votos_compra']
                        })
                        in_position = False
                        break

                    # Take Profit
                    if row['high'] >= take_profit:
                        self.trades.append({
                            'entry_date': entry_signal['timestamp'],
                            'entry_price': entry_price,
                            'exit_date': row['timestamp'],
                            'exit_price': take_profit,
                            'pnl': take_profit - entry_price,
                            'pnl_pct': ((take_profit - entry_price) / entry_price) * 100,
                            'reason': 'TAKE_PROFIT',
                            'votos': entry_signal['votos_compra']
                        })
                        in_position = False
                        break

                # Señal de venta
                if in_position and signal['signal'] == "VENTA":
                    self.trades.append({
                        'entry_date': entry_signal['timestamp'],
                        'entry_price': entry_price,
                        'exit_date': signal['timestamp'],
                        'exit_price': signal['price'],
                        'pnl': signal['price'] - entry_price,
                        'pnl_pct': ((signal['price'] - entry_price) / entry_price) * 100,
                        'reason': 'SEÑAL_VENTA',
                        'votos': entry_signal['votos_compra']
                    })
                    in_position = False

        print(f"✅ {len(self.trades)} operaciones simuladas")
        return self.trades

    def calculate_metrics(self):
        """Calcular métricas de rendimiento"""
        print("📈 Calculando métricas...")

        if not self.trades:
            print("❌ No hay trades para analizar")
            return None

        df_trades = pd.DataFrame(self.trades)

        # Métricas básicas
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100

        # P&L
        total_pnl = df_trades['pnl'].sum()
        total_pnl_pct = df_trades['pnl_pct'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0

        # Profit Factor
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max Drawdown
        cumulative_pnl = df_trades['pnl_pct'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()

        # Sharpe Ratio (simplified)
        returns = df_trades['pnl_pct']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': expectancy,
            'best_trade': df_trades['pnl_pct'].max(),
            'worst_trade': df_trades['pnl_pct'].min()
        }

        return self.metrics

    def print_report(self):
        """Imprimir reporte de resultados"""
        if not self.metrics:
            return

        print("\n" + "=" * 70)
        print(f"  📊 REPORTE DE BACKTESTING - {self.symbol}")
        print("=" * 70)
        print(f"\n  🎯 ESTRATEGIA:")
        print(f"     Votos mínimos: {self.params['votos_min']}/6")
        print(f"     Stop Loss: {self.params['stop_loss_pct']}%")
        print(f"     Ratio R/R: {self.params['rr_ratio']}:1")

        print(f"\n  📈 RESULTADOS GENERALES:")
        print(f"     Total operaciones: {self.metrics['total_trades']}")
        print(f"     Ganadoras: {self.metrics['winning_trades']} ({self.metrics['win_rate']:.1f}%)")
        print(f"     Perdedoras: {self.metrics['losing_trades']}")

        print(f"\n  💰 RENDIMIENTO:")
        print(f"     P&L Total: {self.metrics['total_pnl_pct']:+.2f}%")
        print(f"     Ganancia promedio: +{self.metrics['avg_win']:.2f}%")
        print(f"     Pérdida promedio: {self.metrics['avg_loss']:.2f}%")
        print(f"     Mejor operación: +{self.metrics['best_trade']:.2f}%")
        print(f"     Peor operación: {self.metrics['worst_trade']:.2f}%")

        print(f"\n  📊 MÉTRICAS AVANZADAS:")
        print(f"     Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"     Max Drawdown: -{self.metrics['max_drawdown']:.2f}%")
        print(f"     Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"     Expectancy: {self.metrics['expectancy']:+.2f}%")

        print(f"\n  🎭 INTERPRETACIÓN:")
        if self.metrics['win_rate'] >= 60:
            print("     ✅ Win rate excelente (≥60%)")
        elif self.metrics['win_rate'] >= 50:
            print("     ✅ Win rate bueno (≥50%)")
        else:
            print("     ⚠️  Win rate bajo (<50%)")

        if self.metrics['profit_factor'] >= 2:
            print("     ✅ Profit factor excelente (≥2.0)")
        elif self.metrics['profit_factor'] >= 1.5:
            print("     ✅ Profit factor bueno (≥1.5)")
        elif self.metrics['profit_factor'] >= 1:
            print("     ⚠️  Profit factor marginal (≥1.0)")
        else:
            print("     ❌ Profit factor negativo (<1.0)")

        if self.metrics['sharpe_ratio'] >= 2:
            print("     ✅ Sharpe ratio excelente (≥2.0)")
        elif self.metrics['sharpe_ratio'] >= 1:
            print("     ✅ Sharpe ratio bueno (≥1.0)")
        else:
            print("     ⚠️  Sharpe ratio bajo (<1.0)")

        print("\n" + "=" * 70)

    def run(self, start_date=None, end_date=None):
        """Ejecutar backtest completo"""
        print("\n🚀 Iniciando Backtesting...")
        print("=" * 70)

        # 1. Cargar datos
        df = self.load_historical_data(start_date, end_date)
        if df is None:
            return

        # 2. Calcular indicadores
        df = self.calculate_indicators(df)

        # 3. Generar señales
        signals = self.generate_signals(df)

        # 4. Simular trades
        trades = self.simulate_trades(df, signals)

        # 5. Calcular métricas
        self.calculate_metrics()

        # 6. Mostrar reporte
        self.print_report()

        return self.metrics


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    import sys

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "AMD"

    # Crear y ejecutar backtester
    bt = Backtester(symbol)
    bt.run()

    # Cerrar DB
    bt.db.close()


if __name__ == "__main__":
    main()
