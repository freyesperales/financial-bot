"""
Optimización de Backtesting
Prueba diferentes configuraciones y timeframes para encontrar la mejor estrategia
"""
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'closed') and not sys.stdout.buffer.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (ValueError, AttributeError):
        pass  # Already wrapped or using different encoding system

from database import TradingDatabase
from data_downloader import DataDownloader
from backtester import Backtester
import pandas as pd

print("=" * 70)
print("  🎯 Optimización de Estrategia - Trading Bot")
print("=" * 70)
print()

# Conectar a BD
db = TradingDatabase()
dl = DataDownloader(db)

# 1. Descargar datos horarios (mejor para esta estrategia)
print("📥 Paso 1: Descargar datos horarios (1h)")
print("   Los datos horarios son mejores para esta estrategia intraday")
print()

response = input("¿Descargar datos horarios de AMD (~2 años)? (S/n): ").strip().upper()

if response != 'N':
    # Limpiar datos diarios
    print()
    print("🗑️  Limpiando datos diarios antiguos...")
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM precios WHERE symbol = 'AMD'")
    conn.commit()
    print("✅ Datos antiguos eliminados")
    print()

    # Descargar 1h
    print("📥 Descargando datos horarios (1h)...")
    df = dl.get_max_historical_data("AMD", interval="1h", force_download=True)

    if df is None:
        print("❌ Error descargando datos")
        db.close()
        exit(1)
else:
    print("⏭️  Usando datos existentes")

print()
print("=" * 70)
print("  🔬 Paso 2: Probar Diferentes Configuraciones")
print("=" * 70)
print()

# Configuraciones a probar
configs = [
    # Config 1: Original (restrictivo)
    {
        'name': 'Original',
        'votos_min': 4,
        'stop_loss_pct': 1.5,
        'rr_ratio': 2.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65
    },
    # Config 2: Más permisivo (más señales)
    {
        'name': 'Permisivo',
        'votos_min': 3,
        'stop_loss_pct': 2.0,
        'rr_ratio': 2.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65
    },
    # Config 3: Stop loss más amplio
    {
        'name': 'SL Amplio',
        'votos_min': 4,
        'stop_loss_pct': 2.5,
        'rr_ratio': 2.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65
    },
    # Config 4: RSI más extremo
    {
        'name': 'RSI Extremo',
        'votos_min': 3,
        'stop_loss_pct': 2.0,
        'rr_ratio': 2.0,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    },
    # Config 5: Ratio R/R mayor
    {
        'name': 'RR 3:1',
        'votos_min': 3,
        'stop_loss_pct': 2.0,
        'rr_ratio': 3.0,
        'rsi_oversold': 35,
        'rsi_overbought': 65
    },
]

results = []

for i, config in enumerate(configs, 1):
    print(f"\n{'─' * 70}")
    print(f"  Configuración {i}/{len(configs)}: {config['name']}")
    print(f"{'─' * 70}")
    print(f"  Votos: {config['votos_min']}/6")
    print(f"  Stop Loss: {config['stop_loss_pct']}%")
    print(f"  R/R: {config['rr_ratio']}:1")
    print(f"  RSI: {config['rsi_oversold']}/{config['rsi_overbought']}")
    print()

    # Ejecutar backtest
    bt = Backtester("AMD", strategy_params=config)
    metrics = bt.run()

    if metrics and metrics['total_trades'] > 0:
        # Calcular score (combinación de métricas)
        # Score = win_rate * profit_factor * (1 - max_drawdown/100)
        score = (
            metrics['win_rate'] / 100 *
            max(metrics['profit_factor'], 0.1) *
            max(1 - metrics['max_drawdown'] / 100, 0.1)
        ) * 100

        results.append({
            'config': config['name'],
            'votos_min': config['votos_min'],
            'stop_loss': config['stop_loss_pct'],
            'rr_ratio': config['rr_ratio'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'pnl': metrics['total_pnl_pct'],
            'profit_factor': metrics['profit_factor'],
            'max_dd': metrics['max_drawdown'],
            'sharpe': metrics['sharpe_ratio'],
            'score': score
        })
    else:
        print("⚠️  No se generaron operaciones con esta configuración")

    print()

# Mostrar resultados comparativos
print()
print("=" * 70)
print("  📊 COMPARACIÓN DE RESULTADOS")
print("=" * 70)
print()

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)

    print(df_results.to_string(index=False))
    print()

    # Mejor configuración
    best = df_results.iloc[0]

    print()
    print("=" * 70)
    print("  🏆 MEJOR CONFIGURACIÓN")
    print("=" * 70)
    print()
    print(f"  Nombre: {best['config']}")
    print(f"  Votos mínimos: {best['votos_min']}/6")
    print(f"  Stop Loss: {best['stop_loss']}%")
    print(f"  Ratio R/R: {best['rr_ratio']}:1")
    print()
    print(f"  📊 Resultados:")
    print(f"     Total operaciones: {best['trades']:.0f}")
    print(f"     Win rate: {best['win_rate']:.1f}%")
    print(f"     P&L Total: {best['pnl']:+.2f}%")
    print(f"     Profit Factor: {best['profit_factor']:.2f}")
    print(f"     Max Drawdown: -{best['max_dd']:.2f}%")
    print(f"     Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"     Score: {best['score']:.2f}")
    print()

    # Recomendación
    if best['win_rate'] >= 55 and best['profit_factor'] >= 1.5:
        print("  ✅ RECOMENDACIÓN: Estrategia viable")
        print("     - Win rate >= 55%")
        print("     - Profit factor >= 1.5")
        print("     - Proceder a paper trading")
    elif best['win_rate'] >= 50 and best['profit_factor'] >= 1.2:
        print("  ⚠️  RECOMENDACIÓN: Estrategia marginal")
        print("     - Resultados aceptables pero no óptimos")
        print("     - Considera más optimización")
        print("     - Paper trading con precaución")
    else:
        print("  ❌ RECOMENDACIÓN: Estrategia no viable")
        print("     - Resultados insuficientes")
        print("     - Necesita mejoras significativas")
        print("     - NO usar con dinero real")

    print()
    print("=" * 70)

    # Guardar configuración óptima
    print()
    response = input("¿Guardar mejor configuración como predeterminada? (S/n): ").strip().upper()

    if response != 'N':
        import json

        config_file = {
            'strategy': {
                'votos_min': int(best['votos_min']),
                'stop_loss_pct': float(best['stop_loss']),
                'rr_ratio': float(best['rr_ratio']),
                'rsi_oversold': 35,
                'rsi_overbought': 65
            },
            'backtest_results': {
                'trades': int(best['trades']),
                'win_rate': float(best['win_rate']),
                'pnl': float(best['pnl']),
                'profit_factor': float(best['profit_factor']),
                'sharpe_ratio': float(best['sharpe'])
            }
        }

        with open('optimal_config.json', 'w') as f:
            json.dump(config_file, f, indent=2)

        print("✅ Configuración guardada en: optimal_config.json")
        print("   Puedes usar esta configuración en el bot")
else:
    print("❌ No se pudieron generar resultados")
    print("   Verifica que los datos se hayan descargado correctamente")

print()
print("=" * 70)
print("  ✅ Optimización Completada")
print("=" * 70)

# Cerrar
db.close()
