"""
Generador de Reporte de Inversiones
Analiza 50 acciones y genera reporte completo con recomendaciones
"""
import pandas as pd
from datetime import datetime
from database import TradingDatabase
from stock_analyzer import StockAnalyzer
from stock_universe import STOCK_UNIVERSE, get_sector

def generate_markdown_report(df_results, portfolio, analyzer, db):
    """Genera reporte en formato Markdown"""

    report = []
    report.append("# 📊 Reporte de Análisis de Inversiones")
    report.append(f"\n**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Acciones analizadas:** {len(df_results)}")
    report.append("\n---\n")

    # Resumen por sectores
    report.append("## 🏭 Distribución por Sectores\n")
    sector_summary = df_results.groupby('sector').agg({
        'total_score': 'mean',
        'symbol': 'count'
    }).round(1)
    sector_summary.columns = ['Score Promedio', 'Cantidad']
    sector_summary = sector_summary.sort_values('Score Promedio', ascending=False)

    report.append(sector_summary.to_markdown())
    report.append("\n---\n")

    # Top 20 acciones
    report.append("## 🏆 Top 20 Acciones por Score Técnico\n")
    top_20 = df_results.sort_values('total_score', ascending=False).head(20)

    report.append("| # | Símbolo | Sector | Score | Precio | Cambio 5d | RSI | Recomendación |")
    report.append("|---|---------|--------|-------|--------|-----------|-----|---------------|")

    for i, (idx, row) in enumerate(top_20.iterrows(), 1):
        # Obtener DataFrame para niveles realistas
        df_stock = db.get_precios(row['symbol'])

        analysis_data = {
            'price': row['price'],
            'support_52w': row['support_52w'],
            'resistance_52w': row['resistance_52w'],
            'total_score': row['total_score'],
            'rsi': row['rsi'],
            'change_5d': row['change_5d'],
            'change_20d': row['change_20d']
        }
        rec = analyzer.generate_recommendation(row['symbol'], analysis_data, df_stock if not df_stock.empty else None)

        report.append(
            f"| {i} | **{row['symbol']}** | {row['sector']} | "
            f"{row['total_score']:.1f}/100 | ${row['price']:.2f} | "
            f"{row['change_5d']:+.1f}% | {row['rsi']:.0f} | "
            f"{rec['recommendation']} |"
        )

    report.append("\n---\n")

    # Portfolio diversificado recomendado
    report.append("## 🎯 Portfolio Diversificado Recomendado\n")
    report.append("\n**Criterios de selección:**")
    report.append("- Top scores técnicos")
    report.append("- Baja correlación entre acciones (≤70%)")
    report.append("- Diversificación sectorial\n")

    portfolio_df = df_results[df_results['symbol'].isin(portfolio)].sort_values('total_score', ascending=False)

    report.append("| # | Símbolo | Sector | Score | Precio | Tamaño Posición |")
    report.append("|---|---------|--------|-------|--------|-----------------|")

    for i, (idx, row) in enumerate(portfolio_df.iterrows(), 1):
        df_stock = db.get_precios(row['symbol'])

        analysis_data = {
            'price': row['price'],
            'support_52w': row['support_52w'],
            'resistance_52w': row['resistance_52w'],
            'total_score': row['total_score'],
            'rsi': row['rsi'],
            'change_5d': row['change_5d'],
            'change_20d': row['change_20d']
        }
        rec = analyzer.generate_recommendation(row['symbol'], analysis_data, df_stock if not df_stock.empty else None)

        report.append(
            f"| {i} | **{row['symbol']}** | {row['sector']} | "
            f"{row['total_score']:.1f}/100 | ${row['price']:.2f} | "
            f"{rec['position_size']} |"
        )

    report.append("\n---\n")

    # Detalles de las 10 mejores
    report.append("## 📈 Análisis Detallado - Top 10\n")

    top_10 = df_results.sort_values('total_score', ascending=False).head(10)

    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        report.append(f"### {i}. {row['symbol']} - {row['sector']}")
        report.append(f"\n**Score Total:** {row['total_score']:.1f}/100")
        report.append(f"\n**Precio Actual:** ${row['price']:.2f}")

        # Scores parciales
        report.append("\n**Scores por Componente:**")
        report.append(f"- Fuerza de Tendencia: {row['trend_strength']:.0f}/100 (25%)")
        report.append(f"- Momentum: {row['momentum']:.0f}/100 (20%)")
        report.append(f"- Volatilidad: {row['volatility']:.0f}/100 (15%)")
        report.append(f"- Volumen: {row['volume']:.0f}/100 (15%)")
        report.append(f"- Acción del Precio: {row['price_action']:.0f}/100 (15%)")
        report.append(f"- Soporte/Resistencia: {row['support_resistance']:.0f}/100 (10%)")

        # Métricas
        report.append("\n**Métricas Clave:**")
        report.append(f"- RSI: {row['rsi']:.0f}")
        report.append(f"- Cambio 5 días: {row['change_5d']:+.2f}%")
        report.append(f"- Cambio 20 días: {row['change_20d']:+.2f}%")
        report.append(f"- Soporte 52 semanas: ${row['support_52w']:.2f}")
        report.append(f"- Resistencia 52 semanas: ${row['resistance_52w']:.2f}")

        # Recomendación
        df_stock = db.get_precios(row['symbol'])

        analysis_data = {
            'price': row['price'],
            'support_52w': row['support_52w'],
            'resistance_52w': row['resistance_52w'],
            'total_score': row['total_score'],
            'rsi': row['rsi'],
            'change_5d': row['change_5d'],
            'change_20d': row['change_20d']
        }
        rec = analyzer.generate_recommendation(row['symbol'], analysis_data, df_stock if not df_stock.empty else None)

        report.append(f"\n**Recomendación:** {rec['recommendation']}")
        report.append(f"\n**Tamaño de Posición:** {rec['position_size']}")

        # Agregar timing
        report.append(f"\n**⏰ Timing:** {rec['timing']}")
        report.append(f"\n**Razón:** {rec['timing_reason']}")

        report.append("\n**Niveles de Entrada Sugeridos:**")
        for j, level in enumerate(rec['entry_levels'], 1):
            report.append(f"  - Nivel {j}: ${level:.2f}")

        report.append("\n**Niveles de Salida Sugeridos:**")
        for j, level in enumerate(rec['exit_levels'], 1):
            report.append(f"  - Nivel {j}: ${level:.2f}")

        report.append(f"\n**Stop Loss:** ${rec['stop_loss']:.2f}")
        report.append(f"\n**Risk/Reward:** {rec['rr_ratio']:.2f}:1 (Riesgo: {rec['risk_pct']:.1f}% | Recompensa: {rec['reward_pct']:.1f}%)\n")
        report.append("\n---\n")

    # Notas finales
    report.append("## ⚠️ Advertencias Importantes\n")
    report.append("1. **Este análisis es únicamente técnico** - No considera factores fundamentales")
    report.append("2. **No es asesoría financiera** - Consulta con un profesional antes de invertir")
    report.append("3. **Diversificación es clave** - No inviertas todo en una sola acción")
    report.append("4. **Usa stop losses** - Protege tu capital contra movimientos adversos")
    report.append("5. **Los mercados cambian** - Revisa análisis periódicamente")
    report.append("\n---\n")

    # Metodología
    report.append("## 📖 Metodología\n")
    report.append("\n**Sistema de Scoring (0-100):**\n")
    report.append("- **Fuerza de Tendencia (25%):** EMA 9, 21, 50")
    report.append("- **Momentum (20%):** RSI, MACD")
    report.append("- **Volatilidad (15%):** Bollinger Bands, ATR")
    report.append("- **Volumen (15%):** Volumen relativo vs promedio")
    report.append("- **Acción del Precio (15%):** Cambios 5d, 20d")
    report.append("- **Soporte/Resistencia (10%):** Distancia a máx/mín 52 semanas")
    report.append("\n**Interpretación de Scores:**")
    report.append("- **75-100:** Compra Fuerte (5-10% del portfolio)")
    report.append("- **60-75:** Compra (3-5% del portfolio)")
    report.append("- **50-60:** Compra Moderada (1-3% del portfolio)")
    report.append("- **40-50:** Esperar mejor momento")
    report.append("- **0-40:** Evitar")

    return "\n".join(report)


def main():
    print("=" * 70)
    print("  📊 Generador de Reporte de Inversiones")
    print("  50 Acciones Diversificadas - Análisis Técnico Completo")
    print("=" * 70)
    print()

    # Conectar a BD
    db = TradingDatabase()
    analyzer = StockAnalyzer(db)

    # Paso 1: Descargar datos
    print("📥 Paso 1: Descarga de Datos Históricos")
    print()
    response = input("¿Descargar/actualizar datos de las 50 acciones? (S/n): ").strip().upper()

    if response != 'N':
        success, failed = analyzer.download_all_data(interval="1d", force=True)
        print()

        if success < 30:
            print("⚠️  Muy pocas acciones descargadas. Revisa tu conexión.")
            response = input("¿Continuar de todos modos? (s/N): ").strip().upper()
            if response != 'S':
                db.close()
                return

    # Paso 2: Analizar acciones
    print()
    print("=" * 70)
    print("📊 Paso 2: Análisis Técnico")
    print("=" * 70)
    print()

    df_results = analyzer.analyze_all_stocks()

    if df_results.empty:
        print("❌ No se pudieron analizar acciones suficientes")
        db.close()
        return

    # Paso 3: Portfolio diversificado
    print()
    print("=" * 70)
    print("🎯 Paso 3: Selección de Portfolio Diversificado")
    print("=" * 70)
    print()

    portfolio = analyzer.find_diversified_portfolio(df_results, n=10, max_corr=0.7)

    # Paso 4: Generar reporte
    print()
    print("=" * 70)
    print("📝 Paso 4: Generación de Reporte")
    print("=" * 70)
    print()

    report_content = generate_markdown_report(df_results, portfolio, analyzer, db)

    # Guardar reporte
    filename = f"REPORTE_INVERSIONES_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ Reporte guardado: {filename}")
    print()

    # Guardar también CSV con todos los resultados
    csv_filename = f"analisis_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"✅ Datos CSV guardados: {csv_filename}")
    print()

    # Mostrar resumen en consola
    print("=" * 70)
    print("  🏆 RESUMEN - TOP 10 ACCIONES")
    print("=" * 70)
    print()

    top_10 = df_results.sort_values('total_score', ascending=False).head(10)

    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        df_stock = db.get_precios(row['symbol'])

        analysis_data = {
            'price': row['price'],
            'support_52w': row['support_52w'],
            'resistance_52w': row['resistance_52w'],
            'total_score': row['total_score'],
            'rsi': row['rsi'],
            'change_5d': row['change_5d'],
            'change_20d': row['change_20d']
        }
        rec = analyzer.generate_recommendation(row['symbol'], analysis_data, df_stock if not df_stock.empty else None)

        # Header con data freshness
        data_status = row.get('data_freshness', 'N/A')
        last_update = row.get('last_update', 'N/A')

        print(f"{i:2}. {row['symbol']:6} ({row['sector']:15}) - Score: {row['total_score']:5.1f}/100")
        print(f"    Precio: ${row['price']:7.2f} | RSI: {row['rsi']:5.0f} | Datos: {last_update} {data_status}")
        print(f"    {rec['change_explanation']}")
        print()
        print(f"    → {rec['recommendation']} ({rec['position_size']})")
        print(f"    ⏰ {rec['timing']}")
        print(f"    💡 {rec['timing_reason']}")
        print()
        print(f"    → Comprar: ${rec['entry_levels'][0]:.2f} | ${rec['entry_levels'][1]:.2f} | ${rec['entry_levels'][2]:.2f}")
        print(f"    → Objetivos de Venta:")
        for target in rec['exit_targets']:
            print(f"       TP{target['level']}: ${target['price']:.2f} (+{target['gain_pct']:.1f}% en ~{target['estimated_days']} días)")
        print(f"    🛡️ Stop Loss: ${rec['stop_loss']:.2f} | R/R: {rec['rr_ratio']:.2f}:1")
        print()

    print("=" * 70)
    print()
    print(f"📄 Revisa el reporte completo en: {filename}")
    print()

    db.close()


if __name__ == "__main__":
    main()
