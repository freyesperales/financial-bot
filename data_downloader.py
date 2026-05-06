"""
Data Downloader - Descarga máximo histórico posible de acciones
Intenta obtener el rango más amplio disponible en yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from database import TradingDatabase


class DataDownloader:
    def __init__(self, db=None):
        self.db = db or TradingDatabase()

    def get_max_historical_data(self, symbol, interval="1d", force_download=False):
        """
        Descarga máximo histórico posible para una acción

        Límites de yfinance:
        - 1d: ~20+ años (depende de la acción)
        - 1h: ~730 días (2 años)
        - 5m: ~60 días
        - 1m: ~7 días

        Args:
            symbol: Ticker de la acción (ej: "AMD")
            interval: Timeframe ("1d", "1h", "5m", "1m")
            force_download: Si True, descarga aunque ya exista en DB

        Returns:
            DataFrame con datos históricos
        """
        print(f"📥 Descargando histórico de {symbol} (interval: {interval})...")

        # Verificar si ya tenemos datos en DB
        if not force_download:
            existing = self.db.get_precios(symbol)
            if not existing.empty:
                print(f"  ℹ️  Ya hay {len(existing)} registros en BD")
                print(f"  📅 Desde: {existing['timestamp'].min()}")
                print(f"  📅 Hasta: {existing['timestamp'].max()}")

                response = input("  ¿Descargar de nuevo? (s/N): ").lower()
                if response != 's':
                    return existing

        # Determinar período según interval
        periods = {
            "1d": "max",      # Máximo disponible (~20+ años)
            "1h": "730d",     # 2 años
            "5m": "60d",      # 60 días
            "1m": "7d"        # 7 días
        }

        period = periods.get(interval, "max")

        try:
            # Descargar datos
            print(f"  ⏳ Descargando período: {period}...")
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=True,
                auto_adjust=True
            )

            if df.empty:
                print(f"  ❌ No se obtuvieron datos para {symbol}")
                return None

            # Limpiar MultiIndex si existe
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Renombrar columnas a minúsculas
            df.columns = [c.lower() for c in df.columns]

            # Limpiar NaN
            df = df.dropna()

            print(f"  ✅ Descargados {len(df)} registros")
            print(f"  📅 Desde: {df.index[0]}")
            print(f"  📅 Hasta: {df.index[-1]}")
            print(f"  💵 Rango: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

            # Guardar en base de datos
            if len(df) > 0:
                print(f"  💾 Guardando en base de datos...")
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['timestamp'] = df_copy.index

                # Insertar en BD
                self.db.bulk_insert_precios(df_copy, symbol)
                print(f"  ✅ Guardado en BD")

            return df

        except Exception as e:
            print(f"  ❌ Error descargando {symbol}: {e}")
            return None

    def download_multiple_intervals(self, symbol, intervals=["1d", "1h"]):
        """Descarga múltiples timeframes de una acción"""
        results = {}

        for interval in intervals:
            df = self.get_max_historical_data(symbol, interval)
            if df is not None:
                results[interval] = df

        return results

    def update_recent_data(self, symbol, interval="1d", days=7):
        """
        Actualiza solo los datos más recientes
        Útil para ejecución diaria del bot
        """
        print(f"🔄 Actualizando {symbol} (últimos {days} días)...")

        try:
            # Calcular fecha de inicio
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Descargar
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False
            )

            if df.empty:
                print(f"  ⚠️  No hay datos nuevos")
                return None

            # Procesar y guardar
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.columns = [c.lower() for c in df.columns]
            df = df.dropna()

            # Guardar en BD
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy['timestamp'] = df_copy.index

            self.db.bulk_insert_precios(df_copy, symbol)

            print(f"  ✅ Actualizados {len(df)} registros")
            return df

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

    def get_data_summary(self, symbol):
        """Obtiene resumen de datos disponibles"""
        df = self.db.get_precios(symbol)

        if df.empty:
            print(f"❌ No hay datos de {symbol} en la base de datos")
            return None

        print(f"\n📊 Resumen de datos - {symbol}")
        print(f"  Total registros: {len(df)}")
        print(f"  📅 Fecha inicio: {df['timestamp'].min()}")
        print(f"  📅 Fecha fin: {df['timestamp'].max()}")
        print(f"  💵 Precio mín: ${df['close'].min():.2f}")
        print(f"  💵 Precio máx: ${df['close'].max():.2f}")
        print(f"  💵 Precio actual: ${df['close'].iloc[0]:.2f}")

        # Calcular duración
        start = pd.to_datetime(df['timestamp'].min())
        end = pd.to_datetime(df['timestamp'].max())
        duration = (end - start).days

        print(f"  📆 Duración: {duration} días ({duration/365:.1f} años)")

        return {
            'total_records': len(df),
            'start_date': start,
            'end_date': end,
            'duration_days': duration,
            'min_price': df['close'].min(),
            'max_price': df['close'].max(),
            'current_price': df['close'].iloc[0]
        }


# ─────────────────────────────────────────────
#  CLI para uso directo
# ─────────────────────────────────────────────

def main():
    """Interfaz de línea de comandos"""
    import sys

    print("=" * 60)
    print("  📥 Data Downloader - Trading Bot")
    print("=" * 60)
    print()

    # Crear downloader
    downloader = DataDownloader()

    # Solicitar símbolo
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = input("Símbolo de acción (ej: AMD): ").upper() or "AMD"

    # Solicitar interval
    print("\nIntervalos disponibles:")
    print("  1 - Diario (1d) - ~20+ años")
    print("  2 - Horario (1h) - ~2 años")
    print("  3 - 5 minutos (5m) - ~60 días")
    print("  4 - 1 minuto (1m) - ~7 días")
    print("  5 - Todos los anteriores")

    choice = input("\nElige opción (1-5) [1]: ").strip() or "1"

    intervals_map = {
        "1": ["1d"],
        "2": ["1h"],
        "3": ["5m"],
        "4": ["1m"],
        "5": ["1d", "1h", "5m", "1m"]
    }

    intervals = intervals_map.get(choice, ["1d"])

    # Descargar
    print()
    for interval in intervals:
        df = downloader.get_max_historical_data(symbol, interval, force_download=True)
        if df is not None:
            print(f"  ✅ {interval}: {len(df)} registros descargados")
        print()

    # Mostrar resumen
    downloader.get_data_summary(symbol)

    print()
    print("=" * 60)
    print("  ✅ Descarga completada")
    print("=" * 60)

    # Cerrar DB
    downloader.db.close()


if __name__ == "__main__":
    main()
