"""
Database Manager - SQLite para Trading Bot
Almacena históricos, señales y operaciones
"""
import sqlite3
import pandas as pd
from datetime import datetime
import json

from logging_setup import get_logger

log = get_logger(__name__)


class TradingDatabase:
    def __init__(self, db_path="trading_bot.db", auto_cleanup_snapshots=True,
                 snapshots_keep=20):
        self.db_path = db_path
        self.conn = None
        self._auto_cleanup_snapshots = auto_cleanup_snapshots
        self._snapshots_keep = snapshots_keep
        self.create_tables()
        if auto_cleanup_snapshots:
            try:
                self.cleanup_old_snapshots(keep=snapshots_keep)
            except Exception:
                # No bloquear arranque si la limpieza falla (p. ej. tabla recién creada).
                pass

    def connect(self):
        """Conectar a base de datos.

        Activa modo WAL y synchronous=NORMAL en la primera conexión:
        - WAL permite lecturas concurrentes con una escritura activa, lo que
          encaja mejor con el GUI multi-hilo del proyecto.
        - synchronous=NORMAL es seguro con WAL y ~2-3x más rápido que FULL.
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass
        return self.conn

    def close(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self):
        """Crear tablas si no existen"""
        conn = self.connect()
        cursor = conn.cursor()

        # Tabla de precios históricos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_lower REAL,
                vwap REAL,
                ema_9 REAL,
                ema_21 REAL,
                atr REAL,
                UNIQUE(symbol, timestamp)
            )
        ''')

        # Tabla de señales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS señales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                tipo TEXT NOT NULL,
                precio REAL NOT NULL,
                votos_compra INTEGER,
                votos_venta INTEGER,
                detalle TEXT,
                executed BOOLEAN DEFAULT 0
            )
        ''')

        # Tabla de operaciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operaciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entrada_timestamp DATETIME NOT NULL,
                entrada_precio REAL NOT NULL,
                entrada_votos INTEGER,
                salida_timestamp DATETIME,
                salida_precio REAL,
                salida_razon TEXT,
                pnl REAL,
                pnl_pct REAL,
                duracion_minutos INTEGER,
                stop_loss REAL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL
            )
        ''')

        # Tabla de configuración
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Trade Journal — posiciones reales del usuario
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS journal_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                amount_eur REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                fees_eur REAL DEFAULT 0,
                slippage_bps REAL DEFAULT 0,
                notes TEXT DEFAULT '',
                exit_date TEXT,
                exit_price REAL,
                exit_fees_eur REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                pnl_eur REAL,
                pnl_pct REAL,
                pnl_net_eur REAL,
                mae_pct REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self._migrate_journal_extended(cursor)

        # Tabla de snapshots de señales (para alertas de cambio)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                confidence TEXT,
                confidence_score REAL,
                vol_signal TEXT,
                rs_signal TEXT,
                risk_signal TEXT,
                total_score REAL,
                in_top_buys INTEGER DEFAULT 0,
                fund_signal TEXT,
                price REAL,
                UNIQUE(run_timestamp, symbol)
            )
        ''')
        self._migrate_snapshots_extended(cursor)

        conn.commit()

    # ── Migraciones seguras (ADD COLUMN idempotente) ──────────────────────────

    def _migrate_journal_extended(self, cursor):
        """Añade columnas extendidas a journal_trades si no existen (3.4)."""
        new_cols = [
            ('take_profit',   'REAL'),
            ('fees_eur',      'REAL DEFAULT 0'),
            ('slippage_bps',  'REAL DEFAULT 0'),
            ('exit_fees_eur', 'REAL DEFAULT 0'),
            ('pnl_eur',       'REAL'),
            ('pnl_pct',       'REAL'),
            ('pnl_net_eur',   'REAL'),
            ('mae_pct',       'REAL'),
        ]
        for col, col_type in new_cols:
            try:
                cursor.execute(f'ALTER TABLE journal_trades ADD COLUMN {col} {col_type}')
            except Exception:
                pass  # columna ya existe

    def _migrate_snapshots_extended(self, cursor):
        """Añade fund_signal y price a signal_snapshots si no existen (3.6)."""
        for col, col_type in [('fund_signal', 'TEXT'), ('price', 'REAL')]:
            try:
                cursor.execute(f'ALTER TABLE signal_snapshots ADD COLUMN {col} {col_type}')
            except Exception:
                pass

    # ─────────────────────────────────────────────
    #  SIGNAL SNAPSHOTS
    # ─────────────────────────────────────────────

    def save_signal_snapshot(self, df_results, run_timestamp=None):
        """Guarda un snapshot completo de señales tras cada análisis."""
        if df_results is None or df_results.empty:
            return
        conn = self.connect()
        cursor = conn.cursor()
        ts = run_timestamp or datetime.now().isoformat(timespec='seconds')

        rows = []
        for _, row in df_results.iterrows():
            rows.append((
                ts,
                row.get('symbol', ''),
                row.get('confidence', ''),
                row.get('confidence_score'),
                row.get('vol_signal', ''),
                row.get('rs_signal', ''),
                row.get('risk_signal', ''),
                row.get('total_score'),
                1 if row.get('vol_signal') == 'COMPRAR' else 0,
                row.get('fund_signal', ''),
                row.get('price'),
            ))

        cursor.executemany('''
            INSERT OR REPLACE INTO signal_snapshots
            (run_timestamp, symbol, confidence, confidence_score, vol_signal,
             rs_signal, risk_signal, total_score, in_top_buys, fund_signal, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()

    def get_last_snapshot(self):
        """Devuelve el snapshot más reciente (antes del actual)."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT run_timestamp FROM signal_snapshots
            ORDER BY run_timestamp DESC LIMIT 2
        ''')
        runs = [r[0] for r in cursor.fetchall()]
        if len(runs) < 2:
            return None   # No hay análisis anterior
        prev_ts = runs[1]
        cursor.execute('''
            SELECT * FROM signal_snapshots WHERE run_timestamp = ?
        ''', (prev_ts,))
        rows = cursor.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cursor.description]
        return pd.DataFrame([dict(zip(cols, r)) for r in rows])

    def cleanup_old_snapshots(self, keep=10):
        """Mantiene solo los últimos N análisis para no inflar la DB."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM signal_snapshots WHERE run_timestamp NOT IN (
                SELECT DISTINCT run_timestamp FROM signal_snapshots
                ORDER BY run_timestamp DESC LIMIT ?
            )
        ''', (keep,))
        conn.commit()

    # ─────────────────────────────────────────────
    #  TRADE JOURNAL
    # ─────────────────────────────────────────────

    def journal_open(self, symbol, entry_date, entry_price, quantity,
                     amount_eur, stop_loss=None, take_profit=None,
                     fees_eur=0.0, slippage_bps=0.0, notes=''):
        """Registra una nueva posición abierta."""
        conn = self.connect()
        conn.execute('''
            INSERT INTO journal_trades
            (symbol, entry_date, entry_price, quantity, amount_eur,
             stop_loss, take_profit, fees_eur, slippage_bps, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol.upper(), entry_date, entry_price, quantity, amount_eur,
              stop_loss, take_profit, fees_eur or 0.0, slippage_bps or 0.0, notes))
        conn.commit()

    def journal_close(self, trade_id, exit_date, exit_price, exit_fees_eur=0.0):
        """Cierra una posición abierta y calcula P&L bruto y neto."""
        trade_id = int(trade_id)   # numpy.int64 no es aceptado por sqlite3 en Py3.13
        conn = self.connect()
        cur = conn.execute(
            'SELECT entry_price, amount_eur, fees_eur, slippage_bps, quantity '
            'FROM journal_trades WHERE id=?',
            (trade_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        entry_price  = row[0]
        amount_eur   = row[1]
        entry_fees   = row[2] or 0.0
        slippage_bps = row[3] or 0.0
        quantity     = row[4]

        pnl_pct      = (exit_price - entry_price) / entry_price * 100
        pnl_eur      = amount_eur * pnl_pct / 100

        # Coste de slippage (bps sobre el valor de salida)
        exit_value    = exit_price * quantity
        slippage_cost = exit_value * slippage_bps / 10_000
        total_fees    = (entry_fees or 0.0) + (exit_fees_eur or 0.0) + slippage_cost
        pnl_net_eur   = pnl_eur - total_fees

        conn.execute('''
            UPDATE journal_trades
            SET exit_date=?, exit_price=?, exit_fees_eur=?, status='CLOSED',
                pnl_eur=?, pnl_pct=?, pnl_net_eur=?
            WHERE id=?
        ''', (exit_date, exit_price, exit_fees_eur or 0.0,
              round(pnl_eur, 2), round(pnl_pct, 2), round(pnl_net_eur, 2),
              trade_id))
        conn.commit()

    def journal_update_mae(self, trade_id, current_price):
        """
        Actualiza el Maximum Adverse Excursion (MAE) de una posicion abierta.
        MAE = caida maxima desde el precio de entrada (en %).
        Solo se actualiza si el nuevo drawdown es peor que el registrado.
        """
        trade_id = int(trade_id)
        conn = self.connect()
        cur = conn.execute(
            'SELECT entry_price, mae_pct FROM journal_trades WHERE id=? AND status="OPEN"',
            (trade_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        entry_price = row[0]
        mae_stored  = row[1] or 0.0
        dd_pct = (current_price - entry_price) / entry_price * 100   # negativo = caida
        if dd_pct < mae_stored:   # empeoro el drawdown
            conn.execute(
                'UPDATE journal_trades SET mae_pct=? WHERE id=?',
                (round(dd_pct, 2), trade_id),
            )
            conn.commit()

    def journal_get_stats(self):
        """
        Estadisticas agregadas del journal:
          total_trades, win_rate, avg_pnl_pct, total_pnl_eur,
          total_fees_eur, avg_hold_days, best_trade, worst_trade.
        Solo considera posiciones CLOSED.
        """
        conn = self.connect()
        df = pd.read_sql(
            'SELECT * FROM journal_trades WHERE status="CLOSED"', conn,
        )
        if df.empty:
            return {
                'total_trades': 0, 'win_rate': None, 'avg_pnl_pct': None,
                'total_pnl_eur': 0.0, 'total_fees_eur': 0.0,
                'avg_hold_days': None, 'best_trade': None, 'worst_trade': None,
            }
        wins   = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]

        # Dias de hold
        try:
            df['_entry'] = pd.to_datetime(df['entry_date'])
            df['_exit']  = pd.to_datetime(df['exit_date'])
            df['_days']  = (df['_exit'] - df['_entry']).dt.days
            avg_hold = float(df['_days'].mean())
        except Exception:
            avg_hold = None

        total_fees = float((df['fees_eur'].fillna(0) + df['exit_fees_eur'].fillna(0)).sum())

        best  = df.loc[df['pnl_pct'].idxmax()] if not df.empty else None
        worst = df.loc[df['pnl_pct'].idxmin()] if not df.empty else None

        return {
            'total_trades':  len(df),
            'win_rate':      round(len(wins) / len(df) * 100, 1),
            'avg_pnl_pct':   round(float(df['pnl_pct'].mean()), 2),
            'total_pnl_eur': round(float(df['pnl_net_eur'].fillna(df['pnl_eur']).sum()), 2),
            'total_fees_eur':round(total_fees, 2),
            'avg_hold_days': round(avg_hold, 1) if avg_hold else None,
            'best_trade':    {'symbol': best['symbol'], 'pnl_pct': round(float(best['pnl_pct']), 1)} if best is not None else None,
            'worst_trade':   {'symbol': worst['symbol'], 'pnl_pct': round(float(worst['pnl_pct']), 1)} if worst is not None else None,
        }

    def journal_delete(self, trade_id):
        trade_id = int(trade_id)
        conn = self.connect()
        conn.execute('DELETE FROM journal_trades WHERE id=?', (trade_id,))
        conn.commit()

    def journal_get_open(self):
        conn = self.connect()
        return pd.read_sql('SELECT * FROM journal_trades WHERE status="OPEN" ORDER BY entry_date DESC',
                           conn)

    def journal_get_closed(self):
        conn = self.connect()
        return pd.read_sql('SELECT * FROM journal_trades WHERE status="CLOSED" ORDER BY exit_date DESC',
                           conn)

    # ─────────────────────────────────────────────
    #  PRECIOS
    # ─────────────────────────────────────────────

    def insert_precio(self, symbol, timestamp, open_price, high, low, close, volume,
                     rsi=None, macd=None, macd_signal=None, bb_upper=None, bb_lower=None,
                     vwap=None, ema_9=None, ema_21=None, atr=None):
        """Insertar precio histórico"""
        conn = self.connect()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO precios
                (symbol, timestamp, open, high, low, close, volume, rsi, macd, macd_signal,
                 bb_upper, bb_lower, vwap, ema_9, ema_21, atr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timestamp, open_price, high, low, close, volume, rsi, macd,
                  macd_signal, bb_upper, bb_lower, vwap, ema_9, ema_21, atr))
            conn.commit()
            return True
        except Exception as e:
            log.error("Error insertando precio %s @ %s: %s", symbol, timestamp, e)
            return False

    def bulk_insert_precios(self, df, symbol):
        """Insertar múltiples precios desde DataFrame"""
        conn = self.connect()
        cursor = conn.cursor()

        # Primero, eliminar datos existentes para este símbolo
        cursor.execute("DELETE FROM precios WHERE symbol = ?", (symbol,))
        conn.commit()

        # Preparar datos
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        df_copy['timestamp'] = df_copy.index

        # Renombrar columnas a minúsculas
        df_copy.columns = [c.lower() for c in df_copy.columns]

        # Insertar
        try:
            df_copy.to_sql('precios', conn, if_exists='append', index=False)
            return True
        except Exception as e:
            log.error("Error bulk_insert_precios %s: %s", symbol, e)
            return False

    def get_precios(self, symbol, start_date=None, end_date=None, limit=None):
        """Obtener precios históricos"""
        conn = self.connect()

        query = "SELECT * FROM precios WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql_query(query, conn, params=params)

    def get_last_date(self, symbol):
        """Retorna la última fecha almacenada para un símbolo (string YYYY-MM-DD) o None."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM precios WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        if row and row[0]:
            return str(row[0])[:10]  # YYYY-MM-DD
        return None

    def append_precios(self, df, symbol):
        """Inserta filas nuevas (fecha > última existente) sin borrar las existentes."""
        last = self.get_last_date(symbol)
        conn = self.connect()

        df_copy = df.copy()
        df_copy['symbol'] = symbol
        df_copy['timestamp'] = df_copy.index

        # Filtrar solo filas más recientes que la última guardada
        if last:
            df_copy = df_copy[df_copy['timestamp'].astype(str).str[:10] > last]

        if df_copy.empty:
            return True

        df_copy.columns = [c.lower() for c in df_copy.columns]
        try:
            df_copy.to_sql('precios', conn, if_exists='append', index=False)
            conn.commit()
            return True
        except Exception as e:
            log.error("Error append_precios %s: %s", symbol, e)
            return False

    # ─────────────────────────────────────────────
    #  SEÑALES
    # ─────────────────────────────────────────────

    def insert_señal(self, symbol, timestamp, tipo, precio, votos_compra, votos_venta, detalle=None):
        """Insertar señal"""
        conn = self.connect()
        cursor = conn.cursor()

        detalle_json = json.dumps(detalle) if detalle else None

        cursor.execute('''
            INSERT INTO señales (symbol, timestamp, tipo, precio, votos_compra, votos_venta, detalle)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, timestamp, tipo, precio, votos_compra, votos_venta, detalle_json))

        conn.commit()
        return cursor.lastrowid

    def get_señales(self, symbol=None, tipo=None, limit=100):
        """Obtener señales"""
        conn = self.connect()

        query = "SELECT * FROM señales WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if tipo:
            query += " AND tipo = ?"
            params.append(tipo)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        return pd.read_sql_query(query, conn, params=params)

    # ─────────────────────────────────────────────
    #  OPERACIONES
    # ─────────────────────────────────────────────

    def insert_operacion(self, symbol, entrada_timestamp, entrada_precio, entrada_votos,
                        stop_loss, tp1, tp2, tp3):
        """Insertar nueva operación (entrada)"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO operaciones
            (symbol, entrada_timestamp, entrada_precio, entrada_votos, stop_loss, tp1, tp2, tp3)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, entrada_timestamp, entrada_precio, entrada_votos, stop_loss, tp1, tp2, tp3))

        conn.commit()
        return cursor.lastrowid

    def close_operacion(self, operacion_id, salida_timestamp, salida_precio, salida_razon):
        """Cerrar operación (salida)"""
        conn = self.connect()
        cursor = conn.cursor()

        # Obtener datos de entrada
        cursor.execute("SELECT entrada_timestamp, entrada_precio FROM operaciones WHERE id = ?", (operacion_id,))
        row = cursor.fetchone()

        if row:
            entrada_timestamp = datetime.fromisoformat(row['entrada_timestamp'])
            entrada_precio = row['entrada_precio']

            # Calcular P&L
            pnl = salida_precio - entrada_precio
            pnl_pct = (pnl / entrada_precio) * 100

            # Calcular duración
            salida_dt = datetime.fromisoformat(salida_timestamp)
            duracion = (salida_dt - entrada_timestamp).total_seconds() / 60  # minutos

            # Actualizar
            cursor.execute('''
                UPDATE operaciones
                SET salida_timestamp = ?, salida_precio = ?, salida_razon = ?,
                    pnl = ?, pnl_pct = ?, duracion_minutos = ?
                WHERE id = ?
            ''', (salida_timestamp, salida_precio, salida_razon, pnl, pnl_pct, duracion, operacion_id))

            conn.commit()
            return True

        return False

    def get_operaciones(self, symbol=None, cerradas_only=False, limit=100):
        """Obtener operaciones"""
        conn = self.connect()

        query = "SELECT * FROM operaciones WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if cerradas_only:
            query += " AND salida_timestamp IS NOT NULL"

        query += f" ORDER BY entrada_timestamp DESC LIMIT {limit}"

        return pd.read_sql_query(query, conn, params=params)

    # ─────────────────────────────────────────────
    #  ESTADÍSTICAS
    # ─────────────────────────────────────────────

    def get_estadisticas(self, symbol=None):
        """Obtener estadísticas de operaciones"""
        conn = self.connect()

        query = '''
            SELECT
                symbol,
                COUNT(*) as total_ops,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as ganadoras,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as perdedoras,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(pnl_pct) as total_pnl_pct,
                AVG(duracion_minutos) as avg_duracion_min,
                MAX(pnl_pct) as max_ganancia,
                MIN(pnl_pct) as max_perdida
            FROM operaciones
            WHERE salida_timestamp IS NOT NULL
        '''

        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " GROUP BY symbol"

        df = pd.read_sql_query(query, conn, params=params)

        # Calcular win rate
        if not df.empty:
            df['win_rate'] = (df['ganadoras'] / df['total_ops']) * 100

        return df

    # ─────────────────────────────────────────────
    #  UTILIDADES
    # ─────────────────────────────────────────────

    def vacuum(self):
        """Optimizar base de datos"""
        conn = self.connect()
        conn.execute("VACUUM")

    def get_table_count(self, table):
        """Contar registros en tabla"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def get_table_count_for_symbol(self, symbol):
        """Contar filas de precios para un símbolo específico."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM precios WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        return row[0] if row else 0


# ─────────────────────────────────────────────
#  EJEMPLO DE USO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Crear database
    db = TradingDatabase()

    print("✅ Base de datos creada")
    print(f"  - Precios: {db.get_table_count('precios')} registros")
    print(f"  - Señales: {db.get_table_count('señales')} registros")
    print(f"  - Operaciones: {db.get_table_count('operaciones')} registros")

    # Cerrar
    db.close()
