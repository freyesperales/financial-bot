"""
gui/data_controller.py — DataController (Etapa 4.2).

Encapsula el estado compartido del análisis y expone métodos tipados
que sirven DataFrames listos para cada pestaña.

Las pestañas consumen datos a través del controller en lugar de
acceder directamente a self.df_results, lo que facilita:
  - Testing unitario de la lógica de preparación de datos.
  - Cambiar la fuente de datos sin tocar cada tab.
  - Añadir caché por tab sin duplicar código.

Uso en InvestmentReportGUI:
    self.ctrl = DataController()
    # Al completar un análisis:
    self.ctrl.update(df_results, portfolio, vix_data, spy_data)
    # En un tab:
    df = self.ctrl.get_top_buys(n=10)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np


class DataController:
    """Fuente única de verdad para los datos del análisis más reciente."""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._portfolio: list = []
        self._vix: Optional[dict] = None
        self._spy: Optional[dict] = None
        self._regime: str = 'transition'

    # ── Actualización ─────────────────────────────────────────────────────────

    def update(
        self,
        df_results: Optional[pd.DataFrame],
        portfolio: Optional[list] = None,
        vix: Optional[dict] = None,
        spy: Optional[dict] = None,
        regime: Optional[str] = None,
    ) -> None:
        """Actualiza el estado con los resultados del último análisis."""
        self._df = df_results
        if portfolio is not None:
            self._portfolio = portfolio
        if vix is not None:
            self._vix = vix
        if spy is not None:
            self._spy = spy
        if regime is not None:
            self._regime = regime

    @property
    def has_data(self) -> bool:
        return self._df is not None and not self._df.empty

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """DataFrame completo — solo lectura."""
        return self._df

    @property
    def vix(self) -> Optional[dict]:
        return self._vix

    @property
    def spy(self) -> Optional[dict]:
        return self._spy

    @property
    def portfolio(self) -> list:
        return self._portfolio

    @property
    def regime(self) -> str:
        return self._regime

    # ── Queries por pestaña ───────────────────────────────────────────────────

    def get_top_buys(self, n: int = 15) -> pd.DataFrame:
        """Top n acciones por señal de compra (vol_signal = COMPRAR)."""
        if not self.has_data:
            return pd.DataFrame()
        df = self._df[self._df['vol_signal'] == 'COMPRAR'].copy()
        if df.empty:
            return df
        sort_cols = [c for c in ('confidence_score', 'total_score') if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        return df.head(n).reset_index(drop=True)

    def get_dip_candidates(self, n: int = 15) -> pd.DataFrame:
        """Acciones en dip con señal técnica sólida."""
        if not self.has_data:
            return pd.DataFrame()
        df = self._df.copy()
        score_col = 'dip_score' if 'dip_score' in df.columns else 'total_score'
        df = df.sort_values(score_col, ascending=False)
        return df.head(n).reset_index(drop=True)

    def get_risk_summary(self) -> pd.DataFrame:
        """Tabla de riesgo: sharpe, drawdown, kelly, VaR por símbolo."""
        if not self.has_data:
            return pd.DataFrame()
        risk_cols = [c for c in (
            'symbol', 'sector', 'price',
            'sharpe', 'max_dd', 'ann_vol', 'ann_return',
            'kelly_pct', 'kelly_pct_p5', 'kelly_pct_p95',
            'risk_signal', 'risk_score',
            'beta', 'corr_60d', 'info_ratio',
            'confidence',
        ) if c in self._df.columns]
        df = self._df[risk_cols].copy()
        sort_cols = [c for c in ('risk_score', 'sharpe') if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        return df.reset_index(drop=True)

    def get_portfolio_weights_df(self) -> pd.DataFrame:
        """
        DataFrame con pesos del portfolio actual (symbols en self._portfolio).
        Si no hay portfolio, usa top-buys.
        """
        if not self.has_data:
            return pd.DataFrame()
        symbols = self._portfolio if self._portfolio else []
        if not symbols and self.has_data:
            top = self.get_top_buys(12)
            symbols = top['symbol'].tolist() if not top.empty else []
        if not symbols:
            return pd.DataFrame()
        df_port = self._df[self._df['symbol'].isin(symbols)].copy()
        n = len(df_port)
        if n == 0:
            return pd.DataFrame()
        df_port['weight_ew'] = round(1.0 / n, 4)
        return df_port[['symbol', 'sector', 'total_score', 'confidence',
                         'weight_ew'] +
                        [c for c in ('sharpe', 'max_dd', 'beta') if c in df_port.columns]
                       ].reset_index(drop=True)

    def get_sector_summary(self) -> pd.DataFrame:
        """Resumen por sector: score promedio, nº acciones, % en compra."""
        if not self.has_data:
            return pd.DataFrame()
        df = self._df.copy()
        if 'sector' not in df.columns:
            return pd.DataFrame()
        agg = df.groupby('sector').agg(
            n_symbols=('symbol', 'count'),
            avg_score=('total_score', 'mean'),
            n_buy=('vol_signal', lambda x: (x == 'COMPRAR').sum()),
        ).reset_index()
        agg['buy_pct'] = (agg['n_buy'] / agg['n_symbols'] * 100).round(1)
        agg['avg_score'] = agg['avg_score'].round(1)
        return agg.sort_values('avg_score', ascending=False).reset_index(drop=True)

    def get_returns_matrix(
        self,
        symbols: Optional[list] = None,
        min_rows: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Intenta construir una matriz de retornos diarios desde df_results.
        Devuelve None si no hay suficiente información.

        Nota: los retornos completos no están en df_results — este método es
        un placeholder. La fuente real son los precios de la DB via StockAnalyzer.
        """
        return None   # Se llenará cuando el caller pase la DB directamente

    def get_vol_zscore_df(self) -> pd.DataFrame:
        """Símbolos con vol_zscore disponible, ordenados por sector."""
        if not self.has_data:
            return pd.DataFrame()
        cols = [c for c in ('symbol', 'sector', 'price', 'vol_signal',
                             'vol_ratio', 'vol_zscore', 'total_score', 'confidence')
                if c in self._df.columns]
        df = self._df[cols].dropna(subset=['vol_zscore'] if 'vol_zscore' in cols else [])
        return df.sort_values('sector').reset_index(drop=True) if not df.empty else df

    # ── Estadísticas globales ──────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Métricas de resumen rápido para el Dashboard."""
        if not self.has_data:
            return {}
        df = self._df
        n_buy  = int((df.get('vol_signal', pd.Series()) == 'COMPRAR').sum())
        n_sell = int((df.get('vol_signal', pd.Series()) == 'VENDER').sum())
        avg_sc = float(df['total_score'].mean()) if 'total_score' in df.columns else 0.0
        hi_conf = int((df.get('confidence', pd.Series()).isin({'ALTA', 'MUY ALTA'})).sum())
        return {
            'n_symbols':   len(df),
            'n_buy':       n_buy,
            'n_sell':      n_sell,
            'n_neutral':   len(df) - n_buy - n_sell,
            'avg_score':   round(avg_sc, 1),
            'hi_conf':     hi_conf,
            'regime':      self._regime,
            'vix':         self._vix.get('vix') if self._vix else None,
        }
