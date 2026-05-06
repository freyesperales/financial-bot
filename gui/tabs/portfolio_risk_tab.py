"""
PortfolioRiskTabMixin — new Portfolio Risk tab (Etapa 4.4).
Mixed into InvestmentReportGUI; all methods use self.* from the host class.

Shows: ERC weights, equity curve, drawdown, VaR metrics, stress test table.
Heavy computation runs in a background thread.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd

from gui.colors import COLORS


class PortfolioRiskTabMixin:

    # ─── BUILD ───────────────────────────────

    def _build_portfolio_risk(self):
        p = self.tab_portfolio_risk

        # ── Header ──
        hdr = tk.Frame(p, bg=COLORS['surface'], padx=20, pady=12)
        hdr.pack(fill=tk.X)

        tk.Label(hdr, text='Riesgo del Portafolio',
                 bg=COLORS['surface'], fg=COLORS['text'],
                 font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT)

        self._pr_status = tk.Label(hdr, text='Ejecuta el análisis y pulsa Calcular',
                                   bg=COLORS['surface'], fg=COLORS['text_dim'],
                                   font=('Segoe UI', 9))
        self._pr_status.pack(side=tk.LEFT, padx=(20, 0))

        tk.Button(hdr, text='⚡ Calcular Riesgo',
                  command=self._run_portfolio_risk,
                  bg=COLORS['primary'], fg='white',
                  font=('Segoe UI', 9, 'bold'), relief='flat',
                  padx=14, pady=5, cursor='hand2').pack(side=tk.RIGHT, padx=(0, 4))

        # ── Selector de fuente y horizonte ──
        ctrl = tk.Frame(p, bg=COLORS['bg'], padx=16, pady=6)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text='Fuente:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self._pr_source = tk.StringVar(value='top_buys')
        for val, lbl in (('top_buys', 'Top Compras (hasta 15)'),
                         ('portfolio', 'Portfolio seleccionado'),
                         ('journal', 'Posiciones abiertas')):
            tk.Radiobutton(ctrl, text=lbl, variable=self._pr_source, value=val,
                           bg=COLORS['bg'], fg=COLORS['text_dim'],
                           selectcolor=COLORS['surface2'],
                           activebackground=COLORS['bg'],
                           font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(ctrl, text='  Rebalanceo:',
                 bg=COLORS['bg'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(20, 0))
        self._pr_freq = tk.StringVar(value='monthly')
        for val, lbl in (('weekly', 'Semanal'), ('monthly', 'Mensual')):
            tk.Radiobutton(ctrl, text=lbl, variable=self._pr_freq, value=val,
                           bg=COLORS['bg'], fg=COLORS['text_dim'],
                           selectcolor=COLORS['surface2'],
                           activebackground=COLORS['bg'],
                           font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(4, 0))

        # ── Métricas rápidas ──
        self._pr_metrics_frame = tk.Frame(p, bg=COLORS['surface2'], padx=16, pady=8)
        self._pr_metrics_frame.pack(fill=tk.X, padx=16, pady=(4, 0))
        self._pr_metrics_labels = {}
        for key, txt in (('var', 'VaR-95 1d'), ('cvar', 'CVaR-95 1d'),
                         ('vol', 'Vol anual'), ('ret', 'Retorno sim.'),
                         ('dd', 'Max DD'), ('cost', 'Coste rebalan.')):
            col_f = tk.Frame(self._pr_metrics_frame, bg=COLORS['surface2'])
            col_f.pack(side=tk.LEFT, padx=(0, 28))
            tk.Label(col_f, text=txt, bg=COLORS['surface2'], fg=COLORS['text_dim'],
                     font=('Segoe UI', 8)).pack(anchor='w')
            lbl = tk.Label(col_f, text='—', bg=COLORS['surface2'], fg=COLORS['accent'],
                           font=('Segoe UI', 11, 'bold'))
            lbl.pack(anchor='w')
            self._pr_metrics_labels[key] = lbl

        # ── Área principal ──
        main = tk.Frame(p, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 0))

        # Gráficos (matplotlib) a la izquierda
        self._pr_chart_frame = tk.Frame(main, bg=COLORS['bg'])
        self._pr_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(self._pr_chart_frame,
                 text='Los gráficos aparecerán después de calcular',
                 bg=COLORS['bg'], fg=COLORS['text_muted'],
                 font=('Segoe UI', 10)).pack(expand=True)

        # Tablas a la derecha
        right = tk.Frame(main, bg=COLORS['bg'], width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(right, text='Stress Tests', bg=COLORS['bg'], fg=COLORS['text'],
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 4))

        st_cols = ('period', 'ret', 'dd', 'var')
        st_hdrs = ('Período', 'Retorno', 'Max DD', 'VaR 1d')
        self._pr_stress_tree, st_scroll = self._make_treeview(
            right, st_cols, st_hdrs, col_widths=[110, 72, 72, 72])
        st_scroll.pack(fill=tk.BOTH, expand=True)

        tk.Label(right, text='Pesos ERC óptimos', bg=COLORS['bg'], fg=COLORS['text'],
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 4))

        w_cols = ('symbol', 'weight', 'sector')
        w_hdrs = ('Símbolo', 'Peso %', 'Sector')
        self._pr_weights_tree, w_scroll = self._make_treeview(
            right, w_cols, w_hdrs, col_widths=[72, 58, 110])
        w_scroll.pack(fill=tk.BOTH, expand=True)

        self._pr_canvas = None
        self._pr_fig    = None

    # ─── POPULATE (llamado desde _populate_all_tables) ───────────────

    def _populate_portfolio_risk(self, df):
        """Actualiza el label de estado cuando hay datos nuevos."""
        if df is not None and not df.empty:
            n = len(df)
            self._pr_status.config(
                text=f'{n} símbolos disponibles — pulsa Calcular para analizar riesgo',
                fg=COLORS['text_dim'])

    # ─── CÁLCULO EN BACKGROUND ───────────────

    def _run_portfolio_risk(self):
        if self.df_results is None or self.df_results.empty:
            messagebox.showwarning('Sin datos', 'Ejecuta el análisis primero.')
            return

        self._pr_status.config(text='Calculando… (puede tardar unos segundos)',
                               fg=COLORS['warning'])

        def _worker():
            try:
                result = self._compute_portfolio_risk()
                self.q.put(('portfolio_risk_done', result))
            except Exception as e:
                self.q.put(('portfolio_risk_error', str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_portfolio_risk(self):
        """Runs in background thread. Returns dict with all computed data."""
        from portfolio_optimizer import (
            optimize_erc, simulate_rebalance, portfolio_var, stress_test,
        )

        # ── Seleccionar símbolos según fuente ──
        source = self._pr_source.get()
        if source == 'journal' and self.db is not None:
            df_open = self.db.journal_get_open()
            symbols = df_open['symbol'].tolist() if not df_open.empty else []
        elif source == 'portfolio':
            symbols = list(self.portfolio)
        else:
            df_buys = self.df_results[self.df_results['vol_signal'] == 'COMPRAR']
            symbols = df_buys.nlargest(15, 'total_score')['symbol'].tolist()

        if len(symbols) < 3:
            raise ValueError(
                f'Se necesitan al menos 3 símbolos (seleccionados: {len(symbols)}). '
                'Ejecuta el análisis o añade posiciones al journal.'
            )

        # ── Cargar precios desde DB ──
        db = self.db
        if db is None:
            raise ValueError('Base de datos no disponible.')

        price_data = {}
        for sym in symbols:
            try:
                conn = db.connect()
                df_p = pd.read_sql(
                    "SELECT fecha, close FROM precios WHERE symbol=? ORDER BY fecha ASC",
                    conn, params=(sym,))
                if len(df_p) >= 60:
                    df_p['fecha'] = pd.to_datetime(df_p['fecha'])
                    df_p = df_p.set_index('fecha')['close']
                    price_data[sym] = df_p
            except Exception:
                pass

        if len(price_data) < 3:
            raise ValueError(
                f'Datos insuficientes en DB para {len(price_data)} símbolos. '
                'Descarga datos primero.'
            )

        prices_df = pd.DataFrame(price_data).dropna(how='all')
        prices_df = prices_df.ffill().dropna()
        returns_df = prices_df.pct_change().dropna()
        returns_df = returns_df.tail(252)

        # SPY como benchmark
        spy_rets = None
        try:
            conn = db.connect()
            df_spy = pd.read_sql(
                "SELECT fecha, close FROM precios WHERE symbol='SPY' ORDER BY fecha ASC",
                conn)
            if len(df_spy) >= 60:
                df_spy['fecha'] = pd.to_datetime(df_spy['fecha'])
                df_spy = df_spy.set_index('fecha')['close']
                spy_rets = df_spy.pct_change().dropna()
                spy_rets = spy_rets.reindex(returns_df.index).fillna(0)
        except Exception:
            pass

        # ── Optimización ERC ──
        erc_result  = optimize_erc(returns_df, spy_rets=spy_rets)
        erc_weights = erc_result['weights']   # {sym: float}

        # ── Simulación rebalanceo ──
        freq  = self._pr_freq.get()
        rebal = simulate_rebalance(prices_df, erc_weights, freq=freq, cost_bps=5)

        # ── VaR / CVaR ──
        var_result = portfolio_var(erc_weights, returns_df, confidence=0.95, horizon_days=1)

        # ── Stress tests ──
        stress_result = stress_test(erc_weights, prices_df)

        # ── Sectores ──
        sector_map = {}
        if self.df_results is not None and not self.df_results.empty:
            sector_map = self.df_results.set_index('symbol')['sector'].to_dict()

        valid_syms = list(erc_weights.keys())
        return {
            'symbols':      valid_syms,
            'weights':      erc_weights,       # dict {sym: float}
            'sector_map':   sector_map,
            'rebal':        rebal,
            'var':          var_result,        # values already in %
            'stress':       stress_result,     # values already in %
            'equity_curve': rebal.get('equity_curve'),
            'returns_df':   returns_df,
        }

    # ─── RENDERIZAR RESULTADOS (UI thread) ───

    def _render_portfolio_risk(self, result):
        """Called from _poll_queue on the UI thread after computation finishes."""
        try:
            self._update_pr_metrics(result)
            self._update_pr_stress_table(result['stress'])
            self._update_pr_weights_table(result['weights'], result['sector_map'])
            self._draw_pr_charts(result)
            self._pr_status.config(
                text=f"Análisis completado — {len(result['symbols'])} símbolos",
                fg=COLORS['success'])
        except Exception as e:
            self._pr_status.config(text=f'Error al renderizar: {e}', fg=COLORS['danger'])

    def _update_pr_metrics(self, result):
        var_r = result['var']    # values in %
        rebal = result['rebal']  # annualized/max_drawdown already in %
        lbl   = self._pr_metrics_labels

        def _fmt_pct(v, suffix='%', decimals=2):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '—'
            return f"{v:.{decimals}f}{suffix}"

        lbl['var'].config(text=_fmt_pct(var_r.get('var_historical')))
        lbl['cvar'].config(text=_fmt_pct(var_r.get('cvar_historical')))
        # port_vol_daily is in %, annualise with sqrt(252)
        vol_d = var_r.get('port_vol_daily', 0) or 0
        lbl['vol'].config(text=_fmt_pct(vol_d * (252 ** 0.5)))
        lbl['ret'].config(text=_fmt_pct(rebal.get('annualized')))
        lbl['dd'].config(text=_fmt_pct(rebal.get('max_drawdown')))
        lbl['cost'].config(
            text=f"{rebal.get('total_cost_bps', 0):.1f} bps / {rebal.get('n_rebalances', 0)} reb.")

    def _update_pr_stress_table(self, stress):
        tree = self._pr_stress_tree
        tree.delete(*tree.get_children())
        for period, metrics in stress.items():
            if 'error' in metrics:
                tree.insert('', 'end', values=(period, '—', '—', '—'))
                continue
            ret = metrics.get('cum_return', 0)
            tag = 'buy' if ret >= 0 else 'sell'
            tree.insert('', 'end', tags=(tag,), values=(
                period,
                f"{ret:+.1f}%",
                f"{metrics.get('max_drawdown', 0):.1f}%",
                f"{metrics.get('var_95_1d', 0):.2f}%",
            ))

    def _update_pr_weights_table(self, weights_dict, sector_map):
        tree = self._pr_weights_tree
        tree.delete(*tree.get_children())
        sorted_w = sorted(weights_dict.items(), key=lambda x: -x[1])
        for sym, w in sorted_w:
            tree.insert('', 'end', values=(
                sym,
                f"{w*100:.1f}%",
                sector_map.get(sym, '—'),
            ))

    def _draw_pr_charts(self, result):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            for widget in self._pr_chart_frame.winfo_children():
                widget.destroy()
            tk.Label(self._pr_chart_frame,
                     text='matplotlib no disponible — pip install matplotlib',
                     bg=COLORS['bg'], fg=COLORS['danger'],
                     font=('Segoe UI', 10)).pack(expand=True)
            return

        equity     = result.get('equity_curve')   # pd.Series, starts at 100
        weights    = result['weights']              # {sym: float}
        returns_df = result['returns_df']

        # Limpiar canvas anterior
        if self._pr_canvas is not None:
            self._pr_canvas.get_tk_widget().destroy()
        if self._pr_fig is not None:
            plt.close(self._pr_fig)

        bg       = COLORS['bg']
        surf     = COLORS['surface']
        text_col = COLORS['text']
        grid_col = COLORS['surface3']

        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
        fig.patch.set_facecolor(bg)

        def _style_ax(ax, title):
            ax.set_facecolor(surf)
            ax.set_title(title, color=text_col, fontsize=9, pad=6)
            ax.tick_params(colors=COLORS['text_dim'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_col)
            ax.grid(True, color=grid_col, linewidth=0.5, alpha=0.6)

        # ── 1: Equity curve (starts at 100) ──
        ax1 = axes[0, 0]
        _style_ax(ax1, 'Equity Curve (rebalanceo simulado)')
        if equity is not None and len(equity) > 1:
            ax1.plot(equity.index, equity.values,
                     color=COLORS['primary'], linewidth=1.5)
            ax1.axhline(100.0, color=COLORS['text_muted'], linewidth=0.8, linestyle='--')
        else:
            ax1.text(0.5, 0.5, 'Sin datos de equity', transform=ax1.transAxes,
                     ha='center', va='center', color=COLORS['text_dim'], fontsize=9)

        # ── 2: Drawdown ──
        ax2 = axes[0, 1]
        _style_ax(ax2, 'Drawdown')
        if equity is not None and len(equity) > 1:
            roll_max = equity.cummax()
            dd = (equity - roll_max) / roll_max
            ax2.fill_between(dd.index, dd.values, 0,
                             color=COLORS['danger'], alpha=0.5)
            ax2.plot(dd.index, dd.values, color=COLORS['danger'], linewidth=0.8)
            ax2.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        else:
            ax2.text(0.5, 0.5, 'Sin datos de drawdown', transform=ax2.transAxes,
                     ha='center', va='center', color=COLORS['text_dim'], fontsize=9)

        # ── 3: Pesos ERC ──
        ax3 = axes[1, 0]
        _style_ax(ax3, 'Pesos ERC óptimos')
        sorted_pairs = sorted(weights.items(), key=lambda x: -x[1])
        if sorted_pairs:
            syms_s, ws_s = zip(*sorted_pairs)
            n = len(syms_s)
            avg_w = 1.0 / n if n else 0
            colors_bar = [COLORS['primary'] if w > avg_w else COLORS['text_dim']
                          for w in ws_s]
            ax3.barh(syms_s, [w * 100 for w in ws_s],
                     color=colors_bar, height=0.7)
            ax3.set_xlabel('%', color=COLORS['text_dim'], fontsize=7)
            ax3.invert_yaxis()
            ax3.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax3.grid(axis='x', color=grid_col, linewidth=0.5)
            ax3.grid(axis='y', visible=False)

        # ── 4: Correlación (top 10) ──
        ax4 = axes[1, 1]
        _style_ax(ax4, 'Correlación (subset top 10)')
        try:
            top_syms = [s for s, _ in sorted_pairs[:10]]
            available = [s for s in top_syms if s in returns_df.columns]
            corr = returns_df[available].corr()
            ax4.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            ax4.set_xticks(range(len(available)))
            ax4.set_yticks(range(len(available)))
            ax4.set_xticklabels(available, rotation=45, ha='right',
                                color=COLORS['text_dim'], fontsize=6)
            ax4.set_yticklabels(available, color=COLORS['text_dim'], fontsize=6)
            ax4.grid(False)
        except Exception:
            ax4.text(0.5, 0.5, 'Sin datos de correlación', transform=ax4.transAxes,
                     ha='center', va='center', color=COLORS['text_dim'], fontsize=9)

        fig.tight_layout(pad=1.5)

        for widget in self._pr_chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self._pr_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._pr_canvas = canvas
        self._pr_fig    = fig
