"""
MLTabMixin — Pestaña Modelo ML / SHAP (Etapa 5.5).

Mixed into InvestmentReportGUI.

Muestra:
  - Estado del modelo (AUC, iteraciones, features, fecha)
  - Importancia de features (bar chart LightGBM gain)
  - Valores SHAP (opcional — requiere `pip install shap`)
  - Tabla de resultados CV por fold
  - Comparación Sharpe OOS: modelo vs reglas vs buy-and-hold
  - Botones: Entrenar Modelo / Comparar Sharpe

El entrenamiento y la comparación se ejecutan en hilos background para no
bloquear el GUI. Requiere que haya un dataset en cache/ml_dataset.parquet
(generado con build_dataset) para que el entrenamiento funcione.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
import pandas as pd

from gui.colors import COLORS


# ── Constantes de layout ──────────────────────────────────────────────────────
_FONT_TITLE  = ('Segoe UI', 12, 'bold')
_FONT_LABEL  = ('Segoe UI', 9)
_FONT_SMALL  = ('Segoe UI', 8)
_FONT_VALUE  = ('Segoe UI', 10, 'bold')
_PAD         = dict(padx=10, pady=4)


class MLTabMixin:

    # ─── BUILD ────────────────────────────────────────────────

    def _build_ml_tab(self):
        p = self.tab_ml
        p.configure(bg=COLORS['bg'])

        # ── Header ──────────────────────────────────────────────
        hdr = tk.Frame(p, bg=COLORS['surface'], padx=16, pady=10)
        hdr.pack(fill=tk.X)

        tk.Label(hdr, text='Modelo Predictivo ML',
                 bg=COLORS['surface'], fg=COLORS['text'],
                 font=_FONT_TITLE).pack(side=tk.LEFT)

        self._ml_status_lbl = tk.Label(
            hdr, text='Sin modelo entrenado',
            bg=COLORS['surface'], fg=COLORS['text_dim'],
            font=_FONT_LABEL)
        self._ml_status_lbl.pack(side=tk.LEFT, padx=(16, 0))

        # Botones lado derecho
        btn_frame = tk.Frame(hdr, bg=COLORS['surface'])
        btn_frame.pack(side=tk.RIGHT)

        tk.Button(btn_frame, text='Entrenar Modelo',
                  command=self._run_ml_train,
                  bg=COLORS['primary'], fg='white',
                  font=('Segoe UI', 9, 'bold'), relief='flat',
                  padx=12, pady=4, cursor='hand2').pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text='Comparar Sharpe OOS',
                  command=self._run_ml_compare,
                  bg=COLORS['surface2'], fg=COLORS['text'],
                  font=_FONT_LABEL, relief='flat',
                  padx=12, pady=4, cursor='hand2').pack(side=tk.LEFT, padx=4)

        # ── Cuerpo principal: dos columnas ────────────────────────
        body = tk.Frame(p, bg=COLORS['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── Columna izquierda: gráfico importancia / SHAP ─────────
        left = tk.LabelFrame(body, text=' Feature Importance ',
                              bg=COLORS['bg'], fg=COLORS['text_dim'],
                              font=_FONT_SMALL, bd=1, relief='solid')
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))

        self._ml_plot_frame = tk.Frame(left, bg=COLORS['bg'])
        self._ml_plot_frame.pack(fill=tk.BOTH, expand=True)

        self._ml_no_model_lbl = tk.Label(
            self._ml_plot_frame,
            text='Entrena el modelo para ver la importancia de features.',
            bg=COLORS['bg'], fg=COLORS['text_dim'],
            font=_FONT_LABEL, wraplength=340)
        self._ml_no_model_lbl.pack(expand=True)

        # ── Columna derecha: CV + comparación ────────────────────
        right = tk.Frame(body, bg=COLORS['bg'])
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Tabla CV
        cv_frame = tk.LabelFrame(right, text=' Resultados CV por Fold ',
                                  bg=COLORS['bg'], fg=COLORS['text_dim'],
                                  font=_FONT_SMALL, bd=1, relief='solid')
        cv_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 6))

        cv_cols = ('fold', 'n_train', 'n_test', 'auc', 'iter')
        cv_hdrs = ('Fold', 'Train', 'Test', 'AUC', 'Iter')
        self._ml_cv_tree = ttk.Treeview(cv_frame, columns=cv_cols,
                                         show='headings', height=6)
        for col, hdr in zip(cv_cols, cv_hdrs):
            self._ml_cv_tree.heading(col, text=hdr)
            self._ml_cv_tree.column(col, width=60, anchor='center')
        self._ml_cv_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Comparación Sharpe
        cmp_frame = tk.LabelFrame(right, text=' Comparación Sharpe OOS ',
                                   bg=COLORS['bg'], fg=COLORS['text_dim'],
                                   font=_FONT_SMALL, bd=1, relief='solid')
        cmp_frame.grid(row=1, column=0, sticky='nsew')
        self._ml_cmp_frame = cmp_frame

        self._ml_cmp_lbl = tk.Label(
            cmp_frame,
            text='Pulsa "Comparar Sharpe OOS" después de entrenar.',
            bg=COLORS['bg'], fg=COLORS['text_dim'],
            font=_FONT_LABEL, wraplength=260, justify=tk.LEFT)
        self._ml_cmp_lbl.pack(expand=True, padx=8, pady=8)

        # Cargar modelo si ya existe
        self.root.after(200, self._ml_load_if_exists)

    # ─── CARGAR MODELO EXISTENTE ──────────────────────────────

    def _ml_load_if_exists(self):
        """Al arrancar la pestaña, carga el modelo en caché si existe."""
        try:
            from ml_model import load_model, model_exists
            if model_exists():
                model, cv_info = load_model()
                self._ml_model   = model
                self._ml_cv_info = cv_info
                self._ml_refresh_display(cv_info)
        except Exception:
            pass

    # ─── ENTRENAMIENTO EN BACKGROUND ──────────────────────────

    def _run_ml_train(self):
        """Lanza el entrenamiento en un hilo background."""
        self._ml_status_lbl.config(text='Construyendo dataset...', fg=COLORS['warning'])
        self._ml_status_lbl.update_idletasks()
        threading.Thread(target=self._ml_train_worker, daemon=True).start()

    def _ml_train_worker(self):
        try:
            from ml_dataset import load_or_build_dataset, feature_columns
            from ml_model import train_model, save_model
            from ml_signal import invalidate_model_cache

            self.root.after(0, lambda: self._ml_status_lbl.config(
                text='Cargando dataset...', fg=COLORS['warning']))

            db_path = getattr(self, '_db_path', 'trading_bot.db')
            df = load_or_build_dataset(db_path=db_path)

            if df.empty:
                self.root.after(0, lambda: self._ml_status_lbl.config(
                    text='Dataset vacío — ejecuta el análisis primero.',
                    fg=COLORS['danger']))
                return

            self.root.after(0, lambda: self._ml_status_lbl.config(
                text=f'Entrenando LightGBM ({len(df)} obs)...',
                fg=COLORS['warning']))

            model, cv_info = train_model(df)
            save_model(model, cv_info)
            invalidate_model_cache()  # fuerza recarga en ml_signal

            self._ml_model   = model
            self._ml_cv_info = cv_info

            self.root.after(0, lambda: self._ml_refresh_display(cv_info))

        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: self._ml_status_lbl.config(
                text=f'Error: {msg[:80]}', fg=COLORS['danger']))

    # ─── COMPARACIÓN EN BACKGROUND ────────────────────────────

    def _run_ml_compare(self):
        if not hasattr(self, '_ml_model') or self._ml_model is None:
            messagebox.showinfo('Modelo ML', 'Entrena el modelo primero.')
            return
        if not hasattr(self, 'df_results') or self.df_results is None or self.df_results.empty:
            messagebox.showinfo('Modelo ML',
                                'Ejecuta el análisis completo primero para tener datos OOS.')
            return
        self._ml_status_lbl.config(text='Comparando estrategias...', fg=COLORS['warning'])
        threading.Thread(target=self._ml_compare_worker, daemon=True).start()

    def _ml_compare_worker(self):
        try:
            from ml_dataset import load_or_build_dataset, feature_columns
            from ml_compare import compare_strategies

            model   = self._ml_model
            cv_info = self._ml_cv_info
            feat_cols = cv_info.get("feature_cols", [])

            db_path = getattr(self, '_db_path', 'trading_bot.db')
            df = load_or_build_dataset(db_path=db_path)
            if df.empty:
                return

            feat_cols = [c for c in feat_cols if c in df.columns]
            X = df[feat_cols].values.astype(np.float32)
            df = df.copy()
            df["ml_prob"] = model.predict(X)

            result = compare_strategies(df, oos_months=6)
            self.root.after(0, lambda: self._ml_display_compare(result))

        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: self._ml_status_lbl.config(
                text=f'Error comparando: {msg[:80]}', fg=COLORS['danger']))

    # ─── ACTUALIZAR DISPLAY ──────────────────────────────────

    def _ml_refresh_display(self, cv_info: dict):
        """Actualiza status, tabla CV y gráfico de importancia."""
        mean_auc = cv_info.get("mean_auc", 0)
        std_auc  = cv_info.get("std_auc", 0)
        n_iter   = cv_info.get("best_iteration", 0)
        n_feat   = cv_info.get("n_features", len(cv_info.get("feature_cols", [])))
        n_samp   = cv_info.get("n_samples", 0)

        status = (f'AUC={mean_auc:.3f} ±{std_auc:.3f}  |  '
                  f'iter={n_iter}  feat={n_feat}  obs={n_samp}')
        self._ml_status_lbl.config(text=status, fg=COLORS['success'])

        # Tabla CV
        for row in self._ml_cv_tree.get_children():
            self._ml_cv_tree.delete(row)
        for fold in cv_info.get("fold_results", []):
            self._ml_cv_tree.insert('', 'end', values=(
                fold.get("fold", ""),
                fold.get("n_train", ""),
                fold.get("n_test", ""),
                f"{fold.get('auc', 0):.4f}",
                fold.get("best_iteration", ""),
            ))

        # Gráfico feature importance
        self._ml_plot_importance()

    def _ml_plot_importance(self):
        """Dibuja el gráfico de importancia de features con matplotlib."""
        if not hasattr(self, '_ml_model') or self._ml_model is None:
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from ml_model import feature_importance

            cv_info  = getattr(self, '_ml_cv_info', {})
            feat_cols = cv_info.get("feature_cols", [])
            if not feat_cols:
                return

            fi = feature_importance(self._ml_model, feat_cols)
            top_n = min(20, len(fi))
            fi_top = fi.head(top_n)

            # Limpiar frame anterior
            for w in self._ml_plot_frame.winfo_children():
                w.destroy()

            fig, ax = plt.subplots(figsize=(5.2, max(3.5, top_n * 0.28)))
            fig.patch.set_facecolor(COLORS['bg'])
            ax.set_facecolor(COLORS['surface'])

            bars = ax.barh(
                range(top_n),
                fi_top['importance_gain_norm'].values * 100,
                color='#4f8ef7', edgecolor='none', height=0.72
            )
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(fi_top['feature'].values,
                               fontsize=7, color=COLORS['text'])
            ax.set_xlabel('Importancia (%)', fontsize=8, color=COLORS['text_dim'])
            ax.tick_params(axis='x', colors=COLORS['text_dim'], labelsize=7)
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for sp in ('bottom', 'left'):
                ax.spines[sp].set_color(COLORS['surface2'])

            fig.tight_layout(pad=1.2)

            canvas = FigureCanvasTkAgg(fig, master=self._ml_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

        except ImportError:
            tk.Label(self._ml_plot_frame,
                     text='matplotlib no disponible',
                     bg=COLORS['bg'], fg=COLORS['text_dim'],
                     font=_FONT_LABEL).pack(expand=True)
        except Exception as e:
            tk.Label(self._ml_plot_frame,
                     text=f'Error al graficar: {e}',
                     bg=COLORS['bg'], fg=COLORS['text_dim'],
                     font=_FONT_LABEL).pack(expand=True)

    def _ml_display_compare(self, result: dict):
        """Actualiza el panel de comparación Sharpe."""
        # Limpiar contenido anterior
        for w in self._ml_cmp_frame.winfo_children():
            w.destroy()

        ml = result.get("ml", {})
        ru = result.get("rules", {})
        bh = result.get("buy_hold", {})
        imp = result.get("sharpe_improvement")
        start = result.get("oos_start", "?")
        end   = result.get("oos_end",   "?")

        def _val(d, k, pct=False):
            v = d.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:.2f}{'%' if pct else ''}"

        info = tk.Frame(self._ml_cmp_frame, bg=COLORS['bg'])
        info.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Período OOS
        tk.Label(info, text=f'OOS: {start} → {end}',
                 bg=COLORS['bg'], fg=COLORS['text_dim'],
                 font=_FONT_SMALL).grid(row=0, column=0, columnspan=4,
                                         sticky='w', pady=(0, 6))

        # Cabeceras
        for col_i, hdr in enumerate(('Métrica', 'ML', 'Reglas', 'B&H')):
            tk.Label(info, text=hdr, bg=COLORS['bg'],
                     fg=COLORS['text_dim'],
                     font=('Segoe UI', 8, 'bold')).grid(
                         row=1, column=col_i, padx=6, pady=2, sticky='e')

        metrics_rows = [
            ('Sharpe',     'sharpe',      False),
            ('Win Rate',   'win_rate',    False),
            ('Ret. avg',   'avg_return',  True),
            ('Max DD',     'max_drawdown',True),
            ('N Trades',   'n_trades',    False),
        ]
        for ri, (label, key, pct) in enumerate(metrics_rows, start=2):
            tk.Label(info, text=label, bg=COLORS['bg'],
                     fg=COLORS['text'], font=_FONT_SMALL).grid(
                         row=ri, column=0, padx=6, pady=1, sticky='w')
            for ci, d in enumerate((ml, ru, bh), start=1):
                v = _val(d, key, pct)
                fg = COLORS['text']
                if key == 'sharpe' and ci == 1 and imp is not None and imp > 0:
                    fg = COLORS['success']
                tk.Label(info, text=v, bg=COLORS['bg'],
                         fg=fg, font=_FONT_SMALL).grid(
                             row=ri, column=ci, padx=6, pady=1, sticky='e')

        # Resultado comparativo
        if imp is not None:
            sign   = "+" if imp >= 0 else ""
            winner = "MODELO ML" if imp > 0.05 else "REGLAS" if imp < -0.05 else "SIMILAR"
            color  = COLORS['success'] if imp > 0.05 else (
                COLORS['danger'] if imp < -0.05 else COLORS['text_dim'])
            tk.Label(info,
                     text=f'Mejora Sharpe: {sign}{imp:.3f}  →  {winner}',
                     bg=COLORS['bg'], fg=color,
                     font=('Segoe UI', 8, 'bold')).grid(
                         row=len(metrics_rows) + 2, column=0, columnspan=4,
                         sticky='w', padx=6, pady=(8, 0))

        self._ml_status_lbl.config(text='Comparación completada.', fg=COLORS['success'])
