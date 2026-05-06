"""
JournalTabMixin — trade journal tab extracted from investment_report_gui.py (Etapa 4.1).
Mixed into InvestmentReportGUI; all methods use self.* from the host class.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

import pandas as pd

from gui.colors import COLORS


class JournalTabMixin:

    # ─── TAB: TRADE JOURNAL ──────────────────

    def _build_journal(self):
        p = self.tab_journal

        # ── Formulario nueva operación ──
        form_frame = tk.Frame(p, bg=COLORS['surface'], padx=20, pady=14)
        form_frame.pack(fill=tk.X, padx=24, pady=(16, 0))

        tk.Label(form_frame, text='Registrar nueva compra',
                 bg=COLORS['surface'], fg=COLORS['text'],
                 font=('Segoe UI', 11, 'bold')).grid(
                 row=0, column=0, columnspan=8, sticky='w', pady=(0, 10))

        def lbl(text, col):
            tk.Label(form_frame, text=text, bg=COLORS['surface'],
                     fg=COLORS['text_dim'], font=('Segoe UI', 9)).grid(
                     row=1, column=col, sticky='w', padx=(0, 4))

        def ent(var, col, width=10):
            e = tk.Entry(form_frame, textvariable=var, width=width,
                         bg=COLORS['surface2'], fg=COLORS['text'],
                         insertbackground=COLORS['text'],
                         relief='flat', bd=4, font=('Segoe UI', 9))
            e.grid(row=2, column=col, padx=(0, 12), sticky='w')
            return e

        self._j_sym   = tk.StringVar()
        self._j_date  = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        self._j_price = tk.StringVar()
        self._j_qty   = tk.StringVar()
        self._j_eur   = tk.StringVar()
        self._j_stop  = tk.StringVar()
        self._j_tp    = tk.StringVar()
        self._j_fees  = tk.StringVar(value='0')
        self._j_notes = tk.StringVar()

        lbl('Símbolo',    0); ent(self._j_sym,   0, 8)
        lbl('Fecha',      1); ent(self._j_date,  1, 11)
        lbl('Precio €',   2); ent(self._j_price, 2, 9)
        lbl('Acciones',   3); ent(self._j_qty,   3, 8)
        lbl('Total EUR',  4); ent(self._j_eur,   4, 9)
        lbl('Stop Loss',  5); ent(self._j_stop,  5, 9)
        lbl('Take Profit',6); ent(self._j_tp,    6, 9)
        lbl('Comisión €', 7); ent(self._j_fees,  7, 7)
        lbl('Notas',      8); ent(self._j_notes, 8, 16)

        tk.Button(form_frame, text='✓ Registrar',
                  command=self._journal_add,
                  bg=COLORS['success'], fg='white',
                  font=('Segoe UI', 9, 'bold'), relief='flat',
                  padx=12, pady=4, cursor='hand2').grid(
                  row=2, column=9, padx=(8, 0))

        self._j_sym.trace_add('write', self._journal_autofill)

        # ── Posiciones ABIERTAS ──
        tk.Label(p, text='Posiciones Abiertas',
                 bg=COLORS['bg'], fg=COLORS['text'],
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=24, pady=(14, 4))

        open_cols = ('id', 'symbol', 'entry_date', 'entry_price', 'qty',
                     'amount', 'cur_price', 'pnl_eur', 'pnl_pct', 'days',
                     'stop', 'tp', 'mae', 'signal')
        open_hdrs = ('ID', 'Símbolo', 'Fecha', 'P.Entrada', 'Acciones',
                     'EUR', 'P.Actual', 'P&L €', 'P&L %',
                     'Días', 'SL', 'TP', 'MAE%', 'Señal')

        self.journal_open_tree, scroll_jo = self._make_treeview(
            p, open_cols, open_hdrs,
            col_widths=[32, 68, 88, 90, 68, 80, 90, 78, 68, 45, 78, 78, 55, 75])
        scroll_jo.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 4))
        self.journal_open_tree.bind('<Double-1>', self._journal_close_dialog)

        # Botones acción
        btn_bar = tk.Frame(p, bg=COLORS['bg'])
        btn_bar.pack(fill=tk.X, padx=24, pady=(0, 8))

        tk.Button(btn_bar, text='↓ Cerrar posición (doble-click o aquí)',
                  command=self._journal_close_dialog,
                  bg=COLORS['warning'], fg='white',
                  font=('Segoe UI', 9), relief='flat', padx=10, pady=4,
                  cursor='hand2').pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(btn_bar, text='✕ Eliminar',
                  command=self._journal_delete,
                  bg=COLORS['danger'], fg='white',
                  font=('Segoe UI', 9), relief='flat', padx=10, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)

        tk.Button(btn_bar, text='↺ Actualizar',
                  command=self._populate_journal,
                  bg=COLORS['surface2'], fg=COLORS['text'],
                  font=('Segoe UI', 9), relief='flat', padx=10, pady=4,
                  cursor='hand2').pack(side=tk.RIGHT)

        # ── Resumen P&L ──
        self.journal_summary = tk.Label(p, text='',
                                         bg=COLORS['surface'], fg=COLORS['text'],
                                         font=('Segoe UI', 9, 'bold'),
                                         anchor='w', padx=16, pady=6)
        self.journal_summary.pack(fill=tk.X, padx=24, pady=(0, 4))

        # ── Historial CERRADAS ──
        tk.Label(p, text='Historial — Posiciones Cerradas',
                 bg=COLORS['bg'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=24, pady=(6, 2))

        closed_cols = ('symbol', 'entry_date', 'exit_date', 'entry_price',
                       'exit_price', 'amount', 'pnl_eur', 'pnl_pct', 'pnl_net', 'fees', 'mae')
        closed_hdrs = ('Símbolo', 'Entrada', 'Salida', 'P.Entrada',
                       'P.Salida', 'EUR', 'P&L €', 'P&L %', 'P&L Neto', 'Fees', 'MAE%')

        self.journal_closed_tree, scroll_jc = self._make_treeview(
            p, closed_cols, closed_hdrs,
            col_widths=[68, 88, 88, 88, 88, 80, 78, 68, 80, 58, 55])
        scroll_jc.pack(fill=tk.BOTH, expand=False, padx=24, pady=(0, 4))

        # ── Concentración de cartera ──
        self.journal_concentration = tk.Label(
            p, text='', bg=COLORS['surface2'], fg=COLORS['text_dim'],
            font=('Segoe UI', 9), anchor='w', padx=16, pady=5, wraplength=900)
        self.journal_concentration.pack(fill=tk.X, padx=24, pady=(0, 8))

        # Botón historial de señales
        btn_hist = tk.Frame(p, bg=COLORS['bg'])
        btn_hist.pack(fill=tk.X, padx=24, pady=(0, 8))
        tk.Button(btn_hist, text='📊 Historial de Señales',
                  command=self._show_signal_history_popup,
                  bg=COLORS['surface2'], fg=COLORS['primary'],
                  font=('Segoe UI', 9, 'bold'), relief='flat', padx=10, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)
        tk.Label(btn_hist,
                 text='  Ver cómo evolucionaron los rankings y señales entre análisis',
                 bg=COLORS['bg'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)

        self._populate_journal()

    def _journal_autofill(self, *_):
        sym = self._j_sym.get().strip().upper()
        if not sym or self.df_results is None:
            return
        rows = self.df_results[self.df_results['symbol'] == sym]
        if rows.empty:
            return
        row = rows.iloc[0]
        price   = row.get('price')
        atr_pct = row.get('atr_pct')
        if price and not self._j_price.get():
            self._j_price.set(f"{price:.2f}")
        if price and atr_pct and not self._j_stop.get():
            stop = price - 2 * (price * atr_pct / 100)
            self._j_stop.set(f"{stop:.2f}")
        if price and atr_pct and not self._j_tp.get():
            tp = price + 3 * (price * atr_pct / 100)
            self._j_tp.set(f"{tp:.2f}")

    def _journal_add(self):
        try:
            sym    = self._j_sym.get().strip().upper()
            date   = self._j_date.get().strip()
            price  = float(self._j_price.get().replace(',', '.'))
            qty    = float(self._j_qty.get().replace(',', '.')) if self._j_qty.get() else 0
            eur    = float(self._j_eur.get().replace(',', '.'))
            stop   = float(self._j_stop.get().replace(',', '.')) if self._j_stop.get() else None
            tp     = float(self._j_tp.get().replace(',', '.'))   if self._j_tp.get()   else None
            fees   = float(self._j_fees.get().replace(',', '.')) if self._j_fees.get() else 0.0
            notes  = self._j_notes.get().strip()

            if not sym or not date or price <= 0 or eur <= 0:
                messagebox.showwarning('Datos incompletos',
                                       'Símbolo, fecha, precio y total EUR son obligatorios.')
                return

            if qty == 0:
                qty = eur / price

            self.db.journal_open(sym, date, price, qty, eur, stop, tp, fees, 0.0, notes)
            for v in (self._j_sym, self._j_price, self._j_qty,
                      self._j_eur, self._j_stop, self._j_tp, self._j_notes):
                v.set('')
            self._j_fees.set('0')
            self._populate_journal()
            self._log('ok', f'Trade registrado: {sym} @ ${price:.2f} ({eur:.0f}€)')
        except ValueError as e:
            messagebox.showerror('Error', f'Valor inválido: {e}')

    def _journal_close_dialog(self, event=None):
        sel = self.journal_open_tree.selection()
        if not sel:
            messagebox.showinfo('Selecciona', 'Selecciona una posición abierta primero.')
            return
        vals = self.journal_open_tree.item(sel[0], 'values')
        trade_id = int(vals[0])
        sym      = vals[1]

        win = tk.Toplevel(self.root)
        win.title(f'Cerrar posición — {sym}')
        win.geometry('340x220')
        win.configure(bg=COLORS['bg'])
        win.grab_set()

        tk.Label(win, text=f'Cerrar {sym}', bg=COLORS['bg'], fg=COLORS['text'],
                 font=('Segoe UI', 12, 'bold')).pack(pady=(16, 8))

        frm = tk.Frame(win, bg=COLORS['bg'])
        frm.pack()

        tk.Label(frm, text='Fecha salida:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w')
        exit_date = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        tk.Entry(frm, textvariable=exit_date, width=14,
                 bg=COLORS['surface2'], fg=COLORS['text'],
                 insertbackground=COLORS['text'], relief='flat', bd=4,
                 font=('Segoe UI', 9)).grid(row=0, column=1, padx=8, pady=4)

        tk.Label(frm, text='Precio salida:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w')
        exit_price = tk.StringVar()
        if self.df_results is not None:
            r = self.df_results[self.df_results['symbol'] == sym]
            if not r.empty:
                exit_price.set(f"{r.iloc[0]['price']:.2f}")
        tk.Entry(frm, textvariable=exit_price, width=14,
                 bg=COLORS['surface2'], fg=COLORS['text'],
                 insertbackground=COLORS['text'], relief='flat', bd=4,
                 font=('Segoe UI', 9)).grid(row=1, column=1, padx=8, pady=4)

        tk.Label(frm, text='Comision salida €:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).grid(row=2, column=0, sticky='w')
        exit_fees = tk.StringVar(value='0')
        tk.Entry(frm, textvariable=exit_fees, width=14,
                 bg=COLORS['surface2'], fg=COLORS['text'],
                 insertbackground=COLORS['text'], relief='flat', bd=4,
                 font=('Segoe UI', 9)).grid(row=2, column=1, padx=8, pady=4)

        def do_close():
            try:
                ep   = float(exit_price.get().replace(',', '.'))
                fees = float(exit_fees.get().replace(',', '.')) if exit_fees.get() else 0.0
                self.db.journal_close(trade_id, exit_date.get(), ep, fees)
                win.destroy()
                self._populate_journal()
                self._log('ok', f'Posicion {sym} cerrada @ ${ep:.2f}')
            except ValueError:
                messagebox.showerror('Error', 'Precio invalido')

        tk.Button(win, text='Confirmar cierre', command=do_close,
                  bg=COLORS['danger'], fg='white',
                  font=('Segoe UI', 10, 'bold'), relief='flat',
                  padx=16, pady=6, cursor='hand2').pack(pady=14)

    def _journal_delete(self):
        sel = self.journal_open_tree.selection()
        if not sel:
            return
        vals  = self.journal_open_tree.item(sel[0], 'values')
        trade_id = int(vals[0])
        sym   = vals[1]
        if messagebox.askyesno('Eliminar', f'¿Eliminar operación de {sym}?'):
            self.db.journal_delete(trade_id)
            self._populate_journal()

    def _populate_journal(self):
        if self.db is None:
            return
        tree = self.journal_open_tree
        tree.delete(*tree.get_children())
        df_open = self.db.journal_get_open()

        total_invested = 0.0
        total_pnl = 0.0

        if not df_open.empty:
            price_map = {}
            if self.df_results is not None and not self.df_results.empty:
                price_map = self.df_results.set_index('symbol')['price'].to_dict()
            signal_map = {}
            if self.df_results is not None and not self.df_results.empty:
                signal_map = self.df_results.set_index('symbol')['confidence'].to_dict()

            for _, row in df_open.iterrows():
                sym        = row['symbol']
                entry_p    = row['entry_price']
                qty        = row['quantity']
                amount     = row['amount_eur']
                stop       = row['stop_loss']
                entry_date = row['entry_date']
                cur_p      = price_map.get(sym, entry_p)
                pnl_eur    = (cur_p - entry_p) * qty
                pnl_pct    = (cur_p - entry_p) / entry_p * 100

                try:
                    days = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
                except Exception:
                    days = '?'

                total_invested += amount
                total_pnl      += pnl_eur

                tp  = row.get('take_profit')
                mae = row.get('mae_pct')

                try:
                    self.db.journal_update_mae(row['id'], cur_p)
                    mae = self.db.journal_get_open().set_index('id').at[row['id'], 'mae_pct'] if mae is None else min(mae, pnl_pct)
                except Exception:
                    pass

                tag = 'buy' if pnl_pct >= 0 else 'sell'
                tree.insert('', 'end', tags=(tag,), values=(
                    row['id'], sym, entry_date,
                    f"${entry_p:.2f}", f"{qty:.4f}",
                    f"{amount:.0f}€",
                    f"${cur_p:.2f}",
                    f"{pnl_eur:+.2f}€",
                    f"{pnl_pct:+.1f}%",
                    days,
                    f"${stop:.2f}" if stop else '—',
                    f"${tp:.2f}"   if tp   else '—',
                    f"{mae:.1f}%"  if mae is not None else '—',
                    signal_map.get(sym, '—'),
                ))

        try:
            stats = self.db.journal_get_stats()
            stats_txt = ''
            if stats['total_trades'] > 0:
                stats_txt = (
                    f"  Cerradas: {stats['total_trades']}  |  "
                    f"Win rate: {stats['win_rate']:.0f}%  |  "
                    f"P&L neto: {stats['total_pnl_eur']:+.2f}€  |  "
                    f"Fees: {stats['total_fees_eur']:.2f}€  |  "
                    f"Hold avg: {stats['avg_hold_days']:.0f}d"
                    if stats['avg_hold_days'] else ''
                )
        except Exception:
            stats_txt = ''

        pnl_color = COLORS['success'] if total_pnl >= 0 else COLORS['danger']
        self.journal_summary.config(
            text=(f"  Invertido: {total_invested:.0f}€   |   "
                  f"P&L abierto: {total_pnl:+.2f}€   |   "
                  f"Pos. abiertas: {len(df_open)}"
                  + (f"   |   {stats_txt}" if stats_txt else '')),
            fg=pnl_color if total_pnl != 0 else COLORS['text']
        )

        self._update_concentration_warning(df_open, total_invested)

        ctree = self.journal_closed_tree
        ctree.delete(*ctree.get_children())
        df_closed = self.db.journal_get_closed()

        if not df_closed.empty:
            for _, row in df_closed.iterrows():
                pnl_pct    = row.get('pnl_pct', 0) or 0
                pnl_net    = row.get('pnl_net_eur')
                fees_total = (row.get('fees_eur') or 0) + (row.get('exit_fees_eur') or 0)
                mae        = row.get('mae_pct')
                tag = 'buy' if pnl_pct >= 0 else 'sell'
                ctree.insert('', 'end', tags=(tag,), values=(
                    row['symbol'],
                    row['entry_date'],
                    row.get('exit_date', '—'),
                    f"${row['entry_price']:.2f}",
                    f"${row['exit_price']:.2f}" if row.get('exit_price') else '—',
                    f"{row['amount_eur']:.0f}€",
                    f"{row.get('pnl_eur', 0):+.2f}€" if row.get('pnl_eur') is not None else '—',
                    f"{pnl_pct:+.1f}%",
                    f"{pnl_net:+.2f}€" if pnl_net is not None else '—',
                    f"{fees_total:.2f}€" if fees_total else '—',
                    f"{mae:.1f}%" if mae is not None else '—',
                ))

    def _update_concentration_warning(self, df_open, total_invested):
        label = self.journal_concentration
        if df_open.empty or total_invested <= 0:
            label.config(text='  No hay posiciones abiertas.',
                         fg=COLORS['text_muted'])
            return

        sector_map = {}
        if self.df_results is not None and not self.df_results.empty:
            sector_map = self.df_results.set_index('symbol')['sector'].to_dict()
        if not sector_map:
            try:
                from stock_universe import get_sector
                sector_map = {sym: get_sector(sym) for sym in df_open['symbol']}
            except Exception:
                pass

        sector_inv = {}
        for _, row in df_open.iterrows():
            sym    = row['symbol']
            amount = row['amount_eur']
            sect   = sector_map.get(sym, 'Unknown')
            sector_inv[sect] = sector_inv.get(sect, 0) + amount

        parts = []
        warnings = []
        for sect, inv in sorted(sector_inv.items(), key=lambda x: -x[1]):
            pct = inv / total_invested * 100
            parts.append(f"{sect}: {pct:.0f}%")
            if pct > 50:
                warnings.append(f"⚠️ '{sect}' concentra el {pct:.0f}% de tu cartera (>50%)")
            elif pct > 35:
                warnings.append(f"⚡ '{sect}' concentra el {pct:.0f}% (considera diversificar)")

        dist_txt = "  Distribución sectorial:  " + "  ·  ".join(parts)
        if warnings:
            full_txt  = dist_txt + "     " + "     ".join(warnings)
            fg_color  = COLORS['danger'] if any('⚠️' in w for w in warnings) else COLORS['warning']
        else:
            full_txt  = dist_txt + "     ✅ Cartera bien diversificada"
            fg_color  = COLORS['text_dim']

        label.config(text=full_txt, fg=fg_color)

    def _show_signal_history_popup(self):
        if self.db is None:
            messagebox.showinfo('Sin datos', 'Ejecuta el análisis primero para generar historial.')
            return

        try:
            conn = self.db.connect()
            df_hist = pd.read_sql(
                'SELECT * FROM signal_snapshots ORDER BY run_timestamp ASC', conn)
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo leer el historial: {e}')
            return

        if df_hist.empty:
            messagebox.showinfo('Sin historial',
                                'Todavía no hay análisis previos guardados.\n'
                                'Ejecuta el análisis al menos 2 veces para ver la evolución.')
            return

        win = tk.Toplevel(self.root)
        win.title('📊 Historial de Señales')
        win.geometry('1000x580')
        win.configure(bg=COLORS['bg'])
        win.resizable(True, True)

        hdr = tk.Frame(win, bg=COLORS['surface'], padx=20, pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text='📊  Historial de Señales — Evolución entre análisis',
                 bg=COLORS['surface'], fg=COLORS['primary'],
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w')

        n_runs = df_hist['run_timestamp'].nunique()
        n_syms = df_hist['symbol'].nunique()
        tk.Label(hdr,
                 text=f'{n_runs} análisis guardados · {n_syms} símbolos · '
                      f'Ordenado de más antiguo a más reciente',
                 bg=COLORS['surface'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(2, 0))

        ctrl = tk.Frame(win, bg=COLORS['bg'])
        ctrl.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(ctrl, text='Filtrar símbolo:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).pack(side=tk.LEFT)
        sym_var = tk.StringVar(value='Todos')
        sym_list = ['Todos'] + sorted(df_hist['symbol'].unique().tolist())
        sym_cb = ttk.Combobox(ctrl, textvariable=sym_var, width=10, state='readonly',
                              values=sym_list, font=('Segoe UI', 9))
        sym_cb.pack(side=tk.LEFT, padx=(6, 20))

        tk.Label(ctrl, text='Solo señal COMPRAR:', bg=COLORS['bg'],
                 fg=COLORS['text_dim'], font=('Segoe UI', 9)).pack(side=tk.LEFT)
        only_buy = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, variable=only_buy,
                       bg=COLORS['bg'], fg=COLORS['text'],
                       selectcolor=COLORS['surface2'],
                       activebackground=COLORS['bg']).pack(side=tk.LEFT, padx=(4, 0))

        cols = ('ts', 'sym', 'conf', 'score', 'vol', 'rs', 'risk', 'total', 'top')
        hdrs_h = ('Análisis (fecha)', 'Símbolo', 'Confianza', 'Score Conf.',
                  'Vol MA', 'RS', 'Riesgo', 'Score Total', '¿Top Compras?')
        hist_tree, hist_sf = self._make_treeview(
            win, cols, hdrs_h,
            col_widths=[160, 68, 88, 78, 90, 90, 90, 82, 88])
        hist_sf.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 6))

        def populate_history(*_):
            hist_tree.delete(*hist_tree.get_children())
            df = df_hist.copy()
            sym_filter = sym_var.get()
            if sym_filter and sym_filter != 'Todos':
                df = df[df['symbol'] == sym_filter]
            if only_buy.get():
                df = df[df['vol_signal'] == 'COMPRAR']

            df = df.sort_values(['symbol', 'run_timestamp'], ascending=[True, False])

            prev_sym_score = {}
            for _, row in df.iterrows():
                sym   = row['symbol']
                ts    = row['run_timestamp']
                conf  = row.get('confidence', '—')
                cscore= row.get('confidence_score')
                vol   = row.get('vol_signal', '—')
                rs    = row.get('rs_signal', '—')
                risk  = row.get('risk_signal', '—')
                total = row.get('total_score')
                top   = '✅ SÍ' if row.get('in_top_buys', 0) else '—'

                prev_t = prev_sym_score.get(sym)
                if prev_t is not None and total is not None:
                    diff = (total or 0) - (prev_t or 0)
                    if diff >= 5:
                        trend = '↑'
                    elif diff <= -5:
                        trend = '↓'
                    else:
                        trend = '→'
                    total_str = f"{total:.1f} {trend}" if total is not None else '—'
                else:
                    total_str = f"{total:.1f}" if total is not None else '—'
                prev_sym_score[sym] = total

                tag = ('buy'  if vol == 'COMPRAR'
                       else 'sell' if vol == 'VENDER'
                       else 'neutral')
                hist_tree.insert('', 'end', tags=(tag,), values=(
                    ts[:16], sym, conf,
                    f"{cscore:.1f}" if cscore is not None else '—',
                    vol, rs, risk,
                    total_str, top,
                ))

        sym_cb.bind('<<ComboboxSelected>>', populate_history)
        only_buy.trace_add('write', populate_history)
        populate_history()

        try:
            latest_ts  = df_hist['run_timestamp'].max()
            oldest_ts  = df_hist['run_timestamp'].min()
            df_latest  = df_hist[df_hist['run_timestamp'] == latest_ts]
            df_oldest  = df_hist[df_hist['run_timestamp'] == oldest_ts]
            merged = df_latest.merge(df_oldest, on='symbol', suffixes=('_new', '_old'))
            merged['delta'] = merged['total_score_new'].fillna(0) - merged['total_score_old'].fillna(0)
            top_improv = merged.nlargest(3, 'delta')[['symbol', 'delta']]
            top_improv_txt = '  '.join(
                f"{r['symbol']} ({r['delta']:+.1f})" for _, r in top_improv.iterrows()
            )
            summary_txt = f"  📈 Mayor mejora entre primer y último análisis: {top_improv_txt}"
        except Exception:
            summary_txt = f"  {n_runs} análisis · {n_syms} símbolos registrados"

        tk.Label(win, text=summary_txt,
                 bg=COLORS['surface2'], fg=COLORS['text_dim'],
                 font=('Segoe UI', 9), anchor='w',
                 padx=12, pady=5).pack(fill=tk.X, padx=16, pady=(0, 8))

        tk.Button(win, text='Cerrar', command=win.destroy,
                  bg=COLORS['surface3'], fg=COLORS['text'],
                  relief='flat', cursor='hand2',
                  font=('Segoe UI', 9), padx=12, pady=4).pack(pady=(0, 10))
