"""
AlertsMixin — alert detection methods extracted from investment_report_gui.py (Etapa 4.1).
Mixed into InvestmentReportGUI; all methods use self.* from the host class.
"""
from __future__ import annotations

import tkinter as tk

from gui.colors import COLORS


class AlertsMixin:

    # ─── ALERTAS DE CAMBIO DE SEÑAL ──────────

    def _detect_signal_changes(self, df_current):
        """
        Compara el análisis actual con el snapshot anterior.
        Devuelve lista de dicts con los cambios detectados.
        """
        prev = self.db.get_last_snapshot()
        if prev is None or prev.empty:
            return []

        alerts = []
        prev_map = prev.set_index('symbol').to_dict('index')
        conf_order = {'MUY BAJA': 0, 'BAJA': 1, 'MEDIA': 2, 'ALTA': 3, 'MUY ALTA': 4}

        for _, row in df_current.iterrows():
            sym   = row.get('symbol', '')
            if sym not in prev_map:
                continue
            old   = prev_map[sym]

            conf_new = row.get('confidence', '')
            conf_old = old.get('confidence', '')
            vol_new  = row.get('vol_signal', '')
            vol_old  = old.get('vol_signal', '')
            top_new  = vol_new == 'COMPRAR'
            top_old  = bool(old.get('in_top_buys', 0))

            reasons = []

            lvl_new = conf_order.get(conf_new, -1)
            lvl_old = conf_order.get(conf_old, -1)
            if lvl_new != lvl_old and lvl_new >= 0 and lvl_old >= 0:
                diff = lvl_new - lvl_old
                if abs(diff) >= 2:
                    direction = '↑' if diff > 0 else '↓'
                    reasons.append(f"Confianza {conf_old} → {conf_new} {direction}")

            if top_new and not top_old:
                reasons.append(f"Entró en Top Compras (Vol {vol_new})")
            elif not top_new and top_old:
                reasons.append(f"Salió de Top Compras (Vol {vol_new})")

            if vol_new != vol_old and vol_old not in ('N/A', ''):
                reasons.append(f"Vol MA: {vol_old} → {vol_new}")

            if reasons:
                alerts.append({
                    'symbol':  sym,
                    'sector':  row.get('sector', ''),
                    'price':   row.get('price'),
                    'reasons': ' | '.join(reasons),
                    'type':    'positive' if (lvl_new > lvl_old or top_new) else 'negative',
                })

        alerts.sort(key=lambda a: (0 if a['type'] == 'positive' else 1, a['symbol']))
        return alerts

    def _detect_smart_alerts(self, df_current):
        """
        Alertas inteligentes para posiciones del journal (3.6).

        Detecta:
          - Stop tecnico: precio <= stop_loss (o dentro del 3%)
          - Stop tecnico semanal: tf_alignment del analisis actual = BAJISTA
          - Stop fundamental: fund_signal deterioro a DESFAVORABLE/NEGATIVO
          - Score critico: total_score cayo > 15 pts vs snapshot anterior
        """
        if self.db is None or df_current is None or df_current.empty:
            return []

        try:
            df_open = self.db.journal_get_open()
        except Exception:
            return []

        if df_open.empty:
            return []

        prev = self.db.get_last_snapshot()
        prev_map = prev.set_index('symbol').to_dict('index') if (prev is not None and not prev.empty) else {}

        current_map = df_current.set_index('symbol').to_dict('index')
        alerts = []

        for _, trade in df_open.iterrows():
            sym        = trade['symbol']
            entry_p    = trade['entry_price']
            stop       = trade.get('stop_loss')

            if sym not in current_map:
                continue

            cur = current_map[sym]
            cur_price   = cur.get('price') or entry_p
            tf_align    = cur.get('tf_alignment', '')
            fund_signal = cur.get('fund_signal', '')
            score_now   = cur.get('total_score')
            score_prev  = prev_map.get(sym, {}).get('total_score') if prev_map else None
            fund_prev   = prev_map.get(sym, {}).get('fund_signal', '') if prev_map else ''

            reasons = []
            severity = 'warning'

            if stop and cur_price > 0:
                if cur_price <= stop:
                    reasons.append(f"STOP LOSS ALCANZADO: precio ${cur_price:.2f} <= SL ${stop:.2f}")
                    severity = 'critical'
                elif cur_price <= stop * 1.03:
                    pct_above = (cur_price / stop - 1) * 100
                    reasons.append(f"SL PROXIMO: precio a {pct_above:.1f}% del stop (${stop:.2f})")

            if tf_align in ('BAJISTA', 'ALINEADO BAJISTA'):
                reasons.append(f"Tendencia semanal: {tf_align}")

            if fund_signal in ('DESFAVORABLE', 'NEGATIVO', 'MUY DESFAVORABLE'):
                if fund_prev not in ('DESFAVORABLE', 'NEGATIVO', 'MUY DESFAVORABLE'):
                    reasons.append(f"Fund: {fund_prev or 'anterior'} -> {fund_signal}")
                    severity = 'critical'
                else:
                    reasons.append(f"Fundamentales: {fund_signal}")

            if score_now is not None and score_prev is not None:
                drop = score_prev - score_now
                if drop >= 15:
                    reasons.append(f"Score cayo {drop:.0f} pts ({score_prev:.0f} -> {score_now:.0f})")

            if reasons:
                alerts.append({
                    'symbol':   sym,
                    'sector':   cur.get('sector', ''),
                    'price':    cur_price,
                    'entry':    entry_p,
                    'pnl_pct':  (cur_price / entry_p - 1) * 100 if entry_p else 0,
                    'reasons':  ' | '.join(reasons),
                    'type':     'critical' if severity == 'critical' else 'smart_warning',
                    'trade_id': trade['id'],
                })

        alerts.sort(key=lambda a: (0 if a['type'] == 'critical' else 1, a['symbol']))
        return alerts

    def _show_alerts_popup(self, alerts):
        """Muestra popup con alertas de cambio de señal y alertas inteligentes."""
        if not alerts:
            return

        n_smart    = sum(1 for a in alerts if a['type'] in ('critical', 'smart_warning'))
        n_signal   = len(alerts) - n_smart
        n_critical = sum(1 for a in alerts if a['type'] == 'critical')

        title = f'Alertas ({len(alerts)})'
        if n_critical:
            title = f'ALERTAS CRITICAS ({n_critical}) + Cambios ({n_signal})'

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry('860x480')
        win.configure(bg=COLORS['bg'])
        win.grab_set()

        hdr_color = COLORS['danger'] if n_critical else COLORS['surface']
        parts = []
        if n_critical:
            parts.append(f'{n_critical} criticas (posiciones en riesgo)')
        if n_smart - n_critical:
            parts.append(f'{n_smart - n_critical} advertencias de posicion')
        if n_signal:
            parts.append(f'{n_signal} cambios de senal')
        summary = '  ' + '   |   '.join(parts)

        tk.Label(win, text=summary,
                 bg=hdr_color, fg='white' if n_critical else COLORS['text'],
                 font=('Segoe UI', 11, 'bold'),
                 padx=16, pady=10).pack(fill=tk.X)

        cols = ('tipo', 'symbol', 'sector', 'price', 'cambio')
        hdrs = ('Tipo', 'Simbolo', 'Sector', 'Precio', 'Detalle')
        tree, scroll_f = self._make_treeview(win, cols, hdrs,
                                              col_widths=[80, 72, 140, 80, 400])
        scroll_f.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        _type_labels = {
            'critical':       'CRITICO',
            'smart_warning':  'ADVERTENCIA',
            'positive':       'MEJORA',
            'negative':       'DETERIORO',
        }

        for a in alerts:
            atype     = a['type']
            tag       = 'sell' if atype in ('critical', 'negative') else 'buy'
            price_str = f"${a['price']:.2f}" if a.get('price') else '—'
            tipo_lbl  = _type_labels.get(atype, atype.upper())
            tree.insert('', 'end', tags=(tag,), values=(
                tipo_lbl, a['symbol'], a.get('sector', ''), price_str, a['reasons']))

        tk.Button(win, text='Cerrar', command=win.destroy,
                  bg=COLORS['primary'], fg='white',
                  font=('Segoe UI', 10), relief='flat', padx=20, pady=6,
                  cursor='hand2').pack(pady=(0, 12))
