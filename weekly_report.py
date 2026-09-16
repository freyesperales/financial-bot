"""
weekly_report.py — Informe bi-semanal (lunes y viernes) con seguimiento de predicciones.

Qué hace en cada corrida:
  1. Carga el último análisis (last_analysis.parquet) y rankea con el screener 6m.
  2. TRACKING: guarda el top-20 de hoy en tracking/predictions_log.csv y evalúa
     todos los snapshots anteriores contra los precios actuales — ¿se están
     cumpliendo las predicciones? (retorno realizado vs SPY, hit rate).
  3. Detecta cambios vs el informe anterior (entradas/salidas del top 15, rotación
     sectorial) y redacta una interpretación; si el CLI `claude` está disponible,
     añade un comentario de analista generado por IA (fallback silencioso si no).
  4. Genera HTML → PDF (Microsoft Edge headless, sin dependencias extra) en
     informes/INFORME_6M_YYYY-MM-DD.pdf.
  5. Si existe email_config.json, envía el PDF por correo (SMTP). Plantilla en
     email_config.example.json — las credenciales las rellena el usuario.

Uso:
    python weekly_report.py [--no-ai] [--no-email] [--open]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
TRACK_DIR = BASE / "tracking"
TRACK_CSV = TRACK_DIR / "predictions_log.csv"
REPORT_DIR = BASE / "informes"
SNAPSHOT_N = 20      # cuántos símbolos del top se registran por corrida
MIN_EVAL_DAYS = 5    # días mínimos para evaluar un snapshot

EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


# ── Tracking de predicciones ─────────────────────────────────────────────────

def record_snapshot(ranked: pd.DataFrame, spy_price: float) -> None:
    """Añade el top-N de hoy al log de predicciones (idempotente por fecha)."""
    TRACK_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    if TRACK_CSV.exists():
        with open(TRACK_CSV, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
        if any(r["date"] == today for r in rows):
            return  # ya registrado hoy
    for _, x in ranked.head(SNAPSHOT_N).iterrows():
        rows.append({
            "date": today, "symbol": x["symbol"], "rank": int(x["rank"]),
            "score": f"{x['screener_score']:.1f}", "price": f"{x['price']:.4f}",
            "spy_price": f"{spy_price:.4f}", "region": x["region"],
        })
    with open(TRACK_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "symbol", "rank", "score",
                                          "price", "spy_price", "region"])
        w.writeheader()
        w.writerows(rows)


def evaluate_snapshots(prices_now: dict, spy_now: float) -> list[dict]:
    """
    Evalúa cada snapshot pasado contra los precios actuales.
    Retorna una lista por fecha con retorno medio, retorno SPY, exceso y hit rate.
    """
    if not TRACK_CSV.exists():
        return []
    log = pd.read_csv(TRACK_CSV)
    out = []
    today = pd.Timestamp(datetime.now().date())
    for date, grp in log.groupby("date"):
        days = (today - pd.Timestamp(date)).days
        if days < MIN_EVAL_DAYS:
            continue
        rets, beats = [], 0
        spy_then = float(grp["spy_price"].iloc[0])
        spy_ret = (spy_now / spy_then - 1) * 100 if spy_then > 0 else 0.0
        for _, r in grp.iterrows():
            now = prices_now.get(r["symbol"])
            if now is None or r["price"] <= 0:
                continue
            ret = (now / float(r["price"]) - 1) * 100
            rets.append(ret)
            if ret > spy_ret:
                beats += 1
        if not rets:
            continue
        mean_ret = sum(rets) / len(rets)
        out.append({
            "date": date, "days": days, "n": len(rets),
            "ret": mean_ret, "spy": spy_ret, "excess": mean_ret - spy_ret,
            "hit": beats / len(rets) * 100,
        })
    return sorted(out, key=lambda d: d["date"])


def diff_vs_previous(ranked: pd.DataFrame, top_n: int = 15) -> dict:
    """Entradas y salidas del top-N respecto al snapshot anterior."""
    current = list(ranked.head(top_n)["symbol"])
    if not TRACK_CSV.exists():
        return {"in": [], "out": [], "prev_date": None}
    log = pd.read_csv(TRACK_CSV)
    today = datetime.now().strftime("%Y-%m-%d")
    prev_dates = sorted(d for d in log["date"].unique() if d != today)
    if not prev_dates:
        return {"in": [], "out": [], "prev_date": None}
    prev_date = prev_dates[-1]
    prev = list(log[(log["date"] == prev_date) & (log["rank"] <= top_n)]["symbol"])
    return {
        "in":  [s for s in current if s not in prev],
        "out": [s for s in prev if s not in current],
        "prev_date": prev_date,
    }


# ── Interpretación ───────────────────────────────────────────────────────────

def build_interpretation(ranked, meta, diff, evals) -> str:
    """Interpretación rule-based (siempre disponible)."""
    p = []
    vix = (meta.get("vix_data") or {}).get("vix")
    top15 = ranked.head(15)
    sectors = top15["sector"].value_counts()
    lead = ", ".join(f"{s} ({n})" for s, n in sectors.head(3).items())
    eu = (top15["region"] == "Europa").sum()
    p.append(f"El top 15 está liderado por {lead}. {eu} de 15 posiciones son europeas."
             + (f" VIX en {vix:.1f}." if vix else ""))
    if diff["prev_date"]:
        if diff["in"] or diff["out"]:
            p.append(f"Rotación desde el informe del {diff['prev_date']}: "
                     f"entran {', '.join(diff['in']) or '—'}; "
                     f"salen {', '.join(diff['out']) or '—'}.")
        else:
            p.append(f"Sin cambios en el top 15 desde el {diff['prev_date']} — tendencia estable.")
    if evals:
        mature = [e for e in evals if e["days"] >= 20]
        pool = mature or evals
        avg_exc = sum(e["excess"] for e in pool) / len(pool)
        ok = sum(1 for e in pool if e["excess"] > 0)
        verdict = "las predicciones se están cumpliendo" if avg_exc > 0 else \
                  "las predicciones van por debajo del mercado — precaución"
        p.append(f"Seguimiento: {ok}/{len(pool)} snapshots superan al SPY "
                 f"(exceso medio {avg_exc:+.1f}pp) — {verdict}.")
    return " ".join(p)


def ai_commentary(ranked, interpretation, evals, timeout_s: int = 150) -> str:
    """Comentario de analista vía CLI de Claude Code (opcional, fallback silencioso)."""
    import shutil
    exe = shutil.which("claude")
    if not exe:
        for cand in [Path.home() / ".local" / "bin" / "claude.exe",
                     Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd"]:
            if cand.exists():
                exe = str(cand)
                break
    if not exe:
        return ""
    top = ranked.head(10)[["symbol", "region", "sector", "screener_score"]].to_string(index=False)
    track = "\n".join(f"{e['date']}: exceso {e['excess']:+.1f}pp en {e['days']}d"
                      for e in evals[-6:]) or "sin historial aún"
    prompt = (
        "Eres analista de inversiones. En español, en máximo 150 palabras y sin listas, "
        "interpreta este ranking cuantitativo a 6 meses (síntesis de momentum 6-12m, "
        "riesgo, fundamentales) y su historial de aciertos. Sé concreto y honesto, "
        "menciona riesgos. No repitas los datos literales, interprétalos.\n\n"
        f"Top 10 de hoy:\n{top}\n\nContexto: {interpretation}\n\n"
        f"Historial de exceso vs SPY:\n{track}"
    )
    try:
        res = subprocess.run([exe, "-p", prompt], capture_output=True, text=True,
                             timeout=timeout_s, encoding="utf-8", errors="replace")
        txt = (res.stdout or "").strip()
        return txt if res.returncode == 0 and 50 < len(txt) < 2500 else ""
    except Exception:
        return ""


# ── HTML / PDF ───────────────────────────────────────────────────────────────

def _table(df: pd.DataFrame, cols: list[tuple], n: int) -> str:
    head = "".join(f"<th>{h}</th>" for _, h, _ in cols)
    body = []
    for _, r in df.head(n).iterrows():
        tds = "".join(f"<td>{fmt(r)}</td>" for fmt, _, _ in cols)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(ranked, meta, interpretation, ai_text, evals, diff) -> str:
    e = html.escape
    today = datetime.now().strftime("%d/%m/%Y")
    cols = [
        (lambda r: int(r["rank"]), "#", None),
        (lambda r: f"<b>{e(r['symbol'])}</b>", "Símbolo", None),
        (lambda r: e(str(r["nombre"])[:30]), "Nombre", None),
        (lambda r: e(r["region"]), "Región", None),
        (lambda r: e(r["sector"]), "Sector", None),
        (lambda r: f"<b>{r['screener_score']:.1f}</b>", "Score", None),
        (lambda r: f"{r['c_momentum_6m']:.0f}" if pd.notna(r.get("c_momentum_6m")) else "—", "Mom6m", None),
        (lambda r: f"{r['c_fundamental']:.0f}" if pd.notna(r.get("c_fundamental")) else "—", "Fund", None),
        (lambda r: e(str(r["confidence"])), "Confianza", None),
    ]
    secs = [f"""
    <h1>Informe de Inversiones — horizonte 6 meses</h1>
    <p class="meta">{today} · {len(ranked)} símbolos analizados · screener 6m
    (momentum 6-12m 25% · riesgo 20% · técnico 15% · fundamentales 15% · ML 15% · confianza 10%)</p>
    <div class="box"><b>Interpretación:</b> {e(interpretation)}</div>"""]
    if ai_text:
        secs.append(f'<div class="box ai"><b>Comentario del analista (IA):</b> {e(ai_text)}</div>')

    secs.append("<h2>Top 15 global</h2>")
    secs.append(_table(ranked, cols, 15))
    secs.append("<h2>Top 10 Europa</h2>")
    secs.append(_table(ranked[ranked["region"] == "Europa"], cols, 10))
    secs.append("<h2>Top 10 USA</h2>")
    secs.append(_table(ranked[ranked["region"] == "USA"], cols, 10))

    secs.append("<h2>¿Se están cumpliendo las predicciones?</h2>")
    if evals:
        rows = "".join(
            f"<tr><td>{e(v['date'])}</td><td>{v['days']}d</td><td>{v['n']}</td>"
            f"<td>{v['ret']:+.1f}%</td><td>{v['spy']:+.1f}%</td>"
            f"<td class='{'pos' if v['excess'] > 0 else 'neg'}'>{v['excess']:+.1f}pp</td>"
            f"<td>{v['hit']:.0f}%</td></tr>"
            for v in evals)
        secs.append("<table><thead><tr><th>Snapshot</th><th>Días</th><th>N</th>"
                    "<th>Retorno top</th><th>SPY</th><th>Exceso</th><th>Baten SPY</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table>")
        secs.append("<p class='meta'>Cada fila es un top-20 recomendado en esa fecha, "
                    "valorado a precios de hoy. Exceso positivo = la selección batió al mercado.</p>")
    else:
        secs.append("<p>Primer informe: el seguimiento comienza hoy. Desde el próximo "
                    "informe verás aquí si las recomendaciones anteriores baten al SPY.</p>")

    if diff["prev_date"] and (diff["in"] or diff["out"]):
        secs.append(f"<h2>Rotación vs {e(diff['prev_date'])}</h2>"
                    f"<p><b>Entran al top 15:</b> {e(', '.join(diff['in']) or '—')}<br>"
                    f"<b>Salen:</b> {e(', '.join(diff['out']) or '—')}</p>")

    secs.append("<p class='meta'>Análisis cuantitativo automatizado — no es asesoría financiera. "
                "Verifica cada instrumento en N26. Diversifica y usa stop losses.</p>")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; color: #1a1a1a; font-size: 13px; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    h2 {{ font-size: 16px; margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    .meta {{ color: #666; font-size: 11px; }}
    .box {{ background: #f4f4f0; border-left: 3px solid #8a7ddd; padding: 10px 14px; margin: 12px 0; }}
    .box.ai {{ border-left-color: #1d9e75; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
    th {{ text-align: left; background: #f0f0ec; padding: 5px 8px; font-size: 11px; }}
    td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
    .pos {{ color: #0a7a3d; font-weight: bold; }} .neg {{ color: #b3261e; font-weight: bold; }}
    </style></head><body>{''.join(secs)}</body></html>"""


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    import shutil
    edge = next((p for p in EDGE_PATHS if Path(p).exists()), None) or shutil.which("msedge")
    if not edge:
        return False
    try:
        subprocess.run([edge, "--headless", "--disable-gpu",
                        f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
                        html_path.as_uri()],
                       capture_output=True, timeout=90)
        return pdf_path.exists()
    except Exception:
        return False


# ── Email opcional ───────────────────────────────────────────────────────────

def send_email(pdf_path: Path, subject: str, body: str) -> bool:
    cfg_path = BASE / "email_config.json"
    if not cfg_path.exists():
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(body)
    msg.add_attachment(pdf_path.read_bytes(), maintype="application",
                       subtype="pdf", filename=pdf_path.name)
    with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port", 587))) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ai", action="store_true", help="Sin comentario IA (claude CLI)")
    parser.add_argument("--no-email", action="store_true", help="No enviar email aunque haya config")
    parser.add_argument("--open", action="store_true", help="Abrir el PDF al terminar")
    args = parser.parse_args()

    from screener import build_screener, load_cached_analysis
    from database import TradingDatabase

    df, meta = load_cached_analysis()
    ranked = build_screener(df, horizon="6m")

    # Precios actuales para evaluar snapshots (de la propia corrida de análisis)
    prices_now = dict(zip(ranked["symbol"], ranked["price"].astype(float)))
    db = TradingDatabase()
    spy_df = db.get_precios("SPY").sort_values("timestamp")
    spy_now = float(spy_df["close"].iloc[-1])
    db.close()

    evals = evaluate_snapshots(prices_now, spy_now)
    diff = diff_vs_previous(ranked)
    record_snapshot(ranked, spy_now)

    interpretation = build_interpretation(ranked, meta, diff, evals)
    ai_text = "" if args.no_ai else ai_commentary(ranked, interpretation, evals)

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    html_path = REPORT_DIR / f"INFORME_6M_{stamp}.html"
    pdf_path = REPORT_DIR / f"INFORME_6M_{stamp}.pdf"
    html_path.write_text(build_html(ranked, meta, interpretation, ai_text, evals, diff),
                         encoding="utf-8")

    ok_pdf = html_to_pdf(html_path, pdf_path)
    print(f"Informe: {pdf_path if ok_pdf else html_path}")
    if ok_pdf:
        html_path.unlink(missing_ok=True)  # el PDF es el entregable; no acumular HTML

    sent = False
    if not args.no_email and ok_pdf:
        try:
            sent = send_email(pdf_path, f"Informe inversiones 6m — {stamp}",
                              interpretation + ("\n\n" + ai_text if ai_text else ""))
        except Exception as ex:
            print(f"AVISO: email no enviado ({ex})")
    print(f"Email: {'enviado' if sent else 'no configurado/omitido'}")

    if args.open:
        target = pdf_path if ok_pdf else html_path
        try:
            import os
            os.startfile(target)  # noqa
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
