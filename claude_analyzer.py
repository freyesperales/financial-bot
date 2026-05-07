"""
claude_analyzer.py — Integración con Claude CLI (claude -p)
=============================================================
Enriquece el análisis técnico con perspectiva cualitativa usando
el Claude Code CLI que ya tienes instalado.

NO requiere API key ni SDK de Anthropic.
Usa la ruta completa al ejecutable claude para evitar problemas de PATH.
"""

import json
import os
import shutil
import subprocess
import textwrap
import threading


# ──────────────────────────────────────────────────────────────────────────────
# Localización del ejecutable claude
# ──────────────────────────────────────────────────────────────────────────────

def _find_claude_exe() -> str | None:
    """
    Devuelve la ruta completa a claude.exe / claude, o None si no se encuentra.

    Estrategia (en orden):
      1. shutil.which('claude')  — usa el PATH actual del proceso
      2. shutil.which con PATH extendido que incluye ~/.local/bin y rutas npm
      3. Rutas conocidas en Windows (instalaciones típicas de Claude Code)
    """
    # 1) Búsqueda estándar con PATH actual
    found = shutil.which('claude')
    if found:
        return found

    # 2) Extender el PATH con directorios típicos donde Claude Code se instala
    home = os.path.expanduser('~')
    extra_paths = [
        os.path.join(home, '.local', 'bin'),                          # Linux/Mac y Git Bash Windows
        os.path.join(home, 'AppData', 'Local', 'Programs', 'Claude'), # Instalador oficial
        os.path.join(home, 'AppData', 'Roaming', 'npm'),              # npm global (Windows)
        os.path.join(home, 'AppData', 'Local', 'npm'),
        r'C:\Program Files\Anthropic\Claude',
        r'C:\Program Files (x86)\Anthropic\Claude',
    ]
    sep = os.pathsep
    extended_path = sep.join(extra_paths) + sep + os.environ.get('PATH', '')
    found = shutil.which('claude', path=extended_path)
    if found:
        return found

    # 3) Verificar rutas absolutas directamente (también variantes de extensión)
    candidates = [
        os.path.join(home, '.local', 'bin', 'claude.exe'),
        os.path.join(home, '.local', 'bin', 'claude.EXE'),
        os.path.join(home, '.local', 'bin', 'claude.cmd'),
        os.path.join(home, '.local', 'bin', 'claude'),
        os.path.join(home, 'AppData', 'Roaming', 'npm', 'claude.cmd'),
        os.path.join(home, 'AppData', 'Roaming', 'npm', 'claude.exe'),
        os.path.join(home, 'AppData', 'Roaming', 'npm', 'claude'),
        os.path.join(home, 'AppData', 'Local', 'Programs', 'Claude', 'claude.exe'),
        r'C:\Program Files\Anthropic\Claude\claude.exe',
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # 4) Último recurso: glob en .local/bin y AppData/Roaming/npm
    import glob
    for pattern in [
        os.path.join(home, '.local', 'bin', 'claude*'),
        os.path.join(home, 'AppData', 'Roaming', 'npm', 'claude*'),
    ]:
        matches = [p for p in glob.glob(pattern)
                   if os.path.isfile(p) and not p.endswith('.ps1')]
        if matches:
            return matches[0]

    return None


# Ruta al ejecutable — se resuelve una vez al importar
CLAUDE_EXE: str | None = _find_claude_exe()
CLAUDE_AVAILABLE: bool = CLAUDE_EXE is not None


# ──────────────────────────────────────────────────────────────────────────────
# Entorno limpio para subproceso
# ──────────────────────────────────────────────────────────────────────────────

def _clean_env() -> dict:
    """
    Entorno del proceso actual SIN variables que bloquean llamadas anidadas
    de Claude Code, pero CON rutas extra para que el ejecutable se encuentre.
    """
    env = os.environ.copy()
    # Eliminar TODAS las variables que hacen que Claude detecte sesión anidada
    for key in list(env.keys()):
        if key.upper().startswith('CLAUDE'):
            env.pop(key, None)

    # Añadir ~/.local/bin al PATH de Windows para que subprocesos también lo encuentren
    home = os.path.expanduser('~')
    local_bin = os.path.join(home, '.local', 'bin')
    npm_bin   = os.path.join(home, 'AppData', 'Roaming', 'npm')
    current_path = env.get('PATH', '')
    if local_bin not in current_path:
        env['PATH'] = local_bin + os.pathsep + current_path
    if npm_bin not in current_path:
        env['PATH'] = npm_bin + os.pathsep + env['PATH']

    return env


# ──────────────────────────────────────────────────────────────────────────────
# Prompt de sistema
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
Eres un analista financiero senior especializado en inversión a mediano plazo
(3-12 meses) en mercados de EE.UU. (NYSE/NASDAQ).

Conoces en profundidad la dinámica sectorial, macro, competencia y riesgos
regulatorios de las principales empresas cotizadas.

El usuario quiere superar al S&P 500 (>15%/año) seleccionando 5-15 posiciones
con fundamentos sólidos y momentum técnico confirmado.

REGLAS ESTRICTAS:
- Responde SIEMPRE en español.
- Devuelve SOLAMENTE un array JSON válido, sin markdown, sin texto antes o después.
- No pongas ```json ni ```. Solo el array JSON puro.
- Si no tienes información suficiente sobre una empresa, indícalo en "thesis".
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt de usuario
# ──────────────────────────────────────────────────────────────────────────────

_USER_TEMPLATE = textwrap.dedent("""
El bot de inversión generó el siguiente ranking de Top Candidatos de compra
para mediano plazo. Para cada acción necesito evaluación cualitativa.

DATOS TÉCNICOS/FUNDAMENTALES DEL BOT:
{stocks_json}

Para CADA símbolo devuelve un objeto con esta estructura exacta:
{{
  "symbol": "TICKER",
  "ia_rating": "COMPRAR",
  "ia_confidence": 4,
  "thesis": "Tesis de 2-3 oraciones sobre por qué puede superar al mercado",
  "risks": ["riesgo clave 1", "riesgo clave 2"],
  "catalysts": ["catalizador 1", "catalizador 2"],
  "macro_context": "1 oración sobre entorno macro/sectorial relevante",
  "tech_coherence": "CONFIRMA",
  "tech_coherence_reason": "El score técnico refleja X porque Y"
}}

Valores válidos:
  ia_rating: "COMPRAR" | "VIGILAR" | "EVITAR"
  ia_confidence: 1 (muy baja) a 5 (muy alta)
  tech_coherence: "CONFIRMA" | "CONTRADICE" | "NEUTRO"

Devuelve un array JSON: [ {{...}}, {{...}}, ... ]
""").strip()


def _build_stock_summary(row: dict) -> dict:
    """Extrae los campos más relevantes de una fila del DataFrame de resultados."""

    def fmt(v, dec=1, suf=''):
        if v is None:
            return 'N/D'
        try:
            return f"{float(v):.{dec}f}{suf}"
        except (TypeError, ValueError):
            return str(v)

    return {
        'symbol':         row.get('symbol', ''),
        'sector':         row.get('sector', ''),
        'score_bot':      fmt(row.get('_buy_score') or row.get('total_score')),
        'confianza':      row.get('confidence', 'N/D'),
        'vol_signal':     row.get('vol_signal', 'N/D'),
        'tf_alignment':   row.get('tf_alignment', 'N/D'),
        'rs_vs_spy':      row.get('rs_signal', 'N/D'),
        'rs_252d_pct':    fmt(row.get('rs_252d'), suf='%'),
        'ret_anual_hist': fmt(row.get('ann_return'), suf='%'),
        'sharpe':         fmt(row.get('sharpe'), dec=2),
        'max_drawdown':   fmt(row.get('max_dd'), suf='%'),
        'fund_signal':    row.get('fund_signal', 'N/D'),
        'pe_ratio':       fmt(row.get('fund_pe'), suf='x'),
        'peg_ratio':      fmt(row.get('fund_peg'), dec=2),
        'crecimiento':    fmt(row.get('fund_growth'), suf='%/año'),
        'precio_usd':     fmt(row.get('price'), dec=2, suf='$'),
        'cambio_252d':    fmt(row.get('change_252d'), suf='%'),
        'backtest_wr':    fmt(row.get('bt_win_rate_vol'), dec=0, suf='%'),
        'prediccion':     row.get('pred_signal', 'N/D'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Llamada al CLI
# ──────────────────────────────────────────────────────────────────────────────

def call_claude_cli(
    prompt: str,
    system: str = _SYSTEM_PROMPT,
    model: str = 'sonnet',
    timeout: int = 120,
) -> str:
    """
    Llama a Claude CLI usando la ruta completa al ejecutable.
    Devuelve la respuesta como string o lanza excepción.
    """
    if not CLAUDE_EXE:
        raise RuntimeError(
            "No se encontró el ejecutable de Claude.\n"
            "Verifica que Claude Code esté instalado."
        )

    cmd = [
        CLAUDE_EXE,
        '-p', prompt,
        '--system-prompt', system,
        '--model', model,
        '--output-format', 'text',
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
        env=_clean_env(),
    )

    if result.returncode != 0:
        err = result.stderr.strip()[:400] if result.stderr else '(sin detalle)'
        raise RuntimeError(f"Claude CLI retornó código {result.returncode}:\n{err}")

    return result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Parseo de respuesta JSON
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list[dict]:
    """
    Parsea el array JSON de la respuesta de Claude.
    Tolerante a markdown fences y texto accidental antes del '['.
    """
    text = raw.strip()

    # Eliminar fences markdown si Claude los incluyó
    if '```' in text:
        lines = text.split('\n')
        text = '\n'.join(
            line for line in lines
            if not line.strip().startswith('```')
        ).strip()

    # Intentar primero como array JSON
    start = text.find('[')
    end   = text.rfind(']') + 1
    if start != -1 and end > 0:
        return json.loads(text[start:end])

    # Si no hay array, intentar como objeto único y envolverlo
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start != -1 and end > 0:
        obj = json.loads(text[start:end])
        return [obj] if isinstance(obj, dict) else list(obj.values())

    raise ValueError(
        f"No se encontró JSON en la respuesta.\n"
        f"Primeros 300 caracteres:\n{raw[:300]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeAnalyzer:
    """
    Interfaz principal para la integración Claude CLI en la GUI.

    Uso:
        analyzer = ClaudeAnalyzer()
        if analyzer.available:
            results = analyzer.analyze_top_stocks(df_results, top_n=10)
    """

    def __init__(self):
        self.available: bool = CLAUDE_AVAILABLE
        self.exe_path: str   = CLAUDE_EXE or ''
        self._cache: dict    = {}
        self._lock           = threading.Lock()

    def mode_label(self) -> str:
        if self.available:
            return f'Claude CLI  ({self.exe_path})'
        return 'No disponible'

    def analyze_top_stocks(
        self,
        df_results,
        top_n: int = 10,
        score_col: str = '_buy_score',
        only_buy: bool = True,
        model: str = 'sonnet',
    ) -> list[dict]:
        """
        Analiza los top_n stocks del DataFrame con Claude.
        Devuelve lista de dicts con: ia_rating, ia_confidence, thesis,
        risks, catalysts, macro_context, tech_coherence, tech_coherence_reason.
        """
        if not self.available:
            raise RuntimeError("Claude CLI no disponible.")

        df = df_results.copy()
        if only_buy and 'vol_signal' in df.columns:
            df_filt = df[df['vol_signal'] == 'COMPRAR']
            if df_filt.empty:
                df_filt = df
        else:
            df_filt = df

        if score_col not in df_filt.columns:
            score_col = 'total_score'

        top = df_filt.sort_values(score_col, ascending=False).head(top_n)

        # Caché por conjunto de símbolos
        symbols = frozenset(top['symbol'].tolist())
        with self._lock:
            if symbols in self._cache:
                return self._cache[symbols]

        stocks_data = [_build_stock_summary(row) for _, row in top.iterrows()]
        stocks_json = json.dumps(stocks_data, ensure_ascii=False, indent=2)
        user_prompt = _USER_TEMPLATE.format(stocks_json=stocks_json)

        raw     = call_claude_cli(user_prompt, model=model)
        results = _parse_json_response(raw)

        with self._lock:
            self._cache[symbols] = results

        return results

    def clear_cache(self):
        with self._lock:
            self._cache.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Diagnóstico desde línea de comandos
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Ejecutable encontrado : {CLAUDE_EXE}")
    print(f"Claude disponible     : {CLAUDE_AVAILABLE}")
    if CLAUDE_AVAILABLE:
        print("\nProbando llamada de prueba...")
        try:
            resp = call_claude_cli(
                'Responde solo con este JSON, sin nada más: [{"test": "ok"}]',
                system='Responde solo con JSON puro, sin markdown.',
                timeout=30,
            )
            data = _parse_json_response(resp)
            print(f"✓ Respuesta recibida y parseada: {data}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("\nClaude CLI no encontrado.")
        print("Rutas buscadas:")
        home = os.path.expanduser('~')
        for p in [
            os.path.join(home, '.local', 'bin', 'claude.exe'),
            os.path.join(home, 'AppData', 'Roaming', 'npm', 'claude.cmd'),
        ]:
            print(f"  {'✓' if os.path.isfile(p) else '✗'}  {p}")
