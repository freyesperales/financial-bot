"""Tests del screener consolidado (screener.py)."""
import numpy as np
import pandas as pd
import pytest

from screener import build_screener, compute_screener_score, screener_markdown


def _fake_results():
    return pd.DataFrame([
        # symbol, técnica alta + todo bueno → debe rankear primero
        {"symbol": "AAPL", "sector": "Technology", "total_score": 80.0,
         "confidence_score": 6.0, "ml_prob": 0.7, "fund_score": 80.0,
         "rs_score": 70.0, "sharpe": 2.0, "max_dd": -10.0,
         "confidence": "ALTA", "data_freshness": "OK", "price": 100.0,
         "dip_from_high_pct": -2.0},
        # técnica media
        {"symbol": "SAN", "sector": "Financials", "total_score": 60.0,
         "confidence_score": 0.0, "ml_prob": 0.5, "fund_score": 55.0,
         "rs_score": 45.0, "sharpe": 0.8, "max_dd": -20.0,
         "confidence": "MEDIA", "data_freshness": "OK", "price": 5.0,
         "dip_from_high_pct": -10.0},
        # ETF sin fundamentales ni ML → pesos renormalizados, sin NaN en score
        {"symbol": "VGK", "sector": "ETF - Europa", "total_score": 65.0,
         "confidence_score": 2.0, "ml_prob": np.nan, "fund_score": np.nan,
         "rs_score": 50.0, "sharpe": 1.0, "max_dd": -15.0,
         "confidence": "MEDIA", "data_freshness": "OK", "price": 70.0,
         "dip_from_high_pct": -1.0},
        # técnica baja → último
        {"symbol": "F", "sector": "Consumer Discretionary", "total_score": 35.0,
         "confidence_score": -5.0, "ml_prob": 0.3, "fund_score": 30.0,
         "rs_score": 10.0, "sharpe": -0.5, "max_dd": -45.0,
         "confidence": "MUY BAJA", "data_freshness": "desactualizado",
         "price": 10.0, "dip_from_high_pct": -30.0},
    ])


def test_ranking_order_and_range():
    ranked = build_screener(_fake_results())
    assert ranked.iloc[0]["symbol"] == "AAPL"
    assert ranked.iloc[-1]["symbol"] == "F"
    assert ranked["screener_score"].between(0, 100).all()
    assert not ranked["screener_score"].isna().any()


def test_region_and_type_annotation():
    ranked = build_screener(_fake_results())
    row = ranked.set_index("symbol")
    assert row.loc["SAN", "region"] == "Europa"
    assert row.loc["AAPL", "region"] == "USA"
    assert row.loc["VGK", "tipo"] == "ETF"
    assert row.loc["SAN", "tipo"] == "Acción"


def test_missing_components_renormalized():
    """Un ETF sin ML ni fundamentales no debe quedar con score NaN ni castigado a 0."""
    ranked = build_screener(_fake_results()).set_index("symbol")
    assert pd.notna(ranked.loc["VGK", "screener_score"])
    # VGK (técnica 65) debe superar a F (técnica 35) pese a tener menos componentes
    assert ranked.loc["VGK", "screener_score"] > ranked.loc["F", "screener_score"]


def test_exclude_stale():
    ranked = build_screener(_fake_results(), exclude_stale=True)
    assert "F" not in ranked["symbol"].values
    assert len(ranked) == 3


def test_markdown_contains_sections():
    ranked = build_screener(_fake_results())
    md = screener_markdown(ranked, top_n=10)
    for expected in ["Top Global", "Top USA", "Top Europa", "Top ETFs",
                     "no es asesoría financiera"]:
        assert expected in md
    assert "AAPL" in md and "SAN" in md


def test_score_without_optional_columns():
    """Solo con total_score el screener sigue funcionando."""
    df = pd.DataFrame([
        {"symbol": "MSFT", "sector": "Technology", "total_score": 70.0},
        {"symbol": "KO", "sector": "Consumer Staples", "total_score": 50.0},
    ])
    scores = compute_screener_score(df)
    assert scores.iloc[0] == 70.0
    assert scores.iloc[1] == 50.0


def test_horizon_6m_prioritizes_medium_term_momentum():
    """En perfil 6m, un valor con momentum 60/252d fuerte debe superar a uno
    que solo brilla en el técnico de 20 días."""
    df = pd.DataFrame([
        # Momentum medio plazo fuerte, técnico corto mediocre
        {"symbol": "MOM", "sector": "Technology", "total_score": 55.0,
         "change_60d": 25.0, "change_252d": 45.0, "rs_60d": 10.0, "rs_252d": 20.0,
         "confidence_score": 2.0, "fund_score": 70.0, "sharpe": 1.5, "max_dd": -12.0},
        # Técnico corto brillante, momentum medio plazo plano
        {"symbol": "FLASH", "sector": "Technology", "total_score": 85.0,
         "change_60d": 0.0, "change_252d": -5.0, "rs_60d": -5.0, "rs_252d": -10.0,
         "confidence_score": 2.0, "fund_score": 70.0, "sharpe": 1.5, "max_dd": -12.0},
    ])
    s6 = compute_screener_score(df, horizon="6m")
    s20 = compute_screener_score(df, horizon="20d")
    assert s6.iloc[0] > s6.iloc[1]      # 6m: gana el momentum medio
    assert s20.iloc[1] > s20.iloc[0]    # 20d: gana el técnico corto


def test_horizon_recorded_in_attrs():
    ranked = build_screener(_fake_results(), horizon="6m")
    assert ranked.attrs["horizon"] == "6m"
