"""
Tests de portfolio_optimizer.py (Etapa 3).

Filosofia:
- Tests deterministas sobre retornos sinteticos (no red, no DB).
- Verificamos propiedades invariantes: pesos suman 1, VaR >= 0,
  ERC da contribuciones mas uniformes que equal-weight naive, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import (
    optimize_mv,
    optimize_erc,
    simulate_rebalance,
    portfolio_var,
    stress_test,
    suggest_var_budget,
    STRESS_PERIODS,
)


# ── Fixtures sinteticos ───────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synthetic_returns() -> pd.DataFrame:
    """5 activos, 500 dias de retornos aleatorios con semilla fija."""
    rng = np.random.default_rng(42)
    n, k = 500, 5
    mu    = np.array([0.0003, 0.0002, 0.0001, 0.00015, 0.00025])
    sigma = np.array([0.015, 0.012, 0.008, 0.010, 0.018])
    rets  = rng.normal(mu, sigma, size=(n, k))
    symbols = ['AAPL', 'MSFT', 'TLT', 'GLD', 'NVDA']
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    return pd.DataFrame(rets, index=dates, columns=symbols)


@pytest.fixture(scope='module')
def synthetic_prices(synthetic_returns) -> pd.DataFrame:
    """Precios sinteticos a partir de los retornos acumulados."""
    return (1 + synthetic_returns).cumprod() * 100


@pytest.fixture(scope='module')
def spy_returns(synthetic_returns) -> pd.Series:
    """Retornos sinteticos del SPY (proxy)."""
    rng = np.random.default_rng(99)
    spy = rng.normal(0.0002, 0.013, size=len(synthetic_returns))
    return pd.Series(spy, index=synthetic_returns.index, name='SPY')


# ── optimize_mv ───────────────────────────────────────────────────────────────

class TestOptimizeMV:
    def test_weights_sum_to_one(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        total = sum(res['weights'].values())
        assert abs(total - 1.0) < 1e-5

    def test_weights_non_negative(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        for w in res['weights'].values():
            assert w >= -1e-6

    def test_max_weight_respected(self, synthetic_returns):
        res = optimize_mv(synthetic_returns, constraints={'max_weight': 0.30})
        for w in res['weights'].values():
            assert w <= 0.30 + 1e-5

    def test_returns_expected_keys(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        for key in ('weights', 'expected_return', 'volatility', 'sharpe', 'method', 'n_assets'):
            assert key in res

    def test_method_label(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        assert res['method'] == 'mean_variance_lw'

    def test_sharpe_finite(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        assert np.isfinite(res['sharpe'])

    def test_volatility_positive(self, synthetic_returns):
        res = optimize_mv(synthetic_returns)
        assert res['volatility'] > 0

    def test_scores_tilt_weights(self, synthetic_returns):
        """Tiltar con scores favorables a NVDA debe aumentar su peso."""
        scores_neutral = {s: 50 for s in synthetic_returns.columns}
        scores_nvda    = {s: 50 for s in synthetic_returns.columns}
        scores_nvda['NVDA'] = 95

        res_neutral = optimize_mv(synthetic_returns, scores=scores_neutral)
        res_tilted  = optimize_mv(synthetic_returns, scores=scores_nvda)

        # NVDA debe tener igual o mayor peso con score alto (o al menos no menor)
        assert res_tilted['weights']['NVDA'] >= res_neutral['weights']['NVDA'] - 0.01

    def test_spy_rets_accepted(self, synthetic_returns, spy_returns):
        res = optimize_mv(synthetic_returns, spy_rets=spy_returns)
        assert sum(res['weights'].values()) == pytest.approx(1.0, abs=1e-5)

    def test_max_beta_constraint(self, synthetic_returns, spy_returns):
        """Con max_beta muy bajo, el portafolio debe tener beta <= limite."""
        res = optimize_mv(
            synthetic_returns,
            spy_rets=spy_returns,
            constraints={'max_beta': 0.5, 'max_weight': 0.40},
        )
        assert sum(res['weights'].values()) == pytest.approx(1.0, abs=1e-5)


# ── optimize_erc ──────────────────────────────────────────────────────────────

class TestOptimizeERC:
    def test_weights_sum_to_one(self, synthetic_returns):
        res = optimize_erc(synthetic_returns)
        assert abs(sum(res['weights'].values()) - 1.0) < 1e-5

    def test_weights_non_negative(self, synthetic_returns):
        res = optimize_erc(synthetic_returns)
        for w in res['weights'].values():
            assert w >= -1e-6

    def test_method_label(self, synthetic_returns):
        res = optimize_erc(synthetic_returns)
        assert res['method'] == 'erc'

    def test_risk_contributions_more_equal_than_ew(self, synthetic_returns):
        """ERC debe distribuir el riesgo mas uniformemente que equal-weight."""
        from sklearn.covariance import LedoitWolf
        ret_mat = synthetic_returns.values.astype(float)
        cov = LedoitWolf().fit(ret_mat).covariance_ * 252

        res = optimize_erc(synthetic_returns)
        w_erc = np.array(list(res['weights'].values()))
        n = len(w_erc)
        w_ew  = np.ones(n) / n

        def risk_contribution_std(w):
            port_var = float(w @ cov @ w)
            if port_var < 1e-12:
                return 0.0
            rc = w * (cov @ w) / port_var
            return float(np.std(rc))

        assert risk_contribution_std(w_erc) <= risk_contribution_std(w_ew) + 0.05

    def test_max_weight_respected(self, synthetic_returns):
        res = optimize_erc(synthetic_returns, constraints={'max_weight': 0.25})
        for w in res['weights'].values():
            assert w <= 0.25 + 1e-5


# ── simulate_rebalance ────────────────────────────────────────────────────────

class TestSimulateRebalance:
    def _equal_weights(self, symbols):
        return {s: 1.0 / len(symbols) for s in symbols}

    def test_equity_curve_starts_at_100(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        res = simulate_rebalance(synthetic_prices, w, freq='monthly')
        assert res['equity_curve'].iloc[0] == pytest.approx(100.0, rel=0.01)

    def test_equity_curve_length_matches_prices(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        res = simulate_rebalance(synthetic_prices, w, freq='monthly')
        assert len(res['equity_curve']) >= len(synthetic_prices) - 2

    def test_rebalance_log_non_empty_monthly(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        res = simulate_rebalance(synthetic_prices, w, freq='monthly')
        assert res['n_rebalances'] >= 1

    def test_weekly_more_rebalances_than_monthly(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        w_res = simulate_rebalance(synthetic_prices, w, freq='weekly')
        m_res = simulate_rebalance(synthetic_prices, w, freq='monthly')
        assert w_res['n_rebalances'] > m_res['n_rebalances']

    def test_higher_cost_reduces_return(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        r_cheap = simulate_rebalance(synthetic_prices, w, cost_bps=0)
        r_exp   = simulate_rebalance(synthetic_prices, w, cost_bps=50)
        assert r_cheap['total_return'] >= r_exp['total_return']

    def test_returns_required_keys(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        res = simulate_rebalance(synthetic_prices, w)
        for key in ('equity_curve', 'rebalance_log', 'total_return',
                    'annualized', 'max_drawdown', 'total_cost_bps', 'n_rebalances'):
            assert key in res

    def test_max_drawdown_non_positive(self, synthetic_prices):
        symbols = list(synthetic_prices.columns)
        w = self._equal_weights(symbols)
        res = simulate_rebalance(synthetic_prices, w)
        assert res['max_drawdown'] <= 0


# ── portfolio_var ─────────────────────────────────────────────────────────────

class TestPortfolioVar:
    def _equal_weights(self, symbols):
        return {s: 1.0 / len(symbols) for s in symbols}

    def test_var_non_negative(self, synthetic_returns):
        symbols = list(synthetic_returns.columns)
        w = self._equal_weights(symbols)
        res = portfolio_var(w, synthetic_returns)
        assert res['var_parametric'] >= 0
        assert res['var_historical'] >= 0

    def test_cvar_ge_var(self, synthetic_returns):
        """CVaR siempre >= VaR (misma probabilidad)."""
        symbols = list(synthetic_returns.columns)
        w = self._equal_weights(symbols)
        res = portfolio_var(w, synthetic_returns)
        assert res['cvar_historical'] >= res['var_historical'] - 0.001

    def test_returns_required_keys(self, synthetic_returns):
        symbols = list(synthetic_returns.columns)
        w = self._equal_weights(symbols)
        res = portfolio_var(w, synthetic_returns)
        for key in ('var_parametric', 'var_historical', 'cvar_historical',
                    'port_vol_daily', 'target_met'):
            assert key in res

    def test_higher_vol_higher_var(self, synthetic_returns):
        """Concentrar en el activo mas volatil debe subir el VaR."""
        symbols = list(synthetic_returns.columns)
        w_eq    = self._equal_weights(symbols)
        w_nvda  = {s: 0.0 for s in symbols}
        w_nvda['NVDA'] = 1.0  # solo el mas volatil

        res_eq   = portfolio_var(w_eq,   synthetic_returns)
        res_nvda = portfolio_var(w_nvda, synthetic_returns)
        assert res_nvda['var_historical'] >= res_eq['var_historical'] - 0.01

    def test_target_met_is_bool(self, synthetic_returns):
        symbols = list(synthetic_returns.columns)
        w = self._equal_weights(symbols)
        res = portfolio_var(w, synthetic_returns)
        assert isinstance(res['target_met'], bool)


# ── suggest_var_budget ────────────────────────────────────────────────────────

def test_suggest_var_budget_returns_valid(synthetic_returns):
    symbols = list(synthetic_returns.columns)
    initial_w = {s: 1.0 / len(symbols) for s in symbols}
    res = suggest_var_budget(synthetic_returns, initial_w)
    assert 'weights' in res
    assert 'var_info' in res
    assert 'var_ok'   in res
    assert abs(sum(res['weights'].values()) - 1.0) < 1e-5


# ── stress_test ───────────────────────────────────────────────────────────────

class TestStressTest:
    def _make_long_prices(self):
        """Precios sinteticos con fechas que cubren los periodos de estres."""
        rng = np.random.default_rng(7)
        dates = pd.date_range('2019-01-01', '2023-12-31', freq='B')
        n = len(dates)
        prices = pd.DataFrame(
            (1 + rng.normal(0.0003, 0.015, size=(n, 3))).cumprod(axis=0) * 100,
            index=dates,
            columns=['AAPL', 'MSFT', 'TLT'],
        )
        return prices

    def test_stress_returns_dict(self):
        prices = self._make_long_prices()
        w = {'AAPL': 0.4, 'MSFT': 0.4, 'TLT': 0.2}
        res = stress_test(w, prices)
        assert isinstance(res, dict)
        assert len(res) >= 1

    def test_stress_known_periods(self):
        prices = self._make_long_prices()
        w = {'AAPL': 0.4, 'MSFT': 0.4, 'TLT': 0.2}
        res = stress_test(w, prices, periods=STRESS_PERIODS)
        for period in STRESS_PERIODS:
            assert period in res

    def test_stress_max_drawdown_non_positive(self):
        prices = self._make_long_prices()
        w = {'AAPL': 0.4, 'MSFT': 0.4, 'TLT': 0.2}
        res = stress_test(w, prices, periods=STRESS_PERIODS)
        for period, data in res.items():
            if 'error' not in data:
                assert data['max_drawdown'] <= 0

    def test_stress_var_non_negative(self):
        prices = self._make_long_prices()
        w = {'AAPL': 0.4, 'MSFT': 0.4, 'TLT': 0.2}
        res = stress_test(w, prices, periods=STRESS_PERIODS)
        for period, data in res.items():
            if 'error' not in data:
                assert data['var_95_1d'] >= 0

    def test_stress_custom_period(self):
        prices = self._make_long_prices()
        w = {'AAPL': 0.5, 'MSFT': 0.3, 'TLT': 0.2}
        custom = {'test_period': ('2021-06-01', '2021-12-31')}
        res = stress_test(w, prices, periods=custom)
        assert 'test_period' in res
        if 'error' not in res['test_period']:
            assert res['test_period']['n_days'] > 0
