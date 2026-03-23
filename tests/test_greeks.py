"""
Unit tests for src/greeks.py – Black-Scholes pricing, Greeks, and IV utilities.
"""

import math
import pytest
import numpy as np

from src.greeks import (
    OptionContract,
    Greeks,
    black_scholes_price,
    calculate_greeks,
    implied_volatility,
    historical_volatility,
    iv_rank,
    iv_percentile,
    call_payoff_at_expiry,
    put_payoff_at_expiry,
    pop_long_call,
    pop_long_put,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def atm_call() -> OptionContract:
    """At-the-money call option, classic example from textbooks."""
    return OptionContract(
        underlying_price=100.0,
        strike=100.0,
        time_to_expiry=1.0,       # 1 year
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )


@pytest.fixture
def atm_put() -> OptionContract:
    return OptionContract(
        underlying_price=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="put",
    )


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

class TestBlackScholesPrice:
    def test_atm_call_approximate(self, atm_call):
        """Standard ATM call: S=100, K=100, T=1, r=5%, σ=20% ≈ $10.45."""
        price = black_scholes_price(atm_call)
        assert 9.0 < price < 12.0, f"ATM call price {price:.4f} outside expected range"

    def test_atm_put_approximate(self, atm_put):
        """Standard ATM put ≈ $5.57 (put-call parity difference from call)."""
        price = black_scholes_price(atm_put)
        assert 4.0 < price < 7.0, f"ATM put price {price:.4f} outside expected range"

    def test_put_call_parity(self, atm_call, atm_put):
        """C - P = S - K·e^(-rT)  (put-call parity)."""
        call_price = black_scholes_price(atm_call)
        put_price = black_scholes_price(atm_put)
        S = atm_call.underlying_price
        K = atm_call.strike
        r = atm_call.risk_free_rate
        T = atm_call.time_to_expiry
        parity_rhs = S - K * math.exp(-r * T)
        assert abs((call_price - put_price) - parity_rhs) < 0.01

    def test_deep_itm_call_approaches_intrinsic(self):
        c = OptionContract(100.0, 50.0, 0.01, 0.05, 0.20, "call")
        price = black_scholes_price(c)
        assert price >= 50.0 - 0.5  # close to intrinsic

    def test_deep_otm_call_near_zero(self):
        c = OptionContract(100.0, 200.0, 0.01, 0.05, 0.20, "call")
        price = black_scholes_price(c)
        assert price < 0.01

    def test_expired_option_returns_intrinsic(self):
        c_call = OptionContract(110.0, 100.0, 0.0, 0.05, 0.20, "call")
        assert black_scholes_price(c_call) == pytest.approx(10.0)
        c_put_otm = OptionContract(110.0, 100.0, 0.0, 0.05, 0.20, "put")
        assert black_scholes_price(c_put_otm) == pytest.approx(0.0)

    def test_price_always_non_negative(self):
        for spot in [50, 100, 150, 200]:
            for strike in [50, 100, 150, 200]:
                for opt_type in ("call", "put"):
                    c = OptionContract(spot, strike, 0.5, 0.05, 0.25, opt_type)
                    assert black_scholes_price(c) >= 0.0


# ── Greeks ────────────────────────────────────────────────────────────────────

class TestGreeks:
    def test_call_delta_between_0_and_1(self, atm_call):
        g = calculate_greeks(atm_call)
        assert 0.0 <= g.delta <= 1.0

    def test_put_delta_between_minus1_and_0(self, atm_put):
        g = calculate_greeks(atm_put)
        assert -1.0 <= g.delta <= 0.0

    def test_atm_call_delta_near_half(self, atm_call):
        g = calculate_greeks(atm_call)
        assert 0.45 < g.delta < 0.65

    def test_gamma_positive(self, atm_call):
        g = calculate_greeks(atm_call)
        assert g.gamma > 0

    def test_theta_negative_for_long_option(self, atm_call, atm_put):
        g_call = calculate_greeks(atm_call)
        g_put = calculate_greeks(atm_put)
        assert g_call.theta < 0
        assert g_put.theta < 0

    def test_vega_positive(self, atm_call):
        g = calculate_greeks(atm_call)
        assert g.vega > 0

    def test_call_rho_positive(self, atm_call):
        g = calculate_greeks(atm_call)
        assert g.rho > 0

    def test_put_rho_negative(self, atm_put):
        g = calculate_greeks(atm_put)
        assert g.rho < 0

    def test_deep_itm_call_delta_near_1(self):
        c = OptionContract(200.0, 100.0, 1.0, 0.05, 0.20, "call")
        g = calculate_greeks(c)
        assert g.delta > 0.90

    def test_deep_otm_call_delta_near_0(self):
        c = OptionContract(50.0, 200.0, 0.1, 0.05, 0.20, "call")
        g = calculate_greeks(c)
        assert g.delta < 0.05

    def test_theoretical_price_matches_bs(self, atm_call):
        g = calculate_greeks(atm_call)
        bs_price = black_scholes_price(atm_call)
        assert abs(g.theoretical_price - bs_price) < 1e-8


# ── Implied Volatility ────────────────────────────────────────────────────────

class TestImpliedVolatility:
    def test_roundtrip(self, atm_call):
        """IV solver should recover the original volatility from a BS price."""
        original_vol = 0.25
        atm_call.volatility = original_vol
        market_price = black_scholes_price(atm_call)
        iv = implied_volatility(market_price, atm_call)
        assert iv is not None
        assert abs(iv - original_vol) < 0.001

    def test_different_volatilities_roundtrip(self):
        for sigma in [0.10, 0.20, 0.35, 0.50, 0.80]:
            c = OptionContract(100.0, 100.0, 0.5, 0.05, sigma, "call")
            price = black_scholes_price(c)
            iv = implied_volatility(price, c)
            assert iv is not None, f"IV solver returned None for σ={sigma}"
            assert abs(iv - sigma) < 0.002, f"IV={iv:.4f} vs σ={sigma}"

    def test_put_roundtrip(self, atm_put):
        original_vol = 0.30
        atm_put.volatility = original_vol
        market_price = black_scholes_price(atm_put)
        iv = implied_volatility(market_price, atm_put)
        assert iv is not None
        assert abs(iv - original_vol) < 0.001

    def test_returns_none_for_below_intrinsic(self):
        c = OptionContract(100.0, 90.0, 1.0, 0.05, 0.20, "call")
        # Price below intrinsic – no IV exists
        iv = implied_volatility(-1.0, c)
        assert iv is None


# ── Historical Volatility ─────────────────────────────────────────────────────

class TestHistoricalVolatility:
    def test_constant_prices_zero_vol(self):
        prices = [100.0] * 50
        hv = historical_volatility(prices)
        assert hv == pytest.approx(0.0)

    def test_reasonable_range(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 252)))
        hv = historical_volatility(prices.tolist())
        # Annualised vol from daily σ≈0.01 → ~0.16; allow wide range
        assert 0.05 < hv < 0.50

    def test_short_series_returns_float(self):
        hv = historical_volatility([100.0, 101.0])
        assert isinstance(hv, float)

    def test_single_price_returns_zero(self):
        hv = historical_volatility([100.0])
        assert hv == 0.0


# ── IV Rank / Percentile ──────────────────────────────────────────────────────

class TestIVRankPercentile:
    def test_iv_rank_at_high(self):
        history = [0.10, 0.15, 0.20, 0.25, 0.30]
        assert iv_rank(0.30, history) == pytest.approx(1.0)

    def test_iv_rank_at_low(self):
        history = [0.10, 0.15, 0.20, 0.25, 0.30]
        assert iv_rank(0.10, history) == pytest.approx(0.0)

    def test_iv_rank_midpoint(self):
        history = [0.10, 0.30]
        assert iv_rank(0.20, history) == pytest.approx(0.5)

    def test_iv_percentile_above_all(self):
        history = [0.10, 0.15, 0.20, 0.25]
        assert iv_percentile(0.30, history) == pytest.approx(1.0)

    def test_iv_percentile_below_all(self):
        history = [0.15, 0.20, 0.25, 0.30]
        assert iv_percentile(0.10, history) == pytest.approx(0.0)


# ── Payoff diagrams ───────────────────────────────────────────────────────────

class TestPayoffs:
    def test_call_payoff_above_strike(self):
        # At expiry: spot 120, strike 100, premium 5 → P&L = 120-100-5 = 15
        pnl = call_payoff_at_expiry(100.0, 5.0, [120.0])
        assert pnl[0] == pytest.approx(15.0)

    def test_call_payoff_below_strike(self):
        pnl = call_payoff_at_expiry(100.0, 5.0, [80.0])
        assert pnl[0] == pytest.approx(-5.0)

    def test_put_payoff_below_strike(self):
        pnl = put_payoff_at_expiry(100.0, 5.0, [80.0])
        assert pnl[0] == pytest.approx(15.0)

    def test_put_payoff_above_strike(self):
        pnl = put_payoff_at_expiry(100.0, 5.0, [120.0])
        assert pnl[0] == pytest.approx(-5.0)

    def test_call_max_loss_is_premium(self):
        prices = np.linspace(0, 50, 100)  # all below strike
        pnl = call_payoff_at_expiry(100.0, 7.0, prices)
        assert np.all(pnl == pytest.approx(-7.0))


# ── PoP helpers ───────────────────────────────────────────────────────────────

class TestPoP:
    def test_pop_long_call_atm(self):
        pop = pop_long_call(0.50)
        assert pop == pytest.approx(0.50)

    def test_pop_long_put_atm(self):
        pop = pop_long_put(-0.50)
        assert pop == pytest.approx(0.50)

    def test_pop_long_call_deep_otm(self):
        pop = pop_long_call(0.05)   # delta=0.05 → PoP ≈ 0.95 (rarely right)
        assert pop == pytest.approx(0.95)

    def test_pop_bounded(self):
        for delta in [-1.5, -1.0, 0.0, 1.0, 1.5]:
            assert 0.0 <= pop_long_call(delta) <= 1.0
            assert 0.0 <= pop_long_put(delta) <= 1.0
