"""
Unit tests for src/scanner.py – option chain scanning utilities.

These tests mock yfinance to avoid live network calls, testing the enrichment
logic (Greeks, IV, UOA detection, OI walls, ranking) in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.scanner import (
    _mid_price,
    _spread_pct,
    _trading_days_to_expiry,
    _time_to_expiry_years,
    fetch_option_chain,
    rank_candidates,
    ScanResult,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

class TestMidPrice:
    def test_normal_bid_ask(self):
        assert _mid_price(1.0, 2.0) == pytest.approx(1.5)

    def test_zero_bid_returns_ask(self):
        assert _mid_price(0.0, 3.0) == pytest.approx(3.0)

    def test_zero_ask_returns_bid(self):
        assert _mid_price(2.0, 0.0) == pytest.approx(2.0)

    def test_both_zero_returns_zero(self):
        assert _mid_price(0.0, 0.0) == pytest.approx(0.0)


class TestSpreadPct:
    def test_tight_spread(self):
        # bid=9, ask=11, mid=10 → spread_pct = 2/10 = 0.20
        sp = _spread_pct(9.0, 11.0)
        assert sp == pytest.approx(0.20)

    def test_wide_spread(self):
        sp = _spread_pct(1.0, 5.0)
        assert sp == pytest.approx(1.33, rel=0.01)

    def test_zero_prices_returns_one(self):
        assert _spread_pct(0.0, 0.0) == pytest.approx(1.0)


class TestTimeConversions:
    def test_dte_zero_or_positive(self):
        # Far future expiry
        dte = _trading_days_to_expiry("2030-01-01")
        assert dte > 0

    def test_past_expiry_returns_zero(self):
        dte = _trading_days_to_expiry("2020-01-01")
        assert dte == 0

    def test_time_to_expiry_years_positive(self):
        t = _time_to_expiry_years(30)
        assert t == pytest.approx(30 / 365.0)

    def test_time_to_expiry_zero_dte_returns_small_positive(self):
        t = _time_to_expiry_years(0)
        assert t > 0


# ── fetch_option_chain (mocked) ───────────────────────────────────────────────

def _make_mock_ticker(spot: float = 500.0, expiry: str = "2030-01-01") -> MagicMock:
    """Build a minimal yfinance Ticker mock."""
    mock = MagicMock()

    # History
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=252)
    prices = 500 * np.exp(np.cumsum(rng.normal(0, 0.01, 252)))
    hist_df = pd.DataFrame({"Close": prices}, index=dates)
    mock.history.return_value = hist_df

    # Options expirations
    mock.options = [expiry]

    # Calls
    calls_df = pd.DataFrame({
        "strike": [480.0, 490.0, 500.0, 510.0, 520.0],
        "bid":    [22.0,  15.0,  10.0,  6.0,   3.0],
        "ask":    [24.0,  17.0,  12.0,  8.0,   5.0],
        "lastPrice": [23.0, 16.0, 11.0, 7.0,  4.0],
        "volume":    [500,  800,  2000, 1200,  300],
        "openInterest": [100, 200, 400, 150,  50],
    })

    # Puts
    puts_df = pd.DataFrame({
        "strike": [480.0, 490.0, 500.0, 510.0, 520.0],
        "bid":    [3.0,   6.0,   10.0,  15.0,  22.0],
        "ask":    [5.0,   8.0,   12.0,  17.0,  24.0],
        "lastPrice": [4.0, 7.0,  11.0,  16.0,  23.0],
        "volume":    [300, 1200, 2000,  800,   500],
        "openInterest": [50, 150, 400,  200,   100],
    })

    chain_mock = MagicMock()
    chain_mock.calls = calls_df
    chain_mock.puts = puts_df
    mock.option_chain.return_value = chain_mock

    return mock


@pytest.fixture
def mock_scan_result() -> ScanResult:
    with patch("src.scanner.yf.Ticker", return_value=_make_mock_ticker()):
        result = fetch_option_chain(
            "FAKE",
            dte_range=(0, 3000),   # accept any DTE so our 2030 expiry passes
            min_volume=50,
            max_spread_pct=1.0,    # wide spread allowed in test
        )
    assert result is not None, "fetch_option_chain returned None"
    return result


class TestFetchOptionChain:
    def test_returns_scan_result(self, mock_scan_result):
        assert isinstance(mock_scan_result, ScanResult)

    def test_underlying_price_set(self, mock_scan_result):
        assert mock_scan_result.underlying_price > 0

    def test_hv_computed(self, mock_scan_result):
        assert 0.0 < mock_scan_result.hv_30 < 2.0

    def test_atm_iv_set(self, mock_scan_result):
        assert 0.0 < mock_scan_result.atm_iv < 5.0

    def test_contracts_dataframe_nonempty(self, mock_scan_result):
        assert not mock_scan_result.contracts.empty

    def test_contracts_have_required_columns(self, mock_scan_result):
        required = {"ticker", "expiry", "dte", "option_type", "strike",
                    "iv", "delta", "gamma", "theta", "vega", "is_uoa",
                    "vol_oi_ratio", "spread_pct"}
        assert required.issubset(set(mock_scan_result.contracts.columns))

    def test_iv_rank_in_range(self, mock_scan_result):
        assert 0.0 <= mock_scan_result.iv_rank_value <= 1.0

    def test_iv_percentile_in_range(self, mock_scan_result):
        assert 0.0 <= mock_scan_result.iv_percentile_value <= 1.0

    def test_oi_walls_list(self, mock_scan_result):
        assert isinstance(mock_scan_result.oi_walls, list)

    def test_uoa_flagging(self, mock_scan_result):
        df = mock_scan_result.contracts
        # vol/OI ratio > 5 should be flagged
        flagged = df[df["is_uoa"]]
        high_ratio = df[df["vol_oi_ratio"] >= 5.0]
        assert set(high_ratio.index).issubset(set(flagged.index))

    def test_returns_none_for_empty_history(self):
        mock = _make_mock_ticker()
        mock.history.return_value = pd.DataFrame()
        with patch("src.scanner.yf.Ticker", return_value=mock):
            result = fetch_option_chain("FAKE")
        assert result is None

    def test_returns_none_for_no_options(self):
        mock = _make_mock_ticker()
        mock.options = []
        with patch("src.scanner.yf.Ticker", return_value=mock):
            result = fetch_option_chain("FAKE")
        assert result is None


# ── rank_candidates ───────────────────────────────────────────────────────────

class TestRankCandidates:
    def test_returns_dataframe(self, mock_scan_result):
        ranked = rank_candidates({"FAKE": mock_scan_result}, top_n=10)
        assert isinstance(ranked, pd.DataFrame)

    def test_top_n_respected(self, mock_scan_result):
        ranked = rank_candidates({"FAKE": mock_scan_result}, top_n=3)
        assert len(ranked) <= 3

    def test_ranked_by_score_descending(self, mock_scan_result):
        ranked = rank_candidates({"FAKE": mock_scan_result}, top_n=20)
        if len(ranked) > 1:
            scores = ranked["score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_empty_results_returns_empty_df(self):
        result = rank_candidates({})
        assert result.empty

    def test_required_columns_present(self, mock_scan_result):
        ranked = rank_candidates({"FAKE": mock_scan_result})
        required = {"ticker", "expiry", "dte", "type", "strike", "iv", "delta",
                    "vol_oi_ratio", "score", "is_uoa"}
        assert required.issubset(set(ranked.columns))
