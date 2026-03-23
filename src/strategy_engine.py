"""
Strategy Engine – suggests naked call/put trades and validates risk/reward.

Workflow
--------
1. Accept a market view (bullish / bearish) and a volatility view (high / low)
2. Filter scan results to contracts matching the directional bias
3. Apply risk/reward and PoP gates
4. Return ranked trade recommendations with position sizing

Focus: long naked calls (bullish) and long naked puts (bearish).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .greeks import (
    OptionContract,
    calculate_greeks,
    call_payoff_at_expiry,
    pop_long_call,
    pop_long_put,
    put_payoff_at_expiry,
)

logger = logging.getLogger(__name__)

MarketView = Literal["bullish", "bearish", "neutral"]
VolView = Literal["high", "low", "neutral"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecommendation:
    """A single actionable trade recommendation."""

    ticker: str
    expiry: str
    dte: int
    option_type: str        # "call" or "put"
    strike: float
    underlying_price: float
    entry_price: float      # mid-price of the option
    delta: float
    theta: float
    vega: float
    iv: float
    pop: float              # probability of profit (decimal)
    max_loss: float         # maximum loss per contract (dollar, 1-lot = 100 shares)
    reward_risk: float      # simplified reward-to-risk at 2× target
    is_uoa: bool
    rationale: str = ""


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Recommend naked call/put trades given a directional and volatility view.

    Parameters
    ----------
    market_view:
        ``"bullish"`` → long calls, ``"bearish"`` → long puts, ``"neutral"`` → both.
    vol_view:
        ``"high"`` → prefer contracts where IV > HV (breakout/momentum potential).
        ``"low"``  → prefer contracts where IV < HV (cheap options).
        ``"neutral"`` → no filter.
    min_reward_risk:
        Minimum reward-to-risk ratio at the 2× target.
    pop_range:
        ``(min_pop, max_pop)`` – keeps contracts with reasonable profit probability.
    account_size:
        Account size in dollars for position sizing.
    risk_per_trade:
        Fraction of account to risk per trade.
    """

    def __init__(
        self,
        market_view: MarketView = "bullish",
        vol_view: VolView = "neutral",
        min_reward_risk: float = 2.0,
        pop_range: tuple[float, float] = (0.30, 0.65),
        account_size: float = 25_000,
        risk_per_trade: float = 0.02,
    ) -> None:
        self.market_view = market_view
        self.vol_view = vol_view
        self.min_reward_risk = min_reward_risk
        self.pop_range = pop_range
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, ranked_candidates: pd.DataFrame) -> list[TradeRecommendation]:
        """
        Filter *ranked_candidates* (output of :func:`scanner.rank_candidates`)
        and return a list of :class:`TradeRecommendation` objects.
        """
        if ranked_candidates.empty:
            return []

        recs: list[TradeRecommendation] = []

        for _, row in ranked_candidates.iterrows():
            opt_type = row["type"]

            # Directional filter
            if self.market_view == "bullish" and opt_type != "call":
                continue
            if self.market_view == "bearish" and opt_type != "put":
                continue

            # Volatility view filter
            iv_vs_hv = row.get("iv_vs_hv", 1.0)
            if self.vol_view == "high" and iv_vs_hv < 1.0:
                continue
            if self.vol_view == "low" and iv_vs_hv > 1.0:
                continue

            entry = float(row["mid"])
            delta = float(row["delta"])
            theta = float(row["theta"])
            vega = float(row["vega"])
            iv = float(row["iv"]) / 100.0  # convert back from percent

            # PoP calculation (per-contract, 1 lot = 100 shares)
            if opt_type == "call":
                pop = pop_long_call(delta)
            else:
                pop = pop_long_put(delta)

            if not (self.pop_range[0] <= pop <= self.pop_range[1]):
                continue

            # Risk / reward
            max_loss_per_contract = entry * 100  # entire premium
            target_profit = entry * 2 * 100      # 2× target
            reward_risk = target_profit / max_loss_per_contract if max_loss_per_contract > 0 else 0

            if reward_risk < self.min_reward_risk:
                continue

            rationale = self._build_rationale(row, pop, reward_risk)

            recs.append(
                TradeRecommendation(
                    ticker=row["ticker"],
                    expiry=row["expiry"],
                    dte=int(row["dte"]),
                    option_type=opt_type,
                    strike=float(row["strike"]),
                    underlying_price=float(row["underlying"]),
                    entry_price=entry,
                    delta=delta,
                    theta=theta,
                    vega=vega,
                    iv=iv * 100,
                    pop=pop,
                    max_loss=max_loss_per_contract,
                    reward_risk=reward_risk,
                    is_uoa=bool(row.get("is_uoa", False)),
                    rationale=rationale,
                )
            )

        # Sort by PoP proximity to 0.50 (most balanced), then UOA flag
        recs.sort(key=lambda r: (not r.is_uoa, abs(r.pop - 0.50)))
        return recs

    def position_size(self, entry_price: float) -> int:
        """Return the number of contracts to buy given max dollar risk."""
        max_risk = self.account_size * self.risk_per_trade
        cost_per_contract = entry_price * 100
        if cost_per_contract <= 0:
            return 0
        return max(1, int(max_risk / cost_per_contract))

    # ------------------------------------------------------------------
    # Payoff data for visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def payoff_data(
        rec: TradeRecommendation,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (price_range, pnl) arrays for the P&L-at-expiry diagram.

        Prices span ±40% around the underlying price.
        P&L is per-contract (100 shares).
        """
        spot = rec.underlying_price
        lo = spot * 0.60
        hi = spot * 1.40
        prices = np.linspace(lo, hi, n_points)

        if rec.option_type == "call":
            pnl = call_payoff_at_expiry(rec.strike, rec.entry_price, prices) * 100
        else:
            pnl = put_payoff_at_expiry(rec.strike, rec.entry_price, prices) * 100

        return prices, pnl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rationale(
        self, row: pd.Series, pop: float, reward_risk: float
    ) -> str:
        parts = []
        if row.get("is_uoa"):
            parts.append(f"UOA: {row['vol_oi_ratio']:.1f}× Vol/OI")
        if float(row.get("iv_rank", 0)) > 50:
            parts.append(f"IV Rank {row['iv_rank']:.0f}%")
        if float(row.get("iv_vs_hv", 1.0)) > 1.2:
            parts.append(f"IV/HV={row['iv_vs_hv']:.2f}")
        parts.append(f"PoP≈{pop * 100:.0f}%")
        parts.append(f"R:R={reward_risk:.1f}×")
        return " | ".join(parts)
