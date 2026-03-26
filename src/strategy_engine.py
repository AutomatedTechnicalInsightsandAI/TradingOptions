"""
Strategy Engine – suggests trades with IV-aware strategy selection.

Workflow
--------
1. Accept a market view (bullish / bearish) and a volatility view (high / low)
2. Filter scan results to contracts matching the directional bias
3. Apply risk/reward and PoP gates
4. Select strategy type based on IV Rank regime
5. Return ranked trade recommendations with position sizing

Strategy selection based on IV Rank:
  IV Rank < 30%  → Long Call / Long Put (buy cheap premium)
  IV Rank 30–50% → Bull/Bear Debit Spread
  IV Rank 50–70% → Bull Put / Bear Call Credit Spread
  IV Rank > 70%  → Iron Condor
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
StrategyType = Literal[
    "Long Call", "Long Put",
    "Bull Call Spread", "Bear Put Spread",
    "Bull Put Credit Spread", "Bear Call Credit Spread",
    "Iron Condor", "Long Straddle", "Long Strangle",
]

# IV Rank regime thresholds (0–100 scale)
_IV_LOW = 30.0    # below → buy premium
_IV_MOD = 50.0    # 30–50 → debit spread
_IV_HIGH = 70.0   # 50–70 → credit spread; above → Iron Condor

# Proxy spread width used when exact short-strike prices are unavailable.
# Entry × 1.5 approximates a typical 50% OTM short leg, keeping max-loss
# and reward/risk calculations in a realistic range.
_SPREAD_WIDTH_MULTIPLIER = 1.5


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
    iv_rank: float          # IV Rank (0–100)
    pop: float              # probability of profit (decimal)
    max_loss: float         # maximum loss per contract (dollar, 1-lot = 100 shares)
    reward_risk: float      # simplified reward-to-risk at 2× target
    is_uoa: bool
    strategy_type: StrategyType = "Long Call"
    strategy_legs: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Recommend trades given a directional and volatility view.

    Strategy is selected based on IV Rank:
      IV Rank < 30%  → Long Call / Long Put (buy cheap premium)
      IV Rank 30–50% → Bull/Bear Debit Spread
      IV Rank 50–70% → Bull Put / Bear Call Credit Spread
      IV Rank > 70%  → Iron Condor

    Parameters
    ----------
    market_view:
        ``"bullish"`` → long calls/bull spreads, ``"bearish"`` → long puts/bear spreads,
        ``"neutral"`` → both.
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
            iv_rank = float(row.get("iv_rank", 50))

            # PoP calculation (per-contract, 1 lot = 100 shares)
            if opt_type == "call":
                pop = pop_long_call(delta)
            else:
                pop = pop_long_put(delta)

            if not (self.pop_range[0] <= pop <= self.pop_range[1]):
                continue

            # Select strategy based on IV rank
            strategy_type, strategy_legs = self._select_strategy(opt_type, iv_rank, iv_vs_hv, delta)

            # Risk / reward — credit strategies use spread-width proxy
            is_credit = strategy_type in (
                "Bull Put Credit Spread", "Bear Call Credit Spread", "Iron Condor"
            )
            if is_credit:
                spread_width = entry * _SPREAD_WIDTH_MULTIPLIER
                max_loss_per_contract = (spread_width - entry) * 100
                reward_risk = entry / (spread_width - entry) if (spread_width - entry) > 0 else 0
            else:
                max_loss_per_contract = entry * 100  # entire premium
                target_profit = entry * 2 * 100      # 2× target
                reward_risk = target_profit / max_loss_per_contract if max_loss_per_contract > 0 else 0

            if reward_risk < self.min_reward_risk:
                continue

            rationale = self._build_rationale(row, pop, reward_risk, strategy_type, iv_rank)

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
                    iv_rank=iv_rank,
                    pop=pop,
                    max_loss=max_loss_per_contract,
                    reward_risk=reward_risk,
                    is_uoa=bool(row.get("is_uoa", False)),
                    strategy_type=strategy_type,
                    strategy_legs=strategy_legs,
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
        strike = rec.strike
        entry = rec.entry_price

        if rec.strategy_type == "Long Call":
            pnl = call_payoff_at_expiry(strike, entry, prices) * 100

        elif rec.strategy_type == "Long Put":
            pnl = put_payoff_at_expiry(strike, entry, prices) * 100

        elif rec.strategy_type == "Bull Call Spread":
            short_strike = strike * 1.05
            pnl = (
                call_payoff_at_expiry(strike, entry, prices)
                - call_payoff_at_expiry(short_strike, 0, prices)
            ) * 100

        elif rec.strategy_type == "Bear Put Spread":
            short_strike = strike * 0.95
            pnl = (
                put_payoff_at_expiry(strike, entry, prices)
                - put_payoff_at_expiry(short_strike, 0, prices)
            ) * 100

        elif rec.strategy_type == "Bull Put Credit Spread":
            # Max profit = entry * 100, Max loss = (spread_width - entry) * 100
            spread_width = entry * _SPREAD_WIDTH_MULTIPLIER
            max_profit = entry * 100
            max_loss = (spread_width - entry) * 100
            short_put = strike * 0.95
            pnl = np.where(
                prices >= strike,
                max_profit,
                np.where(
                    prices <= short_put,
                    -max_loss,
                    max_profit - (strike - prices) / (strike - short_put) * (max_profit + max_loss),
                ),
            )

        elif rec.strategy_type == "Bear Call Credit Spread":
            spread_width = entry * _SPREAD_WIDTH_MULTIPLIER
            max_profit = entry * 100
            max_loss = (spread_width - entry) * 100
            short_call = strike * 1.05
            pnl = np.where(
                prices <= strike,
                max_profit,
                np.where(
                    prices >= short_call,
                    -max_loss,
                    max_profit - (prices - strike) / (short_call - strike) * (max_profit + max_loss),
                ),
            )

        elif rec.strategy_type == "Iron Condor":
            spread_width = entry * _SPREAD_WIDTH_MULTIPLIER
            max_profit = entry * 100
            max_loss = (spread_width - entry) * 100
            put_short = strike * 0.95
            call_short = strike * 1.05
            put_long = strike * 0.90
            call_long = strike * 1.10
            pnl = np.where(
                (prices >= put_short) & (prices <= call_short),
                max_profit,
                np.where(
                    prices < put_long,
                    -max_loss,
                    np.where(
                        prices > call_long,
                        -max_loss,
                        np.where(
                            prices < put_short,
                            max_profit - (put_short - prices) / (put_short - put_long) * (max_profit + max_loss),
                            max_profit - (prices - call_short) / (call_long - call_short) * (max_profit + max_loss),
                        ),
                    ),
                ),
            )

        else:
            # Fallback: treat as long call or put
            if rec.option_type == "call":
                pnl = call_payoff_at_expiry(strike, entry, prices) * 100
            else:
                pnl = put_payoff_at_expiry(strike, entry, prices) * 100

        return prices, pnl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        opt_type: str,
        iv_rank: float,
        iv_vs_hv: float,
        delta: float,
    ) -> tuple[StrategyType, str]:
        """
        Select strategy type based on IV Rank regime.

        Returns (strategy_type, strategy_legs_description).
        """
        if iv_rank < _IV_LOW:
            # LOW IV → buy premium
            if opt_type == "call":
                return (
                    "Long Call",
                    "Buy 1 ATM/near-ATM call. IV is cheap — buying premium is advantageous. "
                    "Max loss = premium paid.",
                )
            else:
                return (
                    "Long Put",
                    "Buy 1 ATM/near-ATM put. IV is cheap — buying premium is advantageous. "
                    "Max loss = premium paid.",
                )

        elif iv_rank < _IV_MOD:
            # MODERATE IV → debit spreads
            if opt_type == "call":
                return (
                    "Bull Call Spread",
                    "Buy 1 ATM call + Sell 1 OTM call (~5% above spot). "
                    "Moderate IV — debit spread limits premium cost while capping upside.",
                )
            else:
                return (
                    "Bear Put Spread",
                    "Buy 1 ATM put + Sell 1 OTM put (~5% below spot). "
                    "Moderate IV — debit spread limits premium cost while capping downside capture.",
                )

        elif iv_rank < _IV_HIGH:
            # HIGH IV → credit spreads
            if opt_type == "call":
                return (
                    "Bear Call Credit Spread",
                    "Sell 1 OTM call + Buy 1 further OTM call as hedge. "
                    "IV elevated — selling premium via credit spread to profit from IV crush.",
                )
            else:
                return (
                    "Bull Put Credit Spread",
                    "Sell 1 OTM put + Buy 1 further OTM put as hedge. "
                    "IV elevated — selling premium via credit spread to profit from IV crush.",
                )

        else:
            # VERY HIGH IV → Iron Condor
            return (
                "Iron Condor",
                "Sell OTM call + Sell OTM put + Buy further OTM call wing + Buy further OTM put wing. "
                "IV very high — Iron Condor collects rich premium from both sides.",
            )

    def _build_rationale(
        self,
        row: pd.Series,
        pop: float,
        reward_risk: float,
        strategy_type: StrategyType = "Long Call",
        iv_rank: float = 50.0,
    ) -> str:
        # IV regime note
        if iv_rank < _IV_LOW:
            iv_note = "⬇️ Low IV — Buy premium"
        elif iv_rank < _IV_MOD:
            iv_note = "➡️ Moderate IV — Debit spread"
        elif iv_rank < _IV_HIGH:
            iv_note = "⬆️ High IV — Credit spread (avoid IV crush)"
        else:
            iv_note = "🔥 Very High IV — Iron Condor / sell vol"

        parts = [f"[{strategy_type}]", iv_note]
        if row.get("is_uoa"):
            parts.append(f"UOA: {row['vol_oi_ratio']:.1f}× Vol/OI")
        if float(row.get("iv_rank", 0)) > 50:
            parts.append(f"IV Rank {row['iv_rank']:.0f}%")
        if float(row.get("iv_vs_hv", 1.0)) > 1.2:
            parts.append(f"IV/HV={row['iv_vs_hv']:.2f}")
        parts.append(f"PoP≈{pop * 100:.0f}%")
        parts.append(f"R:R={reward_risk:.1f}×")
        return " | ".join(parts)
