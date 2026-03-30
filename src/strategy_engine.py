"""
Strategy Engine – directional ATM long-option recommendations for a cash account.

Workflow
--------
1. Accept a market view (bullish / bearish / neutral)
2. Determine directional side (call or put) from market view + UOA flow + IV skew
3. Filter ranked candidates to ATM contracts only (within 2 % of spot)
4. Apply max-premium-per-contract cap
5. Return up to 5 ranked ATM trade recommendations with position sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .greeks import (
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

# ATM tolerance: strike must be within this fraction of the underlying price
_ATM_TOLERANCE = 0.02

# Wider ATM window used when comparing call vs put IV for skew analysis
_IV_SKEW_ATM_WINDOW = 0.05

# Threshold ratio for detecting a meaningful IV skew between calls and puts
_IV_SKEW_THRESHOLD = 1.05

# Minimum UOA contract count difference to consider a directional UOA signal significant
_UOA_SIGNIFICANCE_THRESHOLD = 2

# Guard against division-by-zero when underlying price is near zero
_MIN_UNDERLYING_PRICE = 0.01


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
    directional_verdict: str = ""   # e.g. "🟢 BUY CALL — Bullish UOA + IV skew"
    directional_confidence: str = ""  # "High" / "Medium" / "Low"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Recommend directional ATM long-option trades for a cash account.

    Strategy is always Long Call or Long Put — no spreads or credit strategies.
    The recommended side is derived from:
      1. ``market_view`` — primary input (bullish → call, bearish → put)
      2. UOA flow   — unusual call/put activity detected in ``ranked_candidates``
      3. IV skew    — whether call IV or put IV is elevated at ATM

    Parameters
    ----------
    market_view:
        ``"bullish"`` → Long Call, ``"bearish"`` → Long Put,
        ``"neutral"`` → side determined by UOA flow + IV skew.
    vol_view:
        Kept for API compatibility; no longer used to gate recommendations.
    max_premium:
        Maximum option premium per contract in dollars (entry_price × 100).
        Contracts more expensive than this cap are filtered out.
    account_size:
        Account size in dollars for position sizing.
    risk_per_trade:
        Fraction of account to risk per trade.
    """

    def __init__(
        self,
        market_view: MarketView = "bullish",
        vol_view: VolView = "neutral",
        max_premium: float = 500.0,
        account_size: float = 25_000,
        risk_per_trade: float = 0.02,
        # Legacy params kept for backward compatibility — ignored internally
        min_reward_risk: float = 2.0,
        pop_range: tuple[float, float] = (0.30, 0.65),
    ) -> None:
        self.market_view = market_view
        self.vol_view = vol_view
        self.max_premium = max_premium
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, ranked_candidates: pd.DataFrame) -> list[TradeRecommendation]:
        """
        Filter *ranked_candidates* (output of :func:`scanner.rank_candidates`)
        and return a list of :class:`TradeRecommendation` objects.

        Always recommends Long Call or Long Put — no spreads or credit strategies.
        Filters to ATM contracts only (within 2 % of spot) and applies the
        ``max_premium`` cap.  Returns up to 5 recommendations.
        """
        if ranked_candidates.empty:
            return []

        side, verdict, confidence = self._compute_directional_signal(ranked_candidates)
        strategy_type: StrategyType = "Long Call" if side == "call" else "Long Put"

        recs_with_volume: list[tuple[TradeRecommendation, float]] = []

        for _, row in ranked_candidates.iterrows():
            opt_type = row["type"]

            # Only the recommended side
            if opt_type != side:
                continue

            entry = float(row["mid"])
            underlying = float(row["underlying"])
            strike = float(row["strike"])

            # ATM filter: strike within 2 % of underlying
            if underlying > 0 and abs(strike - underlying) / underlying > _ATM_TOLERANCE:
                continue

            # Max premium cap
            if entry * 100 > self.max_premium:
                continue

            delta = float(row["delta"])
            theta = float(row["theta"])
            vega = float(row["vega"])
            iv = float(row["iv"]) / 100.0  # convert back from percent
            iv_rank = float(row.get("iv_rank", 50))
            volume = float(row.get("volume", 0))

            if opt_type == "call":
                pop = pop_long_call(delta)
            else:
                pop = pop_long_put(delta)

            max_loss_per_contract = entry * 100

            rationale = self._build_rationale(
                row, side, verdict, iv_rank, max_loss_per_contract
            )

            rec = TradeRecommendation(
                ticker=row["ticker"],
                expiry=row["expiry"],
                dte=int(row["dte"]),
                option_type=opt_type,
                strike=strike,
                underlying_price=underlying,
                entry_price=entry,
                delta=delta,
                theta=theta,
                vega=vega,
                iv=iv * 100,
                iv_rank=iv_rank,
                pop=pop,
                max_loss=max_loss_per_contract,
                reward_risk=2.0,  # 2× target / 1× cost for long options
                is_uoa=bool(row.get("is_uoa", False)),
                strategy_type=strategy_type,
                strategy_legs=(
                    f"Buy 1 ATM {side}. Max loss = premium paid (${max_loss_per_contract:,.0f})."
                ),
                rationale=rationale,
                directional_verdict=verdict,
                directional_confidence=confidence,
            )
            recs_with_volume.append((rec, volume))

        # Sort: nearest to ATM first, then UOA flag (True first), then volume (desc)
        recs_with_volume.sort(
            key=lambda rv: (
                abs(rv[0].strike - rv[0].underlying_price) / max(rv[0].underlying_price, _MIN_UNDERLYING_PRICE),
                not rv[0].is_uoa,
                -rv[1],
            )
        )
        return [rv[0] for rv in recs_with_volume[:5]]

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
            spread_width = entry * 1.5  # proxy: short leg ~50% OTM
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
            spread_width = entry * 1.5
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
            spread_width = entry * 1.5
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

    def _compute_directional_signal(
        self,
        ranked_candidates: pd.DataFrame,
    ) -> tuple[str, str, str]:
        """
        Determine recommended side (``"call"`` or ``"put"``), a human-readable
        verdict string, and a confidence level (``"High"``, ``"Medium"``,
        ``"Low"``).

        Logic:
        * ``market_view == "bullish"``  → call
        * ``market_view == "bearish"``  → put
        * ``market_view == "neutral"``  → use UOA flow then IV skew to break tie
        """
        # ── UOA flow ─────────────────────────────────────────────────────
        uoa_calls = 0
        uoa_puts = 0
        if not ranked_candidates.empty and "is_uoa" in ranked_candidates.columns:
            uoa_df = ranked_candidates[ranked_candidates["is_uoa"].astype(bool)]
            uoa_calls = int((uoa_df["type"] == "call").sum())
            uoa_puts = int((uoa_df["type"] == "put").sum())

        uoa_signal = ""
        if uoa_calls > uoa_puts:
            uoa_signal = "Bullish UOA"
        elif uoa_puts > uoa_calls:
            uoa_signal = "Bearish UOA"

        # ── IV skew (compare mean ATM call IV vs put IV) ──────────────────
        iv_skew_signal = "neutral"
        iv_skew_note = ""
        if not ranked_candidates.empty and "iv" in ranked_candidates.columns:
            if "underlying" in ranked_candidates.columns:
                atm_mask = (
                    abs(ranked_candidates["strike"] - ranked_candidates["underlying"])
                    / ranked_candidates["underlying"].clip(lower=_MIN_UNDERLYING_PRICE)
                    <= _IV_SKEW_ATM_WINDOW
                )
                atm_df = ranked_candidates[atm_mask]
                if not atm_df.empty:
                    call_iv_mean = atm_df.loc[atm_df["type"] == "call", "iv"].mean()
                    put_iv_mean = atm_df.loc[atm_df["type"] == "put", "iv"].mean()
                    if pd.notna(call_iv_mean) and pd.notna(put_iv_mean):
                        if put_iv_mean > call_iv_mean * _IV_SKEW_THRESHOLD:
                            iv_skew_signal = "bearish"
                            iv_skew_note = "Put IV skew"
                        elif call_iv_mean > put_iv_mean * _IV_SKEW_THRESHOLD:
                            iv_skew_signal = "bullish"
                            iv_skew_note = "Call IV skew"

        # ── Determine side ───────────────────────────────────────────────
        if self.market_view == "bullish":
            side = "call"
        elif self.market_view == "bearish":
            side = "put"
        else:  # neutral — let flow / skew decide
            if uoa_calls > uoa_puts:
                side = "call"
            elif uoa_puts > uoa_calls:
                side = "put"
            elif iv_skew_signal == "bullish":
                side = "call"
            elif iv_skew_signal == "bearish":
                side = "put"
            else:
                side = "call"  # default when no signal

        # ── Confidence ───────────────────────────────────────────────────
        confirming = 0
        conflicting = False
        if self.market_view == "bullish":
            if uoa_signal == "Bullish UOA":
                confirming += 1
            elif uoa_signal == "Bearish UOA":
                conflicting = True
            if iv_skew_signal == "bullish":
                confirming += 1
            elif iv_skew_signal == "bearish":
                conflicting = True
        elif self.market_view == "bearish":
            if uoa_signal == "Bearish UOA":
                confirming += 1
            elif uoa_signal == "Bullish UOA":
                conflicting = True
            if iv_skew_signal == "bearish":
                confirming += 1
            elif iv_skew_signal == "bullish":
                conflicting = True
        else:  # neutral
            if abs(uoa_calls - uoa_puts) >= _UOA_SIGNIFICANCE_THRESHOLD:
                confirming += 1
            if iv_skew_signal != "neutral":
                confirming += 1

        if conflicting:
            confidence = "Low"
        elif confirming >= 2:
            confidence = "High"
        elif confirming == 1 or self.market_view != "neutral":
            confidence = "Medium"
        else:
            confidence = "Low"

        # ── Build verdict string ─────────────────────────────────────────
        reasons: list[str] = []
        if self.market_view != "neutral":
            reasons.append(f"{self.market_view.title()} view")
        if uoa_signal:
            reasons.append(uoa_signal)
        if iv_skew_note:
            reasons.append(iv_skew_note)
        reason_str = " + ".join(reasons) if reasons else "No clear signal"

        if side == "call":
            verdict = f"🟢 BUY CALL — {reason_str}"
        else:
            verdict = f"🔴 BUY PUT — {reason_str}"

        return side, verdict, confidence

    def _build_rationale(
        self,
        row: pd.Series,
        side: str,
        verdict: str,
        iv_rank: float,
        max_loss: float,
    ) -> str:
        """Build a plain-English rationale for a single recommendation."""
        parts: list[str] = [verdict]

        # IV context
        if iv_rank >= 70:
            parts.append("⚠️ Very High IV — premium is expensive, size down")
        elif iv_rank >= 50:
            parts.append("⚠️ Elevated IV — premium may be pricey")
        elif iv_rank < 30:
            parts.append("✅ Low IV — favorable environment to buy premium")

        if row.get("is_uoa"):
            parts.append(f"UOA: {row['vol_oi_ratio']:.1f}× Vol/OI")

        parts.append(
            f"ATM strike ${float(row['strike']):.2f} | expiry {row['expiry']} ({int(row['dte'])} DTE)"
        )
        parts.append(f"Max loss = ${max_loss:,.0f} per contract")
        return " | ".join(parts)
