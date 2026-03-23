"""
Options Chain Scanner – data ingestion, UOA detection, and candidate ranking.

Data source: yfinance (free, no API key required).

Key features
------------
* Fetch live option chains for any ticker
* Calculate Greeks and IV for every contract
* Detect Unusual Options Activity (UOA): Vol/OI ratio > threshold
* Identify Open Interest Walls (price magnets / resistance levels)
* Rank candidates by a composite score (IV rank, UOA, spread quality)
* Calculate Historical Volatility and compare with IV
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .greeks import (
    OptionContract,
    calculate_greeks,
    historical_volatility,
    implied_volatility,
    iv_percentile,
    iv_rank,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridable by the caller)
# ---------------------------------------------------------------------------

DEFAULT_UOA_RATIO = 5.0    # flag if volume / open_interest > this
DEFAULT_MIN_VOLUME = 50
DEFAULT_MAX_SPREAD_PCT = 0.20
DEFAULT_DTE_RANGE = (7, 60)
DEFAULT_HV_WINDOW = 30
DEFAULT_RISK_FREE_RATE = 0.053


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContractRow:
    """Enriched option contract row (one row in the scanner output table)."""

    ticker: str
    expiry: str
    dte: int
    option_type: str          # "call" or "put"
    strike: float
    last_price: float
    bid: float
    ask: float
    mid: float
    spread_pct: float         # (ask - bid) / mid
    volume: int
    open_interest: int
    vol_oi_ratio: float       # volume / open_interest
    iv: float                 # implied volatility (decimal)
    delta: float
    gamma: float
    theta: float
    vega: float
    theoretical_price: float
    is_uoa: bool              # unusual options activity flag
    underlying_price: float


@dataclass
class ScanResult:
    """Full scan result for a single ticker."""

    ticker: str
    underlying_price: float
    hv_30: float              # 30-day historical volatility
    atm_iv: float             # at-the-money IV (nearest strike to spot)
    iv_rank_value: float      # IV rank vs 52-week IV history [0,1]
    iv_percentile_value: float
    iv_vs_hv: float           # atm_iv / hv_30 ratio
    oi_walls: list[float]     # price levels with highest OI concentration
    contracts: pd.DataFrame   # full enriched chain
    uoa_alerts: pd.DataFrame  # filtered UOA rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trading_days_to_expiry(expiry_str: str) -> int:
    """Return approximate calendar days to expiry (used as DTE proxy)."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    return max(0, (exp - now).days)


def _time_to_expiry_years(dte: int) -> float:
    return max(dte / 365.0, 1e-6)


def _mid_price(bid: float, ask: float) -> float:
    if bid <= 0 and ask <= 0:
        return 0.0
    if bid <= 0:
        return ask
    if ask <= 0:
        return bid
    return (bid + ask) / 2.0


def _spread_pct(bid: float, ask: float) -> float:
    mid = _mid_price(bid, ask)
    if mid <= 0:
        return 1.0
    return (ask - bid) / mid


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------

def fetch_option_chain(
    ticker: str,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    uoa_vol_oi_ratio: float = DEFAULT_UOA_RATIO,
    min_volume: int = DEFAULT_MIN_VOLUME,
    max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
    dte_range: tuple[int, int] = DEFAULT_DTE_RANGE,
    hv_window: int = DEFAULT_HV_WINDOW,
) -> Optional[ScanResult]:
    """
    Fetch and enrich the full option chain for *ticker*.

    Returns a :class:`ScanResult` or ``None`` if data is unavailable.
    """
    try:
        tkr = yf.Ticker(ticker)

        # ── Underlying price & history ─────────────────────────────────────
        hist = tkr.history(period="1y")
        if hist.empty:
            logger.warning("No price history for %s", ticker)
            return None

        underlying_price = float(hist["Close"].iloc[-1])
        close_prices = hist["Close"].tolist()

        hv_30 = historical_volatility(close_prices, window=hv_window)

        # Build a rough 52-week IV proxy using rolling HV (IV history requires
        # expensive intraday data; this is a practical free-data approximation)
        iv_history = [
            historical_volatility(close_prices[max(0, i - hv_window): i + 1], window=hv_window)
            for i in range(hv_window, len(close_prices))
        ]
        if not iv_history:
            iv_history = [hv_30]

        # ── Option expiry dates ───────────────────────────────────────────
        expirations = tkr.options
        if not expirations:
            logger.warning("No options data for %s", ticker)
            return None

        rows: list[ContractRow] = []
        all_iv_values: list[float] = []

        for expiry in expirations:
            dte = _trading_days_to_expiry(expiry)
            if not (dte_range[0] <= dte <= dte_range[1]):
                continue

            t = _time_to_expiry_years(dte)

            try:
                chain = tkr.option_chain(expiry)
            except Exception as exc:
                logger.debug("Could not fetch chain for %s %s: %s", ticker, expiry, exc)
                continue

            for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    try:
                        strike = float(row.get("strike", 0))
                        bid = float(row.get("bid", 0) or 0)
                        ask = float(row.get("ask", 0) or 0)
                        volume = int(row.get("volume", 0) or 0)
                        oi = int(row.get("openInterest", 0) or 0)
                        last = float(row.get("lastPrice", 0) or 0)

                        if volume < min_volume:
                            continue

                        mid = _mid_price(bid, ask)
                        sp_pct = _spread_pct(bid, ask)

                        if sp_pct > max_spread_pct and max_spread_pct > 0:
                            continue

                        market_price = mid if mid > 0 else last
                        if market_price <= 0:
                            continue

                        contract = OptionContract(
                            underlying_price=underlying_price,
                            strike=strike,
                            time_to_expiry=t,
                            risk_free_rate=risk_free_rate,
                            volatility=hv_30 or 0.3,
                            option_type=opt_type,
                        )

                        iv = implied_volatility(market_price, contract)
                        if iv is None or iv <= 0:
                            iv = hv_30 or 0.3
                        contract.volatility = iv

                        g = calculate_greeks(contract)

                        # Cap Vol/OI at 999 when OI is zero to avoid inf in
                        # sorting, display, and numerical operations.
                        vol_oi = (volume / oi) if oi > 0 else 999.0
                        is_uoa = vol_oi >= uoa_vol_oi_ratio

                        rows.append(
                            ContractRow(
                                ticker=ticker,
                                expiry=expiry,
                                dte=dte,
                                option_type=opt_type,
                                strike=strike,
                                last_price=last,
                                bid=bid,
                                ask=ask,
                                mid=mid,
                                spread_pct=sp_pct,
                                volume=volume,
                                open_interest=oi,
                                vol_oi_ratio=vol_oi,
                                iv=iv,
                                delta=g.delta,
                                gamma=g.gamma,
                                theta=g.theta,
                                vega=g.vega,
                                theoretical_price=g.theoretical_price,
                                is_uoa=is_uoa,
                                underlying_price=underlying_price,
                            )
                        )
                        all_iv_values.append(iv)

                    except Exception as exc:
                        logger.debug("Row error for %s %s: %s", ticker, expiry, exc)
                        continue

        if not rows:
            logger.warning("No qualifying contracts found for %s", ticker)
            return None

        df_all = pd.DataFrame([vars(r) for r in rows])

        # ── ATM IV ────────────────────────────────────────────────────────
        call_df = df_all[df_all["option_type"] == "call"].copy()
        if not call_df.empty:
            nearest = call_df.iloc[(call_df["strike"] - underlying_price).abs().argsort()[:1]]
            atm_iv = float(nearest["iv"].values[0])
        else:
            atm_iv = hv_30 or 0.3

        # ── IV Rank / Percentile ──────────────────────────────────────────
        ivr = iv_rank(atm_iv, iv_history)
        ivp = iv_percentile(atm_iv, iv_history)

        # ── OI Walls ─────────────────────────────────────────────────────
        oi_by_strike = (
            df_all.groupby("strike")["open_interest"].sum().reset_index()
        )
        top_oi = oi_by_strike.nlargest(5, "open_interest")
        oi_walls = top_oi["strike"].tolist()

        # ── UOA alerts ───────────────────────────────────────────────────
        uoa_df = df_all[df_all["is_uoa"]].copy()

        return ScanResult(
            ticker=ticker,
            underlying_price=underlying_price,
            hv_30=hv_30,
            atm_iv=atm_iv,
            iv_rank_value=ivr,
            iv_percentile_value=ivp,
            iv_vs_hv=atm_iv / hv_30 if hv_30 > 0 else float("nan"),
            oi_walls=oi_walls,
            contracts=df_all,
            uoa_alerts=uoa_df,
        )

    except Exception as exc:
        logger.error("scan_ticker failed for %s: %s", ticker, exc, exc_info=True)
        return None


def scan_watchlist(
    tickers: list[str],
    **kwargs,
) -> dict[str, ScanResult]:
    """
    Scan a list of tickers and return a mapping of ticker → ScanResult.
    """
    results: dict[str, ScanResult] = {}
    for ticker in tickers:
        logger.info("Scanning %s …", ticker)
        result = fetch_option_chain(ticker, **kwargs)
        if result is not None:
            results[ticker] = result
    return results


def rank_candidates(
    results: dict[str, ScanResult],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Build a ranked list of actionable candidates across all scanned tickers.

    Scoring heuristic
    -----------------
    score = uoa_flag * 3
          + iv_rank_value           (higher IV rank → richer premium environment)
          + (1 if iv_vs_hv > 1.2)  (IV elevated vs HV → selling edge or breakout)
          + (1 if vol_oi > 10)      (extreme UOA emphasis)
    """
    rows: list[dict] = []

    for ticker, res in results.items():
        df = res.uoa_alerts if not res.uoa_alerts.empty else res.contracts.head(5)
        for _, row in df.iterrows():
            uoa_bonus = 3 if row["is_uoa"] else 0
            iv_vs_hv_bonus = 1 if res.iv_vs_hv > 1.2 else 0
            extreme_uoa_bonus = 1 if row["vol_oi_ratio"] > 10 else 0
            score = uoa_bonus + res.iv_rank_value + iv_vs_hv_bonus + extreme_uoa_bonus

            rows.append(
                {
                    "ticker": ticker,
                    "expiry": row["expiry"],
                    "dte": row["dte"],
                    "type": row["option_type"],
                    "strike": row["strike"],
                    "underlying": row["underlying_price"],
                    "mid": row["mid"],
                    "volume": row["volume"],
                    "open_interest": row["open_interest"],
                    "vol_oi_ratio": round(row["vol_oi_ratio"], 1),
                    "iv": round(row["iv"] * 100, 1),
                    "iv_rank": round(res.iv_rank_value * 100, 1),
                    "iv_vs_hv": round(res.iv_vs_hv, 2),
                    "delta": round(row["delta"], 3),
                    "theta": round(row["theta"], 4),
                    "vega": round(row["vega"], 4),
                    "spread_pct": round(row["spread_pct"] * 100, 1),
                    "is_uoa": row["is_uoa"],
                    "score": round(score, 3),
                }
            )

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n)
    return ranked.reset_index(drop=True)
