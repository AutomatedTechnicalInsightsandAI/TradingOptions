"""
TradingOptions Dashboard – Streamlit-based command centre.

Run with:
    streamlit run src/dashboard.py

Features
--------
* Sidebar controls: watchlist, market/vol view, account size, strategy params
* Auto-refresh scanner (configurable interval)
* Options chain table with Greeks and IV for each contract
* Volatility smile chart (IV vs Strike)
* Open Interest + Volume by Strike chart
* P&L payoff diagram for selected trade recommendations
* UOA alerts panel
* IV Rank / Percentile gauges
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Allow running directly as `streamlit run src/dashboard.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.greeks import call_payoff_at_expiry, put_payoff_at_expiry
from src.scanner import ScanResult, fetch_option_chain, rank_candidates, scan_watchlist
from src.strategy_engine import StrategyEngine, TradeRecommendation

logging.basicConfig(level=logging.WARNING)

# ── Load configuration ─────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    return {}


CFG = _load_config()

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradingOptions – QQQ Top-10 Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    default_watchlist = CFG.get("watchlist", ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN"])
    watchlist_input = st.text_area(
        "Watchlist (one ticker per line)",
        value="\n".join(default_watchlist),
        height=200,
    )
    tickers = [t.strip().upper() for t in watchlist_input.split("\n") if t.strip()]

    st.markdown("---")
    st.subheader("Strategy")
    market_view = st.selectbox("Market View", ["bullish", "bearish", "neutral"], index=0)
    vol_view = st.selectbox("Volatility View", ["neutral", "high", "low"], index=0)

    st.markdown("---")
    st.subheader("Scanner Params")
    scanner_cfg = CFG.get("scanner", {})
    min_dte = st.slider("Min DTE", 1, 30, scanner_cfg.get("dte_range", [7, 60])[0])
    max_dte = st.slider("Max DTE", 15, 120, scanner_cfg.get("dte_range", [7, 60])[1])
    min_vol = st.number_input("Min Volume", min_value=1, value=int(scanner_cfg.get("min_volume", 50)))
    uoa_ratio = st.number_input("UOA Vol/OI Threshold", min_value=1.0, value=float(scanner_cfg.get("uoa_vol_oi_ratio", 5.0)), step=0.5)
    max_spread = st.slider("Max Spread %", 1, 50, int(float(scanner_cfg.get("max_spread_pct", 0.20)) * 100)) / 100.0
    top_n = st.slider("Top N Candidates", 5, 50, int(scanner_cfg.get("top_n", 20)))

    st.markdown("---")
    st.subheader("Risk / Account")
    strategy_cfg = CFG.get("strategy", {})
    account_size = st.number_input("Account Size ($)", min_value=1000, value=int(strategy_cfg.get("account_size", 25000)), step=1000)
    risk_pct = st.slider("Risk per Trade (%)", 1, 10, int(float(strategy_cfg.get("risk_per_trade", 0.02)) * 100)) / 100.0
    min_rr = st.number_input("Min Reward:Risk", min_value=1.0, value=float(strategy_cfg.get("min_reward_risk", 2.0)), step=0.5)

    st.markdown("---")
    refresh_sec = st.number_input("Auto-refresh (sec, 0=off)", min_value=0, value=int(CFG.get("dashboard", {}).get("refresh_seconds", 300)), step=30)

    run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)

# ── Auto-refresh logic ─────────────────────────────────────────────────────────
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = 0
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = {}
if "ranked" not in st.session_state:
    st.session_state["ranked"] = pd.DataFrame()
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = []

now = time.time()
auto_refresh = refresh_sec > 0 and (now - st.session_state["last_scan_time"]) >= refresh_sec

if run_scan or auto_refresh:
    with st.spinner("Scanning options chains …"):
        results = scan_watchlist(
            tickers,
            uoa_vol_oi_ratio=uoa_ratio,
            min_volume=min_vol,
            max_spread_pct=max_spread,
            dte_range=(min_dte, max_dte),
        )
        ranked = rank_candidates(results, top_n=top_n)

        engine = StrategyEngine(
            market_view=market_view,
            vol_view=vol_view,
            min_reward_risk=min_rr,
            account_size=account_size,
            risk_per_trade=risk_pct,
        )
        recs = engine.recommend(ranked)

        st.session_state["scan_results"] = results
        st.session_state["ranked"] = ranked
        st.session_state["recommendations"] = recs
        st.session_state["last_scan_time"] = now

results: dict[str, ScanResult] = st.session_state["scan_results"]
ranked: pd.DataFrame = st.session_state["ranked"]
recommendations: list[TradeRecommendation] = st.session_state["recommendations"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 TradingOptions – QQQ Top-10 Scanner")

if results:
    last_time = st.session_state["last_scan_time"]
    st.caption(f"Last scan: {time.strftime('%H:%M:%S', time.localtime(last_time))} | {len(results)} ticker(s) scanned")

if not results:
    st.info("👈 Press **Run Scan** in the sidebar to begin.")
    st.stop()

# ── Market Overview ───────────────────────────────────────────────────────────
st.header("Market Overview")

overview_rows = []
for tkr, res in results.items():
    overview_rows.append({
        "Ticker": tkr,
        "Price": f"${res.underlying_price:,.2f}",
        "HV 30d": f"{res.hv_30 * 100:.1f}%",
        "ATM IV": f"{res.atm_iv * 100:.1f}%",
        "IV Rank": f"{res.iv_rank_value * 100:.0f}%",
        "IV %ile": f"{res.iv_percentile_value * 100:.0f}%",
        "IV/HV": f"{res.iv_vs_hv:.2f}",
        "OI Walls": ", ".join(f"${w:,.0f}" for w in res.oi_walls[:3]),
    })

st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

# ── Ranked Candidates Table ───────────────────────────────────────────────────
st.header("🎯 Top Ranked Candidates")
if ranked.empty:
    st.warning("No candidates matched the filter criteria. Try relaxing the spread or volume thresholds.")
else:
    def _highlight_uoa(row: pd.Series) -> list[str]:
        color = "background-color: #1a3a1a" if row.get("is_uoa") else ""
        return [color] * len(row)

    styled = ranked.style.apply(_highlight_uoa, axis=1).format(
        {
            "mid": "${:.2f}",
            "underlying": "${:.2f}",
            "strike": "${:.2f}",
            "iv": "{:.1f}%",
            "delta": "{:.3f}",
            "theta": "{:.4f}",
            "vega": "{:.4f}",
            "spread_pct": "{:.1f}%",
            "score": "{:.2f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Per-Ticker Analysis ───────────────────────────────────────────────────────
st.header("📊 Per-Ticker Analysis")

selected_ticker = st.selectbox("Select ticker for detailed analysis", list(results.keys()))

if selected_ticker:
    res = results[selected_ticker]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Underlying Price", f"${res.underlying_price:,.2f}")
    col2.metric("ATM IV", f"{res.atm_iv * 100:.1f}%")
    col3.metric("IV Rank", f"{res.iv_rank_value * 100:.0f}%")
    col4.metric("IV / HV Ratio", f"{res.iv_vs_hv:.2f}x", delta=f"{(res.iv_vs_hv - 1) * 100:.0f}% vs HV")

    tab1, tab2, tab3, tab4 = st.tabs(["Volatility Smile", "OI & Volume", "Options Chain", "UOA Alerts"])

    # ── Tab 1: Volatility Smile ────────────────────────────────────────────
    with tab1:
        st.subheader("Volatility Smile / Skew")
        df = res.contracts

        for exp in sorted(df["expiry"].unique())[:3]:  # show up to 3 nearest expiries
            exp_df = df[df["expiry"] == exp]
            calls = exp_df[exp_df["option_type"] == "call"].sort_values("strike")
            puts = exp_df[exp_df["option_type"] == "put"].sort_values("strike")

            fig = go.Figure()
            if not calls.empty:
                fig.add_trace(go.Scatter(
                    x=calls["strike"], y=calls["iv"] * 100,
                    mode="lines+markers", name="Calls",
                    line=dict(color="#00cc88", width=2),
                    marker=dict(size=5),
                ))
            if not puts.empty:
                fig.add_trace(go.Scatter(
                    x=puts["strike"], y=puts["iv"] * 100,
                    mode="lines+markers", name="Puts",
                    line=dict(color="#ff4444", width=2),
                    marker=dict(size=5),
                ))

            fig.add_vline(
                x=res.underlying_price,
                line_dash="dash", line_color="white",
                annotation_text=f"Spot ${res.underlying_price:,.0f}",
            )
            fig.update_layout(
                title=f"IV Smile – {selected_ticker} expiry {exp}",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: OI & Volume ────────────────────────────────────────────────
    with tab2:
        st.subheader("Open Interest & Volume by Strike")
        df = res.contracts

        oi_call = df[df["option_type"] == "call"].groupby("strike")[["open_interest", "volume"]].sum()
        oi_put = df[df["option_type"] == "put"].groupby("strike")[["open_interest", "volume"]].sum()

        fig = go.Figure()
        if not oi_call.empty:
            fig.add_trace(go.Bar(
                x=oi_call.index, y=oi_call["open_interest"],
                name="Call OI", marker_color="#00cc88", opacity=0.7,
            ))
        if not oi_put.empty:
            fig.add_trace(go.Bar(
                x=oi_put.index, y=oi_put["open_interest"],
                name="Put OI", marker_color="#ff4444", opacity=0.7,
            ))

        fig.add_vline(
            x=res.underlying_price,
            line_dash="dash", line_color="white",
            annotation_text=f"Spot ${res.underlying_price:,.0f}",
        )
        for wall in res.oi_walls:
            fig.add_vline(x=wall, line_dash="dot", line_color="yellow", opacity=0.5)

        fig.update_layout(
            title=f"Open Interest by Strike – {selected_ticker}",
            xaxis_title="Strike",
            yaxis_title="Open Interest",
            barmode="overlay",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        if not oi_call.empty:
            fig2.add_trace(go.Bar(
                x=oi_call.index, y=oi_call["volume"],
                name="Call Vol", marker_color="#00cc88", opacity=0.7,
            ))
        if not oi_put.empty:
            fig2.add_trace(go.Bar(
                x=oi_put.index, y=oi_put["volume"],
                name="Put Vol", marker_color="#ff4444", opacity=0.7,
            ))
        fig2.add_vline(x=res.underlying_price, line_dash="dash", line_color="white")
        fig2.update_layout(
            title=f"Volume by Strike – {selected_ticker}",
            xaxis_title="Strike",
            yaxis_title="Volume",
            barmode="overlay",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Options Chain ──────────────────────────────────────────────
    with tab3:
        st.subheader("Full Options Chain")
        disp = res.contracts.copy()
        disp["iv_%"] = (disp["iv"] * 100).round(1)
        disp = disp[[
            "expiry", "dte", "option_type", "strike", "bid", "ask", "mid",
            "spread_pct", "volume", "open_interest", "vol_oi_ratio",
            "iv_%", "delta", "gamma", "theta", "vega", "is_uoa",
        ]]
        disp = disp.sort_values(["expiry", "option_type", "strike"])

        def _style_chain(row: pd.Series) -> list[str]:
            c = "background-color: #1a3a1a" if row["is_uoa"] else ""
            return [c] * len(row)

        st.dataframe(
            disp.style.apply(_style_chain, axis=1).format(
                {
                    "strike": "${:.2f}", "bid": "${:.2f}", "ask": "${:.2f}",
                    "mid": "${:.2f}", "spread_pct": "{:.1%}",
                    "vol_oi_ratio": "{:.1f}", "iv_%": "{:.1f}%",
                    "delta": "{:.3f}", "gamma": "{:.4f}",
                    "theta": "{:.4f}", "vega": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 4: UOA Alerts ─────────────────────────────────────────────────
    with tab4:
        st.subheader("🚨 Unusual Options Activity (UOA)")
        uoa = res.uoa_alerts
        if uoa.empty:
            st.success("No UOA detected for the current filter settings.")
        else:
            st.warning(f"{len(uoa)} unusual contracts detected!")
            st.dataframe(
                uoa[["expiry", "dte", "option_type", "strike", "mid", "volume",
                     "open_interest", "vol_oi_ratio", "iv", "delta"]].assign(
                    iv=lambda d: (d["iv"] * 100).round(1)
                ).sort_values("vol_oi_ratio", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

# ── Trade Recommendations ─────────────────────────────────────────────────────
st.header(f"💡 Trade Recommendations ({market_view.title()} View)")

if not recommendations:
    st.info("No recommendations met the current risk/reward and PoP criteria. Try adjusting the strategy parameters.")
else:
    engine = StrategyEngine(
        market_view=market_view,
        vol_view=vol_view,
        min_reward_risk=min_rr,
        account_size=account_size,
        risk_per_trade=risk_pct,
    )

    for i, rec in enumerate(recommendations[:10]):
        contracts = engine.position_size(rec.entry_price)
        total_risk = rec.max_loss * contracts

        with st.expander(
            f"{'🟢' if rec.option_type == 'call' else '🔴'} "
            f"**{rec.ticker}** {rec.strike} {rec.option_type.upper()} "
            f"exp {rec.expiry} | PoP {rec.pop * 100:.0f}% | Entry ${rec.entry_price:.2f}"
            + (" 🚨 UOA" if rec.is_uoa else ""),
            expanded=(i == 0),
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Entry (mid)", f"${rec.entry_price:.2f}")
            c2.metric("Max Loss / contract", f"${rec.max_loss:,.0f}")
            c3.metric("PoP", f"{rec.pop * 100:.0f}%")
            c4.metric("Suggested Contracts", str(contracts))
            c5.metric("Total Risk $", f"${total_risk:,.0f}")

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Rationale:** {rec.rationale}")
                greeks_df = pd.DataFrame([{
                    "Delta": f"{rec.delta:.3f}",
                    "Theta": f"{rec.theta:.4f}",
                    "Vega": f"{rec.vega:.4f}",
                    "IV": f"{rec.iv:.1f}%",
                    "DTE": rec.dte,
                }])
                st.dataframe(greeks_df, use_container_width=True, hide_index=True)

            with col_r:
                # P&L payoff diagram
                prices, pnl = StrategyEngine.payoff_data(rec)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prices, y=pnl,
                    mode="lines", name="P&L at Expiry",
                    line=dict(color="#00cc88" if rec.option_type == "call" else "#ff4444", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,204,136,0.1)" if rec.option_type == "call" else "rgba(255,68,68,0.1)",
                ))
                fig.add_hline(y=0, line_color="white", line_dash="dash")
                fig.add_vline(x=rec.underlying_price, line_color="yellow", line_dash="dot",
                              annotation_text="Spot")
                fig.add_vline(x=rec.strike, line_color="orange", line_dash="dot",
                              annotation_text="Strike")
                fig.update_layout(
                    title=f"P&L at Expiry – {rec.ticker} {rec.option_type.upper()} {rec.strike}",
                    xaxis_title="Underlying Price at Expiry",
                    yaxis_title="P&L per Contract ($)",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Auto-refresh countdown ────────────────────────────────────────────────────
if refresh_sec > 0:
    elapsed = time.time() - st.session_state["last_scan_time"]
    remaining = max(0, int(refresh_sec - elapsed))
    st.caption(f"⏱ Next refresh in {remaining}s")
    if remaining <= 0:
        st.rerun()

st.markdown("---")
st.caption("Data: yfinance (free) • Pricing: Black-Scholes • Greeks: Analytical • Not financial advice.")
