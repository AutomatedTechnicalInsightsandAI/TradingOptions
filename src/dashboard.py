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
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from src.greeks import call_payoff_at_expiry, put_payoff_at_expiry
from src.scanner import ScanResult, fetch_option_chain, rank_candidates, scan_watchlist
from src.strategy_engine import StrategyEngine, TradeRecommendation

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Flow sentiment helper
# ---------------------------------------------------------------------------

def _classify_flow_sentiment(
    price_chg: float, oi_chg: float, opt_type: str, oi_chg_pct: float = 0.0
) -> tuple[str, str, str]:
    """
    Classify options flow sentiment using the Price × OI-Change × Option-Type matrix.

    Returns (sentiment_label, badge_color, description).
    """
    rising_price = price_chg > 0
    rising_oi = oi_chg > 0

    # Flat / Coiled Spring check
    if abs(price_chg) < 0.001:
        if oi_chg_pct > 0.05:
            return (
                "⬛ Coiled Spring – Imminent Breakout",
                "#555555",
                "Price flat but OI rising sharply — a large move is loading",
            )
        return ("⬜ Neutral", "#cccccc", "No directional signal")

    if opt_type == "call":
        if rising_price and rising_oi:
            return (
                "🟢 Long Build-up (True Strength)",
                "#00cc44",
                "Fresh capital entering; sustainable move",
            )
        if rising_price and not rising_oi:
            return (
                "🟡 Short Covering (Hollow Rally)",
                "#cccc00",
                "Shorts buying back; no new buyers; potential bull trap",
            )
        if not rising_price and rising_oi:
            return (
                "💎 Bottom Fishing (Contrarian)",
                "#00bfff",
                "Calls bought on dip; bounce expected",
            )
        # falling price, falling OI, call
        return (
            "🟣 Long Unwinding – Calls",
            "#8844aa",
            "Longs exiting; no new buyers yet",
        )
    else:  # put
        if not rising_price and rising_oi:
            return (
                "🔴 Short Build-up (True Weakness)",
                "#cc0000",
                "New sellers entering; drop backed by new money",
            )
        if not rising_price and not rising_oi:
            return (
                "🟠 Long Unwinding (Bullish Pivot)",
                "#ff8800",
                "Forced liquidation flush; selling exhaustion near bottom",
            )
        if rising_price and not rising_oi:
            return (
                "🔵 Short Covering – Puts",
                "#0088ff",
                "Bears exiting; price pumped by covering",
            )
        # rising price, rising OI, put
        return (
            "⚠️ Bearish Entry (New Puts)",
            "#ff4444",
            "Traders buying puts into rally; top bet",
        )

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
    page_title="TradingOptions – Options Analytics Dashboard",
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

    search_ticker = st.text_input("➕ Add ticker to session", placeholder="e.g. MSTR, SPY …").strip().upper()
    if search_ticker and search_ticker not in tickers:
        tickers.append(search_ticker)
        st.caption(f"✅ {search_ticker} added to this session's watchlist")

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
st.title("📈 TradingOptions – Options Analytics Dashboard")

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

    # ── Tab 2: Options Flow Analysis ──────────────────────────────────────
    with tab2:
        st.subheader("📊 Options Flow Analysis")
        df = res.contracts.copy()

        # ── Section 1: Trend Overlay (Dual-Axis Chart) ─────────────────
        st.markdown("#### 📈 Section 1 — Price vs. Total OI Trend Overlay")
        try:
            import yfinance as yf
            tkr_hist = yf.Ticker(selected_ticker).history(period="1mo")
            price_series = tkr_hist["Close"].dropna() if not tkr_hist.empty else None
        except Exception:
            price_series = None

        total_oi_by_expiry = df.groupby("expiry")["open_interest"].sum().reset_index()
        total_oi_series = df.groupby("expiry")["open_interest"].sum()

        if price_series is not None and not price_series.empty:
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            fig_trend.add_trace(
                go.Scatter(
                    x=price_series.index,
                    y=price_series.values,
                    name="Underlying Price",
                    line=dict(color="#FFD700", width=2),
                    mode="lines",
                ),
                secondary_y=False,
            )

            # Use per-expiry total OI as a proxy time series
            if not total_oi_by_expiry.empty:
                fig_trend.add_trace(
                    go.Scatter(
                        x=total_oi_by_expiry["expiry"],
                        y=total_oi_by_expiry["open_interest"],
                        name="Total OI (by expiry)",
                        line=dict(color="#00BFFF", width=2),
                        mode="lines+markers",
                    ),
                    secondary_y=True,
                )

            # Determine annotation
            price_trend = price_series.iloc[-1] > price_series.iloc[0] if len(price_series) > 1 else True
            oi_total = total_oi_by_expiry["open_interest"].sum() if not total_oi_by_expiry.empty else 0
            oi_trend = oi_total > 0

            if price_trend and oi_trend:
                annotation_text = "✅ Long Build-up: True Strength"
            elif price_trend and not oi_trend:
                annotation_text = "⚠️ Hollow Rally / Bull Trap"
            elif not price_trend and oi_trend:
                annotation_text = "🔴 Short Build-up: True Weakness"
            else:
                annotation_text = "🟠 Bullish Pivot: Selling Exhaustion"

            fig_trend.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.01, y=0.97,
                showarrow=False,
                font=dict(size=13, color="white"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="gray",
                borderwidth=1,
            )
            fig_trend.update_layout(
                title=f"Price vs. Total OI — {selected_ticker}",
                template="plotly_dark",
                height=400,
                legend=dict(x=0.01, y=0.90),
            )
            fig_trend.update_yaxes(title_text="Underlying Price ($)", secondary_y=False)
            fig_trend.update_yaxes(title_text="Total Open Interest", secondary_y=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Price history unavailable for the trend overlay chart.")

        # ── Section 2: Vol/OI Bubble Heatmap ──────────────────────────
        st.markdown("#### 🔥 Section 2 — Vol/OI Bubble Heatmap (Unusual Activity Spotlight)")
        calls_df = df[df["option_type"] == "call"].copy()
        puts_df = df[df["option_type"] == "put"].copy()

        fig_bubble = go.Figure()
        if not calls_df.empty:
            calls_df["vol_oi_capped"] = calls_df["vol_oi_ratio"].clip(upper=50)
            fig_bubble.add_trace(go.Scatter(
                x=calls_df["strike"],
                y=calls_df["iv"] * 100,
                mode="markers",
                name="Calls",
                marker=dict(
                    size=calls_df["volume"].clip(upper=5000) / 50 + 4,
                    color=calls_df["vol_oi_capped"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Vol/OI", x=1.0),
                    line=dict(color="green", width=1),
                    opacity=0.8,
                ),
                text=calls_df.apply(
                    lambda r: f"Strike: {r['strike']}<br>IV: {r['iv']*100:.1f}%<br>"
                              f"Vol: {r['volume']}<br>OI: {r['open_interest']}<br>"
                              f"Vol/OI: {r['vol_oi_ratio']:.1f}",
                    axis=1,
                ),
                hoverinfo="text",
            ))
        if not puts_df.empty:
            puts_df["vol_oi_capped"] = puts_df["vol_oi_ratio"].clip(upper=50)
            fig_bubble.add_trace(go.Scatter(
                x=puts_df["strike"],
                y=puts_df["iv"] * 100,
                mode="markers",
                name="Puts",
                marker=dict(
                    size=puts_df["volume"].clip(upper=5000) / 50 + 4,
                    color=puts_df["vol_oi_capped"],
                    colorscale="RdYlGn_r",
                    showscale=False,
                    line=dict(color="red", width=1),
                    opacity=0.8,
                ),
                text=puts_df.apply(
                    lambda r: f"Strike: {r['strike']}<br>IV: {r['iv']*100:.1f}%<br>"
                              f"Vol: {r['volume']}<br>OI: {r['open_interest']}<br>"
                              f"Vol/OI: {r['vol_oi_ratio']:.1f}",
                    axis=1,
                ),
                hoverinfo="text",
            ))
        fig_bubble.add_vline(
            x=res.underlying_price,
            line_dash="dash", line_color="white",
            annotation_text=f"Spot ${res.underlying_price:,.0f}",
        )
        fig_bubble.update_layout(
            title=f"Vol/OI Bubble Map – Unusual Activity Spotlight",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility (%)",
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        # ── Section 3: Sentiment Distribution ─────────────────────────
        st.markdown("#### 🧠 Section 3 — Sentiment Distribution")

        if not df.empty:
            median_call_oi = df[df["option_type"] == "call"]["open_interest"].median() or 1
            median_put_oi = df[df["option_type"] == "put"]["open_interest"].median() or 1

            sentiments: list[str] = []
            for _, row in df.iterrows():
                opt_type = row["option_type"]
                vol_oi = row["vol_oi_ratio"]
                oi = row["open_interest"]
                d = row["delta"]
                median_oi = median_call_oi if opt_type == "call" else median_put_oi

                # Proxy for price direction: actual per-contract price history is not
                # available from the scanner. We use delta + vol/OI as a simplified
                # indicator: high-volume call activity with positive delta implies
                # the market is pricing in upside; put activity implies downside.
                # This is an approximation — treat the sentiment distribution as
                # directional indication, not a precise measurement.
                if opt_type == "call":
                    price_chg = 1.0 if (d > 0 and vol_oi > 1) else -1.0
                else:
                    price_chg = -1.0 if (abs(d) > 0 and vol_oi > 1) else 1.0

                # OI change proxy: above median = "rising"
                oi_chg = 1.0 if oi > median_oi else -1.0
                oi_chg_pct = (oi - median_oi) / (median_oi or 1)

                label, _, _ = _classify_flow_sentiment(price_chg, oi_chg, opt_type, oi_chg_pct)
                sentiments.append(label)

            sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            _sentiment_colors = {
                "🟢 Long Build-up (True Strength)": "#00cc44",
                "🟡 Short Covering (Hollow Rally)": "#cccc00",
                "🔴 Short Build-up (True Weakness)": "#cc0000",
                "🟠 Long Unwinding (Bullish Pivot)": "#ff8800",
                "🔵 Short Covering – Puts": "#0088ff",
                "🟣 Long Unwinding – Calls": "#8844aa",
                "⚠️ Bearish Entry (New Puts)": "#ff4444",
                "💎 Bottom Fishing (Contrarian)": "#00bfff",
                "⬛ Coiled Spring – Imminent Breakout": "#555555",
                "⬜ Neutral": "#cccccc",
            }
            bar_colors = [
                _sentiment_colors.get(s, "#888888")
                for s in sentiment_counts["Sentiment"]
            ]

            fig_sent = go.Figure(go.Bar(
                x=sentiment_counts["Count"],
                y=sentiment_counts["Sentiment"],
                orientation="h",
                marker_color=bar_colors,
                text=sentiment_counts["Count"],
                textposition="outside",
            ))
            fig_sent.update_layout(
                title="Options Flow Sentiment Distribution",
                xaxis_title="Contract Count",
                template="plotly_dark",
                height=max(300, len(sentiment_counts) * 45 + 80),
                margin=dict(l=10, r=40, t=50, b=10),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            st.markdown("""
**Sentiment Legend**

| Sentiment | Meaning |
|-----------|---------|
| 🟢 Long Build-up (True Strength) | Fresh capital entering; sustainable move |
| 🟡 Short Covering (Hollow Rally) | Shorts buying back; no new buyers; potential bull trap |
| 🔴 Short Build-up (True Weakness) | New sellers entering; drop backed by new money |
| 🟠 Long Unwinding (Bullish Pivot) | Forced liquidation flush; selling exhaustion near bottom |
| 🔵 Short Covering – Puts | Bears exiting; price pumped by covering |
| 🟣 Long Unwinding – Calls | Longs exiting; no new buyers yet |
| ⚠️ Bearish Entry (New Puts) | Traders buying puts into rally; top bet |
| 💎 Bottom Fishing (Contrarian) | Calls bought on dip; bounce expected |
| ⬛ Coiled Spring – Imminent Breakout | Price flat, OI rising sharply — big move loading |
| ⬜ Neutral | No directional signal |
""")

        # ── Section 4: Smart Money vs Day-Trader Table ─────────────────
        st.markdown("#### 🐳 Section 4 — Smart Money vs. Day-Trader Classifier")

        smart_df = res.uoa_alerts if not res.uoa_alerts.empty else df.nlargest(20, "vol_oi_ratio")
        if smart_df.empty:
            st.info("No contracts available for Smart Money classification.")
        else:
            median_oi_all = df["open_interest"].median() or 1

            def _classify_trader(row: pd.Series) -> str:
                if row["vol_oi_ratio"] > 5 and row["open_interest"] > median_oi_all:
                    return "🐳 Smart Money / Institutional"
                elif row["vol_oi_ratio"] > 5 and row["open_interest"] <= median_oi_all:
                    return "⚡ Day Trading / Scalping"
                else:
                    return "👤 Retail"

            smart_display = smart_df[[
                "strike", "option_type", "volume", "open_interest",
                "vol_oi_ratio", "iv", "delta",
            ]].copy()
            trader_classes = smart_df.apply(_classify_trader, axis=1).values
            smart_display.columns = [
                "Strike", "Type", "Volume", "OI", "Vol/OI", "IV (dec)", "Delta",
            ]
            smart_display.insert(5, "Trader Class", trader_classes)
            smart_display["IV%"] = (smart_display["IV (dec)"] * 100).round(1)
            smart_display = smart_display.drop(columns=["IV (dec)"])
            smart_display["Strike"] = smart_display["Strike"].map("${:.2f}".format)
            smart_display["Vol/OI"] = smart_display["Vol/OI"].map("{:.1f}".format)
            smart_display["Delta"] = smart_display["Delta"].map("{:.3f}".format)
            smart_display["IV%"] = smart_display["IV%"].map("{:.1f}%".format)
            st.dataframe(smart_display, use_container_width=True, hide_index=True)

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

                # Strategy badge
                _strat = rec.strategy_type
                if _strat in ("Bull Put Credit Spread", "Bear Call Credit Spread", "Iron Condor"):
                    st.warning(f"📋 Recommended Strategy: **{_strat}**")
                elif _strat in ("Long Call", "Long Put", "Long Straddle", "Long Strangle"):
                    st.success(f"📋 Recommended Strategy: **{_strat}**")
                else:
                    st.info(f"📋 Recommended Strategy: **{_strat}**")

                # Strategy legs explanation
                st.info(rec.strategy_legs)

                # IV regime warning
                if rec.iv_rank >= 50:
                    st.warning(
                        "⚠️ IV Rank is elevated. Buying naked options risks IV crush. "
                        "The recommended credit strategy above accounts for this."
                    )
                elif rec.iv_rank < 30:
                    st.success("✅ IV Rank is low — this is a favorable environment to buy premium.")

                # IV Rank metric
                _iv_r = rec.iv_rank
                if _iv_r < 30:
                    _iv_regime = "Low IV"
                elif _iv_r < 50:
                    _iv_regime = "Mod IV"
                elif _iv_r < 70:
                    _iv_regime = "High IV"
                else:
                    _iv_regime = "Very High IV"
                st.metric("IV Rank", f"{_iv_r:.0f}%", delta=_iv_regime)

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
                    title=f"P&L at Expiry – {rec.ticker} | {rec.strategy_type}",
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
