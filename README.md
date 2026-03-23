# TradingOptions 📈

A Python-based, high-performance options trading analytics tool targeting the **QQQ top-10 holdings**.  
It identifies mispriced volatility, unusual options activity (UOA), and optimal naked call/put setups using a modular data-to-execution pipeline.

---

## Architecture

```
TradingOptions/
├── config.yaml              # Watchlist, scanner params, strategy settings
├── requirements.txt         # Python dependencies
├── src/
│   ├── greeks.py            # Black-Scholes pricing, Greeks, IV solver, HV, IV Rank
│   ├── scanner.py           # yfinance chain ingestion, UOA detection, OI walls, ranking
│   ├── strategy_engine.py   # Naked call/put recommendations, PoP, R:R, position sizing
│   └── dashboard.py         # Streamlit interactive dashboard
└── tests/
    ├── test_greeks.py        # 40 unit tests for the maths engine
    └── test_scanner.py       # 28 unit tests for scanning logic (mocked network)
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `greeks.py` | Black-Scholes formula, analytical Greeks (Δ, Γ, Θ, V, ρ), Newton-Raphson IV solver, Historical Volatility, IV Rank/Percentile, P&L payoff arrays |
| `scanner.py` | Fetches option chains via **yfinance** (free), enriches each contract with Greeks/IV, flags UOA (Vol/OI ≥ 5×), detects OI walls, and ranks candidates |
| `strategy_engine.py` | Filters ranked candidates by directional/vol view, checks PoP range (30–65%) and R:R ≥ 2:1, sizes positions by account risk, builds rationale strings |
| `dashboard.py` | Streamlit app with auto-refresh, volatility smile chart, OI/Volume chart, full options chain table, UOA alerts, and interactive P&L payoff diagrams |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the interactive dashboard

```bash
streamlit run src/dashboard.py
```

Then open **http://localhost:8501** in your browser.

- Select your watchlist, market view (bullish/bearish), and account settings in the sidebar.
- Click **Run Scan** – the scanner fetches live option chains from Yahoo Finance, calculates Greeks and IV, and populates all charts and tables.
- Set **Auto-refresh** seconds for continuous monitoring.

### 3. Use the modules directly

```python
from src.scanner import fetch_option_chain, rank_candidates
from src.strategy_engine import StrategyEngine

# Scan a single ticker
result = fetch_option_chain("NVDA", dte_range=(7, 45), min_volume=100)
print(f"ATM IV: {result.atm_iv*100:.1f}%  IV Rank: {result.iv_rank_value*100:.0f}%")
print(f"OI Walls at: {result.oi_walls}")
print(result.uoa_alerts[["strike","option_type","vol_oi_ratio","iv"]].head())

# Rank across multiple tickers
from src.scanner import scan_watchlist
results = scan_watchlist(["QQQ","NVDA","TSLA"])
ranked = rank_candidates(results, top_n=10)
print(ranked[["ticker","type","strike","iv","delta","score"]].to_string())

# Get trade recommendations
engine = StrategyEngine(market_view="bullish", account_size=25000, risk_per_trade=0.02)
recs = engine.recommend(ranked)
for r in recs:
    contracts = engine.position_size(r.entry_price)
    print(f"{r.ticker} {r.strike} {r.option_type} | PoP={r.pop*100:.0f}% | {contracts}x contracts | {r.rationale}")
```

---

## Key Features

### Black-Scholes Engine (`greeks.py`)
- European option pricing: `C = S·N(d₁) − K·e^{−rT}·N(d₂)`
- Full analytical Greeks: Delta, Gamma, Theta (per calendar day), Vega (per unit σ), Rho
- **Implied Volatility solver** – Newton-Raphson with bisection fallback
- **Historical Volatility** – close-to-close log-return method (annualised)
- **IV Rank** – where current IV sits in its 52-week range [0,1]
- **IV Percentile** – fraction of past readings below current IV

### Options Chain Scanner (`scanner.py`)
- Fetches live data via **yfinance** (no API key required)
- Enriches every contract with IV, Δ, Γ, Θ, V
- **UOA detection**: flags contracts where `Volume / Open Interest ≥ 5×`
- **OI Wall identification**: top 5 strike price levels by aggregate open interest
- Configurable filters: DTE range, minimum volume, maximum bid-ask spread %
- Composite scoring: UOA weight + IV Rank + IV/HV ratio + extreme UOA bonus

### Strategy Engine (`strategy_engine.py`)
- Directional filters: bullish → long calls, bearish → long puts
- Volatility view filters: prefers contracts where IV > HV (high vol view) or IV < HV (cheap options)
- **Probability of Profit** approximation: `PoP ≈ 1 − |Δ|` for calls, `|Δ|` for puts
- **Reward-to-Risk gate**: R:R ≥ 2:1 at a 2× premium target
- **Position sizing**: `floor(account × risk_pct / (entry × 100))`

### Streamlit Dashboard (`dashboard.py`)
- **Market Overview** table: Price, HV30, ATM IV, IV Rank, IV %ile, IV/HV, OI walls
- **Ranked Candidates** table with UOA highlights
- **Volatility Smile chart**: IV vs Strike for up to 3 nearest expiries (Plotly)
- **OI & Volume charts**: Call/Put open interest and volume by strike with OI wall markers
- **Full Options Chain** tab with sortable Greeks columns
- **UOA Alerts** tab with extreme activity highlighted
- **Trade Recommendations** cards with: entry price, max loss, PoP, contract count, total risk, Greeks table, and **interactive P&L payoff diagram**
- Configurable auto-refresh (default every 5 minutes)

---

## Configuration (`config.yaml`)

```yaml
watchlist:
  - QQQ
  - AAPL
  - MSFT
  - NVDA
  - AMZN
  # ... (customise as needed)

scanner:
  uoa_vol_oi_ratio: 5.0      # Flag if Volume/OI ≥ this
  min_volume: 50
  max_spread_pct: 0.20       # Maximum bid-ask spread as % of mid
  dte_range: [7, 60]         # Days-to-expiry window

strategy:
  min_reward_risk: 2.0
  pop_range: [0.35, 0.65]
  account_size: 25000
  risk_per_trade: 0.02       # 2% risk per trade

greeks:
  risk_free_rate: 0.053      # Annualised risk-free rate
  hv_window: 30              # Historical volatility look-back (trading days)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All 68 tests should pass. Tests for the scanner mock `yfinance` so no network connection is needed.

---

## Competitive Edge

This tool finds edge cases where **Implied Volatility is severely dislocated from historical norms**:

| Signal | What it means |
|--------|---------------|
| IV Rank > 70% | Options are historically expensive – breakout potential OR premium-selling opportunity |
| IV/HV > 1.3× | Market is pricing in significantly more risk than realised – often precedes large moves |
| UOA Vol/OI > 5× | Smart-money positioning or block trades – directional catalyst anticipated |
| OI Walls | Large OI clusters act as price magnets or resistance levels near expiry (gamma pinning) |

---

## Data Source & Limitations

- **Data**: Yahoo Finance via `yfinance` – free, no API key, ~15-minute delay on options data
- **IV History**: Approximated from rolling 30-day HV (true IV history requires paid data)
- **Pricing model**: Black-Scholes (European). American options have early-exercise premium not modelled
- **This tool is for educational and research purposes only. It is not financial advice.**

---

## Broker API Integration

To route orders from recommendations, connect a broker API:

| Broker | Library | Notes |
|--------|---------|-------|
| **Tradier** | `requests` (REST) | Options-friendly, free paper trading |
| **TD Ameritrade / Schwab** | `schwab-py` | Full options support |
| **Interactive Brokers** | `ib_insync` | Professional-grade, low commissions |
| **Alpaca** | `alpaca-trade-api` | Free paper trading, stocks only |

Example integration point in `strategy_engine.py`: call the broker's order endpoint inside `StrategyEngine.recommend()` after all gates pass.
