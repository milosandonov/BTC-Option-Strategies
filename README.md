# BTC Covered Call Strategy — Systematic Backtest

A quantitative backtest of a **Bitcoin covered call (call overwriting) strategy** over January 2024 – December 2025. The project prices 3-week, 20-delta call options using **Black-76** (forward-based pricing), implements four strategy variants, and produces a comprehensive performance report.

![Strategy Report](BTC_Option_Stragegy_Project.png)

---

## Overview

The core idea: hold 1 BTC and systematically sell short-dated, out-of-the-money call options to collect premium. The backtest explores whether signal conditioning (volatility regime filters, dynamic delta selection) can improve risk-adjusted returns over a naive baseline.

**Result:** The combined strategy generated **$1,104K total P&L** vs **$936K for long BTC**, delivering **+$168K alpha** with a higher Sharpe ratio (1.08 vs 0.97) and lower maximum drawdown (-$505K vs -$632K).

---

## Strategy Variants

| Strategy | Logic |
|---|---|
| **Baseline (20D)** | Sell 3W 20-delta call every day — no signal conditioning |
| **VRP Filter** | Only sell when Volatility Risk Premium (IV − trailing RV) < 12%; skip otherwise |
| **Dynamic Delta** | Adjust call delta based on IV regime (25D when IV < 42%, 20D mid, 25D when IV ≥ 70%) |
| **Combined** *(recommended)* | Apply VRP filter first, then select delta via IV regime |

---

## Methodology

### Option Pricing
- **Model:** Black-76 (futures/forward based) — avoids spot/forward conflation
- **Strike Selection:** Brent's method to invert Black-76 delta for the target delta
- **Greeks:** Delta, Gamma, Theta, Vega computed analytically

### IV Interpolation (Skew-Aware)
Raw Deribit data provides IV at 1W/1M tenors and 15Δ/25Δ points. Two-step bilinear interpolation:
1. Linear interpolation in **delta space** (15Δ → 25Δ) to get 20Δ IV per tenor
2. Linear interpolation in **√T space** (variance scaling) from 1W → 1M to get 3W IV

### Signal Construction (No Look-Ahead)
- **VRP:** `IV(3W, 20D) − RV(trailing 21d)` — uses only past realized volatility
- **IV Regime:** Based on current IV level at entry date only
- All signals are backward-looking; no future data leaks into entry decisions

### P&L Attribution
Each trade is a 21-day position:
```
Strategy P&L = BTC P&L + Short Call P&L
             = (S_expiry − S_entry) + (Premium − max(0, S_expiry − K))
```

---

## Key Results

| Metric | Baseline (20D) | VRP Filter | Dynamic Delta | **Combined** | Long BTC |
|---|---|---|---|---|---|
| Total P&L | $918K | $1,087K | $946K | **$1,104K** | $936K |
| Alpha vs BTC | −$17K | +$151K | +$11K | **+$168K** | — |
| Sharpe Ratio | 1.11 | 1.07 | 1.15 | **1.08** | 0.97 |
| Max Drawdown | −$505K | −$514K | −$490K | **−$505K** | −$632K |
| Win Rate | 56.6% | 55.0% | 57.3% | **55.3%** | 52.8% |
| Option Trades | 709 | 426 | 709 | **426** | — |
| Premium Collected | $756K | $441K | $824K | **$483K** | — |
| Called Away | 17.8% | 16.0% | 18.6% | **16.7%** | — |

---

## Project Structure

```
btc-covered-call/
├── btc_covered_call_optimized.py   # Main backtest engine
├── BTC_Option_Stragegy_Project.png # Output: strategy optimization report
├── BTC_Covered_Call_Analysis.docx  # Written analysis and findings
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scipy matplotlib
```

Data input: a `.pkl` file containing daily Deribit IV surface data with columns:
`timestamp`, `spot`, `tenor` (1W/1M), `delta_bucket` (0.15/0.25), `iv`, `underlying_price`

---

## Usage

```python
# Update DATA_PATH in __main__ to point to your .pkl file
python btc_covered_call_optimized.py
```

Output: a multi-panel PNG report covering cumulative P&L, alpha vs BTC, drawdown, P&L distribution, rolling Sharpe, IV/RV dynamics, and VRP threshold robustness analysis.

---

## Design Principles

- **No look-ahead bias** — all signals use only information available at entry date
- **Forward-based pricing** — Black-76 throughout, using interpolated 3W forward
- **Skew-aware IV** — IV interpolated per-delta independently before tenor interpolation
- **Robustness-tested** — VRP threshold stable across 8–15% range (all produce positive alpha)
- **Minimal free parameters** — only two regime thresholds (VRP: 12%, IV: 42%/70%)

---

## Findings Summary

The 2024–2025 period was strongly bullish for BTC (~$42K → $100K+), making this a challenging environment for covered calls due to frequent upside capping. Despite this headwind:

- **VRP filtering** was the highest-impact improvement, adding $168K alpha by avoiding selling when the market was correctly pricing large moves
- **Dynamic delta** adds modest alpha but shines in its risk management — lower drawdown and higher win rate in extreme IV regimes
- The strategy is **most effective in sideways or mildly bullish markets** and provides meaningful downside cushion during corrections

---

## Author

Milos | February 2026  
*Data sourced from Deribit options market (IV surfaces, 1W/1M tenors, 15Δ/25Δ)*
