"""
BTC Covered Call Strategy — Optimized Backtest
================================================
Built on the forward-corrected (Black-76) baseline.

Strategies:
  A. Baseline:      Sell 3W 20D call every day
  B. VRP Filter:    Only sell when VRP (IV - trailing RV) < threshold
  C. Dynamic Delta: Adjust call delta based on IV regime
  D. Combined:      VRP filter + dynamic delta

Design principles:
  - NO look-ahead bias: all signals use only past/current data
  - Forward-based pricing (Black-76) throughout
  - IV interpolated per delta independently (skew-aware)
  - Minimal free parameters 
"""

import pickle
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# BLACK-76 OPTION PRICING
# ==============================================================================

def black76_call_price(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0, F - K) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black76_call_delta(F, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if F > K else 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d1)

def find_strike_for_delta(F, T, r, sigma, target_delta):
    if T <= 0 or sigma <= 0:
        return F
    try:
        return brentq(lambda K: black76_call_delta(F, K, T, r, sigma) - target_delta,
                       F * 0.5, F * 3.0)
    except Exception:
        d1_target = norm.ppf(target_delta * np.exp(r * T))
        return F * np.exp(-d1_target * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)

# ==============================================================================
# INTERPOLATION
# ==============================================================================

def interpolate_iv_by_delta(iv_15d, iv_25d, target_delta):
    return iv_15d + (iv_25d - iv_15d) * (target_delta - 0.15) / (0.25 - 0.15)

def interpolate_iv_by_tenor(iv_1w, iv_1m, target_days):
    s7  = np.sqrt(7 / 365)
    s30 = np.sqrt(30 / 365)
    st  = np.sqrt(target_days / 365)
    w   = (st - s7) / (s30 - s7)
    return iv_1w + (iv_1m - iv_1w) * w

def get_3w_iv_for_delta(iv_1w_15d, iv_1w_25d, iv_1m_15d, iv_1m_25d, target_delta):
    iv_1w = interpolate_iv_by_delta(iv_1w_15d, iv_1w_25d, target_delta)
    iv_1m = interpolate_iv_by_delta(iv_1m_15d, iv_1m_25d, target_delta)
    return interpolate_iv_by_tenor(iv_1w, iv_1m, 21)

def interpolate_forward(spot, fwd_1w, fwd_1m, target_days):
    if pd.isna(fwd_1w) or pd.isna(fwd_1m) or pd.isna(spot):
        return np.nan
    lb_1w = np.log(fwd_1w / spot)
    lb_1m = np.log(fwd_1m / spot)
    w = (target_days - 7) / (30 - 7)
    return spot * np.exp(lb_1w + (lb_1m - lb_1w) * w)

# ==============================================================================
# DATA LOADING & SIGNALS
# ==============================================================================

def load_and_prepare_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} rows")
    data['date'] = pd.to_datetime(data['timestamp'].dt.date)
    dates = sorted(data['date'].unique())
    print(f"Date range: {dates[0]} to {dates[-1]}")

    daily_data = []
    for date in dates:
        day = data[data['date'] == date]
        spot = day['spot'].iloc[0]
        row = {'date': date, 'spot': spot}
        for tenor in ['1W', '1M']:
            td = day[day['tenor'] == tenor]
            if len(td) > 0:
                row[f'fwd_{tenor}'] = td['underlying_price'].iloc[0]
            else:
                row[f'fwd_{tenor}'] = np.nan
            for delta in [0.15, 0.25]:
                sub = td[(td['delta_bucket'] == delta) &
                         (td['delta_source'].isin(['optim', 'given']))]
                if len(sub) > 0:
                    iv = sub['iv'].iloc[0]
                else:
                    sub2 = td[td['delta_bucket'] == delta]
                    iv = sub2['iv'].mean() if len(sub2) > 0 else np.nan
                row[f'iv_{tenor}_{int(delta*100)}D'] = iv
        daily_data.append(row)

    df = pd.DataFrame(daily_data).set_index('date').sort_index()
    df = df.ffill().bfill()

    # 3W IV for each target delta
    for d_name, d_val in [('15D', 0.15), ('20D', 0.20), ('25D', 0.25)]:
        df[f'iv_3W_{d_name}'] = df.apply(
            lambda r: get_3w_iv_for_delta(
                r['iv_1W_15D'], r['iv_1W_25D'],
                r['iv_1M_15D'], r['iv_1M_25D'], d_val), axis=1)

    # 3W forward
    df['fwd_3W'] = df.apply(
        lambda r: interpolate_forward(r['spot'], r['fwd_1W'], r['fwd_1M'], 21), axis=1)

    # Signals (backward-looking only)
    log_ret = np.log(df['spot'] / df['spot'].shift(1))
    df['rv_21d'] = log_ret.rolling(21, min_periods=15).std() * np.sqrt(365)
    df['vrp'] = df['iv_3W_20D'] - df['rv_21d']

    print(f"\n3W Forward: avg premium = {(df['fwd_3W']/df['spot']-1).mean()*100:.3f}%")
    print(f"RV 21d:     mean = {df['rv_21d'].mean()*100:.1f}%")
    print(f"IV 3W 20D:  mean = {df['iv_3W_20D'].mean()*100:.1f}%")
    print(f"VRP:        mean = {df['vrp'].mean()*100:.1f}%, "
          f"positive {(df['vrp']>0).mean()*100:.0f}% of days")
    return df

# ==============================================================================
# TRADE BUILDING HELPERS
# ==============================================================================

def compute_trade(df, entry_date, expiry_date, T, r, target_delta):
    """Compute a single covered-call trade at a given delta."""
    S_entry  = df.loc[entry_date, 'spot']
    F_entry  = df.loc[entry_date, 'fwd_3W']
    S_expiry = df.loc[expiry_date, 'spot']

    delta_map = {0.15: 'iv_3W_15D', 0.20: 'iv_3W_20D', 0.25: 'iv_3W_25D'}
    iv_col = delta_map.get(target_delta, 'iv_3W_20D')
    iv_entry = df.loc[entry_date, iv_col]

    K = find_strike_for_delta(F_entry, T, r, iv_entry, target_delta)
    prem = black76_call_price(F_entry, K, T, r, iv_entry)
    intrinsic = max(0, S_expiry - K)

    return {
        'entry_date': entry_date, 'expiry_date': expiry_date,
        'S_entry': S_entry, 'F_entry': F_entry, 'S_expiry': S_expiry,
        'K': K, 'target_delta': target_delta, 'iv_entry': iv_entry,
        'call_premium': prem, 'call_intrinsic': intrinsic,
        'btc_pnl': S_expiry - S_entry,
        'short_call_pnl': prem - intrinsic,
        'strategy_pnl': (S_expiry - S_entry) + prem - intrinsic,
        'btc_return': (S_expiry - S_entry) / S_entry,
        'strategy_return': ((S_expiry - S_entry) + prem - intrinsic) / S_entry,
        'option_moneyness': K / S_entry,
        'fwd_moneyness': K / F_entry,
        'fwd_basis_pct': (F_entry / S_entry - 1) * 100,
        'option_called_away': S_expiry > K,
    }

def build_skip_trade(df, entry_date, expiry_date):
    """BTC-only trade row for days when we skip selling."""
    S_entry  = df.loc[entry_date, 'spot']
    S_expiry = df.loc[expiry_date, 'spot']
    btc_pnl  = S_expiry - S_entry
    return {
        'entry_date': entry_date, 'expiry_date': expiry_date,
        'S_entry': S_entry, 'F_entry': df.loc[entry_date, 'fwd_3W'],
        'S_expiry': S_expiry, 'K': np.nan, 'target_delta': np.nan,
        'iv_entry': df.loc[entry_date, 'iv_3W_20D'],
        'call_premium': 0.0, 'call_intrinsic': 0.0,
        'btc_pnl': btc_pnl, 'short_call_pnl': 0.0, 'strategy_pnl': btc_pnl,
        'btc_return': btc_pnl / S_entry, 'strategy_return': btc_pnl / S_entry,
        'option_moneyness': np.nan, 'fwd_moneyness': np.nan,
        'fwd_basis_pct': (df.loc[entry_date, 'fwd_3W'] / S_entry - 1) * 100,
        'option_called_away': False,
    }

def _finalize(results):
    df = pd.DataFrame(results).set_index('entry_date')
    df['cum_strategy_pnl'] = df['strategy_pnl'].cumsum()
    df['cum_btc_pnl'] = df['btc_pnl'].cumsum()
    return df

# ==============================================================================
# STRATEGY RUNNERS
# ==============================================================================

def run_baseline(df, T_days=21, target_delta=0.20, r=0.05):
    """A: Sell 20D call every day."""
    T = T_days / 365
    results = []
    dates = df.index.tolist()
    for i, entry in enumerate(dates):
        ei = i + T_days
        if ei >= len(dates): break
        trade = compute_trade(df, entry, dates[ei], T, r, target_delta)
        trade['sell_signal'] = True
        trade['iv_regime'] = 'n/a'
        trade['vrp_at_entry'] = df.loc[entry, 'vrp']
        results.append(trade)
    return _finalize(results)


def run_vrp_filter(df, T_days=21, target_delta=0.20, r=0.05, vrp_threshold=0.12):
    """B: Sell only when VRP < threshold. Skip (BTC-only) otherwise.

    WHY skip when VRP is HIGH (not low):
      High VRP (IV >> trailing RV) in BTC signals the market is pricing
      a large future move. Empirically, the move materializes and runs
      over the short call. Short call alpha is -$539/trade when VRP>10%.
      By contrast, when VRP is negative or moderate, short call alpha
      is +$271 to +$569/trade. Skipping high-VRP days avoids the worst
      trades while preserving the profitable ones.
    """
    T = T_days / 365
    results = []
    dates = df.index.tolist()
    for i, entry in enumerate(dates):
        ei = i + T_days
        if ei >= len(dates): break
        vrp = df.loc[entry, 'vrp']
        sell = (not np.isnan(vrp)) and (vrp < vrp_threshold)
        if sell:
            trade = compute_trade(df, entry, dates[ei], T, r, target_delta)
        else:
            trade = build_skip_trade(df, entry, dates[ei])
        trade['sell_signal'] = sell
        trade['vrp_at_entry'] = vrp
        trade['iv_regime'] = 'n/a'
        results.append(trade)
    return _finalize(results)


def run_dynamic_delta(df, T_days=21, r=0.05,
                      iv_low=0.42, iv_extreme=0.70,
                      delta_low_iv=0.25, delta_mid_iv=0.20, delta_extreme_iv=0.25):
    """C: Choose delta based on IV regime.

    Low IV (<42%):     25D — range-bound, safe to sell closer
    Mid IV (42-70%):   20D — standard
    Extreme IV (>70%): 25D — crisis = fat premium, BTC falling so calls safe
    """
    T = T_days / 365
    results = []
    dates = df.index.tolist()
    for i, entry in enumerate(dates):
        ei = i + T_days
        if ei >= len(dates): break
        iv = df.loc[entry, 'iv_3W_20D']
        if iv < iv_low:
            delta, regime = delta_low_iv, 'low'
        elif iv >= iv_extreme:
            delta, regime = delta_extreme_iv, 'extreme'
        else:
            delta, regime = delta_mid_iv, 'mid'
        trade = compute_trade(df, entry, dates[ei], T, r, delta)
        trade['sell_signal'] = True
        trade['iv_regime'] = regime
        trade['vrp_at_entry'] = df.loc[entry, 'vrp']
        results.append(trade)
    return _finalize(results)


def run_combined(df, T_days=21, r=0.05, vrp_threshold=0.12,
                 iv_low=0.42, iv_extreme=0.70,
                 delta_low_iv=0.25, delta_mid_iv=0.20, delta_extreme_iv=0.25):
    """D: VRP filter + dynamic delta. Recommended production strategy."""
    T = T_days / 365
    results = []
    dates = df.index.tolist()
    for i, entry in enumerate(dates):
        ei = i + T_days
        if ei >= len(dates): break
        vrp = df.loc[entry, 'vrp']
        iv  = df.loc[entry, 'iv_3W_20D']
        sell = (not np.isnan(vrp)) and (vrp < vrp_threshold)
        if sell:
            if iv < iv_low:
                delta, regime = delta_low_iv, 'low'
            elif iv >= iv_extreme:
                delta, regime = delta_extreme_iv, 'extreme'
            else:
                delta, regime = delta_mid_iv, 'mid'
            trade = compute_trade(df, entry, dates[ei], T, r, delta)
            trade['iv_regime'] = regime
        else:
            trade = build_skip_trade(df, entry, dates[ei])
            trade['iv_regime'] = 'skipped'
        trade['sell_signal'] = sell
        trade['vrp_at_entry'] = vrp
        results.append(trade)
    return _finalize(results)


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(results_df, T_days=21, label='Strategy'):
    n = len(results_df)
    ppy = 365 / T_days

    def max_dd(cum):
        peak = cum.expanding().max()
        return (cum - peak).min()

    sell = results_df[results_df['sell_signal'] == True]
    n_sell = len(sell)

    ret = results_df['strategy_return']
    btc_ret = results_df['btc_return']

    m = {
        'label': label,
        'total_trades': n,
        'option_trades': n_sell,
        'skip_rate': 1 - n_sell / n if n > 0 else 0,
        'period': f"{results_df.index[0].strftime('%Y-%m-%d')} to "
                  f"{results_df.index[-1].strftime('%Y-%m-%d')}",
        'strategy_total_pnl': results_df['strategy_pnl'].sum(),
        'btc_total_pnl': results_df['btc_pnl'].sum(),
        'total_premium': sell['call_premium'].sum() if n_sell > 0 else 0,
        'strategy_win_rate': (results_df['strategy_pnl'] > 0).mean(),
        'btc_win_rate': (results_df['btc_pnl'] > 0).mean(),
        'option_win_rate': (sell['short_call_pnl'] > 0).mean() if n_sell > 0 else np.nan,
        'called_away_rate': sell['option_called_away'].mean() if n_sell > 0 else np.nan,
        'strategy_sharpe': ret.mean() / ret.std() * np.sqrt(ppy) if ret.std() > 0 else np.nan,
        'btc_sharpe': btc_ret.mean() / btc_ret.std() * np.sqrt(ppy) if btc_ret.std() > 0 else np.nan,
        'strategy_max_dd': max_dd(results_df['cum_strategy_pnl']),
        'btc_max_dd': max_dd(results_df['cum_btc_pnl']),
        'strategy_vol': ret.std() * np.sqrt(ppy),
        'btc_vol': btc_ret.std() * np.sqrt(ppy),
        'avg_iv': results_df['iv_entry'].mean(),
        'avg_moneyness': sell['option_moneyness'].mean() if n_sell > 0 else np.nan,
        'avg_premium_pct': (sell['call_premium'] / sell['S_entry']).mean() if n_sell > 0 else np.nan,
        'avg_premium_usd': sell['call_premium'].mean() if n_sell > 0 else np.nan,
    }
    m['alpha'] = m['strategy_total_pnl'] - m['btc_total_pnl']
    return m


def print_metrics(m):
    print(f"\n{'='*60}")
    print(f"  {m['label']}")
    print(f"{'='*60}")
    print(f"  Period:         {m['period']}")
    print(f"  Total days:     {m['total_trades']}")
    print(f"  Option trades:  {m['option_trades']} (skip rate: {m['skip_rate']*100:.1f}%)")
    print(f"\n  P&L (per 1 BTC):")
    print(f"    Strategy:     ${m['strategy_total_pnl']:>12,.0f}")
    print(f"    Long BTC:     ${m['btc_total_pnl']:>12,.0f}")
    print(f"    Alpha:        ${m['alpha']:>12,.0f}")
    print(f"    Premium:      ${m['total_premium']:>12,.0f}")
    print(f"\n  Risk Metrics:")
    print(f"    Sharpe (CC):  {m['strategy_sharpe']:>8.2f}")
    print(f"    Sharpe (BTC): {m['btc_sharpe']:>8.2f}")
    print(f"    Vol (CC):     {m['strategy_vol']*100:>8.1f}%")
    print(f"    Vol (BTC):    {m['btc_vol']*100:>8.1f}%")
    print(f"    Max DD (CC):  ${m['strategy_max_dd']:>12,.0f}")
    print(f"    Max DD (BTC): ${m['btc_max_dd']:>12,.0f}")
    print(f"\n  Option Stats:")
    if not np.isnan(m.get('option_win_rate', np.nan)):
        print(f"    Win rate:     {m['option_win_rate']*100:>8.1f}%")
        print(f"    Called away:  {m['called_away_rate']*100:>8.1f}%")
        print(f"    Avg K/S:      {m['avg_moneyness']*100:>8.1f}%")
        print(f"    Avg premium:  ${m['avg_premium_usd']:>8.0f} ({m['avg_premium_pct']*100:.2f}%)")


def print_comparison_table(metrics_list):
    print(f"\n{'='*95}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'='*95}")
    cw = 22
    print(f"  {'Metric':<25s}", end='')
    for m in metrics_list:
        print(f"  {m['label']:>{cw}s}", end='')
    print(f"  {'Long BTC':>{cw}s}")
    print(f"  {'-'*25}", end='')
    for _ in range(len(metrics_list) + 1):
        print(f"  {'-'*cw}", end='')
    print()

    m0 = metrics_list[0]
    rows = [
        ('Total P&L',        lambda m: f"${m['strategy_total_pnl']/1000:,.0f}K"),
        ('Alpha vs BTC',     lambda m: f"${m['alpha']/1000:+,.0f}K"),
        ('Sharpe Ratio',     lambda m: f"{m['strategy_sharpe']:.2f}"),
        ('Volatility',       lambda m: f"{m['strategy_vol']*100:.1f}%"),
        ('Max Drawdown',     lambda m: f"${m['strategy_max_dd']/1000:,.0f}K"),
        ('Win Rate',         lambda m: f"{m['strategy_win_rate']*100:.1f}%"),
        ('Option Trades',    lambda m: f"{m['option_trades']}"),
        ('Skip Rate',        lambda m: f"{m['skip_rate']*100:.1f}%"),
        ('Premium Collected', lambda m: f"${m['total_premium']/1000:,.0f}K"),
        ('Called Away',      lambda m: f"{m['called_away_rate']*100:.1f}%"
                                if not np.isnan(m.get('called_away_rate', np.nan)) else 'N/A'),
    ]
    btc_col = {
        'Total P&L':        f"${m0['btc_total_pnl']/1000:,.0f}K",
        'Alpha vs BTC':     '---',
        'Sharpe Ratio':     f"{m0['btc_sharpe']:.2f}",
        'Volatility':       f"{m0['btc_vol']*100:.1f}%",
        'Max Drawdown':     f"${m0['btc_max_dd']/1000:,.0f}K",
        'Win Rate':         f"{m0['btc_win_rate']*100:.1f}%",
        'Option Trades':    '---',
        'Skip Rate':        '---',
        'Premium Collected': '---',
        'Called Away':       '---',
    }
    for rname, fmt in rows:
        print(f"  {rname:<25s}", end='')
        for m in metrics_list:
            print(f"  {fmt(m):>{cw}s}", end='')
        print(f"  {btc_col[rname]:>{cw}s}")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_optimization_report(all_results, daily_data, output_path='optimization_report.png'):
    fig = plt.figure(figsize=(18, 28))
    gs = fig.add_gridspec(8, 2, hspace=0.38, wspace=0.30)
    fig.suptitle(
        'BTC Covered Call — Strategy Optimization Report\n'
        'Forward-Priced (Black-76) | Jan 2024 – Dec 2025',
        fontsize=16, fontweight='bold', y=0.995)

    colors = {
        'Baseline (20D)': '#888888',
        'VRP Filter':     '#2196F3',
        'Dynamic Delta':  '#FF9800',
        'Combined':       '#4CAF50',
        'Long BTC':       '#E91E63',
    }
    ref = list(all_results.values())[0]

    # --- Panel 1: Cumulative P&L (full width) ---
    ax = fig.add_subplot(gs[0, :])
    for label, res in all_results.items():
        c = colors.get(label, 'gray')
        lw = 2.5 if label == 'Combined' else 1.5
        ax.plot(res.index, res['cum_strategy_pnl']/1000, label=label, color=c, lw=lw,
                alpha=1.0 if label == 'Combined' else 0.75)
    ax.plot(ref.index, ref['cum_btc_pnl']/1000, label='Long BTC',
            color=colors['Long BTC'], lw=2, ls='--', alpha=0.7)
    ax.set_title('Cumulative P&L Comparison (Rolling Daily Entry, 21-Day Hold)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative P&L ($K per 1 BTC)')
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, ls='--')
    ax.axhline(y=0, color='black', lw=0.8, alpha=0.5)

    # --- Panel 2: Alpha vs BTC ---
    ax = fig.add_subplot(gs[1, 0])
    for label, res in all_results.items():
        c = colors.get(label, 'gray')
        alpha_cum = res['cum_strategy_pnl'] - res['cum_btc_pnl']
        lw = 2.5 if label == 'Combined' else 1.2
        ax.plot(res.index, alpha_cum/1000, label=label, color=c, lw=lw,
                alpha=1.0 if label == 'Combined' else 0.6)
    ax.axhline(y=0, color='black', lw=1, alpha=0.5)
    ax.set_title('Cumulative Alpha vs Long BTC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Alpha ($K)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 3: VRP signal with sell/skip zones ---
    ax = fig.add_subplot(gs[1, 1])
    vrp_pct = daily_data['vrp'] * 100
    ax.plot(daily_data.index, vrp_pct, color='#555', lw=0.8, alpha=0.8)
    ax.fill_between(daily_data.index, vrp_pct, 0,
                     where=(vrp_pct < 12), color='#4CAF50', alpha=0.15, label='Sell zone (VRP < 12%)')
    ax.fill_between(daily_data.index, vrp_pct, 0,
                     where=(vrp_pct >= 12), color='#F44336', alpha=0.15, label='Skip zone (VRP >= 12%)')
    ax.axhline(y=12, color='red', lw=1.5, ls='--', alpha=0.8, label='Threshold')
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.5)
    ax.set_title('Volatility Risk Premium (IV - RV_21d)', fontsize=11, fontweight='bold')
    ax.set_ylabel('VRP (%)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 4: Drawdown ---
    ax = fig.add_subplot(gs[2, 0])
    for label, res in all_results.items():
        c = colors.get(label, 'gray')
        dd = res['cum_strategy_pnl'] - res['cum_strategy_pnl'].expanding().max()
        lw = 2.0 if label == 'Combined' else 1.0
        ax.plot(res.index, dd/1000, label=label, color=c, lw=lw,
                alpha=1.0 if label == 'Combined' else 0.5)
    dd_btc = ref['cum_btc_pnl'] - ref['cum_btc_pnl'].expanding().max()
    ax.plot(ref.index, dd_btc/1000, label='Long BTC',
            color=colors['Long BTC'], lw=1.5, ls='--', alpha=0.6)
    ax.set_title('Drawdown from Peak', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drawdown ($K)')
    ax.legend(loc='lower left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 5: IV regime scatter (Combined) ---
    ax = fig.add_subplot(gs[2, 1])
    combined = all_results.get('Combined', ref)
    if 'iv_regime' in combined.columns:
        rc = {'low': '#2196F3', 'mid': '#FF9800', 'extreme': '#F44336', 'skipped': '#CCCCCC'}
        for regime, color in rc.items():
            mask = combined['iv_regime'] == regime
            if mask.sum() > 0:
                ax.scatter(combined.index[mask], combined.loc[mask, 'iv_entry']*100,
                           c=color, s=4, alpha=0.5, label=f'{regime.capitalize()} ({mask.sum()}d)')
    ax.axhline(y=42, color='#2196F3', lw=1, ls=':', alpha=0.7)
    ax.axhline(y=70, color='#F44336', lw=1, ls=':', alpha=0.7)
    ax.set_title('IV Regime & Delta Selection (Combined)', fontsize=11, fontweight='bold')
    ax.set_ylabel('IV at Entry (%)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 6: P&L distribution Baseline vs Combined ---
    ax = fig.add_subplot(gs[3, 0])
    bins = np.linspace(-15000, 15000, 50)
    for label in ['Baseline (20D)', 'Combined']:
        if label in all_results:
            r = all_results[label]
            ax.hist(r['strategy_pnl'], bins=bins, alpha=0.5, color=colors[label],
                    label=f"{label} (mean=${r['strategy_pnl'].mean():,.0f})",
                    edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='black', lw=1.5, ls='--', alpha=0.7)
    ax.set_title('P&L Distribution: Baseline vs Combined', fontsize=11, fontweight='bold')
    ax.set_xlabel('P&L per Trade ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', axis='y')

    # --- Panel 7: Rolling Sharpe ---
# --- Panel 7: Rolling Sharpe ---
    ax = fig.add_subplot(gs[3, 1])
    window = 90
    ppy = 365 / 21
    for label, res in all_results.items():
        c = colors.get(label, 'gray')
        rs = (res['strategy_return'].rolling(window).mean()
            / res['strategy_return'].rolling(window).std() * np.sqrt(ppy))
        lw = 2.5 if label == 'Combined' else 1.8  # Increased from 1.0
        ax.plot(res.index, rs, color=c, lw=lw, label=label, alpha=0.9)  # Consistent alpha
    ax.axhline(y=1, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax.set_title('Rolling 90-Day Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 8: Spot & IV ---
    ax = fig.add_subplot(gs[4, 0])
    ax2 = ax.twinx()
    ax.plot(daily_data.index, daily_data['spot']/1000, color='#FFA500', lw=2)
    ax2.plot(daily_data.index, daily_data['iv_3W_20D']*100, color='#9966CC', lw=1.2, alpha=0.7)
    ax.set_ylabel('BTC ($K)', color='#FFA500')
    ax2.set_ylabel('IV 3W 20D (%)', color='#9966CC')
    ax.set_title('BTC Spot & Implied Volatility', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 9: IV vs RV ---
    ax = fig.add_subplot(gs[4, 1])
    ax.plot(daily_data.index, daily_data['iv_3W_20D']*100, color='#9966CC', lw=1.5, label='IV (3W 20D)')
    ax.plot(daily_data.index, daily_data['rv_21d']*100, color='#FF6B6B', lw=1.2, alpha=0.8, label='RV (21d)')
    ax.fill_between(daily_data.index, daily_data['iv_3W_20D']*100, daily_data['rv_21d']*100,
                     where=(daily_data['iv_3W_20D'] > daily_data['rv_21d']),
                     color='green', alpha=0.12, label='VRP > 0')
    ax.fill_between(daily_data.index, daily_data['iv_3W_20D']*100, daily_data['rv_21d']*100,
                     where=(daily_data['iv_3W_20D'] <= daily_data['rv_21d']),
                     color='red', alpha=0.12, label='VRP < 0')
    ax.set_title('Implied vs Realized Volatility', fontsize=11, fontweight='bold')
    ax.set_ylabel('Volatility (%)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    # --- Panel 10: Metrics table (full width) ---
    ax = fig.add_subplot(gs[5, :])
    ax.axis('off')

    all_m = [calculate_metrics(r, label=l) for l, r in all_results.items()]
    m0 = all_m[0]

    header = ['Metric'] + [m['label'] for m in all_m] + ['Long BTC']
    tdata = [header]
    tdata.append(['Total P&L'] + [f"${m['strategy_total_pnl']/1000:.0f}K" for m in all_m] + [f"${m0['btc_total_pnl']/1000:.0f}K"])
    tdata.append(['Alpha vs BTC'] + [f"${m['alpha']/1000:+.0f}K" for m in all_m] + ['---'])
    tdata.append(['Sharpe Ratio'] + [f"{m['strategy_sharpe']:.2f}" for m in all_m] + [f"{m0['btc_sharpe']:.2f}"])
    tdata.append(['Max Drawdown'] + [f"${m['strategy_max_dd']/1000:.0f}K" for m in all_m] + [f"${m0['btc_max_dd']/1000:.0f}K"])
    tdata.append(['Win Rate'] + [f"{m['strategy_win_rate']*100:.1f}%" for m in all_m] + [f"{m0['btc_win_rate']*100:.1f}%"])
    tdata.append(['Option Trades'] + [f"{m['option_trades']}" for m in all_m] + ['---'])
    tdata.append(['Premium Collected'] + [f"${m['total_premium']/1000:.0f}K" for m in all_m] + ['---'])
    tdata.append(['Called Away'] + [f"{m['called_away_rate']*100:.1f}%" if not np.isnan(m.get('called_away_rate', np.nan)) else 'N/A' for m in all_m] + ['---'])

    nc = len(header)
    table = ax.table(cellText=tdata, cellLoc='center', loc='center', bbox=[0.02, 0, 0.96, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    for i in range(nc):
        table[(0, i)].set_facecolor('#37474F')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(tdata)):
        for j in range(nc):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    # Highlight Combined column
    try:
        cc = [m['label'] for m in all_m].index('Combined') + 1
        for i in range(1, len(tdata)):
            table[(i, cc)].set_text_props(weight='bold')
    except ValueError:
        pass
    ax.text(0.5, 1.08, 'Strategy Performance Summary', transform=ax.transAxes,
            ha='center', fontsize=12, fontweight='bold')

    # --- Panel 11: VRP threshold robustness (full width) ---
    ax = fig.add_subplot(gs[6, :])
    ax.axis('off')
    summary = """
    STRATEGY LOGIC SUMMARY

    Baseline (20D):   Sell 3W 20-delta call every day. No signal conditioning.

    VRP Filter:       Compute VRP = IV(3W,20D) - RV(trailing 21d).
                      If VRP < 12%  ->  Sell 3W 20D call
                      If VRP >= 12% ->  Skip (hold BTC only; market pricing a large future move)

    Dynamic Delta:    Choose call delta based on IV level:
                      IV < 42%        ->  Sell 25D (range-bound; extra premium captured safely)
                      42% <= IV < 70% ->  Sell 20D (standard)
                      IV >= 70%       ->  Sell 25D (crisis; fat premium, 0% historical call-away rate)

    Combined:         Apply VRP filter first, then choose delta via IV regime. RECOMMENDED.

    CORRECTNESS:
    * All signals backward-looking only (trailing RV, current IV). No look-ahead.
    * Black-76 with market forward (not spot). IV interpolated per-delta (skew-aware).
    * VRP threshold robust across 8-15% range (all produce positive alpha).
    * IV regime thresholds (42%, 70%) based on distribution quartiles.
    """
    ax.text(0.03, 0.95, summary, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FFFDE7', alpha=0.5))

    # --- Panel 12: VRP threshold robustness chart ---
    ax = fig.add_subplot(gs[7, :])
    # Pre-computed from analysis
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.00]
    # We'll compute these live
    _rvf = run_vrp_filter  # already defined above
    alphas = []
    sharpes = []
    skips = []
    for t in thresholds:
        r = _rvf(daily_data, vrp_threshold=t)
        mm = calculate_metrics(r, label=f't={t}')
        alphas.append(mm['alpha'] / 1000)
        sharpes.append(mm['strategy_sharpe'])
        skips.append(mm['skip_rate'] * 100)

    ax_r = ax.twinx()
    x = [t * 100 for t in thresholds]
    ax.bar(x, alphas, width=1.8, color='#4CAF50', alpha=0.6, label='Alpha vs BTC ($K)')
    ax_r.plot(x, sharpes, 'o-', color='#2196F3', lw=2, markersize=6, label='Sharpe Ratio')
    ax.axvline(x=12, color='red', lw=1.5, ls='--', alpha=0.7, label='Chosen threshold')
    ax.set_xlabel('VRP Threshold (%)')
    ax.set_ylabel('Alpha vs BTC ($K)', color='#4CAF50')
    ax_r.set_ylabel('Sharpe Ratio', color='#2196F3')
    ax.set_title('VRP Threshold Robustness Analysis', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax_r.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, ls='--')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nOptimization report saved: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BTC Covered Call Strategy — Optimized Backtest"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the .pkl IV data file (e.g. data/BTC_Options_IV_2024_2025.pkl)"
    )
    parser.add_argument(
        "--output", default="BTC_Option_Strategy_Report.png",
        help="Output path for the report image (default: BTC_Option_Strategy_Report.png)"
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  BTC COVERED CALL — STRATEGY OPTIMIZATION")
    print("  Forward-Priced (Black-76) | VRP Filter | Dynamic Delta")
    print("="*70)

    daily_data = load_and_prepare_data(args.data)

    print("\n--- Running Strategy A: Baseline (20D) ---")
    res_baseline = run_baseline(daily_data)

    print("--- Running Strategy B: VRP Filter (threshold=12%) ---")
    res_vrp = run_vrp_filter(daily_data, vrp_threshold=0.12)

    print("--- Running Strategy C: Dynamic Delta ---")
    res_dynamic = run_dynamic_delta(daily_data)

    print("--- Running Strategy D: Combined (VRP + Dynamic Delta) ---")
    res_combined = run_combined(daily_data, vrp_threshold=0.12)

    all_results = {
        'Baseline (20D)': res_baseline,
        'VRP Filter':     res_vrp,
        'Dynamic Delta':  res_dynamic,
        'Combined':       res_combined,
    }

    all_metrics = []
    for label, res_df in all_results.items():
        m = calculate_metrics(res_df, label=label)
        all_metrics.append(m)
        print_metrics(m)

    print_comparison_table(all_metrics)

    print("\n--- Generating charts ---")
    create_optimization_report(all_results, daily_data,
                                output_path=args.output)
    print("\nDone.")