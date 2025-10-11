# %% [markdown]
# # Information-Driven Bars Analysis: BTC Flash Crash
#
# **Analysis of 60 days of BTCUSDT data (including 10% flash crash in final 24 hours)**
#
# This notebook compares three information-driven bar sampling methods:
# 1. **Dollar Bars** - Normalizes for price changes in high-liquidity assets
# 2. **Dollar Imbalance Bars** - Captures order flow toxicity and informed trading
# 3. **Dollar Runs Bars** - Detects momentum shifts and trend exhaustion
#
# Based on *Advances in Financial Machine Learning* by Marcos Lopez de Prado (Chapter 2)

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# %%
# Configuration
SYMBOL = 'BTCUSDT'
DATA_DIR = Path('../processed_data')

# Load the bars - use glob to find latest files
print("Loading bar data...")

# Find the most recent bar files
def find_latest_bar_file(symbol, bar_type):
    """Find the most recent bar file for given type"""
    pattern = f'{symbol}_{bar_type}_*.parquet'
    files = sorted(DATA_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    return files[0]

# Dollar bars (standard)
dollar_bars_file = find_latest_bar_file(SYMBOL, 'dollar_bars')
dollar_bars = pd.read_parquet(dollar_bars_file)
print(f"✓ Dollar bars loaded: {len(dollar_bars)} bars (from {dollar_bars_file.name})")

# Dollar imbalance bars
dollar_imbalance_file = find_latest_bar_file(SYMBOL, 'dollar_imbalance_bars')
dollar_imbalance = pd.read_parquet(dollar_imbalance_file)
print(f"✓ Dollar imbalance bars loaded: {len(dollar_imbalance)} bars (from {dollar_imbalance_file.name})")

# Dollar runs bars
dollar_runs_file = find_latest_bar_file(SYMBOL, 'dollar_runs_bars')
dollar_runs = pd.read_parquet(dollar_runs_file)
print(f"✓ Dollar runs bars loaded: {len(dollar_runs)} bars (from {dollar_runs_file.name})")

print("\nData date range:")
print(f"  Dollar bars: {dollar_bars.index.min()} to {dollar_bars.index.max()}")
print(f"  Dollar imbalance: {dollar_imbalance.index.min()} to {dollar_imbalance.index.max()}")
print(f"  Dollar runs: {dollar_runs.index.min()} to {dollar_runs.index.max()}")

# %%
# Calculate returns for each bar type
dollar_bars['returns'] = dollar_bars['close'].pct_change()
dollar_imbalance['returns'] = dollar_imbalance['close'].pct_change()
dollar_runs['returns'] = dollar_runs['close'].pct_change()

# Identify crash period (last 24 hours)
crash_start = dollar_bars.index.max() - pd.Timedelta(hours=24)
print(f"\nCrash period starts: {crash_start}")

# %% [markdown]
# ## 2. Overview: Sampling Comparison

# %%
# Summary statistics
summary = pd.DataFrame({
    'Dollar Bars': [
        len(dollar_bars),
        dollar_bars['volume'].sum(),
        dollar_bars['tick_count'].sum(),
        dollar_bars['returns'].std() * 100
    ],
    'Dollar Imbalance': [
        len(dollar_imbalance),
        dollar_imbalance['volume'].sum(),
        dollar_imbalance['tick_count'].sum(),
        dollar_imbalance['returns'].std() * 100
    ],
    'Dollar Runs': [
        len(dollar_runs),
        dollar_runs['volume'].sum(),
        dollar_runs['tick_count'].sum(),
        dollar_runs['returns'].std() * 100
    ]
}, index=['Number of Bars', 'Total Volume', 'Total Ticks', 'Return Volatility (%)'])

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(summary.round(2))

# %% [markdown]
# ## 3. Price Action Comparison
#
# Visualizing when each sampling method creates bars reveals how they adapt to market conditions.

# %%
# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Plot 1: Dollar Bars
axes[0].plot(dollar_bars.index, dollar_bars['close'], label='Close Price', color='steelblue', linewidth=1.5)
axes[0].scatter(dollar_bars.index, dollar_bars['close'], alpha=0.3, s=10, color='steelblue')
axes[0].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start (T-24h)')
axes[0].set_ylabel('Price (USDT)', fontsize=12)
axes[0].set_title('Dollar Bars - Fixed $5M Threshold', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Dollar Imbalance Bars
axes[1].plot(dollar_imbalance.index, dollar_imbalance['close'], label='Close Price', color='darkorange', linewidth=1.5)
axes[1].scatter(dollar_imbalance.index, dollar_imbalance['close'], alpha=0.3, s=10, color='darkorange')
axes[1].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start (T-24h)')
axes[1].set_ylabel('Price (USDT)', fontsize=12)
axes[1].set_title('Dollar Imbalance Bars - Adaptive to Order Flow Toxicity', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Plot 3: Dollar Runs Bars
axes[2].plot(dollar_runs.index, dollar_runs['close'], label='Close Price', color='seagreen', linewidth=1.5)
axes[2].scatter(dollar_runs.index, dollar_runs['close'], alpha=0.3, s=10, color='seagreen')
axes[2].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start (T-24h)')
axes[2].set_ylabel('Price (USDT)', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_title('Dollar Runs Bars - Adaptive to Momentum Shifts', fontsize=14, fontweight='bold')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Sampling Frequency Analysis
#
# How does each method adapt its sampling rate during volatile periods?

# %%
# Resample to hourly frequency to count bars per hour
def bars_per_hour(df):
    """Count number of bars generated per hour"""
    df_copy = df.copy()
    df_copy['bar_count'] = 1
    return df_copy.resample('1H')['bar_count'].sum()

dollar_bars_hourly = bars_per_hour(dollar_bars)
imbalance_bars_hourly = bars_per_hour(dollar_imbalance)
runs_bars_hourly = bars_per_hour(dollar_runs)

# %%
# Plot sampling frequency
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full period
axes[0].plot(dollar_bars_hourly.index, dollar_bars_hourly, label='Dollar Bars', linewidth=2, alpha=0.8)
axes[0].plot(imbalance_bars_hourly.index, imbalance_bars_hourly, label='Dollar Imbalance', linewidth=2, alpha=0.8)
axes[0].plot(runs_bars_hourly.index, runs_bars_hourly, label='Dollar Runs', linewidth=2, alpha=0.8)
axes[0].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start')
axes[0].set_ylabel('Bars per Hour', fontsize=12)
axes[0].set_title('Sampling Frequency Over Time (Full Period)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Crash period zoom
crash_data_dollar = dollar_bars_hourly[dollar_bars_hourly.index >= crash_start]
crash_data_imb = imbalance_bars_hourly[imbalance_bars_hourly.index >= crash_start]
crash_data_runs = runs_bars_hourly[runs_bars_hourly.index >= crash_start]

axes[1].plot(crash_data_dollar.index, crash_data_dollar, label='Dollar Bars', linewidth=2, alpha=0.8, marker='o')
axes[1].plot(crash_data_imb.index, crash_data_imb, label='Dollar Imbalance', linewidth=2, alpha=0.8, marker='s')
axes[1].plot(crash_data_runs.index, crash_data_runs, label='Dollar Runs', linewidth=2, alpha=0.8, marker='^')
axes[1].set_ylabel('Bars per Hour', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_title('Sampling Frequency During Flash Crash (Last 24 Hours)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nSampling frequency during crash:")
print(f"  Dollar bars: {crash_data_dollar.sum():.0f} bars in 24h (avg {crash_data_dollar.mean():.1f}/hour)")
print(f"  Dollar imbalance: {crash_data_imb.sum():.0f} bars in 24h (avg {crash_data_imb.mean():.1f}/hour)")
print(f"  Dollar runs: {crash_data_runs.sum():.0f} bars in 24h (avg {crash_data_runs.mean():.1f}/hour)")

# %% [markdown]
# ## 5. Dollar Imbalance Analysis: Order Flow Toxicity
#
# Dollar imbalance bars capture informed trading and order flow toxicity.
# High absolute imbalance indicates strong directional pressure.

# %%
# Plot imbalance values over time
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full period imbalance
axes[0].plot(dollar_imbalance.index, dollar_imbalance['imbalance'],
             color='darkorange', linewidth=1, alpha=0.7)
axes[0].fill_between(dollar_imbalance.index, 0, dollar_imbalance['imbalance'],
                      where=dollar_imbalance['imbalance'] > 0,
                      color='green', alpha=0.3, label='Buy Pressure')
axes[0].fill_between(dollar_imbalance.index, 0, dollar_imbalance['imbalance'],
                      where=dollar_imbalance['imbalance'] < 0,
                      color='red', alpha=0.3, label='Sell Pressure')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start')
axes[0].set_ylabel('Dollar Imbalance', fontsize=12)
axes[0].set_title('Order Flow Imbalance Over Time', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Crash period zoom
crash_imbalance = dollar_imbalance[dollar_imbalance.index >= crash_start]
axes[1].plot(crash_imbalance.index, crash_imbalance['imbalance'],
             color='darkorange', linewidth=1.5, alpha=0.8, marker='o', markersize=4)
axes[1].fill_between(crash_imbalance.index, 0, crash_imbalance['imbalance'],
                      where=crash_imbalance['imbalance'] > 0,
                      color='green', alpha=0.3)
axes[1].fill_between(crash_imbalance.index, 0, crash_imbalance['imbalance'],
                      where=crash_imbalance['imbalance'] < 0,
                      color='red', alpha=0.3)
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Dollar Imbalance', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_title('Order Flow Imbalance During Flash Crash', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Imbalance statistics
print("\nImbalance Statistics:")
print(f"  Mean imbalance: ${dollar_imbalance['imbalance'].mean():,.0f}")
print(f"  Std imbalance: ${dollar_imbalance['imbalance'].std():,.0f}")
print(f"  Max buy pressure: ${dollar_imbalance['imbalance'].max():,.0f}")
print(f"  Max sell pressure: ${dollar_imbalance['imbalance'].min():,.0f}")

print("\nDuring crash period:")
print(f"  Mean imbalance: ${crash_imbalance['imbalance'].mean():,.0f}")
print(f"  Std imbalance: ${crash_imbalance['imbalance'].std():,.0f}")
print(f"  % Sell bars: {(crash_imbalance['imbalance'] < 0).sum() / len(crash_imbalance) * 100:.1f}%")

# %% [markdown]
# ## 6. Dollar Runs Analysis: Momentum & Trend Exhaustion
#
# Dollar runs bars detect sustained directional moves (runs).
# Long runs indicate strong momentum; runs ending signal potential reversals.

# %%
# Plot runs over time
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full period runs
axes[0].plot(dollar_runs.index, dollar_runs['max_dollar_run'],
             color='seagreen', linewidth=1.5, alpha=0.7)
axes[0].axvline(crash_start, color='red', linestyle='--', alpha=0.7, label='Crash Start')
axes[0].set_ylabel('Max Dollar Run', fontsize=12)
axes[0].set_title('Dollar Runs Over Time (Momentum Detection)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Crash period zoom
crash_runs = dollar_runs[dollar_runs.index >= crash_start]
axes[1].plot(crash_runs.index, crash_runs['max_dollar_run'],
             color='seagreen', linewidth=2, alpha=0.8, marker='o', markersize=5)
axes[1].set_ylabel('Max Dollar Run', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_title('Dollar Runs During Flash Crash', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Runs statistics
print("\nRuns Statistics:")
print(f"  Mean run: ${dollar_runs['max_dollar_run'].mean():,.0f}")
print(f"  Median run: ${dollar_runs['max_dollar_run'].median():,.0f}")
print(f"  Max run: ${dollar_runs['max_dollar_run'].max():,.0f}")
print(f"  Std run: ${dollar_runs['max_dollar_run'].std():,.0f}")

print("\nDuring crash period:")
print(f"  Mean run: ${crash_runs['max_dollar_run'].mean():,.0f}")
print(f"  Max run: ${crash_runs['max_dollar_run'].max():,.0f}")
print(f"  Run > mean: {(crash_runs['max_dollar_run'] > dollar_runs['max_dollar_run'].mean()).sum()} bars")

# %% [markdown]
# ## 7. Returns Distribution Analysis
#
# Information-driven bars should produce more IID (Independent, Identically Distributed) returns
# compared to time-based sampling.

# %%
# Plot returns distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dollar bars
axes[0].hist(dollar_bars['returns'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Returns', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title(f'Dollar Bars Returns\nμ={dollar_bars["returns"].mean():.6f}, σ={dollar_bars["returns"].std():.6f}',
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Dollar imbalance
axes[1].hist(dollar_imbalance['returns'].dropna(), bins=50, alpha=0.7, color='darkorange', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Returns', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title(f'Dollar Imbalance Returns\nμ={dollar_imbalance["returns"].mean():.6f}, σ={dollar_imbalance["returns"].std():.6f}',
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Dollar runs
axes[2].hist(dollar_runs['returns'].dropna(), bins=50, alpha=0.7, color='seagreen', edgecolor='black')
axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Returns', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title(f'Dollar Runs Returns\nμ={dollar_runs["returns"].mean():.6f}, σ={dollar_runs["returns"].std():.6f}',
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Q-Q plots to check normality
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dollar bars
stats.probplot(dollar_bars['returns'].dropna(), dist="norm", plot=axes[0])
axes[0].set_title('Dollar Bars Q-Q Plot', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Dollar imbalance
stats.probplot(dollar_imbalance['returns'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title('Dollar Imbalance Q-Q Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Dollar runs
stats.probplot(dollar_runs['returns'].dropna(), dist="norm", plot=axes[2])
axes[2].set_title('Dollar Runs Q-Q Plot', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Autocorrelation Analysis
#
# Lower autocorrelation in returns indicates better sampling (more IID).

# %%
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dollar bars ACF
plot_acf(dollar_bars['returns'].dropna(), lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('Dollar Bars - Autocorrelation', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=11)
axes[0].set_ylabel('ACF', fontsize=11)

# Dollar imbalance ACF
plot_acf(dollar_imbalance['returns'].dropna(), lags=40, ax=axes[1], alpha=0.05)
axes[1].set_title('Dollar Imbalance - Autocorrelation', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=11)
axes[1].set_ylabel('ACF', fontsize=11)

# Dollar runs ACF
plot_acf(dollar_runs['returns'].dropna(), lags=40, ax=axes[2], alpha=0.05)
axes[2].set_title('Dollar Runs - Autocorrelation', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Lag', fontsize=11)
axes[2].set_ylabel('ACF', fontsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Volume & Tick Distribution per Bar
#
# Examining how evenly each method distributes market activity.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Volume per bar
axes[0, 0].hist(dollar_bars['volume'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Volume', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Dollar Bars - Volume Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(dollar_imbalance['volume'], bins=50, alpha=0.7, color='darkorange', edgecolor='black')
axes[0, 1].set_xlabel('Volume', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Dollar Imbalance - Volume Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(dollar_runs['volume'], bins=50, alpha=0.7, color='seagreen', edgecolor='black')
axes[0, 2].set_xlabel('Volume', fontsize=11)
axes[0, 2].set_ylabel('Frequency', fontsize=11)
axes[0, 2].set_title('Dollar Runs - Volume Distribution', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Tick count per bar
axes[1, 0].hist(dollar_bars['tick_count'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 0].set_xlabel('Tick Count', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Dollar Bars - Tick Count Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(dollar_imbalance['tick_count'], bins=50, alpha=0.7, color='darkorange', edgecolor='black')
axes[1, 1].set_xlabel('Tick Count', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Dollar Imbalance - Tick Count Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(dollar_runs['tick_count'], bins=50, alpha=0.7, color='seagreen', edgecolor='black')
axes[1, 2].set_xlabel('Tick Count', fontsize=11)
axes[1, 2].set_ylabel('Frequency', fontsize=11)
axes[1, 2].set_title('Dollar Runs - Tick Count Distribution', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Key Insights & Conclusions

# %%
print("="*80)
print("KEY INSIGHTS: INFORMATION-DRIVEN BARS DURING FLASH CRASH")
print("="*80)
print()

print("1. DOLLAR BARS (Fixed $5M Threshold)")
print("   • Provides consistent dollar-value sampling")
print("   • Good for normalizing price changes in BTC")
print("   • Does NOT adapt to market microstructure changes")
print()

print("2. DOLLAR IMBALANCE BARS (Order Flow Toxicity)")
print("   • Samples when informed trading pressure exceeds expectations")
print("   • Captures extreme sell pressure during crash")
print(f"   • Generated {len(crash_imbalance)} bars during 24h crash vs {len(dollar_imbalance)} total")
print("   • ADVANTAGE: Detects when smart money is active")
print()

print("3. DOLLAR RUNS BARS (Momentum Detection)")
print("   • Samples when directional runs exceed thresholds")
print("   • Detects sustained selling pressure (long runs)")
print(f"   • Max run during crash: ${crash_runs['max_dollar_run'].max():,.0f}")
print("   • ADVANTAGE: Identifies trend exhaustion points")
print()

print("STATISTICAL PROPERTIES:")
print(f"   Dollar Bars Return Std: {dollar_bars['returns'].std():.6f}")
print(f"   Imbalance Bars Return Std: {dollar_imbalance['returns'].std():.6f}")
print(f"   Runs Bars Return Std: {dollar_runs['returns'].std():.6f}")
print()

print("RECOMMENDATION FOR BTC TRADING:")
print("   ✓ Use Dollar Imbalance Bars for detecting informed trading")
print("   ✓ Use Dollar Runs Bars for momentum/reversal signals")
print("   ✓ Combine both for comprehensive market microstructure analysis")
print("="*80)

# %%
