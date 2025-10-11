# %% [markdown]
# # Quick Check: Information-Driven Bars
# Simple visualization to verify bar generation

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

%matplotlib inline

# %%
# Load data
DATA_DIR = Path('../processed_data')

def find_latest_bar_file(symbol, bar_type):
    """Find most recent bar file"""
    pattern = f'{symbol}_{bar_type}_*.parquet'
    files = sorted(DATA_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None

# Load bars
dollar_bars_file = find_latest_bar_file('BTCUSDT', 'dollar_bars')
dollar_imbalance_file = find_latest_bar_file('BTCUSDT', 'dollar_imbalance_bars')
dollar_runs_file = find_latest_bar_file('BTCUSDT', 'dollar_runs_bars')

print("Loading data...")
dollar_bars = pd.read_parquet(dollar_bars_file)
dollar_imbalance = pd.read_parquet(dollar_imbalance_file)
dollar_runs = pd.read_parquet(dollar_runs_file)

print(f"\n✓ Dollar bars: {len(dollar_bars):,} bars")
print(f"✓ Dollar imbalance: {len(dollar_imbalance):,} bars")
print(f"✓ Dollar runs: {len(dollar_runs):,} bars")

# %%
# Quick stats
days = (dollar_bars.index.max() - dollar_bars.index.min()).days
print(f"\nDate range: {dollar_bars.index.min()} to {dollar_bars.index.max()}")
print(f"Duration: {days} days")
print(f"\nBars per day:")
print(f"  Dollar bars: {len(dollar_bars) / days:.1f}")
print(f"  Dollar imbalance: {len(dollar_imbalance) / days:.1f}")
print(f"  Dollar runs: {len(dollar_runs) / days:.1f}")

# %%
# Simple price plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Dollar bars
axes[0].plot(dollar_bars.index, dollar_bars['close'], linewidth=1, alpha=0.7, color='steelblue')
axes[0].set_ylabel('Price (USDT)', fontsize=11)
axes[0].set_title(f'Dollar Bars ({len(dollar_bars):,} bars)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Dollar imbalance
axes[1].plot(dollar_imbalance.index, dollar_imbalance['close'], linewidth=1, alpha=0.7, color='darkorange')
axes[1].scatter(dollar_imbalance.index, dollar_imbalance['close'], s=20, alpha=0.5, color='darkorange')
axes[1].set_ylabel('Price (USDT)', fontsize=11)
axes[1].set_title(f'Dollar Imbalance Bars ({len(dollar_imbalance):,} bars)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Dollar runs
if len(dollar_runs) < 10000:  # Only plot if reasonable number
    axes[2].plot(dollar_runs.index, dollar_runs['close'], linewidth=1, alpha=0.7, color='seagreen')
    axes[2].scatter(dollar_runs.index, dollar_runs['close'], s=20, alpha=0.5, color='seagreen')
    axes[2].set_title(f'Dollar Runs Bars ({len(dollar_runs):,} bars)', fontsize=12, fontweight='bold')
else:
    axes[2].text(0.5, 0.5, f'TOO MANY BARS TO PLOT\n{len(dollar_runs):,} bars\n\nThreshold too low!',
                 ha='center', va='center', fontsize=16, color='red', fontweight='bold',
                 transform=axes[2].transAxes)
    axes[2].set_title(f'Dollar Runs Bars - ERROR', fontsize=12, fontweight='bold', color='red')

axes[2].set_ylabel('Price (USDT)', fontsize=11)
axes[2].set_xlabel('Date', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Bar distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bars per day
dollar_bars_daily = dollar_bars.resample('1D').size()
imbalance_bars_daily = dollar_imbalance.resample('1D').size()
runs_bars_daily = dollar_runs.resample('1D').size()

axes[0].plot(dollar_bars_daily.index, dollar_bars_daily, label='Dollar', linewidth=2)
axes[0].plot(imbalance_bars_daily.index, imbalance_bars_daily, label='Imbalance', linewidth=2)
if runs_bars_daily.max() < 10000:
    axes[0].plot(runs_bars_daily.index, runs_bars_daily, label='Runs', linewidth=2)
axes[0].set_ylabel('Bars per Day', fontsize=11)
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_title('Daily Bar Count', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Imbalance values
axes[1].bar(range(len(dollar_imbalance)), dollar_imbalance['imbalance'],
            color=['green' if x > 0 else 'red' for x in dollar_imbalance['imbalance']],
            alpha=0.7)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_xlabel('Bar Index', fontsize=11)
axes[1].set_ylabel('Imbalance', fontsize=11)
axes[1].set_title('Imbalance Values', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Runs distribution
if len(dollar_runs) < 10000:
    axes[2].hist(dollar_runs['max_dollar_run'], bins=30, alpha=0.7, color='seagreen', edgecolor='black')
    axes[2].set_xlabel('Max Dollar Run', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Run Length Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
else:
    axes[2].text(0.5, 0.5, 'TOO MANY BARS\nTO ANALYZE',
                 ha='center', va='center', fontsize=14, color='red', fontweight='bold')
    axes[2].set_title('Runs - ERROR', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.show()

# %%
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nDollar Bars: {len(dollar_bars):,} bars ({len(dollar_bars)/days:.1f}/day)")
print(f"  Status: {'✓ GOOD' if 50 < len(dollar_bars)/days < 500 else '✗ CHECK'}")

print(f"\nDollar Imbalance: {len(dollar_imbalance):,} bars ({len(dollar_imbalance)/days:.1f}/day)")
print(f"  Status: {'✓ GOOD' if 1 < len(dollar_imbalance)/days < 20 else '✗ CHECK'}")

print(f"\nDollar Runs: {len(dollar_runs):,} bars ({len(dollar_runs)/days:.1f}/day)")
print(f"  Status: {'✓ GOOD' if 1 < len(dollar_runs)/days < 20 else '✗ TOO MANY - THRESHOLD TOO LOW'}")
print("="*70)

# %%
