# Binance Cryptocurrency Trade Data Collection with Daily Chunking

"""
Objective: Collect raw tick-level trade data from Binance for quantitative finance analysis

Symbols: BTCUSDT, ETHUSDT, SOLUSDT, POWRUSDT
Time Period: 1-3 months (configurable)
Data Type: Aggregated trades with timestamp, price, volume, and aggressor side

Data Resolution:
This script collects tick-level (trade-by-trade) data at millisecond precision—the highest
resolution available. This meets the standard recommended in Advances in Financial Machine Learning
(Chapter 2), which emphasises storing market data at the tick level to enable transformation into
information-driven bars (dollar bars, volume bars, tick imbalance bars) rather than relying on
fixed-time sampling.

Storage Strategy:
Uses daily chunking to keep memory usage constant and enable easy resumption.
Structure: binance_raw_data/SYMBOL/YYYY-MM-DD.parquet
"""

# Setup and Imports
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

print("✓ Libraries imported")

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'POWRUSDT']
MONTHS = 3
DATA_DIR = Path(__file__).parent.parent / 'binance_raw_data'
BASE_URL = 'https://api.binance.com/api/v3'

print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Period: {MONTHS} months")
print(f"Output: {DATA_DIR}/")

# Helper Functions

def fetch_day_trades(symbol, date):
    """Fetch all trades for a symbol for a single day."""
    start_ms = int(date.timestamp() * 1000)
    end_ms = int((date + timedelta(days=1)).timestamp() * 1000)

    trades = []
    current = start_ms
    batch_count = 0

    print(f"    → Fetching {date.date()} from Binance API...", end='', flush=True)

    while current < end_ms:
        params = {
            'symbol': symbol,
            'startTime': current,
            'endTime': end_ms,
            'limit': 1000
        }

        response = requests.get(f'{BASE_URL}/aggTrades', params=params)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        trades.extend(batch)
        current = batch[-1]['T'] + 1
        batch_count += 1

        # Progress indicator
        if batch_count % 10 == 0:
            print(f".", end='', flush=True)

        time.sleep(0.1)  # Rate limit

    print(f" ✓ ({len(trades):,} trades fetched)")
    return trades

def process_trades(trades, symbol):
    """Convert raw trades to clean DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df = df.rename(columns={
        'T': 'timestamp',
        'p': 'price',
        'q': 'quantity',
        'm': 'is_buyer_maker'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['price'].astype(float)
    df['quantity'] = df['quantity'].astype(float)
    df['dollar_volume'] = df['price'] * df['quantity']
    df['symbol'] = symbol

    return df[['timestamp', 'symbol', 'price', 'quantity', 'dollar_volume', 'is_buyer_maker']]

def save_day(df, symbol, date):
    """Save a single day's data."""
    symbol_dir = DATA_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    filepath = symbol_dir / f"{date.strftime('%Y-%m-%d')}.parquet"

    # File creation feedback
    print(f"    → Creating file: {filepath.name}")
    print(f"    → Writing {len(df):,} trades to disk...", end='', flush=True)

    df.to_parquet(filepath, compression='gzip', index=False)

    # Completion feedback with file size
    file_size_kb = filepath.stat().st_size / 1024
    if file_size_kb > 1024:
        print(f" ✓ ({file_size_kb/1024:.2f} MB)")
    else:
        print(f" ✓ ({file_size_kb:.2f} KB)")

    return filepath

def collect_symbol(symbol, start_date, end_date):
    """Collect all data for a symbol, one day at a time."""
    print(f"\n{symbol}")
    print("=" * 60)

    current_date = start_date
    total_trades = 0

    while current_date < end_date:
        filepath = DATA_DIR / symbol / f"{current_date.strftime('%Y-%m-%d')}.parquet"

        # Skip if already exists
        if filepath.exists():
            existing_df = pd.read_parquet(filepath)
            total_trades += len(existing_df)
            print(f"  {current_date.date()}: {len(existing_df):,} trades (existing)")
        else:
            # Fetch and save
            trades = fetch_day_trades(symbol, current_date)
            df = process_trades(trades, symbol)

            if not df.empty:
                save_day(df, symbol, current_date)
                total_trades += len(df)
                print(f"  {current_date.date()}: {len(df):,} trades (fetched)")
            else:
                print(f"  {current_date.date()}: 0 trades")

        current_date += timedelta(days=1)

    print(f"✓ Total: {total_trades:,} trades")
    return total_trades

print("✓ Functions defined")

# Collect Data

end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - timedelta(days=MONTHS * 30)

print(f"\nCollecting from {start_date.date()} to {end_date.date()}\n")

for symbol in SYMBOLS:
    collect_symbol(symbol, start_date, end_date)

print(f"\n✓ Collection complete: {DATA_DIR}/")

# Validate Data

print("\n" + "="*60)
print("DATA VALIDATION")
print("="*60)

for symbol in SYMBOLS:
    symbol_dir = DATA_DIR / symbol
    if not symbol_dir.exists():
        print(f"\n{symbol}: No data")
        continue

    # Load all daily files
    daily_files = sorted(symbol_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in daily_files]

    if not dfs:
        print(f"\n{symbol}: No data")
        continue

    df = pd.concat(dfs, ignore_index=True)

    print(f"\n{symbol}:")
    print(f"  Days collected: {len(daily_files)}")
    print(f"  Trades: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Avg trade size: ${df['dollar_volume'].mean():.2f}")
    print(f"  Total volume: ${df['dollar_volume'].sum():,.0f}")

    days = (df['timestamp'].max() - df['timestamp'].min()).days
    trades_per_day = len(df) / days if days > 0 else 0
    print(f"  Avg trades/day: {trades_per_day:,.0f}")

# # Detailed statistics for comparison
# print("\n" + "="*60)
# print("DETAILED STATISTICS COMPARISON")
# print("="*60)
# 
# for symbol in SYMBOLS:
#     symbol_dir = DATA_DIR / symbol
#     if not symbol_dir.exists():
#         continue
# 
#     daily_files = sorted(symbol_dir.glob("*.parquet"))
#     if not daily_files:
#         continue
# 
#     dfs = [pd.read_parquet(f) for f in daily_files]
#     df = pd.concat(dfs, ignore_index=True)
# 
#     print(f"\n{symbol} - Price & Volume Statistics:")
#     print(df[['price', 'quantity', 'dollar_volume']].describe())
# 
# # Visualise
# 
# import matplotlib.pyplot as plt
# 
# fig, axes = plt.subplots(len(SYMBOLS), 2, figsize=(14, 3*len(SYMBOLS)))
# 
# for idx, symbol in enumerate(SYMBOLS):
#     symbol_dir = DATA_DIR / symbol
#     if not symbol_dir.exists():
#         continue
# 
#     daily_files = sorted(symbol_dir.glob("*.parquet"))
#     if not daily_files:
#         continue
# 
#     dfs = [pd.read_parquet(f) for f in daily_files]
#     df = pd.concat(dfs, ignore_index=True)
#     df_sorted = df.sort_values('timestamp').set_index('timestamp')
# 
#     # Daily aggregation
#     daily_close = df_sorted['price'].resample('D').last()
#     daily_volume = df_sorted['dollar_volume'].resample('D').sum()
# 
#     # Price chart
#     axes[idx, 0].plot(daily_close.index, daily_close.values, linewidth=1.5)
#     axes[idx, 0].set_ylabel('Price (USD)')
#     axes[idx, 0].set_title(f'{symbol} - Daily Close Price')
#     axes[idx, 0].grid(alpha=0.3)
# 
#     # Volume chart
#     axes[idx, 1].bar(daily_volume.index, daily_volume.values/1e6, alpha=0.7)
#     axes[idx, 1].set_ylabel('Volume (M USD)')
#     axes[idx, 1].set_title(f'{symbol} - Daily Trading Volume')
#     axes[idx, 1].grid(alpha=0.3)
# 
#     if idx == len(SYMBOLS) - 1:
#         axes[idx, 0].set_xlabel('Date')
#         axes[idx, 1].set_xlabel('Date')
# 
# plt.tight_layout()
# plt.show()
# 
# print("\n✓ Visualization complete - Notice liquidity differences across symbols")
# 
# Next Steps
# """
# Transform this tick data into information-driven bars:
# - Dollar bars
# - Volume bars
# - Tick imbalance bars
# 
# ---
# Author: James Eggleston
# Date: October 2025
# """
