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
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Parse Command-Line Arguments
parser = argparse.ArgumentParser(
    description='Collect tick-level trade data from Binance',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python src/data_collection.py --symbols BTCUSDT --months 12
  python src/data_collection.py --symbols BTCUSDT ETHUSDT SOLUSDT --months 6
  python src/data_collection.py  # uses defaults
    """
)

parser.add_argument(
    '--symbols',
    nargs='+',
    default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'POWRUSDT'],
    help='Symbols to collect (default: BTCUSDT ETHUSDT SOLUSDT POWRUSDT)'
)

parser.add_argument(
    '--months',
    type=int,
    default=3,
    help='Number of months to collect (default: 3)'
)

parser.add_argument(
    '--output-dir',
    type=str,
    default='binance_raw_data',
    help='Output directory for data (default: binance_raw_data)'
)

parser.add_argument(
    '--skip-validation',
    action='store_true',
    help='Skip the data validation step at the end'
)

args = parser.parse_args()

print("✓ Libraries imported")

# Configuration
SYMBOLS = args.symbols
MONTHS = args.months
DATA_DIR = Path(__file__).parent.parent / args.output_dir
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

if not args.skip_validation:
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


