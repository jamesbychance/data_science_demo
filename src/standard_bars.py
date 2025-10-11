#!/usr/bin/env python3
"""
Chapter 2: Standard Information-Driven Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements standard alternative sampling methods:
- Tick Bars: Sample every T ticks
- Volume Bars: Sample every T volume
- Dollar Bars: Sample every T dollar value traded

THRESHOLD GUIDANCE:
-------------------
The default thresholds are calibrated for BTCUSDT based on 60 days of historical data.
For other assets or different sampling frequencies, you should adjust these thresholds.

TARGET: Aim for 50-300 bars per day depending on your use case:
  - Low frequency (50-100 bars/day): Swing trading, daily analysis
  - Medium frequency (100-200 bars/day): Intraday strategies
  - High frequency (200-300 bars/day): Scalping, microstructure analysis

HOW TO CALIBRATE:
1. Analyze your tick data to get:
   - Total ticks per day
   - Total volume per day
   - Total dollar value per day

2. Calculate thresholds:
   - Tick threshold = ticks_per_day / target_bars_per_day
   - Volume threshold = volume_per_day / target_bars_per_day
   - Dollar threshold = dollar_per_day / target_bars_per_day

3. Test and iterate:
   - Run bar generation on sample period (7 days)
   - Count actual bars generated per day
   - Adjust thresholds if needed

EXAMPLES (for reference):
  BTCUSDT (price ~$115k, ~780k ticks/day):
    --tick-threshold 2500      # → ~310 bars/day
    --volume-threshold 125.0   # → ~120 bars/day
    --dollar-threshold 10000000 # → ~170 bars/day

  ETHUSDT (price ~$2.5k): Scale thresholds proportionally
    --dollar-threshold ~370000  # Scales with price difference

  Lower volume assets (POWRUSDT): Reduce all thresholds significantly
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import numpy as np
import pandas as pd
from utils.bar_utils import save_bars
from utils.data_loader import load_binance_data, get_available_symbols


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def _accumulate_bars(df: pd.DataFrame, threshold: float, value_column: str = None) -> pd.DataFrame:
    """
    Vectorized bar accumulation based on a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    threshold : float
        Threshold value to trigger bar creation
    value_column : str, optional
        Column to accumulate ('volume', 'dollar', or None for tick count)

    Returns
    -------
    pd.DataFrame
        OHLCV bars with tick_count
    """
    # Calculate increment for each row
    if value_column == 'dollar':
        increments = (df['price'] * df['volume']).values
    elif value_column == 'volume':
        increments = df['volume'].values
    else:  # tick bars
        increments = np.ones(len(df))
    
    # Cumulative sum and find bar boundaries
    cumsum = np.cumsum(increments)
    bar_indices = np.searchsorted(cumsum, np.arange(threshold, cumsum[-1], threshold), side='right')
    
    # Add start and end
    bar_indices = np.concatenate([[0], bar_indices, [len(df)]])
    
    # Build bars
    bars = []
    for i in range(len(bar_indices) - 1):
        start_idx = bar_indices[i]
        end_idx = bar_indices[i + 1]
        
        if start_idx >= end_idx:
            continue
            
        bar_slice = df.iloc[start_idx:end_idx]
        bars.append({
            'timestamp': bar_slice['timestamp'].iloc[-1],
            'open': bar_slice['price'].iloc[0],
            'high': bar_slice['price'].max(),
            'low': bar_slice['price'].min(),
            'close': bar_slice['price'].iloc[-1],
            'volume': bar_slice['volume'].sum(),
            'tick_count': len(bar_slice)
        })
    
    return pd.DataFrame(bars).set_index('timestamp')


def tick_bars(df: pd.DataFrame, threshold: int = 1000) -> pd.DataFrame:
    """Sample tick bars: aggregate every T ticks."""
    return _accumulate_bars(df, threshold, value_column=None)

def volume_bars(df: pd.DataFrame, threshold: float = 1e6) -> pd.DataFrame:
    """Sample volume bars: aggregate every T volume traded."""
    return _accumulate_bars(df, threshold, value_column='volume')


def dollar_bars(df: pd.DataFrame, threshold: float = 1e7) -> pd.DataFrame:
    """Sample dollar bars: aggregate every T dollars traded."""
    return _accumulate_bars(df, threshold, value_column='dollar')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate standard bars (tick, volume, dollar) from Binance data'
    )
    parser.add_argument(
        'symbol',
        type=str,
        nargs='?',
        default='BTCUSDT',
        help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT, SOLUSDT, POWRUSDT)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of most recent days to process (default: 7)'
    )
    parser.add_argument(
        '--tick-threshold',
        type=int,
        default=2500,
        help='Tick bar threshold (default: 2500)'
    )
    parser.add_argument(
        '--volume-threshold',
        type=float,
        default=125.0,
        help='Volume bar threshold (default: 125.0)'
    )
    parser.add_argument(
        '--dollar-threshold',
        type=float,
        default=10e6,
        help='Dollar bar threshold (default: 10000000)'
    )

    args = parser.parse_args()

    # Display available symbols
    available = get_available_symbols()
    print("\n" + "=" * 70)
    print(f"Available symbols: {', '.join(available)}")
    print("=" * 70 + "\n")

    # Load tick data from Binance
    try:
        tick_data = load_binance_data(args.symbol, days=args.days)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    print("=" * 70)

    # Generate tick bars
    print(f"\nGenerating tick bars (threshold: {args.tick_threshold:,})...")
    t_bars = tick_bars(tick_data, threshold=args.tick_threshold)
    print(f"✓ Tick bars generated: {len(t_bars)}")
    print(t_bars.head(3))

    # Generate volume bars
    print(f"\nGenerating volume bars (threshold: {args.volume_threshold:,.2f})...")
    v_bars = volume_bars(tick_data, threshold=args.volume_threshold)
    print(f"✓ Volume bars generated: {len(v_bars)}")
    print(v_bars.head(3))

    # Generate dollar bars
    print(f"\nGenerating dollar bars (threshold: ${args.dollar_threshold:,.0f})...")
    d_bars = dollar_bars(tick_data, threshold=args.dollar_threshold)
    print(f"✓ Dollar bars generated: {len(d_bars)}")
    print(d_bars.head(3))

    # Save to parquet
    print("\n" + "=" * 70)
    print("\nSaving bars...\n")

    save_bars(t_bars, args.symbol, 'tick_bars', threshold=args.tick_threshold)
    save_bars(v_bars, args.symbol, 'volume_bars', threshold=args.volume_threshold)
    save_bars(d_bars, args.symbol, 'dollar_bars', threshold=args.dollar_threshold)

    print("\n" + "=" * 70)
    print(f"\n✓ Standard bars for {args.symbol} generated and saved successfully!")
    print(f"\nSummary: {len(t_bars)} tick, {len(v_bars)} volume, {len(d_bars)} dollar bars")
    print("\n" + "=" * 70)
