#!/usr/bin/env python3
"""
Chapter 2: Standard Information-Driven Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements standard alternative sampling methods:
- Tick Bars: Sample every T ticks
- Volume Bars: Sample every T volume
- Dollar Bars: Sample every T dollar value traded
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
    Generic bar accumulation based on a threshold.

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
    bars = []
    accumulator = 0.0
    bar_data = []

    for _, row in df.iterrows():
        bar_data.append(row)

        # Calculate increment based on bar type
        if value_column == 'dollar':
            increment = row['price'] * row.get('volume', 0.0)
        elif value_column == 'volume':
            increment = row.get('volume', 0.0)
        else:  # tick bars
            increment = 1.0

        accumulator += increment

        # Create bar when threshold reached
        if accumulator >= threshold:
            bar_df = pd.DataFrame(bar_data)
            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df)
            })
            accumulator = 0.0
            bar_data = []

    # Add partial bar if exists
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bars.append({
            'timestamp': bar_df['timestamp'].iloc[-1],
            'open': bar_df['price'].iloc[0],
            'high': bar_df['price'].max(),
            'low': bar_df['price'].min(),
            'close': bar_df['price'].iloc[-1],
            'volume': bar_df.get('volume', pd.Series([0])).sum(),
            'tick_count': len(bar_df)
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
        default=1000,
        help='Tick bar threshold (default: 1000)'
    )
    parser.add_argument(
        '--volume-threshold',
        type=float,
        default=50.0,
        help='Volume bar threshold (default: 50.0)'
    )
    parser.add_argument(
        '--dollar-threshold',
        type=float,
        default=5e6,
        help='Dollar bar threshold (default: 5000000)'
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
