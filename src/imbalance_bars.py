#!/usr/bin/env python3
"""
Chapter 2: Imbalance Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements imbalance-based sampling (Section 2.3.2):
- Tick Imbalance Bars (TIB)
- Volume Imbalance Bars (VIB)
- Dollar Imbalance Bars (DIB)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import numpy as np
import pandas as pd
from utils.bar_utils import apply_tick_rule, save_bars
from utils.data_loader import load_binance_data, get_available_symbols


# ============================================================================
# IMBALANCE BARS
# ============================================================================

def tick_imbalance_bars(df: pd.DataFrame,
                        num_prev_bars: int = 3,
                        expected_imbalance_window: int = 10000) -> pd.DataFrame:
    """
    Tick Imbalance Bars (TIB) - Sample when tick imbalances exceed expectations.

    Based on de Prado Chapter 2.3.2.1

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars to use for EWMA of expected T
    expected_imbalance_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        TIB bars
    """
    # Apply tick rule
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    theta_t = 0  # Cumulative tick imbalance
    bar_data = []

    # Track for EWMA calculations
    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for idx, row in df.iterrows():
        bar_data.append(row)
        theta_t += row['b_t']

        # Calculate expected imbalance: E[θ_T] = E[T] * |2P[b_t=1] - 1|
        # Estimate P[b_t=1] from recent history
        if len(prev_imbalance_values) > 0:
            # EWMA of (2*P[b_t=1] - 1)
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        # Sample when |θ_T| >= threshold
        if abs(theta_t) >= max(threshold, expected_T * 0.5):  # Use minimum threshold
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'imbalance': theta_t
            })

            # Update expected values with EWMA
            prev_T_values.append(len(bar_df))
            prev_imbalance_values.append(bar_df['b_t'].mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            # Reset
            theta_t = 0
            bar_data = []

    # Handle remaining data
    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bars.append({
            'timestamp': bar_df['timestamp'].iloc[-1],
            'open': bar_df['price'].iloc[0],
            'high': bar_df['price'].max(),
            'low': bar_df['price'].min(),
            'close': bar_df['price'].iloc[-1],
            'volume': bar_df.get('volume', pd.Series([0])).sum(),
            'tick_count': len(bar_df),
            'imbalance': theta_t
        })

    return pd.DataFrame(bars).set_index('timestamp')


def volume_imbalance_bars(df: pd.DataFrame,
                          num_prev_bars: int = 3,
                          expected_imbalance_window: int = 10000) -> pd.DataFrame:
    """
    Volume Imbalance Bars (VIB) - Sample when volume imbalances exceed expectations.

    Based on de Prado Chapter 2.3.2.2

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars for EWMA
    expected_imbalance_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        VIB bars
    """
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    theta_t = 0  # Cumulative volume imbalance: Σ(b_t * v_t)
    bar_data = []

    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for idx, row in df.iterrows():
        bar_data.append(row)
        volume = row.get('volume', 0.0)
        theta_t += row['b_t'] * volume

        # Expected imbalance calculation
        if len(prev_imbalance_values) > 0:
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        if abs(theta_t) >= max(threshold, expected_T * 0.5):
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'imbalance': theta_t
            })

            prev_T_values.append(len(bar_df))
            # Track b_t * v_t average
            prev_imbalance_values.append((bar_df['b_t'] * bar_df.get('volume', 0)).mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            theta_t = 0
            bar_data = []

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bars.append({
            'timestamp': bar_df['timestamp'].iloc[-1],
            'open': bar_df['price'].iloc[0],
            'high': bar_df['price'].max(),
            'low': bar_df['price'].min(),
            'close': bar_df['price'].iloc[-1],
            'volume': bar_df.get('volume', pd.Series([0])).sum(),
            'tick_count': len(bar_df),
            'imbalance': theta_t
        })

    return pd.DataFrame(bars).set_index('timestamp')


def dollar_imbalance_bars(df: pd.DataFrame,
                          num_prev_bars: int = 3,
                          expected_imbalance_window: int = 10000) -> pd.DataFrame:
    """
    Dollar Imbalance Bars (DIB) - Sample when dollar imbalances exceed expectations.

    Based on de Prado Chapter 2.3.2.2

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars for EWMA
    expected_imbalance_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        DIB bars
    """
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    theta_t = 0  # Cumulative dollar imbalance: Σ(b_t * v_t * price_t)
    bar_data = []

    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for idx, row in df.iterrows():
        bar_data.append(row)
        dollar_value = row['price'] * row.get('volume', 0.0)
        theta_t += row['b_t'] * dollar_value

        if len(prev_imbalance_values) > 0:
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        if abs(theta_t) >= max(threshold, expected_T * 0.5):
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'imbalance': theta_t
            })

            prev_T_values.append(len(bar_df))
            prev_imbalance_values.append((bar_df['b_t'] * bar_df['price'] * bar_df.get('volume', 0)).mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            theta_t = 0
            bar_data = []

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bars.append({
            'timestamp': bar_df['timestamp'].iloc[-1],
            'open': bar_df['price'].iloc[0],
            'high': bar_df['price'].max(),
            'low': bar_df['price'].min(),
            'close': bar_df['price'].iloc[-1],
            'volume': bar_df.get('volume', pd.Series([0])).sum(),
            'tick_count': len(bar_df),
            'imbalance': theta_t
        })

    return pd.DataFrame(bars).set_index('timestamp')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate imbalance bars (tick, volume, dollar) from Binance data'
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
        '--num-prev-bars',
        type=int,
        default=3,
        help='Number of previous bars for EWMA (default: 3)'
    )
    parser.add_argument(
        '--expected-window',
        type=int,
        default=10000,
        help='Initial expected number of ticks per bar (default: 10000)'
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
    print("\n### IMBALANCE BARS ###\n")

    # Generate imbalance bars
    print(f"Generating tick imbalance bars (window: {args.expected_window:,})...")
    tib = tick_imbalance_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_imbalance_window=args.expected_window
    )
    print(f"✓ Tick Imbalance Bars generated: {len(tib)}")
    print(tib.head(3))

    print(f"\nGenerating volume imbalance bars (window: {args.expected_window:,})...")
    vib = volume_imbalance_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_imbalance_window=args.expected_window
    )
    print(f"✓ Volume Imbalance Bars generated: {len(vib)}")
    print(vib.head(3))

    print(f"\nGenerating dollar imbalance bars (window: {args.expected_window:,})...")
    dib = dollar_imbalance_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_imbalance_window=args.expected_window
    )
    print(f"✓ Dollar Imbalance Bars generated: {len(dib)}")
    print(dib.head(3))

    # Save to parquet
    print("\n" + "=" * 70)
    print("\nSaving bars...\n")

    save_bars(tib, args.symbol, 'tick_imbalance_bars')
    save_bars(vib, args.symbol, 'volume_imbalance_bars')
    save_bars(dib, args.symbol, 'dollar_imbalance_bars')

    print("\n" + "=" * 70)
    print(f"\n✓ Imbalance bars for {args.symbol} generated and saved successfully!")
    print(f"\nSummary: {len(tib)} tick, {len(vib)} volume, {len(dib)} dollar imbalance bars")
    print("\n" + "=" * 70)
