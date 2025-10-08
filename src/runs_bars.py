#!/usr/bin/env python3
"""
Chapter 2: Runs Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements runs-based sampling (Section 2.3.3):
- Tick Runs Bars
- Volume Runs Bars
- Dollar Runs Bars
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
# RUNS BARS
# ============================================================================

def tick_runs_bars(df: pd.DataFrame,
                   num_prev_bars: int = 3,
                   expected_runs_window: int = 10000) -> pd.DataFrame:
    """
    Tick Runs Bars - Sample when the length of runs exceeds expectations.

    Monitors sequences of buys in overall volume. Based on de Prado Chapter 2.3.2.3

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars for EWMA
    expected_runs_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        Tick runs bars
    """
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    bar_data = []

    expected_T = expected_runs_window
    prev_T_values = []
    prev_P_buy = []

    buy_run = 0
    sell_run = 0

    for idx, row in df.iterrows():
        bar_data.append(row)

        # Update run counts
        if row['b_t'] == 1:
            buy_run += 1
            sell_run = 0
        else:
            sell_run += 1
            buy_run = 0

        theta_t = max(buy_run, sell_run)

        # Expected run: E[θ_T] = E[T] * max{P[b_t=1], 1-P[b_t=1]}
        if len(prev_P_buy) > 0:
            P_buy = np.mean(prev_P_buy[-100:]) if len(prev_P_buy) >= 100 else np.mean(prev_P_buy)
        else:
            P_buy = 0.5

        expected_run = expected_T * max(P_buy, 1 - P_buy)

        if theta_t >= max(expected_run, expected_T * 0.3):
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'max_run': theta_t
            })

            prev_T_values.append(len(bar_df))
            prev_P_buy.append((bar_df['b_t'] == 1).sum() / len(bar_df))

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_data = []
            buy_run = 0
            sell_run = 0

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
            'max_run': max(buy_run, sell_run)
        })

    return pd.DataFrame(bars).set_index('timestamp')


def volume_runs_bars(df: pd.DataFrame,
                     num_prev_bars: int = 3,
                     expected_runs_window: int = 10000) -> pd.DataFrame:
    """
    Volume Runs Bars - Sample based on volume-weighted runs.

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars for EWMA
    expected_runs_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        Volume runs bars
    """
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    bar_data = []

    expected_T = expected_runs_window
    prev_T_values = []

    buy_volume_run = 0
    sell_volume_run = 0

    for idx, row in df.iterrows():
        bar_data.append(row)
        volume = row.get('volume', 0.0)

        if row['b_t'] == 1:
            buy_volume_run += volume
            sell_volume_run = 0
        else:
            sell_volume_run += volume
            buy_volume_run = 0

        theta_t = max(buy_volume_run, sell_volume_run)

        # Simple threshold based on expected T
        threshold = expected_T * 100  # Scale for volume

        if theta_t >= threshold:
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'max_volume_run': theta_t
            })

            prev_T_values.append(len(bar_df))
            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_data = []
            buy_volume_run = 0
            sell_volume_run = 0

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
            'max_volume_run': max(buy_volume_run, sell_volume_run)
        })

    return pd.DataFrame(bars).set_index('timestamp')


def dollar_runs_bars(df: pd.DataFrame,
                     num_prev_bars: int = 3,
                     expected_runs_window: int = 10000) -> pd.DataFrame:
    """
    Dollar Runs Bars - Sample based on dollar-value-weighted runs.

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with columns: timestamp, price, volume
    num_prev_bars : int
        Number of previous bars for EWMA
    expected_runs_window : int
        Initial expected number of ticks per bar

    Returns
    -------
    pd.DataFrame
        Dollar runs bars
    """
    df = df.copy()
    df['b_t'] = apply_tick_rule(df['price'])

    bars = []
    bar_data = []

    expected_T = expected_runs_window
    prev_T_values = []

    buy_dollar_run = 0
    sell_dollar_run = 0

    for idx, row in df.iterrows():
        bar_data.append(row)
        dollar_value = row['price'] * row.get('volume', 0.0)

        if row['b_t'] == 1:
            buy_dollar_run += dollar_value
            sell_dollar_run = 0
        else:
            sell_dollar_run += dollar_value
            buy_dollar_run = 0

        theta_t = max(buy_dollar_run, sell_dollar_run)

        # Scale threshold for dollar values
        threshold = expected_T * 10000  # Adjust based on typical dollar values

        if theta_t >= threshold:
            bar_df = pd.DataFrame(bar_data)

            bars.append({
                'timestamp': bar_df['timestamp'].iloc[-1],
                'open': bar_df['price'].iloc[0],
                'high': bar_df['price'].max(),
                'low': bar_df['price'].min(),
                'close': bar_df['price'].iloc[-1],
                'volume': bar_df.get('volume', pd.Series([0])).sum(),
                'tick_count': len(bar_df),
                'max_dollar_run': theta_t
            })

            prev_T_values.append(len(bar_df))
            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_data = []
            buy_dollar_run = 0
            sell_dollar_run = 0

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
            'max_dollar_run': max(buy_dollar_run, sell_dollar_run)
        })

    return pd.DataFrame(bars).set_index('timestamp')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate runs bars (tick, volume, dollar) from Binance data'
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
    print("\n### RUNS BARS ###\n")

    # Generate runs bars
    print(f"Generating tick runs bars (window: {args.expected_window:,})...")
    trb = tick_runs_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_runs_window=args.expected_window
    )
    print(f"✓ Tick Runs Bars generated: {len(trb)}")
    print(trb.head(3))

    print(f"\nGenerating volume runs bars (window: {args.expected_window:,})...")
    vrb = volume_runs_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_runs_window=args.expected_window
    )
    print(f"✓ Volume Runs Bars generated: {len(vrb)}")
    print(vrb.head(3))

    print(f"\nGenerating dollar runs bars (window: {args.expected_window:,})...")
    drb = dollar_runs_bars(
        tick_data,
        num_prev_bars=args.num_prev_bars,
        expected_runs_window=args.expected_window
    )
    print(f"✓ Dollar Runs Bars generated: {len(drb)}")
    print(drb.head(3))

    # Save to parquet
    print("\n" + "=" * 70)
    print("\nSaving bars...\n")

    save_bars(trb, args.symbol, 'tick_runs_bars')
    save_bars(vrb, args.symbol, 'volume_runs_bars')
    save_bars(drb, args.symbol, 'dollar_runs_bars')

    print("\n" + "=" * 70)
    print(f"\n✓ Runs bars for {args.symbol} generated and saved successfully!")
    print(f"\nSummary: {len(trb)} tick, {len(vrb)} volume, {len(drb)} dollar runs bars")
    print("\n" + "=" * 70)
