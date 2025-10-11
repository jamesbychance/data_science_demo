#!/usr/bin/env python3
"""
Chapter 2: Runs Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements runs-based sampling (Section 2.3.3):
- Tick Runs Bars
- Volume Runs Bars
- Dollar Runs Bars

THRESHOLD GUIDANCE:
-------------------
Runs bars sample when consecutive buys/sells exceed expected run lengths.
They detect momentum, trend exhaustion, and directional flow.

The --expected-window parameter sets the baseline, but the actual thresholds
are calculated using HARDCODED MULTIPLIERS in the code:
  - Volume runs: threshold = expected_window * 10
  - Dollar runs: threshold = expected_window * 400

TARGET OUTPUT:
  - Dollar Runs Bars: ~100-500 bars over 60 days
  - Captures momentum shifts and sustained directional moves
  - MORE bars during trending markets (strong runs)
  - FEWER bars during choppy, directionless markets

HOW TO CALIBRATE:
1. Start with expected-window = 100 (default for BTCUSDT)
2. Run on sample data (7-60 days)
3. Check output:
   - Too few bars (<50 total)? DECREASE expected-window OR reduce multipliers
   - Too many bars (>1000 total)? INCREASE expected-window OR increase multipliers
   - Just right (100-500)? Keep current setting

IMPORTANT - HARDCODED MULTIPLIERS:
The multipliers (10 for volume, 400 for dollar) are set in the code at:
  - Line ~232: threshold = expected_T * 10  (volume runs)
  - Line ~331: threshold = expected_T * 400 (dollar runs)

To adjust for different assets:
  - High volatility assets (SOL, POWR): Use LOWER multipliers (200-300)
  - Low volatility assets (BTC, ETH): Use HIGHER multipliers (400-600)
  - The multiplier controls how long a run must be to trigger a bar

EXAMPLES:
  BTCUSDT (stable, high liquidity):
    --expected-window 100
    # With multiplier=400 → Dollar threshold = $40k per run

  SOLUSDT (more volatile):
    --expected-window 100
    # Consider reducing multiplier to 300 in code for more bars

  Low liquidity assets:
    --expected-window 150
    # And consider reducing multipliers in code
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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    bar_start_idx = 0

    expected_T = expected_runs_window
    prev_T_values = []
    prev_P_buy = []

    buy_run = 0
    sell_run = 0

    for i in range(len(df)):
        # Update run counts
        if b_t_values[i] == 1:
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
            bar_end_idx = i + 1

            # Extract bar data using array slicing
            bar_prices = prices[bar_start_idx:bar_end_idx]
            bar_volumes = volumes[bar_start_idx:bar_end_idx]
            bar_b_t = b_t_values[bar_start_idx:bar_end_idx]

            bars.append({
                'timestamp': timestamps[i],
                'open': bar_prices[0],
                'high': bar_prices.max(),
                'low': bar_prices.min(),
                'close': bar_prices[-1],
                'volume': bar_volumes.sum(),
                'tick_count': bar_end_idx - bar_start_idx,
                'max_run': theta_t
            })

            prev_T_values.append(bar_end_idx - bar_start_idx)
            prev_P_buy.append((bar_b_t == 1).sum() / len(bar_b_t))

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_start_idx = bar_end_idx
            buy_run = 0
            sell_run = 0

    # Handle remaining data
    if bar_start_idx < len(df):
        bar_prices = prices[bar_start_idx:]
        bar_volumes = volumes[bar_start_idx:]

        bars.append({
            'timestamp': timestamps[-1],
            'open': bar_prices[0],
            'high': bar_prices.max(),
            'low': bar_prices.min(),
            'close': bar_prices[-1],
            'volume': bar_volumes.sum(),
            'tick_count': len(df) - bar_start_idx,
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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    bar_start_idx = 0

    expected_T = expected_runs_window
    prev_T_values = []

    buy_volume_run = 0
    sell_volume_run = 0

    for i in range(len(df)):
        # Update volume-weighted runs
        if b_t_values[i] == 1:
            buy_volume_run += volumes[i]
            sell_volume_run = 0
        else:
            sell_volume_run += volumes[i]
            buy_volume_run = 0

        theta_t = max(buy_volume_run, sell_volume_run)

        # Simple threshold based on expected T
        threshold = expected_T * 10  # Scale for volume

        if theta_t >= threshold:
            bar_end_idx = i + 1

            # Extract bar data using array slicing
            bar_prices = prices[bar_start_idx:bar_end_idx]
            bar_volumes = volumes[bar_start_idx:bar_end_idx]

            bars.append({
                'timestamp': timestamps[i],
                'open': bar_prices[0],
                'high': bar_prices.max(),
                'low': bar_prices.min(),
                'close': bar_prices[-1],
                'volume': bar_volumes.sum(),
                'tick_count': bar_end_idx - bar_start_idx,
                'max_volume_run': theta_t
            })

            prev_T_values.append(bar_end_idx - bar_start_idx)
            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_start_idx = bar_end_idx
            buy_volume_run = 0
            sell_volume_run = 0

    # Handle remaining data
    if bar_start_idx < len(df):
        bar_prices = prices[bar_start_idx:]
        bar_volumes = volumes[bar_start_idx:]

        bars.append({
            'timestamp': timestamps[-1],
            'open': bar_prices[0],
            'high': bar_prices.max(),
            'low': bar_prices.min(),
            'close': bar_prices[-1],
            'volume': bar_volumes.sum(),
            'tick_count': len(df) - bar_start_idx,
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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    bar_start_idx = 0

    expected_T = expected_runs_window
    prev_T_values = []

    buy_dollar_run = 0
    sell_dollar_run = 0

    for i in range(len(df)):
        dollar_value = prices[i] * volumes[i]

        # Update dollar-weighted runs
        if b_t_values[i] == 1:
            buy_dollar_run += dollar_value
            sell_dollar_run = 0
        else:
            sell_dollar_run += dollar_value
            buy_dollar_run = 0

        theta_t = max(buy_dollar_run, sell_dollar_run)

        # Scale threshold for dollar values
        threshold = expected_T * 400  # Adjust based on typical dollar values

        if theta_t >= threshold:
            bar_end_idx = i + 1

            # Extract bar data using array slicing
            bar_prices = prices[bar_start_idx:bar_end_idx]
            bar_volumes = volumes[bar_start_idx:bar_end_idx]

            bars.append({
                'timestamp': timestamps[i],
                'open': bar_prices[0],
                'high': bar_prices.max(),
                'low': bar_prices.min(),
                'close': bar_prices[-1],
                'volume': bar_volumes.sum(),
                'tick_count': bar_end_idx - bar_start_idx,
                'max_dollar_run': theta_t
            })

            prev_T_values.append(bar_end_idx - bar_start_idx)
            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            bar_start_idx = bar_end_idx
            buy_dollar_run = 0
            sell_dollar_run = 0

    # Handle remaining data
    if bar_start_idx < len(df):
        bar_prices = prices[bar_start_idx:]
        bar_volumes = volumes[bar_start_idx:]

        bars.append({
            'timestamp': timestamps[-1],
            'open': bar_prices[0],
            'high': bar_prices.max(),
            'low': bar_prices.min(),
            'close': bar_prices[-1],
            'volume': bar_volumes.sum(),
            'tick_count': len(df) - bar_start_idx,
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
        default=100,
        help='Initial expected number of ticks per bar (default: 100)'
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
