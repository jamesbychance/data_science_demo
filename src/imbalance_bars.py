#!/usr/bin/env python3
"""
Chapter 2: Imbalance Bars
Advances in Financial Machine Learning, Marcos Lopez de Prado

Implements imbalance-based sampling (Section 2.3.2):
- Tick Imbalance Bars (TIB)
- Volume Imbalance Bars (VIB)
- Dollar Imbalance Bars (DIB)

THRESHOLD GUIDANCE:
-------------------
Imbalance bars use an ADAPTIVE threshold based on expected window size.
They sample when order flow imbalance exceeds expectations.

The --expected-window parameter sets the baseline for adaptation.
Unlike standard bars, imbalance bars will generate FEWER bars (10-200 total
over 60 days) because they only trigger when significant order flow imbalances occur.

TARGET OUTPUT:
  - Dollar Imbalance Bars: ~50-200 bars over 60 days
  - Captures periods of informed trading and order flow toxicity
  - Expect MORE bars during volatile periods (crashes, rallies)
  - Expect FEWER bars during calm, balanced markets

HOW TO CALIBRATE:
1. Start with expected-window = 100 (default for BTCUSDT)
2. Run on sample data (7-60 days)
3. Check output:
   - Too few bars (<20 total)? DECREASE expected-window to 50-75
   - Too many bars (>500 total)? INCREASE expected-window to 200-500
   - Just right (50-200)? Keep current setting

KEY INSIGHT:
The EWMA mechanism makes these bars naturally adaptive. The expected-window
is just the starting point - the algorithm learns from the data and adjusts
thresholds dynamically based on recent imbalance patterns.

EXAMPLES:
  BTCUSDT (high liquidity, ~780k ticks/day):
    --expected-window 100  # Good balance for detecting imbalances

  Lower liquidity assets:
    --expected-window 50  # More sensitive to smaller imbalances

  Very high liquidity (during major events):
    --expected-window 200  # Reduce noise, focus on major imbalances
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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    theta_t = 0  # Cumulative tick imbalance
    bar_start_idx = 0

    # Track for EWMA calculations
    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for i in range(len(df)):
        theta_t += b_t_values[i]

        # Calculate expected imbalance: E[θ_T] = E[T] * |2P[b_t=1] - 1|
        # Estimate P[b_t=1] from recent history
        if len(prev_imbalance_values) > 0:
            # EWMA of (2*P[b_t=1] - 1)
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        # Sample when |θ_T| >= threshold
        if abs(theta_t) >= max(threshold, expected_T * 0.1):  # Use minimum threshold
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
                'imbalance': theta_t
            })

            # Update expected values with EWMA
            prev_T_values.append(bar_end_idx - bar_start_idx)
            prev_imbalance_values.append(bar_b_t.mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            # Reset
            theta_t = 0
            bar_start_idx = bar_end_idx

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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    theta_t = 0  # Cumulative volume imbalance: Σ(b_t * v_t)
    bar_start_idx = 0

    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for i in range(len(df)):
        theta_t += b_t_values[i] * volumes[i]

        # Expected imbalance calculation
        if len(prev_imbalance_values) > 0:
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        if abs(theta_t) >= max(threshold, expected_T * 0.1):
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
                'imbalance': theta_t
            })

            prev_T_values.append(bar_end_idx - bar_start_idx)
            # Track b_t * v_t average
            prev_imbalance_values.append((bar_b_t * bar_volumes).mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            theta_t = 0
            bar_start_idx = bar_end_idx

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

    # Extract numpy arrays for faster iteration
    timestamps = df['timestamp'].values
    prices = df['price'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    b_t_values = df['b_t'].values

    bars = []
    theta_t = 0  # Cumulative dollar imbalance: Σ(b_t * v_t * price_t)
    bar_start_idx = 0

    expected_T = expected_imbalance_window
    prev_T_values = []
    prev_imbalance_values = []

    for i in range(len(df)):
        dollar_value = prices[i] * volumes[i]
        theta_t += b_t_values[i] * dollar_value

        if len(prev_imbalance_values) > 0:
            expected_imbalance = np.mean(prev_imbalance_values[-100:]) if len(prev_imbalance_values) >= 100 else np.mean(prev_imbalance_values)
        else:
            expected_imbalance = 0.0

        threshold = expected_T * abs(expected_imbalance)

        if abs(theta_t) >= max(threshold, expected_T * 0.1):
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
                'imbalance': theta_t
            })

            prev_T_values.append(bar_end_idx - bar_start_idx)
            prev_imbalance_values.append((bar_b_t * bar_prices * bar_volumes).mean())

            if len(prev_T_values) > num_prev_bars:
                expected_T = np.mean(prev_T_values[-num_prev_bars:])

            theta_t = 0
            bar_start_idx = bar_end_idx

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
