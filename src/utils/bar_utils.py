#!/usr/bin/env python3
"""
Shared utility functions for bar sampling.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Dict, Any


def apply_tick_rule(price_series: pd.Series) -> pd.Series:
    """
    Apply the tick rule to classify trades as buy (1) or sell (-1).

    The tick rule:
    - b_t = b_{t-1} if ΔP_t = 0
    - b_t = |ΔP_t| / ΔP_t if ΔP_t ≠ 0

    Parameters
    ----------
    price_series : pd.Series
        Price series from tick data

    Returns
    -------
    pd.Series
        Series of 1 (buy) or -1 (sell) classifications
    """
    price_diff = price_series.diff()
    b_t = pd.Series(index=price_series.index, dtype=float)

    # Initialize first value as 1 (arbitrary, but common choice)
    b_t.iloc[0] = 1.0

    for i in range(1, len(price_diff)):
        if price_diff.iloc[i] != 0:
            b_t.iloc[i] = np.sign(price_diff.iloc[i])
        else:
            b_t.iloc[i] = b_t.iloc[i-1]

    return b_t


def save_bars(bars: pd.DataFrame,
              symbol: str,
              bar_type: str,
              threshold: Optional[float] = None,
              output_dir: Optional[str] = None) -> None:
    """
    Save bars to parquet format with metadata.

    Parameters
    ----------
    bars : pd.DataFrame
        Bar data to save
    symbol : str
        Asset symbol (e.g., 'BTCUSDT')
    bar_type : str
        Type of bars (e.g., 'dollar_bars', 'tick_imbalance_bars')
    threshold : float, optional
        Threshold parameter used for bar construction
    output_dir : str, optional
        Output directory (default: '../processed_data' relative to src/)
    """
    if output_dir is None:
        # Default to project root's processed_data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'processed_data')

    os.makedirs(output_dir, exist_ok=True)

    # Extract date range from bars index
    start_date = bars.index[0].strftime('%Y%m%d')
    end_date = bars.index[-1].strftime('%Y%m%d')

    # Save bars with date range in filename
    filename = f'{symbol}_{bar_type}_{start_date}_{end_date}.parquet'
    output_path = os.path.join(output_dir, filename)
    bars.to_parquet(output_path, compression='snappy')

    # Save metadata
    metadata: Dict[str, Any] = {
        'symbol': symbol,
        'bar_type': bar_type,
        'num_bars': len(bars),
        'date_range': [str(bars.index[0]), str(bars.index[-1])],
        'columns': list(bars.columns)
    }

    if threshold is not None:
        metadata['threshold'] = threshold

    metadata_path = output_path.replace('.parquet', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Saved {len(bars)} {bar_type} for {symbol} to {output_path}')
