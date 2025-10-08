#!/usr/bin/env python3
"""
Data loading utilities for Binance raw data
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import glob


def load_binance_data(
    symbol: str,
    days: int = 7,
    data_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load Binance tick data from parquet files.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'POWRUSDT')
    days : int, optional
        Number of most recent days to load (default: 7)
    data_dir : str, optional
        Base directory containing binance data (default: '../../binance_raw_data' relative to src/)

    Returns
    -------
    pd.DataFrame
        Tick data with columns: timestamp, price, volume
        Sorted by timestamp, ready for bar generation
    """

    if data_dir is None:
        # Default to project root's binance_raw_data directory
        script_dir = Path(__file__).parent
        data_dir = str(script_dir / '..' / '..' / 'binance_raw_data')

    # Get all parquet files for this symbol
    symbol_dir = Path(data_dir) / symbol

    if not symbol_dir.exists():
        raise FileNotFoundError(
            f"No data found for {symbol}. "
            f"Available symbols: {[d.name for d in Path(data_dir).iterdir() if d.is_dir()]}"
        )

    parquet_files = sorted(symbol_dir.glob('*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {symbol_dir}")

    # Take last N days
    parquet_files = parquet_files[-days:]

    print(f"Loading {len(parquet_files)} days of {symbol} data...")
    print(f"Date range: {parquet_files[0].stem} to {parquet_files[-1].stem}")

    # Load and concatenate all files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Rename columns to match expected format
    df = df.rename(columns={'quantity': 'volume'})

    # Keep only needed columns
    df = df[['timestamp', 'price', 'volume']].copy()

    # Ensure sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(df):,} ticks")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Total volume: {df['volume'].sum():,.2f}")
    print()

    return df


def get_available_symbols(data_dir: Optional[str] = None) -> list:
    """
    Get list of available trading symbols.

    Parameters
    ----------
    data_dir : str, optional
        Base directory containing binance data (default: '../../binance_raw_data' relative to src/)

    Returns
    -------
    list
        List of available symbol directories
    """
    if data_dir is None:
        # Default to project root's binance_raw_data directory
        script_dir = Path(__file__).parent
        data_dir = str(script_dir / '..' / '..' / 'binance_raw_data')

    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    return [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != 'superseded']
