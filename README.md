# Data Science Demo

End-to-end quantitative research pipeline transforming raw cryptocurrency market data into ML-ready features. Demonstrates advanced ETL capabilities, market microstructure analysis, and information-driven data sampling techniques from *Advances in Financial Machine Learning* by Marcos López de Prado.

## What This Project Does

This project solves a fundamental problem in quantitative trading: **traditional time-based sampling (hourly/daily bars) loses critical information during volatile periods**. Instead, this pipeline:

1. **Collects raw tick-level trade data** from Binance at millisecond precision
2. **Transforms tick data into information-driven bars** (dollar bars, volume bars, imbalance bars, runs bars)
3. **Engineers ML-ready features** from market microstructure that have superior statistical properties for prediction models

**Why this matters:** Information-driven bars provide better statistical properties (closer to normality, more stable variance) compared to time bars, leading to more robust machine learning models for:
- Price prediction and return forecasting
- Market regime detection (trending vs mean-reverting states)
- Risk management and volatility prediction
- Systematic trading strategy development

## Project Structure

```
data_science_demo/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Exclude data files
│
├── finML_book/                        # Completed chapters from Financial ML by De Prado, learn first then apply
│
├── notebooks/
│    ├── 01_data_collection.ipynb      # ✅ Tick data collection from Binance 
│    └── 02_information_bars.ipynb     # ✅ Better statiscical properties
│
├── src/                               # Production-ready scripts
│   ├── data_collection.py             # Automated data collection
│   ├── standard_bars.py               # Tick/volume/dollar bars
│   ├── imbalance_bars.py              # Tick/volume/dollar imbalance bars
│   ├── runs_bars.py                   # Tick/volume/dollar runs bars
│   └── utils/
│       ├── bar_utils.py               # Shared bar construction utilities
│       └── data_loader.py             # Data loading utilities
│
├── binance_raw_data/                  # Raw tick data (not in git)
│   ├── BTCUSDT_2m.parquet
│   ├── ETHUSDT_2m.parquet
│   ├── SOLUSDT_2m.parquet
│   └── POWRUSDT_2m.parquet
│
└── processed_data/                    # Transformed bars (not in git)
    └── *_bars.parquet
```

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `requests` - API calls to Binance
- `pandas` - Data manipulation
- `pyarrow` - Parquet file format
- `matplotlib` - Visualisation
- `jupyterlab` - Interactive notebooks

## How to Use

### Data Collection

**Option 1: Using Jupyter Notebook** (Recommended for exploration)
```bash
jupyter lab
# Open notebooks/01_data_collection.ipynb and run cells
```

**Option 2: Using Python Script** (Automated/production)
```bash
# Default: 4 symbols, 3 months
python src/data_collection.py

# Custom: single symbol, 12 months
python src/data_collection.py --symbols BTCUSDT --months 12

# Custom: multiple symbols, 6 months
python src/data_collection.py --symbols BTCUSDT ETHUSDT SOLUSDT --months 6

# Custom output directory
python src/data_collection.py --symbols BTCUSDT --months 12 --output-dir custom_data

# Skip validation step (faster)
python src/data_collection.py --symbols BTCUSDT --months 12 --skip-validation

# View all options
python src/data_collection.py --help
```

**Command-line arguments:**
- `--symbols` - Symbols to collect (default: BTCUSDT ETHUSDT SOLUSDT POWRUSDT)
- `--months` - Number of months to collect (default: 3)
- `--output-dir` - Output directory for data (default: binance_raw_data)
- `--skip-validation` - Skip the data validation step at the end

**Default collection:** 3 months of tick-level trade data for 4 assets with different liquidity profiles:
- **BTCUSDT** - High liquidity (~100k trades/day)
- **ETHUSDT** - High liquidity (~80k trades/day)
- **SOLUSDT** - Medium liquidity (~30k trades/day)
- **POWRUSDT** - Low liquidity (~10k trades/day)

**Features:**
- Incremental collection (resume from interruptions)
- Rate limiting and error handling
- Saves to `binance_raw_data/{SYMBOL}/YYYY-MM-DD.parquet` (daily files)

### Data Processing - Information-Driven Bars

After collecting raw data, transform it into information-driven bars:

**Standard Bars (Time, Tick, Volume, Dollar):**
```bash
# Default: BTCUSDT, last 7 days
python src/standard_bars.py

# Specify asset
python src/standard_bars.py ETHUSDT

# Specify asset and time window
python src/standard_bars.py SOLUSDT --days 14

# Custom thresholds
python src/standard_bars.py BTCUSDT --tick-threshold 1000 --volume-threshold 100 --dollar-threshold 1000000
```

**Imbalance Bars (Tick/Volume/Dollar Imbalance):**
```bash
python src/imbalance_bars.py BTCUSDT
python src/imbalance_bars.py ETHUSDT --days 14
```

**Runs Bars (Tick/Volume/Dollar Runs):**
```bash
python src/runs_bars.py BTCUSDT
python src/runs_bars.py SOLUSDT --days 7
```

**Available Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, POWRUSDT

All processed bars are saved to `processed_data/{SYMBOL}_{bar_type}.parquet`

### Understanding the Bar Types

1. **Time Bars** - Traditional sampling at fixed intervals (baseline for comparison)
2. **Tick Bars** - Sample every N trades
3. **Volume Bars** - Sample every N contracts traded
4. **Dollar Bars** - Sample every $X volume traded
5. **Tick Imbalance Bars** - Sample when cumulative buy/sell imbalance exceeds threshold
6. **Volume Imbalance Bars** - Sample when volume-weighted imbalance exceeds threshold
7. **Dollar Imbalance Bars** - Sample when dollar-weighted imbalance exceeds threshold
8. **Runs Bars** - Sample based on sequences of consecutive buy or sell trades

Information-driven bars (4-8) provide superior statistical properties compared to time bars for machine learning applications.

## Current Status

### ✅ Completed
- **Part 1: Data Collection** - Automated ETL pipeline collecting 2M+ trades across 4 assets
- **Part 2: Data Transformation** - Implementation of 7 different information-driven bar types


## Future Plans

The following sections from the project roadmap will be implemented:

### Feature engineering and analysis
### Multi-Asset analysis

## Key References

- **Advances in Financial Machine Learning** (Marcos López de Prado)
- **Binance API Documentation** - `/api/v3/aggTrades` endpoint

