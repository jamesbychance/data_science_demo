# Quantitative Finance Pipeline: Data Curation & Feature Engineering

Technical demonstration of institutional-grade quantitative research infrastructure, implementing Stations 1-2 of the systematic trading assembly line as described in *Advances in Financial Machine Learning* by Marcos López de Prado. This repository showcases advanced market microstructure data processing, information-driven sampling techniques, and statistical feature engineering applied to cryptocurrency markets.

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

## Scope & Context

This repository demonstrates the **first two stations** of a complete quantitative trading assembly line:

### **Station 1: Data Curation** (Chapter 1)
- High-frequency tick-level data collection from Binance API
- ETL pipeline handling 1.13B+ trade events across multiple assets
- Data validation, cleaning, and indexing
- Market microstructure awareness (liquidity profiles, tick sizes)

### **Station 2: Feature Engineering** (Chapter 2 - Partial)
- Information-driven bar construction (tick, volume, dollar bars)
- Advanced sampling methods (imbalance bars, runs bars)
- Statistical property analysis and threshold calibration
- Transformation of raw ticks into structured features

**Not Yet Implemented** (remaining Station 2 work):
- Volatility estimation using EWMSTD (Exponentially Weighted Moving Standard Deviation)
- Triple barrier labeling method for ML classification
- Fractionally differentiated features
- Additional microstructure features (VPIN, order flow imbalance, etc.)

### **Subsequent Stations** (Private Development)
The complete quantitative research pipeline includes three additional stations beyond this demonstration:
- **Station 3: Strategy Development** - Transformation of features into investment algorithms
- **Station 4: Backtesting & Validation** - Historical performance simulation and overfitting analysis
- **Station 5: Deployment & Production** - Low-latency implementation with vectorization and parallel computing

This public repository focuses on foundational data infrastructure, demonstrating technical competencies in market microstructure, distributed data processing, and statistical feature engineering that form the basis of systematic trading research.

## Project Structure

```
data_science_demo/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Exclude data files
│
├── notebooks/                         # Jupyter notebooks for learning and analysis
│   ├── Ch1_overview.ipynb             # Chapter 1: Overview of financial ML
│   ├── Ch2_info_bars_BTC_polars.ipynb # Chapter 2: BTC information bars (polars)
│   ├── Ch2_info_bars_ETH_polars.ipynb # Chapter 2: ETH information bars (polars)
│   ├── Ch2_info_bars_SOL_polars.ipynb # Chapter 2: SOL information bars (polars)
│   ├── Ch2_info_bars_POWR_polars.ipynb # Chapter 2: POWR information bars (polars)
│   ├── Ch2_info_bar_*.pdf             # Exported PDF analysis reports
│   └── superseded/                    # Archived notebook versions
│
├── src/                               # Production-ready scripts
│   ├── data_collection.py             # Automated data collection
│   ├── standard_bars.py               # Tick/volume/dollar bars
│   ├── imbalance_bars.py              # Tick/volume/dollar imbalance bars
│   ├── runs_bars.py                   # Tick/volume/dollar runs bars
│   ├── utils/                         # Shared utilities
│   │   ├── bar_utils.py               # Bar construction utilities
│   │   └── data_loader.py             # Data loading utilities
│   └── superseded/                    # Legacy code versions
│
├── binance_raw_data/                  # Raw tick data (not in git)
│   ├── BTCUSDT/                       # Daily parquet files by symbol
│   ├── ETHUSDT/
│   ├── SOLUSDT/
│   └── POWRUSDT/
│
├── processed_bars/                    # Processed information-driven bars (not in git)
│   ├── BTCUSDT/
│   │   ├── tick_bars_YYYYMMDD_YYYYMMDD.parquet
│   │   ├── volume_bars_YYYYMMDD_YYYYMMDD.parquet
│   │   └── dollar_bars_YYYYMMDD_YYYYMMDD.parquet
│   ├── ETHUSDT/
│   ├── SOLUSDT/
│   └── POWRUSDT/
│
├── threshold_configs/                 # Bar threshold calibrations
│   └── {symbol}_thresholds_polars.json
│
└── superseded/                        # Archive of superseded files
    ├── notebooks/
    ├── processed_data/                # Old pandas-based bars
    └── TODOs.md
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
- **BTCUSDT** - High liquidity (~1.4M trades/day)
- **ETHUSDT** - High liquidity (~1.5M trades/day)
- **SOLUSDT** - Medium liquidity (~500k trades/day)
- **POWRUSDT** - Low liquidity (~9k trades/day)

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

All processed bars are saved to `processed_bars/{SYMBOL}/{bar_type}_{start_date}_{end_date}.parquet`

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

## Technical Highlights

This demonstration showcases several institutional-grade capabilities:

### Data Engineering
- **Scale**: 1.13B+ tick-level trades processed (366 days BTC, 288 days ETH, 390 days SOL/POWR)
- **ETL Architecture**: Incremental collection with rate limiting, error handling, and resume-from-interruption
- **Storage Format**: Parquet columnar storage for efficient analytical queries
- **Multi-Asset Handling**: Cross-liquidity analysis from high-liquidity (BTC: 1.4M trades/day) to low-liquidity (POWR: 9k trades/day) markets

### Statistical Feature Engineering
- **9 Bar Types Implemented**: Time, tick, volume, dollar, tick/volume/dollar imbalance, tick/volume/dollar runs
- **Adaptive Thresholds**: Dynamic calibration based on Exponentially Weighted Moving Average (EWMA)
- **Statistical Validation**: Normality tests, variance stability analysis, serial correlation testing
- **Performance**: Polars-based implementation processing 1M+ ticks per second

### Code Quality
- Production-ready Python modules with separation of concerns
- Comprehensive Jupyter notebooks demonstrating methodology and analysis
- Automated threshold configuration persistence
- Modular design enabling easy extension to additional bar types or assets

## Next Steps for Extension

This repository provides foundational infrastructure for quantitative research. To build a complete pipeline, the following extensions would be natural next steps:

### Completing Station 2 (Feature Engineering)
1. **Labeling Methods** (Chapter 3)
   - Triple barrier method for classification labels
   - Meta-labeling for position sizing
   - Trend-scanning labels

2. **Volatility Estimation** (Chapter 3)
   - EWMSTD (Exponentially Weighted Moving Standard Deviation)
   - Parkinson's high-low volatility estimator
   - Dynamic threshold adaptation based on volatility regimes

3. **Advanced Features** (Chapters 4-9)
   - Fractional differentiation for stationarity
   - Microstructure features (VPIN, Kyle's lambda, order flow imbalance)
   - Entropy-based features
   - Structural breaks detection

### Station 3-5 Implementation
- Strategy formulation and hypothesis testing
- Walk-forward backtesting with purging and embargo
- Combinatorial Purged Cross-Validation (CPCV)
- Low-latency production deployment

## Key References

- **Advances in Financial Machine Learning** (Marcos López de Prado)
- **Binance API Documentation** - `/api/v3/aggTrades` endpoint

