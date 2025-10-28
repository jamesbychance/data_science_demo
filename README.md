# Quantitative Finance Pipeline: Data Curation

Technical demonstration of institutional-grade quantitative research infrastructure, sharing only Station 1 of the systematic trading assembly line. This repository showcases advanced market microstructure data processing and information-driven sampling techniques applied to cryptocurrency markets.

## What This Project Does

This project solves a fundamental problem in quantitative trading: **traditional time-based sampling (hourly/daily bars) loses critical information during volatile periods**. Instead, this pipeline:

1. **Collects raw tick-level trade data** from Binance at millisecond precision
2. **Transforms tick data into information-driven bars** (dollar bars, volume bars, imbalance bars, runs bars)
3. **Raw data ready for feature engineering** revealing market microstructure that has superior statistical properties for prediction models

## Scope & Context

This repository demonstrates the **first station** of a complete quantitative trading assembly line:

### **Station 1: Data Curation**
- High-frequency tick-level data collection from Binance API
- ETL pipeline handling 1.13B+ trade events across multiple assets
- Data validation, cleaning, and indexing
- Market microstructure awareness (liquidity profiles, tick sizes)

### **Subsequent Stations** (Private Development)
The complete quantitative research pipeline includes four additional stations beyond this demonstration:
- **Station 2: Feature Engineering** - Converts non-stationary and non-IID data into ML ready features
- **Station 3: Strategy Development** - Transformation of features into investment algorithms
- **Station 4: Backtesting & Validation** - Historical performance simulation and overfitting analysis
- **Station 5: Deployment & Production** - Low-latency implementation with vectorization and parallel computing

## Project Structure

```
data_science_demo/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Exclude data files
├── notebooks/                         # Jupyter notebooks for learning and analysis
├── src/                               # Production-ready scripts
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
└── threshold_configs/                 # Bar threshold calibrations
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

**Directory location:**
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

## Key References

- **Binance API Documentation** - `/api/v3/aggTrades` endpoint

