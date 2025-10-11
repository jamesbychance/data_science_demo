# TODOs

**Saturday, 11 October 2025**

The plan for today...

1. get the latest raw btc data - there was a market crash so should make for interesting analysis!
2. process the data for BTCUSDT over the last 60 days
```
time python3 src/standard_bars.py BTCUSDT --days 60
time python3 src/imbalance_bars.py BTCUSDT --days 60
time python3 src/runs_bars.py BTCUSDT --days 60
```
3. I have excellent data, so create a notebook to visualise these bars to understand what is going on. Will turn that into a very breif report discussing.


       Your BTCUSDT Dataset:

        - 92 days of raw tick data (Jul 8 - Oct 8, 2025)
        - ~1M trades/day on average (high liquidity)
        - Captures the recent crash - ideal for analyzing regime changes

       How Much Processed Data You Need:

       For feature engineering & analysis, process the last 60 days (Aug 9 - Oct 8):
        - Gives you ~60M raw trades
        - Leaves first 30 days for future out-of-sample testing
        - Captures both normal market + crash period

       Which Bar Types to Process:

       DON'T process all 7 - that's overkill. Here's the optimal set:

       ✅ Process these 4:
        1. Dollar bars - Best for high-liquidity assets like BTC (normalizes for price changes)
        2. Dollar imbalance bars - Captures order flow toxicity during crash
        3. Dollar runs bars - Detects momentum shifts and trend exhaustion
        4. Tick imbalance - Skip, dollar imbalance is superior for BTC

       ❌ Skip these 3:
        - Time bars - Poor statistical properties (you already know this)
        - Tick bars - Redundant when you have dollar bars
        - Volume bars - Dollar bars already account for this

       Why dollar-based only? BTC price can change 10%+ during a crash - dollar-based bars automatically adjust thresholds, while tick/volume bars don't.

       Command:
        python src/standard_bars.py BTCUSDT --days 60
        python src/imbalance_bars.py BTCUSDT --days 60
        python src/runs_bars.py BTCUSDT --days 60

       This gives you ~3 complementary bar types capturing different market microstructure signals
       for robust feature engineering.

4. Begin the feature engineering phase! as outlined in the project roadmap!
