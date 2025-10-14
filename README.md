# Stock Market Estimation Agent

This project provides a configurable agent capable of delivering stock price estimations using
verified market data, technical indicators, and recent news flow. It is designed to mirror the
workflow a professional technical analyst would follow while also allowing you to upload your own
datasets to augment the estimation process.

## Features

- **Verified market data** sourced from Yahoo! Finance and other reputable providers, constrained to the most recent
  completed U.S. trading session.
- **News awareness** powered by NewsAPI or local JSON files for offline use.
- **Comprehensive technical analysis** with moving averages, MACD, RSI, Bollinger Bands, and On-Balance Volume.
- **Historical response testing** that evaluates how similar indicator profiles behaved on the following session so trend
  insights are grounded in realised market reactions instead of stale projections.
- **Top gainer intelligence** leveraging Webull and Robinhood leaderboards to extract common traits, learn 100-day
  pre-breakout indicator patterns, and surface a sidebar of potential momentum candidates with at least 70% pattern
  confidence.
- **User uploaded datasets** that are merged with official market data prior to analysis.
- **Transparent outputs** including the data sources used and a narrative summary of the drivers
  behind the estimation.
- **U.S. Central Time rollovers** with a helper that flags when a new previous-day session is
  available so you can refresh prices and news as soon as they settle.

## Installation

```bash
pip install -r requirements.txt
```

The default agent depends on the following libraries:

- `pandas`
- `numpy`
- `yfinance`
- `requests`

Install them manually or via the provided requirements file.

## Usage

### Online mode (default)

```bash
python main.py AAPL --news-api-key YOUR_NEWSAPI_KEY --output result.json
```

### Generating daily recommendations

To surface the strongest setups detected across the latest previous U.S. trading
day, enable recommendation mode. The agent will pull the prior session's
prices, evaluate top gainers, and return a list of symbols whose indicator
patterns align with the learned breakout profile and satisfy the configured
confidence threshold (70% by default).

```bash
python main.py --recommend --news-api-key YOUR_NEWSAPI_KEY --recommend-count 10 --min-confidence 0.7
```

Add `--screen-symbol SYMBOL` for each extra ticker you want assessed alongside
the brokerage top gainer universe. Results are emitted as a JSON array, with
each entry including the target price, potential upside, risk banding, and the
verified sources used in the evaluation.

### Offline mode

Provide your own historical price CSV (indexed by date) and a JSON file of news articles. The news
file should contain a list of objects with at least `title`, `publishedAt`, and an optional
`symbols` array listing the tickers each headline references.

```bash
python main.py AAPL --offline --historical-csv data/aapl.csv --news-json data/aapl_news.json \
    --top-gainers-json data/top_gainers.json
```

When running offline you may optionally provide a JSON array of recent top gainers containing
`symbol`, `name`, `last_price`, `percent_change`, and optional volume/sector metadata so the agent can
continue to profile market leadership without making external calls.

You can add proprietary indicators or alternative datasets with the `--user-dataset` flag. Each
file is merged by timestamp before indicators are computed.

```bash
python main.py AAPL --news-api-key YOUR_NEWSAPI_KEY --user-dataset data/my_alpha.csv
```

Results are printed as JSON to stdout unless `--output` is provided. When
`--recommend` is active, the JSON payload contains an array of recommendation
objects mirroring the `AgentRecommendation` dataclass exposed by the library so
you can use the API directly.

### Working with offline research snapshots

If you capture your own previous-trading-day research pack you can still take advantage of the
dataclasses exposed in `stock_estimation_agent.offline`. Create a JSON payload matching the structure
expected by `load_offline_recommendations`, making sure the `market_date` reflects the most recent
completed U.S. session and that every cited article links to a verified, up-to-date source. Example:

```python
from pathlib import Path

from stock_estimation_agent.offline import load_offline_recommendations

recommendations = load_offline_recommendations(Path("my_latest_snapshot.json"))
for item in recommendations:
    print(item.symbol, item.pattern_confidence, item.potential_upside_pct)
```

This keeps the project free of stale reference data while still allowing you to archive and review
your own latest analytics when operating without network access.

Each estimation includes a `potential_top_gainers` sidebar summarising U.S. equities that currently exhibit the
pre-breakout indicator profile learned from the prior 100 days of brokerage leaders. Only symbols meeting the
learned thresholds with at least 70% confidence are surfaced, and each entry lists the aligned pattern characteristics
alongside the agent's confidence score.

### Tracking U.S. Central trading day rollovers

Whenever you need to determine if a new set of previous-day data is available, call
`StockEstimationAgent.detect_new_market_day()`. The helper uses America/Chicago time so the transition aligns with
the U.S. equities calendar and market close.

```python
from datetime import datetime

from stock_estimation_agent.builder import create_default_agent

agent = create_default_agent(news_api_key="YOUR_NEWSAPI_KEY")
is_new_day, market_date = agent.detect_new_market_day()
if is_new_day:
    print(f"Refresh caches using the {market_date.date()} session before running new estimations.")
```

You can supply your own timestamp—for example, to process historical archives—and the method will return the correct
previous trading day as of that moment. This allows schedulers to wait until the Central Time market close before
pulling the freshly completed session.

### Accuracy and maintenance guidelines

- **Previous trading day focus** – Every estimation is anchored to the last completed U.S. trading session. Historical
  candles earlier than that are only used to study how similar indicator states behaved, not to extrapolate stale price
  targets.
- **Confidence scores** – Pattern confidence reflects how closely a symbol matches the technical factors the agent learned
  from the prior top-gainer cohort; it is a probabilistic heuristic rather than an audited forecast. Treat the numbers as a
  relative ranking signal that should be validated with your own analysis instead of a guarantee of future performance.
- **Source verification** – All cited headlines originate from NewsAPI's catalogue of reputable outlets (Reuters,
  Bloomberg, CNBC, The Wall Street Journal, etc.) or user supplied datasets. Always confirm the articles are still current
  and check for follow-on developments.
- **Live mode expectations** – When run online, the agent pulls the previous trading day's prices and the most recent
  verified headlines available at execution time. Reliability is bounded by the upstream APIs (Yahoo! Finance, NewsAPI,
  Webull, Robinhood) and your network connectivity. Monitor for rate limits or provider outages and cross-check results if
  anything appears inconsistent.
