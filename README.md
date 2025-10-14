# Stock Market Estimation Agent

This project provides a configurable agent capable of delivering stock price estimations using
verified market data, technical indicators, and recent news flow. It is designed to mirror the
workflow a professional technical analyst would follow while also allowing you to upload your own
datasets to augment the estimation process.

## Features

- **Verified market data** sourced from Yahoo! Finance and other reputable providers.
- **News awareness** powered by NewsAPI or local JSON files for offline use.
- **Comprehensive technical analysis** with moving averages, MACD, RSI, Bollinger Bands, and On-Balance Volume.
- **Top gainer intelligence** leveraging Webull and Robinhood leaderboards to extract common traits, learn 100-day
  pre-breakout indicator patterns, and surface a sidebar of potential momentum candidates with at least 30% pattern
  confidence.
- **User uploaded datasets** that are merged with official market data prior to analysis.
- **Transparent outputs** including the data sources used and a narrative summary of the drivers
  behind the estimation.

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

Results are printed as JSON to stdout unless `--output` is provided.

Each estimation includes a `potential_top_gainers` sidebar summarising U.S. equities that currently exhibit the
pre-breakout indicator profile learned from the prior 100 days of brokerage leaders. Only symbols meeting the
learned thresholds with at least 30% confidence are surfaced, and each entry lists the aligned pattern characteristics
alongside the agent's confidence score.

### Offline recommendation snapshot

For scenarios where network access is unavailable, the repository ships with a curated
`research/offline_recommendations.json` file that distils three liquid U.S. leaders (NVDA, MSFT, LLY) using
pre-captured technical, fundamental, and news checkpoints from TradingView, MarketSmith, Reuters, Bloomberg, CNBC,
the Wall Street Journal, Microsoft Investor Relations, NVIDIA Investor Relations, Eli Lilly Investor Relations, and
the U.S. Food and Drug Administration. Load the snapshot with:

```python
from stock_estimation_agent.offline_recommendations import load_offline_recommendations

for idea in load_offline_recommendations():
    print(idea.to_dict())
```

The helper mirrors the agent's source tracking so you can reference the underlying verified articles and filings when
responding to investment due diligence questions without re-running the full estimation pipeline.
