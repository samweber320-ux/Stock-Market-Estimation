# Stock Market Estimation Agent

This project provides a configurable agent capable of delivering stock price estimations using
verified market data, technical indicators, and recent news flow. It is designed to mirror the
workflow a professional technical analyst would follow while also allowing you to upload your own
datasets to augment the estimation process.

## Features

- **Verified market data** sourced from Yahoo! Finance and other reputable providers.
- **News awareness** powered by NewsAPI or local JSON files for offline use.
- **Comprehensive technical analysis** with moving averages, MACD, RSI, Bollinger Bands, and On-Balance Volume.
- **Top gainer intelligence** leveraging Webull and Robinhood leaderboards to extract common traits and surface a
  sidebar of potential momentum candidates.
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
