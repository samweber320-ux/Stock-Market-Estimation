# Stock Market Estimation Toolkit

This repository provides a lightweight command-line helper for running
multi-indicator technical analysis on any Yahoo Finance ticker.

## Features

The `stock_analysis.py` script downloads historical price and volume
data that is anchored to current US Central Time and calculates several
popular indicators:

* Average Directional Index (ADX)
* Moving Average Convergence Divergence (MACD)
* Relative Strength Index (RSI)
* On-Balance Volume (OBV)
* Accumulation/Distribution (A/D) Line
* Aroon Indicator
* Stochastic Oscillator

The tool prints the most recent indicator readings, highlights whether
signals align or conflict, and encourages combining indicators instead
of relying on a single signal. Every download is cached locally so that
subsequent analyses can build on the data previously collected. Each
run also stores a memory of the indicator snapshot, interpretations, and
any lesson you attach, letting you review prior observations like a
research journal.

## Getting Started

```bash
pip install -r requirements.txt
```

## Usage

Download the latest few weeks of data (anchored in US Central time) and
review the indicator summary:

```bash
python stock_analysis.py AAPL
```

Specify a custom date range:

```bash
python stock_analysis.py MSFT --start 2023-01-01 --end 2024-01-01
```

Focus on the latest market action by limiting the automatic lookback
window (minimum one day):

```bash
python stock_analysis.py TSLA --lookback-days 5
```

Store a takeaway alongside the indicators and review your last few
analyses to simulate a learning workflow:

```bash
python stock_analysis.py NVDA --note "Watch for MACD crossovers." --show-history --history-limit 3
```

## Notes

* Yahoo Finance data retrieval requires an active internet connection.
* The script is intended for educational analysis only—it does not
  provide trading advice.
* Cached CSV files are written to `data_store/` for future reference.
* Stored memories live under `data_store/memories/` and capture each
  indicator snapshot plus any note you supply.

