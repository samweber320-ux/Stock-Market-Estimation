"""Command line entry point for running the stock estimation agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from stock_estimation_agent.agent import AgentConfig, StockEstimationAgent
from stock_estimation_agent.builder import create_default_agent
from stock_estimation_agent.data_sources import StaticHistoricalFetcher
from stock_estimation_agent.estimation import EstimationEngine, EstimationResult
from stock_estimation_agent.indicators import IndicatorCalculator
from stock_estimation_agent.offline import LocalNewsFetcher


def _load_user_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Parquet file must use a DatetimeIndex.")
    else:
        raise ValueError("Unsupported file type. Provide CSV or Parquet.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _serialize_result(result: EstimationResult) -> Dict[str, Any]:
    return {
        "symbol": result.symbol,
        "generated_at": result.generated_at.isoformat(),
        "narrative": result.narrative,
        "target_price": result.target_price,
        "confidence": result.confidence,
        "supporting_indicators": result.supporting_indicators,
        "news_headlines": result.news_headlines,
        "sources": [source.__dict__ for source in result.sources],
    }


def build_agent(args: argparse.Namespace, estimation_engine: EstimationEngine) -> StockEstimationAgent:
    if args.offline:
        if args.historical_csv is None:
            raise ValueError("--historical-csv must be provided in offline mode.")
        if args.news_json is None:
            raise ValueError("--news-json must be provided in offline mode.")

        history = pd.read_csv(args.historical_csv, index_col=0, parse_dates=True)
        historical_fetcher = StaticHistoricalFetcher({args.symbol: history})
        news_fetcher = LocalNewsFetcher(args.news_json)
        agent = StockEstimationAgent(
            historical_fetcher=historical_fetcher,
            news_fetcher=news_fetcher,
            indicator_calculator=IndicatorCalculator(),
            estimation_engine=estimation_engine,
        )
    else:
        if args.news_api_key is None:
            raise ValueError("--news-api-key is required unless --offline is set.")
        agent = create_default_agent(news_api_key=args.news_api_key)
        agent.estimation_engine = estimation_engine

    agent.config = AgentConfig(
        history_lookback=args.lookback,
        news_window_days=args.news_window,
        min_data_points=args.min_history,
    )
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stock estimation agent")
    parser.add_argument("symbol", help="Ticker symbol to estimate")
    parser.add_argument("--news-api-key", dest="news_api_key", help="NewsAPI key for fetching headlines")
    parser.add_argument("--lookback", type=int, default=365, help="Historical lookback window in days")
    parser.add_argument("--news-window", type=int, default=7, help="Number of days of news to include")
    parser.add_argument("--min-history", type=int, default=60, help="Minimum data points required for estimation")
    parser.add_argument(
        "--user-dataset",
        type=Path,
        action="append",
        default=[],
        help="Optional CSV/Parquet file with user-provided indicators indexed by datetime.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode with user supplied historical prices and news JSON.",
    )
    parser.add_argument("--historical-csv", type=Path, help="Historical prices CSV for offline mode.")
    parser.add_argument("--news-json", type=Path, help="News JSON file for offline mode.")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimation_engine = EstimationEngine()
    agent = build_agent(args, estimation_engine)

    for dataset_path in args.user_dataset:
        dataset = _load_user_dataset(dataset_path)
        agent.register_user_dataset(args.symbol, dataset)

    result = agent.estimate(args.symbol)

    if args.output:
        args.output.write_text(json.dumps(_serialize_result(result), indent=2))
    else:
        print(json.dumps(_serialize_result(result), indent=2))


if __name__ == "__main__":
    main()
