"""Command line entry point for running the stock estimation agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from stock_estimation_agent.agent import AgentConfig, StockEstimationAgent
from stock_estimation_agent.builder import create_default_agent
from stock_estimation_agent.data_sources import (
    StaticHistoricalFetcher,
    StaticTopGainersFetcher,
    TopGainer,
)
from stock_estimation_agent.estimation import EstimationEngine, EstimationResult
from stock_estimation_agent.indicators import IndicatorCalculator
from stock_estimation_agent.offline import LocalNewsFetcher
from stock_estimation_agent.top_gainers import TopGainerAnalytics


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
        "market_date": result.market_date.isoformat(),
        "reference_close": result.reference_close,
        "narrative": result.narrative,
        "target_price": result.target_price,
        "confidence": result.confidence,
        "supporting_indicators": result.supporting_indicators,
        "news_headlines": result.news_headlines,
        "sources": [source.__dict__ for source in result.sources],
        "top_gainer_factors": result.top_gainer_factors,
        "top_gainer_narrative": result.top_gainer_narrative,
        "potential_top_gainers": result.potential_top_gainers,
    }


def _load_top_gainers(path: Path) -> list[TopGainer]:
    payload = json.loads(path.read_text())
    gainers: list[TopGainer] = []
    for entry in payload:
        def _maybe_int(value):
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        def _maybe_float(value):
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        gainers.append(
            TopGainer(
                symbol=entry.get("symbol", ""),
                name=entry.get("name", ""),
                last_price=_maybe_float(entry.get("last_price")) or 0.0,
                percent_change=_maybe_float(entry.get("percent_change")) or 0.0,
                volume=_maybe_int(entry.get("volume")),
                average_volume=_maybe_int(entry.get("average_volume")),
                market_cap=_maybe_float(entry.get("market_cap")),
                source=entry.get("source", "user"),
                sector=entry.get("sector"),
            )
        )
    return gainers


def build_agent(args: argparse.Namespace, estimation_engine: EstimationEngine) -> StockEstimationAgent:
    if args.offline:
        if args.symbol is None:
            raise ValueError("A symbol must be provided when using offline mode.")
        if args.historical_csv is None:
            raise ValueError("--historical-csv must be provided in offline mode.")
        if args.news_json is None:
            raise ValueError("--news-json must be provided in offline mode.")

        history = pd.read_csv(args.historical_csv, index_col=0, parse_dates=True)
        historical_fetcher = StaticHistoricalFetcher({args.symbol: history})
        news_fetcher = LocalNewsFetcher(args.news_json)
        top_gainers = _load_top_gainers(args.top_gainers_json) if args.top_gainers_json else []
        agent = StockEstimationAgent(
            historical_fetcher=historical_fetcher,
            news_fetcher=news_fetcher,
            indicator_calculator=IndicatorCalculator(),
            estimation_engine=estimation_engine,
            top_gainers_fetcher=StaticTopGainersFetcher(top_gainers),
            top_gainer_analytics=TopGainerAnalytics(),
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
    parser.add_argument("symbol", nargs="?", help="Ticker symbol to estimate")
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
    parser.add_argument(
        "--top-gainers-json",
        type=Path,
        help="Optional JSON array describing recent top gainers for offline mode.",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Screen for top recommendations using current previous-day data instead of estimating a single symbol.",
    )
    parser.add_argument(
        "--recommend-count",
        type=int,
        default=10,
        help="Maximum number of recommendations to return when --recommend is set.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold when generating recommendations.",
    )
    parser.add_argument(
        "--screen-symbol",
        dest="screen_symbols",
        action="append",
        default=[],
        help="Additional symbol to include in the screening universe (can be provided multiple times).",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimation_engine = EstimationEngine()
    agent = build_agent(args, estimation_engine)

    if args.symbol:
        for dataset_path in args.user_dataset:
            dataset = _load_user_dataset(dataset_path)
            agent.register_user_dataset(args.symbol, dataset)

    if args.recommend:
        if not args.news_api_key and not args.offline:
            # build_agent would have raised for missing API key in online mode if symbol was required,
            # but recommendation mode may skip symbol entirely. Enforce the requirement explicitly.
            raise ValueError("--news-api-key is required to generate recommendations in online mode.")

        recommendations = agent.recommend_top_symbols(
            limit=args.recommend_count,
            min_confidence=args.min_confidence,
            extra_symbols=args.screen_symbols,
        )
        payload = [rec.to_dict() for rec in recommendations]

        if args.output:
            args.output.write_text(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return

    result = agent.estimate(args.symbol)

    if args.output:
        args.output.write_text(json.dumps(_serialize_result(result), indent=2))
    else:
        print(json.dumps(_serialize_result(result), indent=2))


if __name__ == "__main__":
    main()
