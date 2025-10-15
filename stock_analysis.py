"""Command-line tool for multi-indicator technical analysis.

This module downloads historical price data using the ``yfinance``
package and calculates a handful of classic technical indicators:

* Average Directional Index (ADX)
* Moving Average Convergence Divergence (MACD)
* Relative Strength Index (RSI)
* On-Balance Volume (OBV)
* Accumulation/Distribution (A/D) Line
* Aroon Indicator
* Stochastic Oscillator

The CLI prints the latest readings together with a lightweight
interpretation so that analysts can quickly gauge whether the
indicators agree or diverge.  The tool is intentionally educational –
it helps the user apply a blended indicator approach as recommended by
many technicians, without claiming predictive certainty.

Example
-------
>>> python stock_analysis.py AAPL --lookback-days 5

Note
----
Network access is required to download data from Yahoo Finance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


try:  # pragma: no cover - optional dependency handling
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "yfinance is required for this script. Install it with 'pip install yfinance'."
    ) from exc


Number = float


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Return the exponential moving average with Wilder style smoothing."""

    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram."""

    ema_fast = _ema(close, span=12)
    ema_slow = _ema(close, span=26)
    macd = ema_fast - ema_slow
    signal = _ema(macd, span=9)
    histogram = macd - signal
    return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""

    delta = close.diff().fillna(0.0)
    direction = delta.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    obv = (direction * volume).cumsum()
    return obv.fillna(0.0)


def calculate_accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Calculate the Accumulation/Distribution (A/D) line."""

    price_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / price_range
    clv = clv.fillna(0.0)
    return (clv * volume).cumsum()


def calculate_tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate the True Range (TR)."""

    previous_close = close.shift(1)
    ranges = pd.DataFrame(
        {
            "hl": high - low,
            "hc": (high - previous_close).abs(),
            "lc": (low - previous_close).abs(),
        }
    )
    return ranges.max(axis=1)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Average Directional Index (ADX)."""

    tr = calculate_tr(high, low, close)

    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0.0)


def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Calculate the Aroon Up and Down indicators."""

    highs = high.to_numpy()
    lows = low.to_numpy()
    aroon_up: List[Number] = []
    aroon_down: List[Number] = []

    for i in range(len(highs)):
        start = max(0, i - period + 1)
        window_high = highs[start : i + 1]
        window_low = lows[start : i + 1]

        idx_high = window_high.argmax()
        idx_low = window_low.argmin()

        periods_since_high = i - (start + idx_high)
        periods_since_low = i - (start + idx_low)

        aroon_up_value = 100 * (period - periods_since_high) / period
        aroon_down_value = 100 * (period - periods_since_low) / period

        aroon_up.append(aroon_up_value)
        aroon_down.append(aroon_down_value)

    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down}, index=high.index)


def calculate_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smoothing: int = 3
) -> pd.DataFrame:
    """Calculate the Stochastic Oscillator (%K and %D)."""

    rolling_low = low.rolling(window=period, min_periods=period).min()
    rolling_high = high.rolling(window=period, min_periods=period).max()
    denominator = (rolling_high - rolling_low).replace(0, np.nan)
    percent_k = 100 * (close - rolling_low) / denominator
    percent_d = percent_k.rolling(window=smoothing, min_periods=smoothing).mean()
    return pd.DataFrame({"percent_k": percent_k.fillna(0.0), "percent_d": percent_d.fillna(0.0)})


@dataclass
class IndicatorSnapshot:
    as_of: pd.Timestamp
    close: float
    rsi: float
    macd: float
    macd_signal: float
    adx: float
    obv_trend: float
    ad_trend: float
    aroon_up: float
    aroon_down: float
    stochastic_k: float
    stochastic_d: float


def summarize_indicators(data: pd.DataFrame) -> IndicatorSnapshot:
    """Compile the latest readings from each indicator."""

    macd_df = calculate_macd(data["Close"])
    rsi = calculate_rsi(data["Close"])
    adx = calculate_adx(data["High"], data["Low"], data["Close"])
    obv = calculate_obv(data["Close"], data["Volume"])
    ad_line = calculate_accumulation_distribution(
        data["High"], data["Low"], data["Close"], data["Volume"]
    )
    aroon_df = calculate_aroon(data["High"], data["Low"])
    stochastic_df = calculate_stochastic(data["High"], data["Low"], data["Close"])

    obv_trend = obv.iloc[-1] - obv.iloc[max(-6, -len(obv))]
    ad_trend = ad_line.iloc[-1] - ad_line.iloc[max(-6, -len(ad_line))]

    as_of = data.index[-1]
    if getattr(as_of, "tzinfo", None) is None:
        as_of = pd.Timestamp(as_of, tz="UTC")

    return IndicatorSnapshot(
        as_of=as_of,
        close=float(data["Close"].iloc[-1]),
        rsi=float(rsi.iloc[-1]),
        macd=float(macd_df["macd"].iloc[-1]),
        macd_signal=float(macd_df["signal"].iloc[-1]),
        adx=float(adx.iloc[-1]),
        obv_trend=float(obv_trend),
        ad_trend=float(ad_trend),
        aroon_up=float(aroon_df["aroon_up"].iloc[-1]),
        aroon_down=float(aroon_df["aroon_down"].iloc[-1]),
        stochastic_k=float(stochastic_df["percent_k"].iloc[-1]),
        stochastic_d=float(stochastic_df["percent_d"].iloc[-1]),
    )


def interpret_snapshot(snapshot: IndicatorSnapshot) -> List[str]:
    """Return human-readable interpretations for the indicator snapshot."""

    statements: List[str] = []

    if snapshot.rsi >= 70:
        statements.append("RSI is overbought (>=70); momentum may be stretched.")
    elif snapshot.rsi <= 30:
        statements.append("RSI is oversold (<=30); momentum may be washed out.")
    else:
        statements.append("RSI is neutral, pointing to balanced momentum.")

    if snapshot.macd > snapshot.macd_signal and snapshot.macd > 0:
        statements.append("MACD is above its signal and zero line, supporting bullish momentum.")
    elif snapshot.macd < snapshot.macd_signal and snapshot.macd < 0:
        statements.append("MACD is below its signal and zero line, indicating bearish momentum.")
    else:
        statements.append("MACD is mixed; watch for clear crossovers or moves above/below zero.")

    if snapshot.adx >= 40:
        statements.append("ADX >= 40 signals a strong trend environment.")
    elif snapshot.adx <= 20:
        statements.append("ADX <= 20 suggests a weak or range-bound market.")
    else:
        statements.append("ADX between 20 and 40 implies a developing but moderate trend.")

    if snapshot.obv_trend > 0:
        statements.append("OBV has risen over the last week, hinting at accumulation.")
    elif snapshot.obv_trend < 0:
        statements.append("OBV has fallen over the last week, pointing to distribution.")
    else:
        statements.append("OBV is flat, showing little directional volume pressure.")

    if snapshot.ad_trend > 0:
        statements.append("The A/D line is rising, signaling buying pressure.")
    elif snapshot.ad_trend < 0:
        statements.append("The A/D line is falling, signaling selling pressure.")
    else:
        statements.append("The A/D line is flat, showing muted accumulation/distribution cues.")

    if snapshot.aroon_up > snapshot.aroon_down:
        statements.append("Aroon Up exceeds Aroon Down, favouring an uptrend.")
    elif snapshot.aroon_up < snapshot.aroon_down:
        statements.append("Aroon Down exceeds Aroon Up, favouring a downtrend.")
    else:
        statements.append("Aroon signals are tied, reflecting indecision.")

    if snapshot.stochastic_k >= 80 and snapshot.stochastic_d >= 80:
        statements.append("Stochastic Oscillator is overbought (>80); monitor for reversals.")
    elif snapshot.stochastic_k <= 20 and snapshot.stochastic_d <= 20:
        statements.append("Stochastic Oscillator is oversold (<20); a bounce is possible.")
    else:
        statements.append("Stochastic readings are neutral, implying no extreme condition.")

    return statements


DATA_STORE_DIR = Path("data_store")
MEMORY_DIR = DATA_STORE_DIR / "memories"
DEFAULT_MEMORY_RETENTION_DAYS = 7


def _memory_path(symbol: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR / f"{symbol.upper()}.json"


def load_memory(symbol: str) -> list[dict[str, object]]:
    """Load prior analyses stored for the symbol."""

    path = _memory_path(symbol)
    if not path.exists():
        return []

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def filter_recent_history(
    history: list[dict[str, object]], retention_days: int
) -> list[dict[str, object]]:
    """Keep only entries that fall within the retention window."""

    cutoff = datetime.now(tz=ZoneInfo("America/Chicago")) - timedelta(days=retention_days)
    recent: list[dict[str, object]] = []

    for entry in history:
        as_of_str = entry.get("as_of")
        if not isinstance(as_of_str, str):
            continue

        try:
            as_of = datetime.fromisoformat(as_of_str)
        except ValueError:
            continue

        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=ZoneInfo("UTC"))

        if as_of >= cutoff:
            recent.append(entry)

    return recent


def store_memory(
    symbol: str,
    snapshot: IndicatorSnapshot,
    interpretations: Iterable[str],
    note: Optional[str],
    data_store_path: Path,
    retention_days: int,
) -> tuple[Path, int]:
    """Persist the analysis snapshot so future runs can learn from it."""

    path = _memory_path(symbol)
    history = load_memory(symbol)
    recent_history = filter_recent_history(history, retention_days)
    pruned_count = len(history) - len(recent_history)

    record = {
        "as_of": snapshot.as_of.isoformat(),
        "close": snapshot.close,
        "rsi": snapshot.rsi,
        "macd": snapshot.macd,
        "macd_signal": snapshot.macd_signal,
        "adx": snapshot.adx,
        "obv_trend": snapshot.obv_trend,
        "ad_trend": snapshot.ad_trend,
        "aroon_up": snapshot.aroon_up,
        "aroon_down": snapshot.aroon_down,
        "stochastic_k": snapshot.stochastic_k,
        "stochastic_d": snapshot.stochastic_d,
        "interpretations": list(interpretations),
        "note": note or "",
        "data_cache": str(data_store_path),
        "stored_at": datetime.now(tz=ZoneInfo("America/Chicago")).isoformat(),
    }

    recent_history.append(record)
    path.write_text(json.dumps(recent_history, indent=2))
    return path, pruned_count


def determine_central_date_range(
    start: Optional[str], end: Optional[str], lookback_days: int
) -> tuple[str, str]:
    """Return start/end ISO strings anchored to US Central time if not provided."""

    tz = ZoneInfo("America/Chicago")
    now_central = datetime.now(tz)

    end_date = end
    if not end_date:
        end_dt = now_central + timedelta(days=1)
        end_date = end_dt.date().isoformat()

    start_date = start
    if not start_date:
        minimum_days = max(lookback_days, 60)
        anchor = now_central - timedelta(days=minimum_days)
        start_date = anchor.date().isoformat()

    return start_date, end_date


def download_price_history(
    symbol: str, start: Optional[str], end: Optional[str], lookback_days: int
) -> pd.DataFrame:
    """Download price data from Yahoo Finance."""

    derived_start, derived_end = determine_central_date_range(start, end, lookback_days)

    data = yf.download(
        symbol,
        start=derived_start,
        end=derived_end,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError("No data returned. Check the symbol or date range.")

    return data


def store_price_history(symbol: str, data: pd.DataFrame) -> Path:
    """Persist downloaded data for future reference."""

    DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_STORE_DIR / f"{symbol.upper()}.csv"

    serialisable = data.copy()
    serialisable.index.name = "Date"

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        combined = pd.concat([existing, serialisable])
    else:
        combined = serialisable

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_csv(path)
    return path


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def print_memory_history(
    symbol: str,
    history: list[dict[str, object]],
    limit: Optional[int],
    retention_days: int,
) -> None:
    """Display previously stored analyses for reference."""

    recent_history = filter_recent_history(history, retention_days)

    if not recent_history:
        print(
            f"No stored memory within the last {retention_days} day(s) for {symbol.upper()} yet."
        )
        return

    if limit is not None and limit > 0:
        to_show = recent_history[-limit:]
    else:
        to_show = recent_history

    retained_message = (
        f"Showing {len(to_show)} stored analyses from the last {retention_days} day(s)."
    )
    print(retained_message)
    print("Stored analyses")
    print("---------------")
    for entry in to_show:
        print(f"As of: {entry['as_of']} | Close: {format_currency(entry['close'])}")
        print(f"RSI: {entry['rsi']:.2f} | MACD: {entry['macd']:.4f} vs Signal {entry['macd_signal']:.4f}")
        print(f"ADX: {entry['adx']:.2f} | OBV Δ5: {entry['obv_trend']:,.0f} | A/D Δ5: {entry['ad_trend']:,.0f}")
        print(
            "Aroon Up/Down: "
            f"{entry['aroon_up']:.2f}/{entry['aroon_down']:.2f} | "
            f"Stoch %K/%D: {entry['stochastic_k']:.2f}/{entry['stochastic_d']:.2f}"
        )
        if entry.get("note"):
            print(f"Note: {entry['note']}")
        print(f"Stored at: {entry['stored_at']}")
        print("-" * 40)


def print_report(
    symbol: str,
    snapshot: IndicatorSnapshot,
    interpretations: Iterable[str],
    data_store_path: Path,
    memory_path: Path,
    retention_days: int,
    pruned_count: int,
) -> None:
    """Pretty-print the indicator snapshot and interpretations."""

    central_time = snapshot.as_of.tz_convert(ZoneInfo("America/Chicago"))

    print(f"Technical analysis for {symbol.upper()}")
    print("=" * (22 + len(symbol)))
    print(f"Data through: {central_time:%Y-%m-%d %H:%M %Z}")
    print(f"Last close: {format_currency(snapshot.close)}")
    print(f"RSI: {snapshot.rsi:.2f}")
    print(f"MACD: {snapshot.macd:.4f} (Signal: {snapshot.macd_signal:.4f})")
    print(f"ADX: {snapshot.adx:.2f}")
    print(f"Aroon Up: {snapshot.aroon_up:.2f} | Aroon Down: {snapshot.aroon_down:.2f}")
    print(f"Stochastic %K: {snapshot.stochastic_k:.2f} | %D: {snapshot.stochastic_d:.2f}")
    print(f"OBV 5-day change: {snapshot.obv_trend:,.0f}")
    print(f"A/D 5-day change: {snapshot.ad_trend:,.0f}")
    print()
    print(f"Data cached at: {data_store_path}")
    memory_line = (
        f"Memory stored at: {memory_path} (retaining {retention_days} day window"
        f"; removed {pruned_count} stale entr{'y' if pruned_count == 1 else 'ies'})."
    )
    print(memory_line)
    print("Information sources: Yahoo Finance price/volume data via yfinance; "
          "local cache and memory files for prior analyses.")
    print("Interpretation")
    print("-------------")
    for statement in interpretations:
        print(f"- {statement}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-indicator technical analysis tool")
    parser.add_argument("symbol", help="Ticker symbol to analyse (e.g., AAPL)")
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD). Overrides the automatic lookback window.",
    )
    parser.add_argument("--end", help="End date (YYYY-MM-DD).", default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=21,
        help=(
            "Number of calendar days to reference when automatically selecting the start date. "
            "Anchored to US Central time."
        ),
    )
    parser.add_argument(
        "--note",
        help="Store a reminder or lesson alongside the generated analysis for future runs.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Display previously stored analyses for this symbol to simulate learning over time.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=5,
        help="Limit how many prior analyses are displayed when --show-history is used (default: 5).",
    )
    parser.add_argument(
        "--memory-retention-days",
        type=int,
        default=DEFAULT_MEMORY_RETENTION_DAYS,
        help=(
            "Number of days of stored analyses to retain. Older entries are pruned automatically "
            "to ensure only up-to-date information is recycled."
        ),
    )
    return parser


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    lookback_days = max(parsed.lookback_days, 1)
    retention_days = max(parsed.memory_retention_days, 1)
    if parsed.show_history:
        history = load_memory(parsed.symbol)
        print_memory_history(parsed.symbol, history, parsed.history_limit, retention_days)
        print()

    data = download_price_history(
        parsed.symbol,
        parsed.start,
        parsed.end,
        lookback_days,
    )
    data_store_path = store_price_history(parsed.symbol, data)
    snapshot = summarize_indicators(data)
    interpretations = interpret_snapshot(snapshot)
    memory_path, pruned_count = store_memory(
        parsed.symbol,
        snapshot,
        interpretations,
        parsed.note,
        data_store_path,
        retention_days,
    )
    print_report(
        parsed.symbol,
        snapshot,
        interpretations,
        data_store_path,
        memory_path,
        retention_days,
        pruned_count,
    )


if __name__ == "__main__":
    main()

