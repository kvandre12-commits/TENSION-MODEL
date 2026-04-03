#!/usr/bin/env python3
"""
build_tension_model_v1.py

Build a simple tension model:
- structural_score = weak/strong condition
- participation_score = attention/engagement
- tension_score = participation - structure
- setup_flag = weak structure + strong participation
- target_5d_abs_15 = 1 if abs(5-day forward return) >= 15%

Expected source table:
    bars_daily(date TEXT, symbol TEXT, close REAL, volume REAL)

Output table:
    tension_features_daily

Usage:
    python build_tension_model_v1.py --db data/spy_truth.db --source bars_daily --symbol MAXN

Or for all symbols:
    python build_tension_model_v1.py --db data/spy_truth.db --source bars_daily
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


OUTPUT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tension_features_daily (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,

    close REAL,
    volume REAL,

    ret_1d REAL,
    ret_20d REAL,
    sma20 REAL,
    dist_sma20 REAL,
    avg_vol20 REAL,
    rel_volume REAL,

    structural_score REAL,
    participation_score REAL,
    tension_score REAL,

    setup_flag INTEGER DEFAULT 0,

    fwd_5d_return REAL,
    fwd_5d_abs_return REAL,
    target_5d_abs_15 INTEGER DEFAULT 0,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);
"""

OUTPUT_INDEXES_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_tension_symbol_date
    ON tension_features_daily(symbol, date);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tension_setup_flag
    ON tension_features_daily(setup_flag);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tension_target
    ON tension_features_daily(target_5d_abs_15);
    """,
]


@dataclass
class Thresholds:
    structural_max: float = -0.50
    participation_min: float = 0.50
    abs_move_threshold: float = 0.15


def rolling_zscore(series: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def clip_score(series: pd.Series, low: float = -3.0, high: float = 3.0) -> pd.Series:
    clipped = series.clip(lower=low, upper=high)
    return clipped / 3.0  # maps roughly into [-1, 1]


def ensure_output_table(conn: sqlite3.Connection) -> None:
    conn.execute(OUTPUT_TABLE_SQL)
    for sql in OUTPUT_INDEXES_SQL:
        conn.execute(sql)
    conn.commit()


def load_price_data(
    conn: sqlite3.Connection,
    source_table: str,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    query = f"""
    SELECT date, symbol, close, volume
    FROM {source_table}
    WHERE close IS NOT NULL
      AND volume IS NOT NULL
    """
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)

    query += " ORDER BY symbol, date"

    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        raise ValueError("No rows found in source table.")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close", "volume"]).copy()

    return df


def build_features(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    out = df.copy()

    def per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()

        # Base features
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_20d"] = g["close"].pct_change(20)
        g["sma20"] = g["close"].rolling(20, min_periods=20).mean()
        g["dist_sma20"] = (g["close"] / g["sma20"]) - 1.0

        g["avg_vol20"] = g["volume"].rolling(20, min_periods=20).mean()
        g["rel_volume"] = g["volume"] / g["avg_vol20"]

        # Forward label
        g["fwd_5d_return"] = g["close"].shift(-5) / g["close"] - 1.0
        g["fwd_5d_abs_return"] = g["fwd_5d_return"].abs()
        g["target_5d_abs_15"] = (g["fwd_5d_abs_return"] >= thresholds.abs_move_threshold).astype("Int64")

        # Normalize with rolling z-scores
        g["ret_20d_z"] = clip_score(rolling_zscore(g["ret_20d"], window=60, min_periods=20))
        g["dist_sma20_z"] = clip_score(rolling_zscore(g["dist_sma20"], window=60, min_periods=20))
        g["rel_volume_z"] = clip_score(rolling_zscore(g["rel_volume"], window=60, min_periods=20))
        g["ret_1d_z"] = clip_score(rolling_zscore(g["ret_1d"], window=60, min_periods=20))

        # Scores
        g["structural_score"] = 0.5 * g["ret_20d_z"] + 0.5 * g["dist_sma20_z"]
        g["participation_score"] = 0.5 * g["rel_volume_z"] + 0.5 * g["ret_1d_z"]
        g["tension_score"] = g["participation_score"] - g["structural_score"]

        # Setup rule
        g["setup_flag"] = (
            (g["structural_score"] <= thresholds.structural_max) &
            (g["participation_score"] >= thresholds.participation_min)
        ).astype(int)

        return g

    out = out.groupby("symbol", group_keys=False).apply(per_symbol)

    # Clean final columns
    keep_cols = [
        "date", "symbol", "close", "volume",
        "ret_1d", "ret_20d", "sma20", "dist_sma20",
        "avg_vol20", "rel_volume",
        "structural_score", "participation_score", "tension_score",
        "setup_flag",
        "fwd_5d_return", "fwd_5d_abs_return", "target_5d_abs_15",
    ]
    out = out[keep_cols].copy()

    # SQLite-friendly dates
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    return out


def write_output(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str = "tension_features_daily") -> None:
    if df.empty:
        return

    records = df.where(pd.notnull(df), None).to_dict("records")

    upsert_sql = f"""
    INSERT INTO {table_name} (
        date, symbol, close, volume,
        ret_1d, ret_20d, sma20, dist_sma20,
        avg_vol20, rel_volume,
        structural_score, participation_score, tension_score,
        setup_flag,
        fwd_5d_return, fwd_5d_abs_return, target_5d_abs_15
    ) VALUES (
        :date, :symbol, :close, :volume,
        :ret_1d, :ret_20d, :sma20, :dist_sma20,
        :avg_vol20, :rel_volume,
        :structural_score, :participation_score, :tension_score,
        :setup_flag,
        :fwd_5d_return, :fwd_5d_abs_return, :target_5d_abs_15
    )
    ON CONFLICT(date, symbol) DO UPDATE SET
        close = excluded.close,
        volume = excluded.volume,
        ret_1d = excluded.ret_1d,
        ret_20d = excluded.ret_20d,
        sma20 = excluded.sma20,
        dist_sma20 = excluded.dist_sma20,
        avg_vol20 = excluded.avg_vol20,
        rel_volume = excluded.rel_volume,
        structural_score = excluded.structural_score,
        participation_score = excluded.participation_score,
        tension_score = excluded.tension_score,
        setup_flag = excluded.setup_flag,
        fwd_5d_return = excluded.fwd_5d_return,
        fwd_5d_abs_return = excluded.fwd_5d_abs_return,
        target_5d_abs_15 = excluded.target_5d_abs_15,
        created_at = CURRENT_TIMESTAMP
    ;
    """

    with conn:
        conn.executemany(upsert_sql, records)


def print_summary(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["target_5d_abs_15"]).copy()
    if valid.empty:
        print("No valid labeled rows to summarize yet.")
        return

    baseline_rate = valid["target_5d_abs_15"].mean()

    flagged = valid[valid["setup_flag"] == 1].copy()
    flagged_rate = flagged["target_5d_abs_15"].mean() if not flagged.empty else np.nan

    print("\n=== Tension Model Summary ===")
    print(f"Rows: {len(valid):,}")
    print(f"Baseline hit rate (15%+ abs move in 5d): {baseline_rate:.2%}")

    if flagged.empty:
        print("Flagged setups: 0")
    else:
        print(f"Flagged setups: {len(flagged):,}")
        print(f"Flagged hit rate: {flagged_rate:.2%}")
        print(f"Lift vs baseline: {(flagged_rate / baseline_rate):.2f}x" if baseline_rate > 0 else "Lift vs baseline: n/a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--source", default="bars_daily", help="Source table with date, symbol, close, volume")
    parser.add_argument("--symbol", default=None, help="Optional single symbol filter")
    parser.add_argument("--structural-max", type=float, default=-0.50)
    parser.add_argument("--participation-min", type=float, default=0.50)
    parser.add_argument("--abs-move-threshold", type=float, default=0.15)
    args = parser.parse_args()

    thresholds = Thresholds(
        structural_max=args.structural_max,
        participation_min=args.participation_min,
        abs_move_threshold=args.abs_move_threshold,
    )

    conn = sqlite3.connect(args.db)
    try:
        ensure_output_table(conn)
        raw = load_price_data(conn, args.source, args.symbol)
        feat = build_features(raw, thresholds)
        write_output(conn, feat)
        print_summary(feat)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
