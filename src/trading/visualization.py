"""Execution-point visualisations for the trading product outputs."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.trading.trade_labeling import pattern_direction


def render_product_visualisations(
    df_ohlcv: pd.DataFrame,
    cfg: dict,
    product_root: str,
    stage1_threshold: float,
) -> None:
    """Create stock-price charts for Stage 2 test trades and paper replay trades."""
    pattern_name = cfg["labeling"]["active_pattern"]
    direction = pattern_direction(pattern_name)

    stage2_csv = os.path.join(product_root, "stage2", "test_trade_candidates.csv")
    if os.path.exists(stage2_csv):
        stage2_df = pd.read_csv(stage2_csv)
        _augment_trade_rows_with_timestamps(stage2_df, df_ohlcv, entry_col="anchor_bar")
        stage2_df["stage1_gate"] = stage2_df["stage1_prob"] >= stage1_threshold

        _plot_execution_overview(
            df_ohlcv=df_ohlcv,
            trades=stage2_df,
            direction=direction,
            out_path=os.path.join(product_root, "stage2", "test_execution_overview.png"),
            title="Stage 2 Test Set — Price With Execution Points",
            entry_bar_col="anchor_bar",
            include_stage1_candidates=True,
        )
        _plot_trade_zoom_panels(
            df_ohlcv=df_ohlcv,
            trades=stage2_df[stage2_df["trade_signal"]].copy(),
            direction=direction,
            out_path=os.path.join(product_root, "stage2", "test_execution_panels.png"),
            title="Stage 2 Executed Test Trades",
            entry_bar_col="anchor_bar",
        )

    paper_csv = os.path.join(product_root, "paper_replay", "paper_trades.csv")
    if os.path.exists(paper_csv):
        try:
            paper_df = pd.read_csv(paper_csv)
        except EmptyDataError:
            paper_df = pd.DataFrame()
        _augment_trade_rows_with_timestamps(paper_df, df_ohlcv, entry_col="entry_bar")

        _plot_execution_overview(
            df_ohlcv=df_ohlcv,
            trades=paper_df,
            direction=direction,
            out_path=os.path.join(product_root, "paper_replay", "paper_execution_overview.png"),
            title="Paper Replay — Price With Execution Points",
            entry_bar_col="entry_bar",
            include_stage1_candidates=False,
        )
        _plot_trade_zoom_panels(
            df_ohlcv=df_ohlcv,
            trades=paper_df.copy(),
            direction=direction,
            out_path=os.path.join(product_root, "paper_replay", "paper_execution_panels.png"),
            title="Paper Replay Trades",
            entry_bar_col="entry_bar",
        )


def _augment_trade_rows_with_timestamps(
    trades: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    entry_col: str,
) -> None:
    """Attach timestamps for entry and exit bar indices in-place."""
    if trades.empty:
        return

    index = df_ohlcv.index
    trades["entry_time"] = [str(index[int(bar)]) for bar in trades[entry_col]]
    trades["exit_time"] = [str(index[int(bar)]) for bar in trades["exit_bar"]]


def _plot_execution_overview(
    df_ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    direction: int,
    out_path: str,
    title: str,
    entry_bar_col: str,
    include_stage1_candidates: bool,
) -> None:
    """Plot one overview chart with Stage 1 candidates, entries, and exits."""
    if trades.empty:
        return

    if include_stage1_candidates:
        stage1_rows = trades[trades["stage1_gate"]].copy()
        focus_rows = trades[trades["trade_signal"]].copy()
    else:
        stage1_rows = pd.DataFrame(columns=trades.columns)
        focus_rows = trades.copy()

    if focus_rows.empty and stage1_rows.empty:
        return

    focus_source = focus_rows if not focus_rows.empty else stage1_rows
    pad = 40
    lo = max(0, int(focus_source[entry_bar_col].min()) - pad)
    hi = min(len(df_ohlcv) - 1, int(focus_source["exit_bar"].max()) + pad)
    segment = df_ohlcv.iloc[lo : hi + 1].copy()

    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(segment.index, segment["Close"], color="black", lw=1.3, label="Close")
    ax_price.fill_between(
        segment.index,
        segment["Low"].to_numpy(dtype=float),
        segment["High"].to_numpy(dtype=float),
        color="lightgray",
        alpha=0.25,
        label="High/Low range",
    )

    if include_stage1_candidates and not stage1_rows.empty:
        rows = stage1_rows[
            (stage1_rows[entry_bar_col] >= lo) & (stage1_rows[entry_bar_col] <= hi)
        ]
        ax_price.scatter(
            [df_ohlcv.index[int(bar)] for bar in rows[entry_bar_col]],
            rows["entry_price"].to_numpy(dtype=float),
            s=28,
            color="gold",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label="Stage 1 candidate",
            zorder=4,
        )

    if not focus_rows.empty:
        rows = focus_rows[
            (focus_rows[entry_bar_col] >= lo) & (focus_rows[entry_bar_col] <= hi)
        ]
        if not rows.empty:
            entry_marker = "v" if direction == -1 else "^"
            entry_color = "crimson" if direction == -1 else "seagreen"

            ax_price.scatter(
                [df_ohlcv.index[int(bar)] for bar in rows[entry_bar_col]],
                rows["entry_price"].to_numpy(dtype=float),
                s=90,
                marker=entry_marker,
                color=entry_color,
                edgecolor="black",
                linewidth=0.5,
                label="Executed entry",
                zorder=5,
            )

            win_rows = rows[rows["net_return"] > 0]
            lose_rows = rows[rows["net_return"] <= 0]

            if not win_rows.empty:
                ax_price.scatter(
                    [df_ohlcv.index[int(bar)] for bar in win_rows["exit_bar"]],
                    win_rows["exit_price"].to_numpy(dtype=float),
                    s=65,
                    marker="o",
                    color="royalblue",
                    edgecolor="black",
                    linewidth=0.4,
                    label="Profitable exit",
                    zorder=5,
                )
            if not lose_rows.empty:
                ax_price.scatter(
                    [df_ohlcv.index[int(bar)] for bar in lose_rows["exit_bar"]],
                    lose_rows["exit_price"].to_numpy(dtype=float),
                    s=65,
                    marker="X",
                    color="firebrick",
                    edgecolor="black",
                    linewidth=0.4,
                    label="Losing exit",
                    zorder=5,
                )

            for row in rows.itertuples(index=False):
                entry_bar = int(getattr(row, entry_bar_col))
                exit_bar = int(row.exit_bar)
                ax_price.plot(
                    [df_ohlcv.index[entry_bar], df_ohlcv.index[exit_bar]],
                    [float(row.entry_price), float(row.exit_price)],
                    color="seagreen" if float(row.net_return) > 0 else "firebrick",
                    alpha=0.35,
                    lw=1.0,
                )

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="best", fontsize=9)

    volume_colors = np.where(
        segment["Close"].to_numpy(dtype=float) >= segment["Open"].to_numpy(dtype=float),
        "seagreen",
        "firebrick",
    )
    ax_volume.bar(segment.index, segment["Volume"], color=volume_colors, width=0.01, alpha=0.55)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")


def _plot_trade_zoom_panels(
    df_ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    direction: int,
    out_path: str,
    title: str,
    entry_bar_col: str,
) -> None:
    """Plot a small panel per executed trade for easier inspection."""
    if trades.empty:
        return

    trades = trades.sort_values(entry_bar_col).copy().head(12)
    n = len(trades)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, nrows * 3.5)), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    pad = 18
    for ax, row in zip(axes.flat, trades.itertuples(index=False)):
        entry_bar = int(getattr(row, entry_bar_col))
        exit_bar = int(row.exit_bar)
        lo = max(0, entry_bar - pad)
        hi = min(len(df_ohlcv) - 1, exit_bar + pad)
        segment = df_ohlcv.iloc[lo : hi + 1]

        ax.plot(segment.index, segment["Close"], color="black", lw=1.2)
        ax.fill_between(
            segment.index,
            segment["Low"].to_numpy(dtype=float),
            segment["High"].to_numpy(dtype=float),
            color="lightgray",
            alpha=0.2,
        )

        entry_marker = "v" if direction == -1 else "^"
        entry_color = "crimson" if direction == -1 else "seagreen"
        exit_marker = "o" if float(row.net_return) > 0 else "X"
        exit_color = "royalblue" if float(row.net_return) > 0 else "firebrick"

        ax.scatter(
            df_ohlcv.index[entry_bar],
            float(row.entry_price),
            marker=entry_marker,
            s=85,
            color=entry_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            df_ohlcv.index[exit_bar],
            float(row.exit_price),
            marker=exit_marker,
            s=70,
            color=exit_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.plot(
            [df_ohlcv.index[entry_bar], df_ohlcv.index[exit_bar]],
            [float(row.entry_price), float(row.exit_price)],
            color="seagreen" if float(row.net_return) > 0 else "firebrick",
            lw=1.1,
            alpha=0.55,
        )
        ax.set_title(
            f"Entry {df_ohlcv.index[entry_bar].strftime('%Y-%m-%d %H:%M')} | "
            f"net {100 * float(row.net_return):.2f}%"
        )
        ax.grid(alpha=0.25)

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(title, y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
