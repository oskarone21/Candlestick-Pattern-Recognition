from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_patterns.config import ensure_dir
from chart_patterns.domain import (
    GALLERY_BUCKET_FN,
    GALLERY_BUCKET_FP,
    GALLERY_BUCKET_TP,
)


def _plot_window(
    px: pd.DataFrame,
    end_idx: int,
    out_path: Path,
    title: str,
    lookback: int,
    lookforward: int,
) -> None:
    start_idx = max(0, end_idx - lookback + 1)
    stop_idx = min(len(px) - 1, end_idx + lookforward)
    window = px.iloc[start_idx : stop_idx + 1].copy()

    if window.empty:
        return

    window = window.set_index("ts_event")

    try:
        import mplfinance as mpf

        mpf_df = window[["open", "high", "low", "close", "volume"]].rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        fig, _ = mpf.plot(
            mpf_df,
            type="candle",
            style="charles",
            title=title,
            volume=True,
            returnfig=True,
            tight_layout=True,
            warn_too_much_data=5000,
        )
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(window.index, window["close"], color="black", linewidth=1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)


def render_pattern_gallery(
    price_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    threshold: float,
    output_dir: str | Path,
    pattern: str,
    lookback: int = 80,
    lookforward: int = 20,
    max_per_bucket: int = 12,
) -> dict[str, int]:
    """Render TP/FP/FN chart gallery for manual verification."""
    px = price_df.sort_values("ts_event").reset_index(drop=True).copy()
    preds = pred_df.copy()

    if preds.empty:
        return {GALLERY_BUCKET_TP: 0, GALLERY_BUCKET_FP: 0, GALLERY_BUCKET_FN: 0}

    preds["y_pred"] = (preds["proba"] >= threshold).astype(int)

    buckets = {
        GALLERY_BUCKET_TP: preds[(preds["label"] == 1) & (preds["y_pred"] == 1)],
        GALLERY_BUCKET_FP: preds[(preds["label"] == 0) & (preds["y_pred"] == 1)],
        GALLERY_BUCKET_FN: preds[(preds["label"] == 1) & (preds["y_pred"] == 0)],
    }

    out_root = ensure_dir(output_dir)
    counts: dict[str, int] = {}

    for bucket, frame in buckets.items():
        bucket_dir = ensure_dir(Path(out_root) / pattern / bucket)
        sampled = frame.head(max_per_bucket)

        records: list[dict] = []
        for i, row in sampled.iterrows():
            end_idx = int(row["window_end_idx"])
            filename = f"{pattern}_{bucket}_{i}.png"
            out_path = bucket_dir / filename

            _plot_window(
                px=px,
                end_idx=end_idx,
                out_path=out_path,
                title=f"{pattern} {bucket.upper()} p={row['proba']:.3f}",
                lookback=lookback,
                lookforward=lookforward,
            )

            records.append(
                {
                    "file": str(out_path),
                    "window_end_idx": int(row["window_end_idx"]),
                    "window_end_ts": row["window_end_ts"],
                    "proba": float(row["proba"]),
                    "label": int(row["label"]),
                    "y_pred": int(row["y_pred"]),
                }
            )

        pd.DataFrame(records).to_csv(bucket_dir / "index.csv", index=False)
        counts[bucket] = int(len(sampled))

    return counts
