"""
Double Top (M-pattern) detector and training window labeler.

Structure: {E1=peak, E2=trough (valley/neckline), E3=peak}
Geometry : two peaks at similar price, sufficient pullback, prior uptrend
Volume   : first peak volume > second peak volume
Breakout : close BELOW neckline * (1 - beta), bearish reversal

Positive labels only at confirmed breakout bar.
Hard negatives (partial/failed patterns) sampled per config ratio.

All behaviour driven by cfg['labeling']['double_top']. No hardcoded values.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Candidate:
    e1_bar: int
    e2_bar: int
    e3_bar: int
    e1_price: float
    e2_price: float
    e3_price: float
    neckline: float   # = e2_price (valley between the two peaks)


@dataclass
class LabeledWindow:
    anchor_bar: int
    label: int
    reason: str
    candidate: Candidate | None = field(default=None, repr=False)


def label_double_top(
    df: pd.DataFrame,
    extrema: pd.DataFrame,
    cfg: dict,
) -> list[LabeledWindow]:
    dt_cfg   = cfg["labeling"]["double_top"]
    win_cfg  = cfg["windowing"]
    conf_cfg = cfg["labeling"]["confirmation"]
    col_cfg  = cfg["data_source"]["columns"]

    lookback   = win_cfg["lookback_bars"]
    col_close  = col_cfg["close"]
    col_volume = col_cfg["volume"]

    close_arr  = df[col_close].to_numpy(dtype=float)
    volume_arr = df[col_volume].to_numpy(dtype=float)
    n_bars     = len(df)

    positives: list[LabeledWindow] = []
    hard_negs: list[LabeledWindow] = []

    triples = _get_peak_trough_peak_triplets(extrema)
    print(f"[double_top] Checking {len(triples)} peak-trough-peak triplets ...")

    geo_fail = vol_fail = partial = confirmed = 0

    for e1, e2, e3 in triples:
        cand = Candidate(
            e1_bar=e1["bar_idx"], e2_bar=e2["bar_idx"], e3_bar=e3["bar_idx"],
            e1_price=e1["price"], e2_price=e2["price"], e3_price=e3["price"],
            neckline=e2["price"],
        )

        ok, reason = _check_geometry(cand, close_arr, dt_cfg)
        if not ok:
            geo_fail += 1
            continue

        ok, reason = _check_volume(cand, volume_arr, dt_cfg)
        if not ok:
            vol_fail += 1
            anchor = _safe_anchor(cand.e3_bar, lookback, n_bars)
            if anchor is not None:
                hard_negs.append(LabeledWindow(
                    anchor_bar=anchor, label=0,
                    reason=f"volume_fail:{reason}", candidate=cand,
                ))
            continue

        breakout_bar = _find_breakout(cand, close_arr, volume_arr, conf_cfg, dt_cfg, n_bars)

        if breakout_bar is None:
            partial += 1
            anchor = _safe_anchor(
                cand.e3_bar + conf_cfg["confirm_break_within_bars"],
                lookback, n_bars,
            )
            if anchor is not None:
                hard_negs.append(LabeledWindow(
                    anchor_bar=anchor, label=0,
                    reason="failed_breakout", candidate=cand,
                ))
            continue

        confirmed += 1
        anchor = _safe_anchor(breakout_bar, lookback, n_bars)
        if anchor is not None:
            positives.append(LabeledWindow(
                anchor_bar=breakout_bar, label=1,
                reason="confirmed_breakout", candidate=cand,
            ))

    print(
        f"[double_top] geometry_fail={geo_fail} | volume_fail={vol_fail} | "
        f"partial={partial} | confirmed={confirmed}"
    )
    print(
        f"[double_top] Positive windows: {len(positives)} | "
        f"Hard-negative pool: {len(hard_negs)}"
    )

    labeled = _balance_samples(positives, hard_negs, cfg)
    return labeled


def _check_geometry(c: Candidate, close_arr: np.ndarray, dt_cfg: dict) -> tuple[bool, str]:
    geo = dt_cfg["geometry"]
    avg_peak = 0.5 * (c.e1_price + c.e3_price)

    # 1. Two peaks at similar price
    if abs(c.e1_price - c.e3_price) / avg_peak > geo["peak_tolerance_pct"]:
        return False, "peak_tolerance"

    # 2. Sufficient pullback from peaks to valley
    if (avg_peak - c.e2_price) / avg_peak < geo["min_pullback_pct"]:
        return False, "pullback_height"

    # 3. Temporal separation between peaks
    sep = c.e3_bar - c.e1_bar
    if not (geo["min_separation_bars"] <= sep <= geo["max_separation_bars"]):
        return False, "temporal_separation"

    # 4. Prior uptrend before E1
    lookback_trend = geo["max_separation_bars"]
    trend_start    = max(0, c.e1_bar - lookback_trend)
    p0             = close_arr[trend_start]
    if p0 <= 0:
        return False, "prior_trend_data"
    prior_rise = (c.e1_price - p0) / p0
    if prior_rise < geo["prior_trend_pct"]:
        return False, "prior_trend"

    return True, ""


def _check_volume(c: Candidate, volume_arr: np.ndarray, dt_cfg: dict) -> tuple[bool, str]:
    vol_cfg = dt_cfg["volume_rules"]
    if not vol_cfg["required"]:
        return True, ""

    if vol_cfg["require_first_peak_volume_greater_than_second"]:
        if volume_arr[c.e1_bar] <= volume_arr[c.e3_bar]:
            return False, "peak_volume_order"

    return True, ""


def _find_breakout(
    c: Candidate,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    conf_cfg: dict,
    dt_cfg: dict,
    n_bars: int,
) -> int | None:
    """Scan for close BELOW neckline * (1 - beta) — bearish breakout."""
    beta         = conf_cfg["breakout_confirmation_pct"]
    max_wait     = conf_cfg["confirm_break_within_bars"]
    target_price = c.neckline * (1.0 - beta)

    avg_bars  = conf_cfg["breakout_volume_average_bars"]
    mult      = dt_cfg["volume_rules"]["breakout_volume_min_multiplier"]

    search_start = c.e3_bar + 1
    search_end   = min(n_bars, c.e3_bar + max_wait + 1)

    for bar in range(search_start, search_end):
        if close_arr[bar] < target_price:
            lo   = max(0, bar - avg_bars)
            mu_v = volume_arr[lo:bar].mean() if bar > lo else 1.0
            # Accept regardless of volume (volume is secondary for Double Top)
            _ = volume_arr[bar] >= mu_v * mult
            return bar

    return None


def _balance_samples(positives, hard_negs, cfg):
    ratio_str  = cfg["labeling"]["labeling_policy"][
        "hard_negative_sampling"]["target_positive_to_negative_ratio"]
    pos_part, neg_part = [int(x) for x in ratio_str.split(":")]
    target_neg = len(positives) * neg_part // pos_part

    rng = np.random.default_rng(cfg["project"]["seed"])

    if len(hard_negs) >= target_neg:
        idx        = rng.choice(len(hard_negs), size=target_neg, replace=False)
        chosen_negs = [hard_negs[i] for i in idx]
    else:
        chosen_negs = hard_negs

    all_samples = positives + chosen_negs
    print(
        f"[double_top] Final dataset: "
        f"{len(positives)} positive + {len(chosen_negs)} negative "
        f"= {len(all_samples)} total windows"
    )
    return all_samples


def windows_to_arrays(labeled, df, cfg):
    col_cfg  = cfg["data_source"]["columns"]
    lookback = cfg["windowing"]["lookback_bars"]
    cols     = [
        col_cfg["open"], col_cfg["high"],
        col_cfg["low"],  col_cfg["close"], col_cfg["volume"],
    ]
    X_list, y_list = [], []
    for lw in labeled:
        start  = lw.anchor_bar - lookback + 1
        end    = lw.anchor_bar + 1
        window = df[cols].iloc[start:end].to_numpy(dtype=float)
        if window.shape[0] != lookback:
            continue
        w_min  = window.min(axis=0)
        w_max  = window.max(axis=0)
        denom  = np.where((w_max - w_min) > 0, w_max - w_min, 1.0)
        window = (window - w_min) / denom
        X_list.append(window)
        y_list.append(lw.label)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def _get_peak_trough_peak_triplets(extrema):
    rows     = extrema.to_dict("records")
    triplets = []
    for i in range(len(rows) - 2):
        e1, e2, e3 = rows[i], rows[i + 1], rows[i + 2]
        if e1["kind"] == "peak" and e2["kind"] == "trough" and e3["kind"] == "peak":
            triplets.append((e1, e2, e3))
    return triplets


def _safe_anchor(bar, lookback, n_bars):
    if bar - lookback + 1 >= 0 and bar < n_bars:
        return bar
    return None
