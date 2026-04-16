"""
Head & Shoulders (H&S) detector and training window labeler.

Structure: {E1=peak(LS), E2=trough, E3=peak(Head), E4=trough, E5=peak(RS)}
Geometry : head above both shoulders, shoulder price symmetry,
           neckline (E2-E4 line) with limited slope
Volume   : V_E1 > V_E3 > V_E5 (decaying peak volumes)
Breakout : close BELOW neckline at E5 bar * (1 - beta), bearish reversal

Positive labels only at confirmed breakout bar.
Hard negatives (partial/failed patterns) sampled per config ratio.

All behaviour driven by cfg['labeling']['head_shoulders']. No hardcoded values.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Candidate:
    e1_bar: int;   e2_bar: int;   e3_bar: int;   e4_bar: int;   e5_bar: int
    e1_price: float; e2_price: float; e3_price: float; e4_price: float; e5_price: float
    neckline_at_e5: float   # neckline value projected to E5 bar


@dataclass
class LabeledWindow:
    anchor_bar: int
    label: int
    reason: str
    candidate: Candidate | None = field(default=None, repr=False)


def label_head_shoulders(
    df: pd.DataFrame,
    extrema: pd.DataFrame,
    cfg: dict,
) -> list[LabeledWindow]:
    hs_cfg   = cfg["labeling"]["head_shoulders"]
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

    non_consec = hs_cfg.get("non_consecutive_scan", False)
    max_sep    = hs_cfg["geometry"]["max_peak_separation_bars"]
    if non_consec:
        quintuples = _get_nonconsec_peak_quintuples(extrema, max_sep)
        print(f"[head_shoulders] NON-CONSECUTIVE scan: {len(quintuples)} peak triplet candidates ...")
    else:
        quintuples = _get_peak_trough_peak_trough_peak_quintuples(extrema)
        print(f"[head_shoulders] Checking {len(quintuples)} peak-trough-peak-trough-peak quintuples ...")

    geo_fail = vol_fail = partial = confirmed = 0

    for e1, e2, e3, e4, e5 in quintuples:
        cand = Candidate(
            e1_bar=e1["bar_idx"], e2_bar=e2["bar_idx"], e3_bar=e3["bar_idx"],
            e4_bar=e4["bar_idx"], e5_bar=e5["bar_idx"],
            e1_price=e1["price"], e2_price=e2["price"], e3_price=e3["price"],
            e4_price=e4["price"], e5_price=e5["price"],
            neckline_at_e5=_neckline_at(e2, e4, e5["bar_idx"]),
        )

        ok, reason = _check_geometry(cand, hs_cfg)
        if not ok:
            geo_fail += 1
            continue

        ok, reason = _check_volume(cand, volume_arr, hs_cfg)
        if not ok:
            vol_fail += 1
            anchor = _safe_anchor(cand.e5_bar, lookback, n_bars)
            if anchor is not None:
                hard_negs.append(LabeledWindow(
                    anchor_bar=anchor, label=0,
                    reason=f"volume_fail:{reason}", candidate=cand,
                ))
            continue

        breakout_bar = _find_breakout(cand, close_arr, volume_arr, conf_cfg, hs_cfg, n_bars)

        if breakout_bar is None:
            partial += 1
            anchor = _safe_anchor(
                cand.e5_bar + conf_cfg["confirm_break_within_bars"],
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
        f"[head_shoulders] geometry_fail={geo_fail} | volume_fail={vol_fail} | "
        f"partial={partial} | confirmed={confirmed}"
    )
    print(
        f"[head_shoulders] Positive windows: {len(positives)} | "
        f"Hard-negative pool: {len(hard_negs)}"
    )

    return _balance_samples(positives, hard_negs, cfg)


def _neckline_at(e2: dict, e4: dict, bar: int) -> float:
    """Linearly interpolate/extrapolate the neckline (E2->E4) to a given bar."""
    if e4["bar_idx"] == e2["bar_idx"]:
        return e2["price"]
    slope = (e4["price"] - e2["price"]) / (e4["bar_idx"] - e2["bar_idx"])
    return e2["price"] + slope * (bar - e2["bar_idx"])


def _check_geometry(c: Candidate, hs_cfg: dict) -> tuple[bool, str]:
    geo = hs_cfg["geometry"]

    # 1. Head must be above both shoulders
    if not (c.e3_price > c.e1_price and c.e3_price > c.e5_price):
        return False, "head_not_above_shoulders"

    # 2. Shoulder price symmetry
    avg_sh = 0.5 * (c.e1_price + c.e5_price)
    if avg_sh <= 0:
        return False, "shoulder_price_zero"
    if abs(c.e1_price - c.e5_price) / avg_sh > geo["shoulder_symmetry_tolerance_pct"]:
        return False, "shoulder_symmetry"

    # 3. Neckline trough symmetry (E2, E4 price similar)
    avg_neck = 0.5 * (c.e2_price + c.e4_price)
    if avg_neck <= 0:
        return False, "neckline_price_zero"
    if abs(c.e2_price - c.e4_price) / avg_neck > geo["neckline_point_symmetry_tolerance_pct"]:
        return False, "neckline_symmetry"

    # 4. Peak separation (E1->E3 and E3->E5)
    sep_left  = c.e3_bar - c.e1_bar
    sep_right = c.e5_bar - c.e3_bar
    min_sep   = geo["min_peak_separation_bars"]
    max_sep   = geo["max_peak_separation_bars"]
    if not (min_sep <= sep_left <= max_sep and min_sep <= sep_right <= max_sep):
        return False, "peak_separation"

    # 5. Neckline slope limit
    if c.e4_bar > c.e2_bar:
        slope_pct = abs(c.e4_price - c.e2_price) / (c.e2_price * (c.e4_bar - c.e2_bar))
        if slope_pct > geo["neckline_max_slope"]:
            return False, "neckline_slope"

    return True, ""


def _check_volume(c: Candidate, volume_arr: np.ndarray, hs_cfg: dict) -> tuple[bool, str]:
    vol_cfg = hs_cfg["volume_rules"]
    if not vol_cfg["required"]:
        return True, ""

    v_e1, v_e3, v_e5 = volume_arr[c.e1_bar], volume_arr[c.e3_bar], volume_arr[c.e5_bar]

    # V_E1 > V_E3 > V_E5
    if vol_cfg.get("require_peak_volume_decay", True):
        if not (v_e1 > v_e3 > v_e5):
            return False, "peak_volume_decay"

    return True, ""


def _find_breakout(
    c: Candidate,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    conf_cfg: dict,
    hs_cfg: dict,
    n_bars: int,
) -> int | None:
    """Scan for close BELOW neckline * (1 - beta) — bearish breakout."""
    beta     = conf_cfg["breakout_confirmation_pct"]
    max_wait = conf_cfg["confirm_break_within_bars"]
    avg_bars = conf_cfg["breakout_volume_average_bars"]

    search_start = c.e5_bar + 1
    search_end   = min(n_bars, c.e5_bar + max_wait + 1)

    for bar in range(search_start, search_end):
        neckline_now = _neckline_at(
            {"bar_idx": c.e2_bar, "price": c.e2_price},
            {"bar_idx": c.e4_bar, "price": c.e4_price},
            bar,
        )
        target = neckline_now * (1.0 - beta)
        if close_arr[bar] < target:
            return bar

    return None


def _balance_samples(positives, hard_negs, cfg):
    ratio_str  = cfg["labeling"]["labeling_policy"][
        "hard_negative_sampling"]["target_positive_to_negative_ratio"]
    pos_part, neg_part = [int(x) for x in ratio_str.split(":")]
    target_neg  = len(positives) * neg_part // pos_part
    rng         = np.random.default_rng(cfg["project"]["seed"])

    if len(hard_negs) >= target_neg:
        idx         = rng.choice(len(hard_negs), size=target_neg, replace=False)
        chosen_negs = [hard_negs[i] for i in idx]
    else:
        chosen_negs = hard_negs

    all_samples = positives + chosen_negs
    print(
        f"[head_shoulders] Final dataset: "
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


def _get_peak_trough_peak_trough_peak_quintuples(extrema):
    """Extract all consecutive peak-trough-peak-trough-peak quintuples."""
    rows       = extrema.to_dict("records")
    quintuples = []
    for i in range(len(rows) - 4):
        e1, e2, e3, e4, e5 = rows[i], rows[i+1], rows[i+2], rows[i+3], rows[i+4]
        if (e1["kind"] == "peak"   and e2["kind"] == "trough" and
                e3["kind"] == "peak"   and e4["kind"] == "trough" and
                e5["kind"] == "peak"):
            quintuples.append((e1, e2, e3, e4, e5))
    return quintuples


def _get_nonconsec_peak_quintuples(extrema, max_sep_bars: int):
    """Non-consecutive H&S scan.

    All peak triplets (E1, E3, E5) where each inter-peak gap <= max_sep_bars.
    E2 = deepest trough between E1 and E3.
    E4 = deepest trough between E3 and E5.
    Deduplicates by keeping only the first occurrence per breakout-bar region.
    """
    import numpy as np
    rows    = extrema.to_dict("records")
    peaks   = [r for r in rows if r["kind"] == "peak"]
    troughs = [r for r in rows if r["kind"] == "trough"]
    tbars   = np.array([t["bar_idx"] for t in troughs])

    quintuples = []
    seen_e3_e5 = set()  # deduplicate by (e3_bar, e5_bar) — same head/RS pair

    np_peaks = len(peaks)
    for i in range(np_peaks):
        e1 = peaks[i]
        for j in range(i + 1, np_peaks):
            e3 = peaks[j]
            if e3["bar_idx"] - e1["bar_idx"] > max_sep_bars:
                break
            mask12 = (tbars > e1["bar_idx"]) & (tbars < e3["bar_idx"])
            if not mask12.any():
                continue
            e2 = min((troughs[k] for k in np.where(mask12)[0]), key=lambda t: t["price"])
            for k in range(j + 1, np_peaks):
                e5 = peaks[k]
                if e5["bar_idx"] - e3["bar_idx"] > max_sep_bars:
                    break
                key = (e3["bar_idx"], e5["bar_idx"])
                if key in seen_e3_e5:
                    continue
                mask34 = (tbars > e3["bar_idx"]) & (tbars < e5["bar_idx"])
                if not mask34.any():
                    continue
                e4 = min((troughs[m] for m in np.where(mask34)[0]), key=lambda t: t["price"])
                seen_e3_e5.add(key)
                quintuples.append((e1, e2, e3, e4, e5))

    return quintuples


def _safe_anchor(bar, lookback, n_bars):
    if bar - lookback + 1 >= 0 and bar < n_bars:
        return bar
    return None
