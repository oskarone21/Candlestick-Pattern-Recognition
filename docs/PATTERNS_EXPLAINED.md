# Chart Patterns: Mathematical Definitions (15-Minute Contract)

This document defines the strict labeling and trade-mapping schema used by this repository.

It is the implementation contract for `labeling.schema_version: 2` in [configs/config.yaml](../configs/config.yaml).

## Scope and timeframe assumptions

This implementation is explicitly designed for:
- source bars: **1-minute intraday SPY data**
- training bars: **15-minute regular-session bars**
- session: `09:30` to `16:00` America/New_York
- labeling anchor: **confirmed neckline breakout bar**

Timezone policy:
1. Parse source timestamps.
2. Require explicit `timezone_in_data` if timestamps are naive.
3. Convert to `America/New_York`.
4. Validate DST transitions and monotonic ordering.

## Detection pipeline

```text
Kaggle 1m SPY bars
  -> schema normalization
  -> timezone audit (America/Denver -> America/New_York)
  -> session filter + 1m -> 15m resample
  -> Nadaraya-Watson smoothing on close
  -> derivative-based extrema extraction
  -> pattern geometry checks
  -> breakout confirmation checks
  -> hard-negative generation (near_miss, failed_breakout, partial)
  -> breakout-anchored training windows
```

## Core notation

- `P_t`: close price at 15-minute bar `t`.
- `m_hat_h(t)`: smoothed close (Nadaraya-Watson with bandwidth `h`).
- `E_k`: kth local extremum (alternating max/min sequence).
- `t_k`: bar index of `E_k`.
- `N(t)`: neckline value at bar `t`.
- `beta`: breakout confirmation percentage.
- `V(x)`: volume near point `x`.
- `mu_V`: rolling breakout volume average.

## 1) Extrema detection

### 1.1 Smoothing model

Use Nadaraya-Watson estimator:

`m_hat_h(t) = [sum_i K_h(t - t_i) * P_i] / [sum_i K_h(t - t_i)]`

Gaussian kernel:

`K(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)`

`K_h(z) = (1/h) * K(z/h)`

### 1.2 Bandwidth policy

- Config field: `labeling.extrema_detection.smoothing.bandwidth`.
- `selection_method: aicc` is the documented policy.
- Implementation uses config-driven defaults and bounded bandwidth range.

### 1.3 Extrema conditions

For smoothed function `m_hat_h(t)`:
- local max: `m_hat_h'(tau)=0` and `m_hat_h''(tau)<0`
- local min: `m_hat_h'(tau)=0` and `m_hat_h''(tau)>0`

Additional constraints:
- minimum extrema separation in bars
- alternating sequence enforcement

## 2) Head and Shoulders (bearish)

### 2.1 Structure

`{E1, E2, E3, E4, E5} = {peak, trough, peak, trough, peak}`

### 2.2 Geometry

- head prominence: `E3 > E1` and `E3 > E5`
- shoulder symmetry:
  `|E1 - E5| / (0.5*(E1 + E5)) <= shoulder_symmetry_tolerance_pct`
- neckline-point symmetry:
  `|E2 - E4| / (0.5*(E2 + E4)) <= neckline_point_symmetry_tolerance_pct`
- neckline slope bounded by `neckline_max_slope`

Neckline:

`N(t) = E2 + ((E4 - E2)/(t4 - t2)) * (t - t2)`

### 2.3 Breakout confirmation

Confirmed bearish breakout only if:

`P_t < N(t) * (1 - beta)`

within `confirm_break_within_bars` after right shoulder.

## 3) Inverse Head and Shoulders (bullish)

### 3.1 Structure

`{E1, E2, E3, E4, E5} = {trough, peak, trough, peak, trough}`

### 3.2 Geometry

- head depth: `E3 < E1` and `E3 < E5`
- shoulder and neckline symmetry constraints as configured

### 3.3 Breakout confirmation

Confirmed bullish breakout only if:

`P_t > N(t) * (1 + beta)`

within configured confirmation horizon.

## 4) Double Top (bearish)

### 4.1 Structure

`{E1, E2, E3} = {peak, trough, peak}`

### 4.2 Geometry

- peak equivalence:
  `|E1 - E3| / (0.5*(E1 + E3)) <= peak_tolerance_pct`
- pullback depth:
  `((0.5*(E1 + E3)) - E2) / (0.5*(E1 + E3)) >= min_pullback_pct`
- temporal separation between peaks in config bounds
- prior uptrend threshold in config

### 4.3 Breakout confirmation

Neckline is horizontal at `E2`. Confirmed if:

`P_t < E2 * (1 - beta)`

## 5) Double Bottom (bullish)

### 5.1 Structure

`{E1, E2, E3} = {trough, peak, trough}`

### 5.2 Geometry

- trough equivalence:
  `|E1 - E3| / (0.5*(E1 + E3)) <= trough_tolerance_pct`
- bounce depth:
  `(E2 - 0.5*(E1 + E3)) / (0.5*(E1 + E3)) >= min_bounce_pct`
- temporal separation and prior downtrend thresholds from config

### 5.3 Breakout confirmation

Neckline is horizontal at `E2`. Confirmed if:

`P_t > E2 * (1 + beta)`

## 6) Label assignment policy

This repository uses **breakout-time labeling**:

- positive label (`1`) only at confirmed breakout bars
- window anchor = breakout bar
- non-confirmed candidates become hard negatives (`0`) with reason:
  - `failed_breakout`
  - `near_miss`
  - `partial`

## 7) Hard negatives and class balance

Because confirmed patterns are sparse, hard negatives are mandatory for robust training.

Configured behavior:
- include partial formations
- include failed breakouts
- include geometric near-misses
- target positive:negative ratio configurable (default `1:3`)

## 8) Leakage controls

To keep reported metrics reliable:
- time-based split only
- enforced minimum embargo around split boundaries
- no future bars in feature windows
- causal smoothing/extrema detection for event generation
- hard-negative downsampling on train only (validation/test untouched)
- threshold selection on validation only
- champion/model selection on validation only
- final score reported once on held-out test

## 9) Breakout-to-trade mapping (for evaluation)

After model predicts a positive breakout:
- **entry**: next 15-minute bar open
- **primary TP**: measured move from neckline by pattern height
- **secondary TP sensitivity**: nearest swing level (reported optionally)
- **SL**: pattern invalidation level plus ATR buffer
- **time stop**: fixed max holding bars

Tracked metrics:
- total PnL
- win rate
- Sharpe ratio
- profit factor
- max drawdown
- expectancy

## 10) Configuration keys (authoritative)

Use [configs/config.yaml](../configs/config.yaml) as the source of truth for:

- `data_source.*` (Kaggle handle, file path, timestamp mapping)
- `resampling.*` (1m -> 15m)
- `labeling.*` (pattern constraints)
