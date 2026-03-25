# Chart Patterns: Academic Mathematical Definitions

This document defines the strict labeling schema used by this project.

It replaces heuristic/fractal pattern labeling with an academically grounded formulation intended for low-noise supervised learning targets.

The implementation contract is `labeling.schema_version: 2` in `configs/config.yaml`.

## Why this definition set

The project uses these rules because they move chart pattern detection from subjective visual interpretation to reproducible inequalities and confirmation events:

- Lo, Mamaysky, Wang (2000): formalized technical analysis with nonparametric smoothing and objective pattern rules. Link: `https://doi.org/10.1111/0022-1082.00265`
- Osler, Chang (1995): objective algorithmic head-and-shoulders detection in exchange rates. Link: `https://www.newyorkfed.org/research/staff_reports/sr4.html`
- Savin, Weller, Zvingelis (2007): predictive power of head-and-shoulders in U.S. equities. Link: `https://doi.org/10.1093/jjfinec/nbl012`
- Nadaraya (1964): kernel regression estimator foundation. Link: `https://doi.org/10.1137/1109020`
- Hurvich, Simonoff, Tsai (1998): AICc bandwidth selection for nonparametric regression. Link: `https://doi.org/10.1111/1467-9868.00125`
- Bulkowski (3rd ed.): practical breakout/volume confirmation ranges used as operational defaults.

## Detection pipeline

```text
Raw OHLCV Data
      -> Resampling (1min -> 15min)
      -> Nadaraya-Watson smoothing of close
      -> Derivative-based extrema extraction
      -> Pattern geometry checks
      -> Volume confirmation checks
      -> Breakout confirmation checks
      -> Label assignment at breakout bar
      -> Sliding window extraction for model training
```

## Core notation

- `P_t`: raw close at bar `t`.
- `m_hat_h(t)`: smoothed close from Nadaraya-Watson estimator.
- `E_k`: kth local extremum in an alternating sequence.
- `t_k`: bar index of `E_k`.
- `N(t)`: neckline value at bar `t`.
- `beta`: breakout confirmation percentage.
- `V(x)`: local/smoothed volume around point `x`.
- `mu_V`: rolling average volume over recent bars (default 20).

## 1) Extrema detection (required)

### 1.1 Smoothing model

Use the Nadaraya-Watson estimator:

`m_hat_h(t) = [sum_i K_h(t - t_i) * P_i] / [sum_i K_h(t - t_i)]`

with Gaussian kernel:

`K(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)`

and scaled kernel:

`K_h(z) = (1/h) * K(z/h)`

### 1.2 Bandwidth selection

- Primary method: AICc (`selection_method: aicc`).
- Dynamic bandwidth is preferred over fixed fractal bar counts.

### 1.3 Extrema conditions

For smoothed function `m_hat_h(t)`:

- Local maximum at `tau` if:
  - `m_hat_h'(tau) = 0`
  - `m_hat_h''(tau) < 0`
- Local minimum at `tau` if:
  - `m_hat_h'(tau) = 0`
  - `m_hat_h''(tau) > 0`

Extracted extrema must alternate max/min and respect minimum temporal spacing.

## 2) Head and Shoulders (bearish)

### 2.1 Structure

Use five consecutive extrema:

`{E1, E2, E3, E4, E5} = {peak, trough, peak, trough, peak}`

### 2.2 Geometry

- Head prominence:
  - `E3 > E1` and `E3 > E5`
- Shoulder symmetry:
  - `|E1 - E5| / (0.5*(E1 + E5)) <= 0.015`
- Neckline-point symmetry:
  - `|E2 - E4| / (0.5*(E2 + E4)) <= 0.015`

Neckline:

`N(t) = E2 + ((E4 - E2)/(t4 - t2)) * (t - t2)`

Slope must satisfy configured cap (`neckline_max_slope`).

### 2.3 Volume rules (mandatory)

- Peak decay profile:
  - `V(E1) > V(E3) > V(E5)`
- Breakout expansion:
  - `V(P_breakout) >= mu_V * 1.2`

### 2.4 Breakout confirmation

Pattern is positive only when:

`P_t < N(t) * (1 - beta)`

with default `beta = 0.04`.

## 3) Inverse Head and Shoulders (bullish)

### 3.1 Structure

Use five consecutive extrema:

`{E1, E2, E3, E4, E5} = {trough, peak, trough, peak, trough}`

### 3.2 Geometry

- Head depth:
  - `E3 < E1` and `E3 < E5`
- Shoulder symmetry:
  - `|E1 - E5| / (0.5*(E1 + E5)) <= 0.015`
- Neckline-point symmetry:
  - `|E2 - E4| / (0.5*(E2 + E4)) <= 0.015`

### 3.3 Volume rules (mandatory)

- Mid-peak weakening:
  - `V(E2) > V(E4)`
- Breakout expansion:
  - `V(P_breakout) >= mu_V * 1.2`

### 3.4 Breakout confirmation

Positive only when:

`P_t > N(t) * (1 + beta)`

with default `beta = 0.04`.

## 4) Double Top (bearish "M")

### 4.1 Structure

`{E1, E2, E3} = {peak, trough, peak}`

### 4.2 Geometry

- Peak equivalence:
  - `|E1 - E3| / (0.5*(E1 + E3)) <= 0.025`
- Pullback depth:
  - `((0.5*(E1 + E3)) - E2) / (0.5*(E1 + E3)) >= 0.05`
- Temporal separation:
  - `min_separation_bars <= (t3 - t1) <= max_separation_bars`
- Prior uptrend:
  - `(E1 - P0) / P0 >= 0.15`

### 4.3 Volume rules (mandatory)

- First peak on higher volume:
  - `V(E1) > V(E3)`
- Breakdown expansion:
  - `V(P_breakout) >= mu_V * 1.2`

### 4.4 Breakout confirmation

Neckline is horizontal at `E2`.

Positive only when:

`P_t < E2 * (1 - beta)`

with default `beta = 0.04`.

## 5) Double Bottom (bullish "W")

### 5.1 Structure

`{E1, E2, E3} = {trough, peak, trough}`

### 5.2 Geometry

- Trough equivalence:
  - `|E1 - E3| / (0.5*(E1 + E3)) <= 0.025`
- Bounce height:
  - `(E2 - 0.5*(E1 + E3)) / (0.5*(E1 + E3)) >= 0.05`
- Temporal separation:
  - `min_separation_bars <= (t3 - t1) <= max_separation_bars`
- Prior downtrend:
  - `(P0 - E1) / P0 >= 0.15`

### 5.3 Volume rules (mandatory)

- Trough contraction:
  - `V(E1) > V(E3)`
- Bullish breakout surge:
  - `V(P_breakout) >= mu_V * 1.5`

### 5.4 Breakout confirmation

Neckline is horizontal at `E2`.

Positive only when:

`P_t > E2 * (1 + beta)`

with default `beta = 0.04`.

## 6) Label assignment policy

This repository uses breakout-time labeling (not formation-time labeling):

- Positive class (`1`) is assigned only at confirmed breakout bars.
- Window anchor is the breakout bar (`window_anchor: breakout_bar`).
- For `lookback_bars: 80`, the positive sample window ends at the breakout bar.
- Any geometric candidate without breakout confirmation is labeled negative (`0`).

This policy follows the same rationale as continuous trend labeling style methods: the event to learn is the actionable confirmation point.

## 7) Hard negatives and imbalance

Because confirmed patterns are sparse, class imbalance is expected.

The schema enforces hard-negative inclusion:

- partial but incomplete formations,
- failed breakout attempts,
- near-miss geometries.

Default target training ratio in config:

- positive:negative = `1:3`.

## 8) Configuration contract (strict schema v2)

All keys below are authoritative and must be used exactly as named:

```yaml
labeling:
  schema_version: 2
  active_pattern: head_shoulders
  strategy: one_vs_rest_binary

  extrema_detection:
    method: nadaraya_watson
    smoothing:
      kernel: gaussian
      bandwidth:
        selection_method: aicc

  confirmation:
    breakout_confirmation_pct: 0.04
    breakout_volume_average_bars: 20

  head_shoulders:
    geometry:
      shoulder_symmetry_tolerance_pct: 0.015
      neckline_point_symmetry_tolerance_pct: 0.015
    volume_rules:
      required: true
      peak_volume_order: V_E1_gt_V_E3_gt_V_E5
      breakout_volume_min_multiplier: 1.2

  inverse_head_shoulders:
    geometry:
      shoulder_symmetry_tolerance_pct: 0.015
      neckline_point_symmetry_tolerance_pct: 0.015

  double_top:
    geometry:
      peak_tolerance_pct: 0.025
      min_pullback_pct: 0.05
      prior_trend_pct: 0.15

  double_bottom:
    geometry:
      trough_tolerance_pct: 0.025
      min_bounce_pct: 0.05
      prior_trend_pct: 0.15
```

## 9) Parameter defaults and justification

- `method = nadaraya_watson`: reduces microstructure noise vs fractal pivots.
- symmetry tolerance `0.015` for H&S shoulders/neckline points: aligns with strict econometric formulations in the literature stream built on Lo et al.
- double-top/bottom pair tolerance `0.025`: keeps retest levels close enough to represent the same support/resistance zone.
- `min_pullback_pct` and `min_bounce_pct = 0.05`: rejects shallow intraday noise reversals.
- `breakout_confirmation_pct = 0.04`: sits in the common empirical 3%-5% confirmation range.
- mandatory volume rules: reduce false positives from low-conviction geometry-only setups.

## 10) Implementation requirements checklist

- [ ] Use only `labeling.schema_version: 2` keys.
- [ ] Do not use fractal pivot logic.
- [ ] Enforce alternating extrema.
- [ ] Require geometric, volume, and breakout constraints together.
- [ ] Assign positive label only at breakout confirmation bar.
- [ ] Keep time-based split and no future-data leakage in features.

## References

1. Lo, A. W., Mamaysky, H., Wang, J. (2000). Foundations of Technical Analysis. `https://doi.org/10.1111/0022-1082.00265`
2. Osler, C. L., Chang, P. H. K. (1995). Head and Shoulders: Not Just a Flaky Pattern. `https://www.newyorkfed.org/research/staff_reports/sr4.html`
3. Savin, G., Weller, P., Zvingelis, J. (2007). The Predictive Power of "Head-and-Shoulders" Price Patterns in the U.S. Stock Market. `https://doi.org/10.1093/jjfinec/nbl012`
4. Nadaraya, E. A. (1964). On Estimating Regression. `https://doi.org/10.1137/1109020`
5. Hurvich, C. M., Simonoff, J. S., Tsai, C.-L. (1998). Smoothing Parameter Selection in Nonparametric Regression Using an Improved Akaike Information Criterion. `https://doi.org/10.1111/1467-9868.00125`
6. Bulkowski, T. N. Encyclopedia of Chart Patterns (3rd Edition).
