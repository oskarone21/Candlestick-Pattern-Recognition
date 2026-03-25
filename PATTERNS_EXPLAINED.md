# Chart Patterns: Mathematical Definitions

This document defines the mathematical criteria for identifying chart patterns in our candlestick pattern recognition system. All team members must use these uniform definitions when implementing detection algorithms or training models.

## Overview

Our pattern detection uses a **rule-based labeling approach** combined with deep learning classification. This document specifies the exact mathematical constraints that define each pattern type.

---

## Pattern Detection Pipeline

```
Raw OHLCV Data
      ↓
Resampling (1min → 15min)
      ↓
Pivot Detection (Fractal method)
      ↓
Pattern Matching (Geometric constraints)
      ↓
Label Assignment
      ↓
Window Extraction → Model Training
```

---

## 1. Head and Shoulders (H&S) Pattern

### Definition
A bearish reversal pattern consisting of three peaks: left shoulder, head (highest), and right shoulder, followed by a neckline breakout.

### Mathematical Constraints

**Structure Requirements:**
- **Three peaks required**: Left Shoulder (LS), Head (H), Right Shoulder (RS)
- **Head must be highest**: `H > LS` AND `H > RS`
- **Shoulder symmetry**: `|LS - RS| / avg(LS, RS) ≤ shoulder_height_tolerance_pct`
  - Default: 3% (0.03)
- **Head prominence**: `(H - avg(LS, RS)) / avg(LS, RS) ≥ head_prominence_pct`
  - Default: 5% (0.05) - Head must be noticeably higher than both shoulders

**Temporal Requirements:**
- **Minimum separation**: `min_peak_separation_bars` between consecutive peaks
  - Default: 5 bars
- **Maximum separation**: `max_peak_separation_bars` between first and last peak
  - Default: 40 bars

**Neckline Requirements:**

The neckline is a critical support level that determines pattern validity:

![Head and Shoulders Pattern](docs/images/head_shoulders_pattern.png)

- **Definition**: Trend line connecting the two valleys between the three peaks (Valley 1 between LS→H, Valley 2 between H→RS)
- **Formula**: Linear interpolation between Valley 1 and Valley 2
  - `neckline_price(t) = V1 + (V2 - V1) × (t - t1) / (t2 - t1)`
  - Where V1, V2 are valley prices at times t1, t2
- **Slope constraint**: `|neckline_slope| ≤ neckline_max_slope`
  - Default: 0.15 (nearly horizontal)
  - Upward sloping neckline: Right valley must be lower than left shoulder peak
  - Downward sloping neckline: More bearish signal
- **Support function**: Acts as support during formation; breaking it confirms reversal

**Pattern Completion (CRITICAL):**
- The pattern is **NOT complete** until the neckline is broken
- Without neckline break, it's just a "potential" pattern, not a confirmed signal
- **Breakout requirement**: Price must close **below** neckline by `breakout_confirmation_pct`
  - Default: 3% (0.03) penetration required
- **Timing**: Must occur within `confirm_break_within_bars` after right shoulder formation
  - Default: 8 bars
- **Volume confirmation**: Higher volume on breakout = stronger signal (optional but ideal)

**Trade Trigger:** Only enter short positions AFTER neckline break with confirmation

**Confirmation:**

### Label Assignment
- **Positive label**: When pattern completes (neckline broken)
- **Window center**: At the right shoulder peak
- **Lookback**: Include bars from left shoulder start to confirmation

---

## 2. Inverse Head and Shoulders Pattern

### Definition
A bullish reversal pattern - the mirror image of H&S with three troughs.

### Mathematical Constraints

**Structure Requirements:**
- **Three troughs required**: Left Shoulder (LS), Head (lowest), Right Shoulder (RS)
- **Head must be lowest**: `H < LS` AND `H < RS`
- **Shoulder symmetry**: `|LS - RS| / avg(LS, RS) ≤ shoulder_depth_tolerance_pct`
  - Default: 3% (0.03)
- **Head prominence**: `(avg(LS, RS) - H) / avg(LS, RS) ≥ head_depth_prominence_pct`
  - Default: 2.5% (0.025)

**Temporal Requirements:**
- **Minimum separation**: `min_trough_separation_bars` between consecutive troughs
  - Default: 5 bars
- **Maximum separation**: `max_trough_separation_bars` between first and last trough
  - Default: 40 bars

**Neckline Requirements:**
- Drawn across the two peaks between the three troughs
- **Slope constraint**: `|neckline_slope| ≤ neckline_max_slope`
  - Default: 0.15

**Confirmation:**
- **Breakout**: Price must close above neckline by `breakout_confirmation_pct`
  - Default: 3% (0.03)
- **Breakout window**: Within `confirm_break_within_bars` after right shoulder formation
  - Default: 8 bars
- **Volume**: Ideally higher on breakout

### Label Assignment
- **Positive label**: When pattern completes (neckline broken)
- **Window center**: At the right shoulder trough

---

## 3. Double Top Pattern

### Definition
A bearish reversal pattern with two peaks at approximately the same price level, separated by a pullback.

### Mathematical Constraints

**Structure Requirements:**
- **Two peaks required**: Peak 1 (P1), Peak 2 (P2)
- **Peak similarity**: `|P1 - P2| / avg(P1, P2) ≤ peak_tolerance_pct`
  - Default: 2.5% (0.025)
- **Minimum pullback**: Valley between peaks must be at least `min_pullback_pct` below the peaks
  - Default: 1.5% (0.015) minimum, ideally 10-20%

**Temporal Requirements:**
- **Minimum separation**: `min_separation_bars` between peaks
  - Default: 5 bars
- **Maximum separation**: `max_separation_bars` between peaks
  - Default: 50 bars
- **Time diversity**: Peaks must not be consecutive bars (need pullback)

**Trend Requirements:**
- **Prior uptrend**: Price should have risen at least `prior_trend_pct` before first peak
  - Default: 15% (0.15)
- Lookback period: 20-50 bars before P1

**Confirmation:**
- **Neckline**: Horizontal line at the valley low between the two peaks
- **Breakout**: Price must close below neckline by `breakout_confirmation_pct`
  - Default: 3% (0.03)
- **Breakout window**: Within `confirm_break_within_bars`
  - Default: 8 bars
- **Volume**: First peak on higher volume, second peak on lower volume (classic pattern)

### Label Assignment
- **Positive label**: When neckline is broken
- **Window center**: At second peak

---

## 4. Double Bottom Pattern

### Definition
A bullish reversal pattern with two troughs at approximately the same price level, separated by a bounce.

### Mathematical Constraints

**Structure Requirements:**
- **Two troughs required**: Trough 1 (T1), Trough 2 (T2)
- **Trough similarity**: `|T1 - T2| / avg(T1, T2) ≤ trough_tolerance_pct`
  - Default: 2.5% (0.025), ideally within 3-4%
- **Minimum bounce**: Peak between troughs must be at least `min_bounce_pct` above the troughs
  - Default: 1.5% (0.015) minimum, ideally 10-20%

**Temporal Requirements:**
- **Minimum separation**: `min_separation_bars` between troughs
  - Default: 5 bars
- **Maximum separation**: `max_separation_bars` between troughs
  - Default: 50 bars

**Trend Requirements:**
- **Prior downtrend**: Price should have fallen at least `prior_trend_pct` before first trough
  - Default: 15% (0.15)

**Confirmation:**
- **Neckline**: Horizontal line at the peak high between the two troughs
- **Breakout**: Price must close above neckline by `breakout_confirmation_pct`
  - Default: 3% (0.03)
- **Breakout window**: Within `confirm_break_within_bars`
  - Default: 8 bars
- **Volume**: Should increase on second bounce and breakout

### Label Assignment
- **Positive label**: When neckline is broken
- **Window center**: At second trough

---

## 5. No Pattern Class

### Definition
Windows that do not contain any of the above completed patterns.

### Assignment Criteria
- **Majority class**: Most windows will be "no_pattern"
- **Balance**: Consider undersampling or class weighting during training
- **Clean windows**: Exclude windows with partial pattern formations to reduce noise

---

## Configuration Parameters Reference

All pattern parameters are configurable in `configs/config.yaml` under the `labeling` section:

### Pivot Detection
```yaml
labeling:
  pivot:
    method: fractal                          # Detection algorithm
    left_bars: 3                             # Bars to confirm pivot (left side)
    right_bars: 3                            # Bars to confirm pivot (right side)
    min_swing_pct: 0.015                     # Minimum price swing to qualify as pivot
```

### Head & Shoulders Parameters
```yaml
  head_shoulders:
    shoulder_height_tolerance_pct: 0.03      # Max shoulder price difference
    head_prominence_pct: 0.025               # Min head height above shoulders
    min_peak_separation_bars: 5              # Min bars between peaks
    max_peak_separation_bars: 40             # Max bars between peaks
    neckline_max_slope: 0.15                 # Max absolute neckline slope
    breakout_confirmation_pct: 0.03          # Price must break neckline by this %
    confirm_break_within_bars: 8             # Breakout must occur within this window
```

### Inverse H&S Parameters
```yaml
  inverse_head_shoulders:
    shoulder_depth_tolerance_pct: 0.03       # Max shoulder depth difference
    head_depth_prominence_pct: 0.025         # Min head depth below shoulders
    min_trough_separation_bars: 5            # Min bars between troughs
    max_trough_separation_bars: 40           # Max bars between troughs
    neckline_max_slope: 0.15                 # Max absolute neckline slope
    breakout_confirmation_pct: 0.03          # Price must break neckline by this %
    confirm_break_within_bars: 8             # Breakout must occur within this window
```

### Double Top Parameters
```yaml
  double_top:
    peak_tolerance_pct: 0.025                # Max peak price difference
    min_pullback_pct: 0.015                  # Min pullback between peaks
    prior_trend_pct: 0.15                    # Required uptrend before pattern
    min_separation_bars: 5                   # Min bars between peaks
    max_separation_bars: 50                  # Max bars between peaks
    breakout_confirmation_pct: 0.03          # Price must break neckline by this %
    confirm_break_within_bars: 8             # Breakout must occur within this window
```

### Double Bottom Parameters
```yaml
  double_bottom:
    trough_tolerance_pct: 0.025              # Max trough price difference
    min_bounce_pct: 0.015                    # Min bounce between troughs
    prior_trend_pct: 0.15                    # Required downtrend before pattern
    min_separation_bars: 5                   # Min bars between troughs
    max_separation_bars: 50                  # Max bars between troughs
    breakout_confirmation_pct: 0.03          # Price must break neckline by this %
    confirm_break_within_bars: 8             # Breakout must occur within this window
```

---

## Implementation Notes

### 1. Pivot Detection
Use fractal pivots: A pivot high at index `i` satisfies:
```
high[i] > high[i-1] AND high[i] > high[i-2] AND high[i] > high[i+1] AND high[i] > high[i+2]
```
(for `left_bars=2, right_bars=2`)

### 2. Pattern Scanning Algorithm
```python
for each window in time_series:
    1. Detect all pivots in window
    2. Try to match H&S pattern:
       - Find 3 consecutive peaks
       - Check all geometric constraints
       - Verify neckline breakout
    3. Try to match other patterns similarly
    4. If any pattern confirmed: label accordingly
    5. Else: label as "no_pattern"
```

### 3. Training Strategy: Binary Classifiers per Pattern

**Important:** We train SEPARATE models for each pattern type to avoid label conflicts and ambiguous training data.

**Why Separate Runs?**
- Multiple patterns can appear in the same window → conflicting labels
- Binary classification focuses the model on one pattern geometry
- Easier debugging and model interpretation
- Teammates can work on different patterns in parallel

**The 4 Training Runs:**

```yaml
Run 1 - H&S Detection:
  Positive class: head_shoulders (pattern detected)
  Negative class: other (inverse H&S, double top, double bottom, no pattern)
  
Run 2 - Inverse H&S Detection:
  Positive class: inverse_head_shoulders
  Negative class: other (H&S, double top, double bottom, no pattern)
  
Run 3 - Double Top Detection:
  Positive class: double_top
  Negative class: other (H&S, inverse H&S, double bottom, no pattern)
  
Run 4 - Double Bottom Detection:
  Positive class: double_bottom
  Negative class: other (H&S, inverse H&S, double top, no pattern)
```

**Labeling for Each Run:**
```python
# Example for Run 1 (H&S detection)
if pattern_detected == "head_shoulders":
    label = 1  # positive
else:
    label = 0  # negative (includes all other patterns + no_pattern)
```

**Configuration:**
Set `labeling.active_pattern` in config.yaml to specify which pattern to detect:
```yaml
labeling:
  active_pattern: head_shoulders  # or inverse_head_shoulders, double_top, double_bottom
```

**Ensemble Approach (Future):**
Once all 4 models are trained, you can ensemble them:
- Run all 4 models on new data
- Combine probability scores
- Flag when multiple patterns detected (may indicate high-confidence signal)

### 4. Tolerance Flexibility
- Real markets are noisy; strict geometric perfection is rare
- Use tolerance percentages to allow realistic variations
- Make all tolerances configurable for experimentation

### 5. Validation Checks
Always verify:
- [ ] Peaks/troughs are properly ordered in time
- [ ] Pattern formations respect min/max separation constraints
- [ ] Neckline breakout is confirmed before labeling
- [ ] No future data leakage (only use past data for detection)

---

## References

1. **Osler & Chang (1995)** - "Head and Shoulders: Not Just a Flaky Pattern" - Federal Reserve Bank of New York
2. **Bulkowski (2021)** - "Encyclopedia of Chart Patterns" - 3rd Edition
3. **Investopedia** - Chart Pattern Definitions and Trading Guidelines
4. **Wikipedia** - Technical Analysis Chart Patterns
5. **Lo, Mamaysky & Wang (2000)** - "Foundations of Technical Analysis" - Journal of Finance

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-XX | Initial pattern definitions based on academic research |

---

## Questions?

For clarifications on pattern definitions or implementation details, refer to:
- TEAM_STANDARDS.md for coding conventions
- configs/config.yaml for adjustable parameters
- This document as the single source of truth for pattern geometry
