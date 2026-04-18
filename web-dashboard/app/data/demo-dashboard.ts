import type {
  AggregateCurve,
  ClassificationDetailRow,
  ClassificationSummaryRow,
  DashboardSnapshot,
  GalleryBucket,
  GallerySample,
  PairCurve,
  PairMetrics,
} from "~/utils/dashboard";

const DATE_STAMPS = [
  "2018-08-02T09:45:00-04:00",
  "2018-10-23T10:15:00-04:00",
  "2019-08-05T14:45:00-04:00",
  "2020-03-17T14:45:00-04:00",
  "2021-03-25T10:00:00-04:00",
];

function curvePoints(pattern: string, values: number[]) {
  return values.map((cumulative_pnl, index) => {
    const previous = index === 0 ? 0 : values[index - 1];
    const net_pnl = cumulative_pnl - previous;

    return {
      trade_number: index + 1,
      entry_ts: DATE_STAMPS[Math.max(index - 1, 0)],
      exit_ts: DATE_STAMPS[index],
      net_pnl,
      gross_pnl: net_pnl + 0.18,
      cumulative_pnl,
      return: net_pnl / 100,
      direction: net_pnl >= 0 ? "long" : "short",
      exit_reason: net_pnl >= 0 ? "tp" : "stop",
      proba: 0.54 + index * 0.07,
      label: 1,
      pattern,
    } as const;
  });
}

function pairCurve(
  model: string,
  pattern: string,
  values: number[],
  trades: number,
): PairCurve {
  return {
    scope: "pair",
    series_id: `${model}::${pattern}`,
    model,
    pattern,
    trades,
    total_pnl: values[values.length - 1] ?? 0,
    points: curvePoints(pattern, values),
  };
}

function aggregateCurve(
  scope: "model" | "pattern",
  label: string,
  values: number[],
): AggregateCurve {
  return {
    scope,
    series_id: label,
    label,
    model: scope === "model" ? label : undefined,
    pattern: scope === "pattern" ? label : undefined,
    points: curvePoints(scope === "pattern" ? label : "portfolio", values),
  };
}

function sample(
  pattern: string,
  bucket: GalleryBucket,
  suffix: string,
  proba: number,
  window_end_ts: string,
  y_pred: 0 | 1,
  label: 0 | 1,
): GallerySample {
  const image_path = `outputs/gallery/demo/${pattern}/${bucket}/${suffix}.png`;

  return {
    image_path,
    image_src: `../../gallery/demo/${pattern}/${bucket}/${suffix}.png`,
    label,
    proba,
    window_end_ts,
    y_pred,
  };
}

const pairRows: PairMetrics[] = [
  { pattern: "double_bottom", model: "hgb", selection_f1: 0.42, selection_precision: 0.67, selection_recall: 0.30, f1: 0.27, precision: 0.48, recall: 0.19, pr_auc: 0.44, threshold: 0.06, support_positive: 75, support_negative: 375, trades: 29, total_pnl: 30.91, win_rate: 0.45, sharpe: 5.51, profit_factor: 3.18, max_drawdown: -6.79, expectancy: 1.07 },
  { pattern: "double_bottom", model: "logreg", selection_f1: 0.50, selection_precision: 0.54, selection_recall: 0.46, f1: 0.39, precision: 0.44, recall: 0.35, pr_auc: 0.48, threshold: 0.21, support_positive: 75, support_negative: 375, trades: 68, total_pnl: 12.78, win_rate: 0.52, sharpe: 1.11, profit_factor: 1.35, max_drawdown: -13.24, expectancy: 0.19 },
  { pattern: "double_bottom", model: "lstm", selection_f1: 0.14, selection_precision: 0.07, selection_recall: 1.0, f1: 0.29, precision: 0.17, recall: 1.0, pr_auc: 0.19, threshold: 0.05, support_positive: 75, support_negative: 375, trades: 449, total_pnl: 35.44, win_rate: 0.55, sharpe: 0.94, profit_factor: 1.18, max_drawdown: -25.31, expectancy: 0.08 },
  { pattern: "double_bottom", model: "tcn", selection_f1: 0.47, selection_precision: 0.58, selection_recall: 0.39, f1: 0.33, precision: 0.37, recall: 0.30, pr_auc: 0.41, threshold: 0.12, support_positive: 75, support_negative: 375, trades: 74, total_pnl: 8.12, win_rate: 0.51, sharpe: 0.78, profit_factor: 1.13, max_drawdown: -15.42, expectancy: 0.11 },
  { pattern: "double_bottom", model: "transformer", selection_f1: 0.38, selection_precision: 0.29, selection_recall: 0.54, f1: 0.31, precision: 0.22, recall: 0.54, pr_auc: 0.32, threshold: 0.09, support_positive: 75, support_negative: 375, trades: 141, total_pnl: -4.88, win_rate: 0.47, sharpe: -0.24, profit_factor: 0.94, max_drawdown: -19.83, expectancy: -0.03 },
  { pattern: "double_top", model: "hgb", selection_f1: 0.57, selection_precision: 0.61, selection_recall: 0.54, f1: 0.48, precision: 0.50, recall: 0.45, pr_auc: 0.58, threshold: 0.19, support_positive: 84, support_negative: 366, trades: 58, total_pnl: 9.43, win_rate: 0.53, sharpe: 1.29, profit_factor: 1.28, max_drawdown: -11.36, expectancy: 0.16 },
  { pattern: "double_top", model: "logreg", selection_f1: 0.63, selection_precision: 0.57, selection_recall: 0.70, f1: 0.56, precision: 0.52, recall: 0.61, pr_auc: 0.69, threshold: 0.31, support_positive: 84, support_negative: 366, trades: 94, total_pnl: 18.34, win_rate: 0.57, sharpe: 1.82, profit_factor: 1.46, max_drawdown: -10.42, expectancy: 0.20 },
  { pattern: "double_top", model: "lstm", selection_f1: 0.37, selection_precision: 0.24, selection_recall: 0.88, f1: 0.41, precision: 0.28, recall: 0.82, pr_auc: 0.34, threshold: 0.07, support_positive: 84, support_negative: 366, trades: 214, total_pnl: 21.76, win_rate: 0.56, sharpe: 0.86, profit_factor: 1.17, max_drawdown: -21.5, expectancy: 0.10 },
  { pattern: "double_top", model: "tcn", selection_f1: 0.59, selection_precision: 0.51, selection_recall: 0.71, f1: 0.53, precision: 0.46, recall: 0.63, pr_auc: 0.60, threshold: 0.16, support_positive: 84, support_negative: 366, trades: 102, total_pnl: 6.95, win_rate: 0.50, sharpe: 0.64, profit_factor: 1.08, max_drawdown: -17.27, expectancy: 0.07 },
  { pattern: "double_top", model: "transformer", selection_f1: 0.48, selection_precision: 0.36, selection_recall: 0.73, f1: 0.45, precision: 0.31, recall: 0.78, pr_auc: 0.49, threshold: 0.11, support_positive: 84, support_negative: 366, trades: 166, total_pnl: -7.16, win_rate: 0.44, sharpe: -0.35, profit_factor: 0.89, max_drawdown: -23.6, expectancy: -0.04 },
  { pattern: "head_shoulders", model: "hgb", selection_f1: 0.61, selection_precision: 0.67, selection_recall: 0.56, f1: 0.58, precision: 0.62, recall: 0.54, pr_auc: 0.66, threshold: 0.22, support_positive: 162, support_negative: 136, trades: 49, total_pnl: 4.12, win_rate: 0.49, sharpe: 0.39, profit_factor: 1.03, max_drawdown: -12.06, expectancy: 0.08 },
  { pattern: "head_shoulders", model: "logreg", selection_f1: 0.69, selection_precision: 0.58, selection_recall: 0.85, f1: 0.72, precision: 0.58, recall: 0.97, pr_auc: 0.76, threshold: 0.26, support_positive: 162, support_negative: 136, trades: 118, total_pnl: 14.46, win_rate: 0.58, sharpe: 1.77, profit_factor: 1.41, max_drawdown: -14.85, expectancy: 0.12 },
  { pattern: "head_shoulders", model: "lstm", selection_f1: 0.44, selection_precision: 0.30, selection_recall: 0.88, f1: 0.49, precision: 0.33, recall: 0.93, pr_auc: 0.43, threshold: 0.08, support_positive: 162, support_negative: 136, trades: 228, total_pnl: 10.33, win_rate: 0.52, sharpe: 0.52, profit_factor: 1.09, max_drawdown: -28.2, expectancy: 0.05 },
  { pattern: "head_shoulders", model: "tcn", selection_f1: 0.67, selection_precision: 0.59, selection_recall: 0.78, f1: 0.61, precision: 0.54, recall: 0.71, pr_auc: 0.71, threshold: 0.18, support_positive: 162, support_negative: 136, trades: 85, total_pnl: 16.24, win_rate: 0.59, sharpe: 1.66, profit_factor: 1.44, max_drawdown: -10.22, expectancy: 0.19 },
  { pattern: "head_shoulders", model: "transformer", selection_f1: 0.70, selection_precision: 0.59, selection_recall: 0.86, f1: 0.39, precision: 0.77, recall: 0.27, pr_auc: 0.73, threshold: 0.50, support_positive: 162, support_negative: 136, trades: 56, total_pnl: -1.12, win_rate: 0.54, sharpe: -0.39, profit_factor: 0.98, max_drawdown: -18.52, expectancy: -0.02 },
  { pattern: "inverse_head_shoulders", model: "hgb", selection_f1: 0.52, selection_precision: 0.60, selection_recall: 0.46, f1: 0.47, precision: 0.53, recall: 0.43, pr_auc: 0.55, threshold: 0.17, support_positive: 98, support_negative: 352, trades: 44, total_pnl: 5.65, win_rate: 0.50, sharpe: 0.77, profit_factor: 1.15, max_drawdown: -8.74, expectancy: 0.13 },
  { pattern: "inverse_head_shoulders", model: "logreg", selection_f1: 0.58, selection_precision: 0.55, selection_recall: 0.63, f1: 0.51, precision: 0.48, recall: 0.56, pr_auc: 0.62, threshold: 0.27, support_positive: 98, support_negative: 352, trades: 73, total_pnl: 11.37, win_rate: 0.55, sharpe: 1.13, profit_factor: 1.29, max_drawdown: -9.63, expectancy: 0.16 },
  { pattern: "inverse_head_shoulders", model: "lstm", selection_f1: 0.28, selection_precision: 0.18, selection_recall: 0.69, f1: 0.34, precision: 0.22, recall: 0.72, pr_auc: 0.31, threshold: 0.08, support_positive: 98, support_negative: 352, trades: 159, total_pnl: 7.92, win_rate: 0.53, sharpe: 0.42, profit_factor: 1.08, max_drawdown: -16.91, expectancy: 0.05 },
  { pattern: "inverse_head_shoulders", model: "tcn", selection_f1: 0.60, selection_precision: 0.53, selection_recall: 0.69, f1: 0.55, precision: 0.47, recall: 0.67, pr_auc: 0.66, threshold: 0.14, support_positive: 98, support_negative: 352, trades: 88, total_pnl: 13.98, win_rate: 0.58, sharpe: 1.41, profit_factor: 1.39, max_drawdown: -11.08, expectancy: 0.17 },
  { pattern: "inverse_head_shoulders", model: "transformer", selection_f1: 0.41, selection_precision: 0.33, selection_recall: 0.55, f1: 0.38, precision: 0.28, recall: 0.59, pr_auc: 0.40, threshold: 0.10, support_positive: 98, support_negative: 352, trades: 143, total_pnl: -5.42, win_rate: 0.45, sharpe: -0.28, profit_factor: 0.93, max_drawdown: -20.11, expectancy: -0.02 },
];

const champions = [
  { pattern: "double_bottom", model: "lstm", selection_split: "val", selection_f1: 0.14, threshold: 0.05, test_f1: 0.29, test_precision: 0.17, test_recall: 1.0, test_pr_auc: 0.19, trades: 449, total_pnl: 35.44, win_rate: 0.55, sharpe: 0.94, profit_factor: 1.18, max_drawdown: -25.31, expectancy: 0.08, precision: 0.17, recall: 1.0, pr_auc: 0.19, f1: 0.29, selection_precision: 0.07, selection_recall: 1.0, support_positive: 75, support_negative: 375 },
  { pattern: "double_top", model: "logreg", selection_split: "val", selection_f1: 0.63, threshold: 0.31, test_f1: 0.56, test_precision: 0.52, test_recall: 0.61, test_pr_auc: 0.69, trades: 94, total_pnl: 18.34, win_rate: 0.57, sharpe: 1.82, profit_factor: 1.46, max_drawdown: -10.42, expectancy: 0.20, precision: 0.52, recall: 0.61, pr_auc: 0.69, f1: 0.56, selection_precision: 0.57, selection_recall: 0.70, support_positive: 84, support_negative: 366 },
  { pattern: "head_shoulders", model: "tcn", selection_split: "val", selection_f1: 0.67, threshold: 0.18, test_f1: 0.61, test_precision: 0.54, test_recall: 0.71, test_pr_auc: 0.71, trades: 85, total_pnl: 16.24, win_rate: 0.59, sharpe: 1.66, profit_factor: 1.44, max_drawdown: -10.22, expectancy: 0.19, precision: 0.54, recall: 0.71, pr_auc: 0.71, f1: 0.61, selection_precision: 0.59, selection_recall: 0.78, support_positive: 162, support_negative: 136 },
  { pattern: "inverse_head_shoulders", model: "tcn", selection_split: "val", selection_f1: 0.60, threshold: 0.14, test_f1: 0.55, test_precision: 0.47, test_recall: 0.67, test_pr_auc: 0.66, trades: 88, total_pnl: 13.98, win_rate: 0.58, sharpe: 1.41, profit_factor: 1.39, max_drawdown: -11.08, expectancy: 0.17, precision: 0.47, recall: 0.67, pr_auc: 0.66, f1: 0.55, selection_precision: 0.53, selection_recall: 0.69, support_positive: 98, support_negative: 352 },
];

const summaryRows: ClassificationSummaryRow[] = pairRows.map((row) => ({
  pattern: row.pattern,
  model: row.model,
  selection_split: "val",
  threshold: row.threshold,
  selection_f1: row.selection_f1,
  selection_precision: row.selection_precision,
  selection_recall: row.selection_recall,
  f1: row.f1,
  precision: row.precision,
  recall: row.recall,
  pr_auc: row.pr_auc,
  support_positive: row.support_positive ?? 0,
  support_negative: row.support_negative ?? 0,
}));

const detailRows: ClassificationDetailRow[] = pairRows.slice(0, 8).map((row) => ({
  pattern: row.pattern,
  model: row.model,
  artifact_path: `outputs/metrics/demo/${row.pattern}/${row.model}.json`,
  threshold: row.threshold,
  accuracy: Math.min(0.9, row.f1 + 0.14),
  confusion_matrix: [84, 13, 21, 56],
  f1_ci_low: Math.max(0, row.f1 - 0.06),
  f1_ci_high: Math.min(1, row.f1 + 0.05),
  precision_ci_low: Math.max(0, row.precision - 0.08),
  precision_ci_high: Math.min(1, row.precision + 0.07),
  recall_ci_low: Math.max(0, row.recall - 0.08),
  recall_ci_high: Math.min(1, row.recall + 0.07),
}));

const pairCurves: PairCurve[] = [
  pairCurve("lstm", "double_bottom", [2.4, 10.2, 21.6, 33.8, 35.44], 449),
  pairCurve("logreg", "double_top", [1.2, 5.8, 10.6, 16.1, 18.34], 94),
  pairCurve("tcn", "head_shoulders", [-0.8, 2.4, 8.7, 14.8, 16.24], 85),
  pairCurve("tcn", "inverse_head_shoulders", [0.9, 3.1, 7.8, 11.9, 13.98], 88),
  pairCurve("hgb", "double_bottom", [0.8, 5.4, 12.8, 21.7, 30.91], 29),
  pairCurve("logreg", "head_shoulders", [0.7, 4.6, 8.2, 12.5, 14.46], 118),
  pairCurve("lstm", "double_top", [-2.1, 1.6, 8.9, 18.4, 21.76], 214),
  pairCurve("transformer", "head_shoulders", [-1.6, -4.2, -5.7, -2.9, -1.12], 56),
  pairCurve("transformer", "double_top", [-0.9, -3.8, -4.6, -5.8, -7.16], 166),
  pairCurve("transformer", "inverse_head_shoulders", [-0.5, -1.3, -2.1, -4.3, -5.42], 143),
];

const aggregateCurves: AggregateCurve[] = [
  aggregateCurve("model", "hgb", [0.4, 3.2, 5.8, 9.7, 12.6]),
  aggregateCurve("model", "logreg", [0.8, 4.4, 10.3, 15.6, 20.8]),
  aggregateCurve("model", "lstm", [0.3, 5.1, 13.2, 24.9, 28.7]),
  aggregateCurve("model", "tcn", [0.1, 3.8, 9.8, 16.4, 19.9]),
  aggregateCurve("model", "transformer", [-0.8, -3.2, -5.5, -8.4, -10.2]),
  aggregateCurve("pattern", "double_bottom", [0.5, 5.2, 12.1, 22.5, 28.9]),
  aggregateCurve("pattern", "double_top", [-0.2, 2.8, 7.9, 14.6, 16.7]),
  aggregateCurve("pattern", "head_shoulders", [-0.4, 1.5, 7.1, 12.8, 15.6]),
  aggregateCurve("pattern", "inverse_head_shoulders", [0.2, 2.7, 5.9, 9.4, 12.4]),
];

export const demoDashboardSnapshot: DashboardSnapshot = {
  meta: {
    run_name: "rough_screen_all_models_2026_04_18_relaxed",
    generated_at: "2026-04-18T11:22:00+01:00",
    output_dir: "outputs/dashboard/rough_screen_all_models_2026_04_18_relaxed",
    models: ["hgb", "logreg", "lstm", "tcn", "transformer"],
    patterns: [
      "double_bottom",
      "double_top",
      "head_shoulders",
      "inverse_head_shoulders",
    ],
    source_paths: [
      "outputs/metrics/rough_screen_all_models_2026_04_18_relaxed",
      "outputs/backtest/rough_screen_all_models_2026_04_18_relaxed",
      "outputs/gallery/rough_screen_all_models_2026_04_18_relaxed",
    ],
  },
  hero: {
    best_backtest_pair: pairRows.find(
      (row) => row.pattern === "double_bottom" && row.model === "lstm",
    )!,
    best_model_by_mean_f1: { model: "logreg", mean_f1: 0.545 },
    best_test_f1_pair: summaryRows.find(
      (row) => row.pattern === "head_shoulders" && row.model === "logreg",
    )!,
    champion_positive_patterns: 4,
    champion_total_patterns: 4,
    macro: {
      macro_f1: 0.52,
      macro_precision: 0.43,
      macro_recall: 0.67,
      patterns_covered: 4,
    },
    model_pair_count: pairRows.length,
    patterns_covered: 4,
  },
  classification: {
    champions,
    detail_rows: detailRows,
    summary_rows: summaryRows,
  },
  backtest: {
    champion_rows: champions,
    champion_summary_rows: champions,
    pair_rows: pairRows,
    pair_curves: pairCurves,
    aggregate_curves: aggregateCurves,
  },
  gallery: {
    summary: {
      double_bottom: { tp: 12, fp: 4, fn: 5 },
      double_top: { tp: 14, fp: 6, fn: 4 },
      head_shoulders: { tp: 11, fp: 5, fn: 7 },
      inverse_head_shoulders: { tp: 9, fp: 4, fn: 5 },
    },
    samples: {
      double_bottom: {
        tp: [
          sample("double_bottom", "tp", "double_bottom_tp_01", 0.92, "2020-03-17T14:45:00-04:00", 1, 1),
        ],
        fp: [
          sample("double_bottom", "fp", "double_bottom_fp_01", 0.61, "2018-10-30T15:15:00-04:00", 1, 0),
        ],
        fn: [
          sample("double_bottom", "fn", "double_bottom_fn_01", 0.16, "2018-11-27T14:00:00-05:00", 0, 1),
        ],
      },
      double_top: {
        tp: [
          sample("double_top", "tp", "double_top_tp_01", 0.88, "2020-04-01T13:00:00-04:00", 1, 1),
        ],
        fp: [
          sample("double_top", "fp", "double_top_fp_01", 0.57, "2019-03-28T12:00:00-04:00", 1, 0),
        ],
        fn: [
          sample("double_top", "fn", "double_top_fn_01", 0.23, "2018-09-06T14:45:00-04:00", 0, 1),
        ],
      },
      head_shoulders: {
        tp: [
          sample("head_shoulders", "tp", "head_shoulders_tp_01", 0.84, "2021-03-19T09:45:00-04:00", 1, 1),
        ],
        fp: [
          sample("head_shoulders", "fp", "head_shoulders_fp_01", 0.64, "2017-11-16T10:00:00-05:00", 1, 0),
        ],
        fn: [
          sample("head_shoulders", "fn", "head_shoulders_fn_01", 0.14, "2018-08-02T09:45:00-04:00", 0, 1),
        ],
      },
      inverse_head_shoulders: {
        tp: [
          sample("inverse_head_shoulders", "tp", "inverse_head_shoulders_tp_01", 0.79, "2020-06-08T13:45:00-04:00", 1, 1),
        ],
        fp: [
          sample("inverse_head_shoulders", "fp", "inverse_head_shoulders_fp_01", 0.55, "2019-06-06T11:30:00-04:00", 1, 0),
        ],
        fn: [
          sample("inverse_head_shoulders", "fn", "inverse_head_shoulders_fn_01", 0.18, "2018-12-18T13:15:00-05:00", 0, 1),
        ],
      },
    },
  },
};
