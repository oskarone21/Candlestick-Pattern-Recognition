export type LatestFinishedRun = {
  runName: string;
  modifiedAt: string;
  sourcePaths: {
    metricsSummary: string;
    champions: string;
    backtestSummary: string;
    gallerySummary: string;
  };
};

export type ClassificationRow = {
  pattern: string;
  model: string;
  selection_split: string;
  selection_metric?: string;
  selection_metric_value?: number;
  selection_f1: number;
  selection_f2?: number;
  selection_precision: number;
  selection_recall: number;
  f1: number;
  f2?: number;
  precision: number;
  recall: number;
  pr_auc: number;
  threshold: number;
  support_positive: number;
  support_negative: number;
  train_positive_support?: number;
  val_positive_support?: number;
  test_positive_support?: number;
  presentation_eligible?: boolean;
};

export type ClassificationDetailRow = {
  pattern: string;
  model: string;
  artifact_path?: string | null;
  threshold: number;
  accuracy?: number | null;
  f1_ci_low?: number | null;
  f1_ci_high?: number | null;
  precision_ci_low?: number | null;
  precision_ci_high?: number | null;
  recall_ci_low?: number | null;
  recall_ci_high?: number | null;
  confusion_matrix?: Record<string, number>;
  train_positive_support?: number;
  val_positive_support?: number;
  test_positive_support?: number;
  presentation_eligible?: boolean;
};

export type ChampionRow = {
  pattern: string;
  model: string;
  selection_split: string;
  selection_f1: number;
  threshold: number;
  test_f1: number;
  test_precision: number;
  test_recall: number;
  test_pr_auc: number;
  train_positive_support?: number;
  val_positive_support?: number;
  test_positive_support?: number;
  presentation_eligible?: boolean;
  trades: number;
  total_pnl: number;
  win_rate: number;
  sharpe: number;
  profit_factor: number;
  max_drawdown: number;
  expectancy: number;
};

export type BacktestPairRow = {
  pattern: string;
  model: string;
  selection_f1: number;
  selection_precision: number;
  selection_recall: number;
  f1: number;
  precision: number;
  recall: number;
  pr_auc: number;
  threshold: number;
  support_positive: number;
  support_negative: number;
  train_positive_support?: number;
  val_positive_support?: number;
  test_positive_support?: number;
  presentation_eligible?: boolean;
  trades: number;
  total_pnl: number;
  win_rate: number;
  sharpe: number;
  profit_factor: number;
  max_drawdown: number;
  expectancy: number;
};

export type CurvePoint = {
  trade_number: number;
  exit_ts: string;
  net_pnl: number;
  cumulative_pnl: number;
  entry_ts?: string | null;
  gross_pnl?: number;
  return?: number;
  direction?: string;
  exit_reason?: string;
  proba?: number;
  label?: number;
  pattern?: string;
  model?: string;
};

export type CurveSeries = {
  scope: "pair" | "model" | "pattern";
  series_id: string;
  label?: string;
  model?: string;
  pattern?: string;
  trades?: number;
  total_pnl?: number;
  points: CurvePoint[];
};

export type GallerySample = {
  image_path: string;
  image_src?: string;
  window_end_ts?: string | null;
  proba?: number | null;
  label: number;
  y_pred: number;
};

export type DashboardSnapshot = {
  meta: {
    run_name: string;
    generated_at: string;
    output_dir: string;
    source_paths: {
      metrics: string;
      backtest: string;
      gallery: string;
    };
    models: string[];
    patterns: string[];
  };
  hero: {
    patterns_covered: number;
    model_pair_count: number;
    champion_positive_patterns: number;
    champion_total_patterns: number;
    best_test_f1_pair: Partial<ClassificationRow>;
    best_backtest_pair: Partial<BacktestPairRow>;
    best_model_by_mean_f1: {
      model?: string;
      f1?: number;
    };
    macro?: Record<string, unknown>;
  };
  presentation: {
    presentation_eligible: boolean;
    visible_patterns: string[];
    hidden_patterns: string[];
    quality_score: number;
    presentation_reason: string;
    mean_visible_champion_f1: number;
    mean_visible_champion_pr_auc: number;
  };
  classification: {
    summary_rows: ClassificationRow[];
    detail_rows: ClassificationDetailRow[];
    champions: ChampionRow[];
  };
  backtest: {
    champion_rows: ChampionRow[];
    champion_summary_rows: Record<string, unknown>[];
    pair_rows: BacktestPairRow[];
    pair_curves: CurveSeries[];
    aggregate_curves: CurveSeries[];
  };
  gallery: {
    summary: Record<string, unknown>;
    samples: Record<string, Record<string, GallerySample[]>>;
  };
};

export type MetricKey = "f1" | "precision" | "recall" | "pr_auc";

export const metricLabels: Record<MetricKey, string> = {
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
  pr_auc: "PR AUC",
};

const percentFormatter = new Intl.NumberFormat("en-GB", {
  style: "percent",
  maximumFractionDigits: 1,
});

const numberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 2,
});

const compactNumberFormatter = new Intl.NumberFormat("en-GB", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const currencyFormatter = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const compactCurrencyFormatter = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

function simplifyUsdSymbol(value: string) {
  return value.replace("US$", "$");
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en-GB", {
    style: "percent",
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatCurrency(value: number | null | undefined, compact = false) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  const rendered = compact ? compactCurrencyFormatter.format(value) : currencyFormatter.format(value);
  return simplifyUsdSymbol(rendered);
}

export function formatSignedCurrency(value: number | null | undefined, compact = false) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  const base = compact ? compactCurrencyFormatter : currencyFormatter;
  const rendered = simplifyUsdSymbol(base.format(Math.abs(value)));
  if (value > 0) {
    return `+${rendered}`;
  }
  if (value < 0) {
    return `-${rendered}`;
  }
  return rendered;
}

export function formatNumber(value: number | null | undefined, compact = false) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return compact ? compactNumberFormatter.format(value) : numberFormatter.format(value);
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function humanizePattern(pattern: string) {
  return pattern
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

export function metricTone(value: number) {
  if (value >= 0.6) {
    return "positive";
  }
  if (value >= 0.5) {
    return "neutral";
  }
  return "caution";
}

export function pnlTone(value: number) {
  if (value > 0) {
    return "positive";
  }
  if (value < 0) {
    return "negative";
  }
  return "neutral";
}

export function buildGalleryAssetUrl(relativePath: string) {
  return `/resources/gallery?src=${encodeURIComponent(relativePath)}`;
}

export function averageByModel(rows: ClassificationRow[]) {
  const buckets = new Map<
    string,
    { model: string; count: number; f1: number; precision: number; recall: number; pr_auc: number }
  >();

  for (const row of rows) {
    const existing =
      buckets.get(row.model) ??
      { model: row.model, count: 0, f1: 0, precision: 0, recall: 0, pr_auc: 0 };
    existing.count += 1;
    existing.f1 += row.f1;
    existing.precision += row.precision;
    existing.recall += row.recall;
    existing.pr_auc += row.pr_auc;
    buckets.set(row.model, existing);
  }

  return [...buckets.values()]
    .map((bucket) => ({
      model: bucket.model,
      count: bucket.count,
      f1: bucket.f1 / bucket.count,
      precision: bucket.precision / bucket.count,
      recall: bucket.recall / bucket.count,
      pr_auc: bucket.pr_auc / bucket.count,
    }))
    .sort((left, right) => right.f1 - left.f1);
}

export function visiblePatterns(snapshot: DashboardSnapshot) {
  return snapshot.presentation ? snapshot.presentation.visible_patterns : snapshot.meta.patterns;
}

export function hiddenPatterns(snapshot: DashboardSnapshot) {
  return snapshot.presentation?.hidden_patterns ?? [];
}

export function visibleSummaryRows(snapshot: DashboardSnapshot) {
  const visible = new Set(visiblePatterns(snapshot));
  return snapshot.classification.summary_rows.filter((row) => visible.has(row.pattern));
}

export function visiblePairRows(snapshot: DashboardSnapshot) {
  const visible = new Set(visiblePatterns(snapshot));
  return snapshot.backtest.pair_rows.filter((row) => visible.has(row.pattern));
}

export function patternModelSeriesId(pattern: string, model: string) {
  return `${model}::${pattern}`;
}

export function championSeriesIds(snapshot: DashboardSnapshot) {
  return new Set(
    snapshot.classification.champions.map((row) => patternModelSeriesId(row.pattern, row.model)),
  );
}

export function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

export function mean(values: number[]) {
  if (values.length === 0) {
    return 0;
  }
  return sum(values) / values.length;
}
