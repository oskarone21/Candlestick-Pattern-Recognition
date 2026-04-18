export const MODEL_COLORS = {
  hgb: "#a86036",
  logreg: "#1f6b63",
  lstm: "#284c8f",
  tcn: "#7a4d7d",
  transformer: "#b5852d",
} as const;

export const PATTERN_COLORS = {
  double_bottom: "#21845c",
  double_top: "#b65e3a",
  head_shoulders: "#944660",
  inverse_head_shoulders: "#34598c",
} as const;

export const SCOPE_OPTIONS = [
  { value: "champion", label: "Champion curves" },
  { value: "pair", label: "All model-pattern pairs" },
  { value: "model", label: "Model rollups" },
  { value: "pattern", label: "Pattern rollups" },
] as const;

export const METRIC_OPTIONS = [
  { value: "f1", label: "Test F1" },
  { value: "precision", label: "Precision" },
  { value: "recall", label: "Recall" },
  { value: "pr_auc", label: "PR AUC" },
  { value: "selection_f1", label: "Validation F1" },
  { value: "total_pnl", label: "Total PnL" },
  { value: "win_rate", label: "Win Rate" },
] as const;

export type ScopeOption = (typeof SCOPE_OPTIONS)[number]["value"];
export type MetricOption = (typeof METRIC_OPTIONS)[number]["value"];
export type ModelSlug = keyof typeof MODEL_COLORS;
export type PatternSlug = keyof typeof PATTERN_COLORS;
export type GalleryBucket = "tp" | "fp" | "fn";

export type HeroMacro = {
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  patterns_covered: number;
};

export type PairMetrics = {
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
  support_positive?: number;
  support_negative?: number;
  trades: number;
  total_pnl: number;
  win_rate: number;
  sharpe: number;
  profit_factor: number;
  max_drawdown: number;
  expectancy: number;
  selection_split?: string;
};

export type ChampionRow = PairMetrics & {
  selection_split: string;
  test_f1: number;
  test_precision: number;
  test_recall: number;
  test_pr_auc: number;
};

export type ClassificationSummaryRow = {
  pattern: string;
  model: string;
  selection_split: string;
  threshold: number;
  selection_f1: number;
  selection_precision: number;
  selection_recall: number;
  f1: number;
  precision: number;
  recall: number;
  pr_auc: number;
  support_positive: number;
  support_negative: number;
};

export type ClassificationDetailRow = {
  pattern: string;
  model: string;
  artifact_path: string;
  threshold: number;
  accuracy: number;
  confusion_matrix: [number, number, number, number];
  f1_ci_low: number;
  f1_ci_high: number;
  precision_ci_low: number;
  precision_ci_high: number;
  recall_ci_low: number;
  recall_ci_high: number;
};

export type ProfitPoint = {
  trade_number: number;
  exit_ts: string;
  entry_ts?: string;
  net_pnl: number;
  gross_pnl?: number;
  cumulative_pnl: number;
  return?: number;
  direction?: "long" | "short";
  exit_reason: string;
  proba?: number;
  label?: number;
  pattern?: string;
};

export type PairCurve = {
  scope: "pair";
  series_id: string;
  model: string;
  pattern: string;
  trades: number;
  total_pnl: number;
  points: ProfitPoint[];
};

export type AggregateCurve = {
  scope: "model" | "pattern";
  series_id: string;
  label: string;
  model?: string;
  pattern?: string;
  points: ProfitPoint[];
};

export type GallerySample = {
  image_path: string;
  image_src: string;
  label: 0 | 1;
  proba: number | null;
  window_end_ts: string;
  y_pred: 0 | 1;
};

export type GallerySamples = Partial<Record<GalleryBucket, GallerySample[]>>;

export type DashboardSnapshot = {
  meta: {
    run_name: string;
    generated_at: string;
    output_dir: string;
    models: string[];
    patterns: string[];
    source_paths: string[];
  };
  hero: {
    best_backtest_pair: PairMetrics;
    best_model_by_mean_f1: { model: string; mean_f1: number };
    best_test_f1_pair: ClassificationSummaryRow;
    champion_positive_patterns: number;
    champion_total_patterns: number;
    macro: HeroMacro;
    model_pair_count: number;
    patterns_covered: number;
  };
  classification: {
    champions: ChampionRow[];
    detail_rows: ClassificationDetailRow[];
    summary_rows: ClassificationSummaryRow[];
  };
  backtest: {
    champion_rows: PairMetrics[];
    champion_summary_rows: PairMetrics[];
    pair_rows: PairMetrics[];
    pair_curves: PairCurve[];
    aggregate_curves: AggregateCurve[];
  };
  gallery: {
    summary: Record<string, Record<GalleryBucket, number>>;
    samples: Record<string, GallerySamples>;
  };
};

export type ModelRollup = {
  model: string;
  avgF1: number;
  avgPrecision: number;
  avgRecall: number;
  avgPnl: number;
  positivePairs: number;
  totalPairs: number;
  championCount: number;
  bestPattern: string;
};

export type PatternRollup = {
  pattern: string;
  avgF1: number;
  avgPnl: number;
  positivePairs: number;
  championModel: string | null;
};

export function average(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function prettyLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function formatMetric(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toFixed(digits);
}

export function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMoney(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "USD",
    signDisplay: "always",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatDateTime(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

export function getAccentColor(key: string) {
  return (
    MODEL_COLORS[key as ModelSlug] ??
    PATTERN_COLORS[key as PatternSlug] ??
    "#23344e"
  );
}

export function valueTone(value: number) {
  if (value > 0) return "text-emerald-700";
  if (value < 0) return "text-rose-700";
  return "text-slate-700";
}

export function championSeries(snapshot: DashboardSnapshot) {
  const championIds = new Set(
    snapshot.classification.champions.map(
      (row) => `${row.model}::${row.pattern}`,
    ),
  );

  return snapshot.backtest.pair_curves.filter((row) =>
    championIds.has(row.series_id),
  );
}

export function selectCurves(
  snapshot: DashboardSnapshot,
  scope: ScopeOption,
  models: string[],
  patterns: string[],
) {
  if (scope === "champion") {
    return championSeries(snapshot).filter(
      (row) => models.includes(row.model) && patterns.includes(row.pattern),
    );
  }

  if (scope === "pair") {
    return snapshot.backtest.pair_curves.filter(
      (row) => models.includes(row.model) && patterns.includes(row.pattern),
    );
  }

  if (scope === "model") {
    return snapshot.backtest.aggregate_curves.filter(
      (row) => row.scope === "model" && row.model && models.includes(row.model),
    );
  }

  return snapshot.backtest.aggregate_curves.filter(
    (row) =>
      row.scope === "pattern" && row.pattern && patterns.includes(row.pattern),
  );
}

export function getModelRollups(snapshot: DashboardSnapshot): ModelRollup[] {
  return snapshot.meta.models.map((model) => {
    const rows = snapshot.backtest.pair_rows.filter((row) => row.model === model);
    const bestRow = rows
      .slice()
      .sort((left, right) => right.total_pnl - left.total_pnl)[0];

    return {
      model,
      avgF1: average(rows.map((row) => row.f1)),
      avgPrecision: average(rows.map((row) => row.precision)),
      avgRecall: average(rows.map((row) => row.recall)),
      avgPnl: average(rows.map((row) => row.total_pnl)),
      positivePairs: rows.filter((row) => row.total_pnl > 0).length,
      totalPairs: rows.length,
      championCount: snapshot.classification.champions.filter(
        (row) => row.model === model,
      ).length,
      bestPattern: bestRow?.pattern ?? snapshot.meta.patterns[0],
    };
  });
}

export function getPatternRollups(snapshot: DashboardSnapshot): PatternRollup[] {
  return snapshot.meta.patterns.map((pattern) => {
    const rows = snapshot.backtest.pair_rows.filter(
      (row) => row.pattern === pattern,
    );
    const champion = snapshot.classification.champions.find(
      (row) => row.pattern === pattern,
    );

    return {
      pattern,
      avgF1: average(rows.map((row) => row.f1)),
      avgPnl: average(rows.map((row) => row.total_pnl)),
      positivePairs: rows.filter((row) => row.total_pnl > 0).length,
      championModel: champion?.model ?? null,
    };
  });
}

export function getPatternRows(snapshot: DashboardSnapshot, pattern: string) {
  return snapshot.backtest.pair_rows.filter((row) => row.pattern === pattern);
}

export function getPatternChampion(
  snapshot: DashboardSnapshot,
  pattern: string,
) {
  return snapshot.classification.champions.find((row) => row.pattern === pattern);
}

export function getGalleryForPattern(
  snapshot: DashboardSnapshot,
  pattern: string,
) {
  return snapshot.gallery.samples[pattern] ?? {};
}

export function isKnownPattern(snapshot: DashboardSnapshot, pattern: string) {
  return snapshot.meta.patterns.includes(pattern);
}

export function axisDomain(series: Array<PairCurve | AggregateCurve>) {
  const points = series.flatMap((row) => row.points);
  const xValues = points.map((point) => new Date(point.exit_ts).getTime());
  const yValues = points.map((point) => point.cumulative_pnl);

  return {
    xMin: Math.min(...xValues),
    xMax: Math.max(...xValues),
    yMin: Math.min(...yValues, 0),
    yMax: Math.max(...yValues, 0),
  };
}
