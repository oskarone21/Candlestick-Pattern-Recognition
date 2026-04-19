import { useState } from "react";
import { useOutletContext } from "react-router";

import type { Route } from "./+types/profitability";
import { ProfitCurveChart, TradeFlowChart, type ChartSeries } from "../components/charts";
import { PageHeader, Pill, SectionCard, StatCard, ToggleGroup } from "../components/ui";
import {
  championSeriesIds,
  formatNumber,
  formatPercent,
  formatSignedCurrency,
  hiddenPatterns,
  humanizePattern,
  mean,
  patternModelSeriesId,
  pnlTone,
  sum,
  type BacktestPairRow,
  type CurveSeries,
  visiblePatterns,
  visiblePairRows as presentationPairRows,
} from "../lib/dashboard";
import type { DashboardOutletContext } from "../root";

type Scope = "champion" | "pair" | "model" | "pattern";

const profitabilityPalette = [
  "#0f766e",
  "#ea580c",
  "#2563eb",
  "#7c3aed",
  "#dc2626",
  "#0891b2",
  "#84cc16",
  "#f59e0b",
];

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Profitability · Candlestick Results" },
    {
      name: "description",
      content: "Cumulative profitability, trade timeline overlays, and filters across models and patterns.",
    },
  ];
}

function aggregateSeries(series: CurveSeries[], scope: "model" | "pattern"): CurveSeries[] {
  const groups = new Map<
    string,
    {
      scope: "model" | "pattern";
      series_id: string;
      label: string;
      model?: string;
      pattern?: string;
      points: Map<string, { exit_ts: string; net_pnl: number }>;
    }
  >();

  for (const curve of series) {
    const key = scope === "model" ? String(curve.model) : String(curve.pattern);
    const label = scope === "model" ? String(curve.model).toUpperCase() : humanizePattern(String(curve.pattern));
    const group =
      groups.get(key) ??
      {
        scope,
        series_id: key,
        label,
        model: scope === "model" ? key : undefined,
        pattern: scope === "pattern" ? key : undefined,
        points: new Map(),
      };

    for (const point of curve.points) {
      const existing = group.points.get(point.exit_ts) ?? { exit_ts: point.exit_ts, net_pnl: 0 };
      existing.net_pnl += point.net_pnl ?? 0;
      group.points.set(point.exit_ts, existing);
    }

    groups.set(key, group);
  }

  return [...groups.values()].map((group) => {
    let cumulative = 0;
    const points = [...group.points.values()]
      .sort((left, right) => new Date(left.exit_ts).getTime() - new Date(right.exit_ts).getTime())
      .map((point, index) => {
        cumulative += point.net_pnl;
        return {
          trade_number: index + 1,
          exit_ts: point.exit_ts,
          net_pnl: point.net_pnl,
          cumulative_pnl: cumulative,
        };
      });

    return {
      scope,
      series_id: group.series_id,
      label: group.label,
      model: group.model,
      pattern: group.pattern,
      trades: points.length,
      total_pnl: cumulative,
      points,
    };
  });
}

function checkboxTone(active: boolean) {
  return active
    ? "border-slate-950 bg-slate-950 text-white shadow-[0_18px_36px_-24px_rgba(15,23,42,0.75)]"
    : "border-white/60 bg-white/70 text-slate-600 hover:bg-white";
}

export default function ProfitabilityRoute() {
  const { snapshot } = useOutletContext<DashboardOutletContext>();
  const [scope, setScope] = useState<Scope>("champion");
  const presentationPatterns = snapshot ? visiblePatterns(snapshot) : [];
  const [selectedModels, setSelectedModels] = useState<string[]>(snapshot?.meta.models ?? []);
  const [selectedPatterns, setSelectedPatterns] = useState<string[]>(presentationPatterns);

  if (!snapshot) {
    return null;
  }

  const lowSupportHiddenPatterns = hiddenPatterns(snapshot);
  const championIds = championSeriesIds(snapshot);
  const filteredPairRows = presentationPairRows(snapshot).filter(
    (row) => selectedModels.includes(row.model) && selectedPatterns.includes(row.pattern),
  );
  const filteredPairCurves = snapshot.backtest.pair_curves.filter(
    (curve) =>
      presentationPatterns.includes(String(curve.pattern)) &&
      selectedModels.includes(String(curve.model)) &&
      selectedPatterns.includes(String(curve.pattern)),
  );

  const visiblePairRows =
    scope === "champion"
      ? filteredPairRows.filter((row) => championIds.has(patternModelSeriesId(row.pattern, row.model)))
      : filteredPairRows;

  const visibleSeries = (() => {
    if (scope === "champion") {
      return filteredPairCurves.filter((curve) => championIds.has(curve.series_id));
    }
    if (scope === "pair") {
      return filteredPairCurves;
    }
    return aggregateSeries(filteredPairCurves, scope);
  })();

  const chartSeries: ChartSeries[] = visibleSeries.map((series, index) => ({
    id: series.series_id,
    label:
      scope === "pair" || scope === "champion"
        ? `${humanizePattern(String(series.pattern))} · ${String(series.model).toUpperCase()}`
        : String(series.label ?? series.series_id),
    color: profitabilityPalette[index % profitabilityPalette.length],
    points: series.points.map((point) => ({
      exit_ts: point.exit_ts,
      cumulative_pnl: point.cumulative_pnl,
      net_pnl: point.net_pnl,
    })),
  }));

  const weightedTrades = sum(visiblePairRows.map((row) => row.trades));
  const totalPnl = sum(visiblePairRows.map((row) => row.total_pnl));
  const weightedWinRate =
    weightedTrades > 0 ? sum(visiblePairRows.map((row) => row.win_rate * row.trades)) / weightedTrades : 0;
  const averageSharpe = mean(visiblePairRows.map((row) => row.sharpe));
  const averageProfitFactor = mean(visiblePairRows.map((row) => row.profit_factor));
  const worstDrawdown = visiblePairRows.length ? Math.min(...visiblePairRows.map((row) => row.max_drawdown)) : 0;
  const allVisiblePairsNegative = visiblePairRows.length > 0 && visiblePairRows.every((row) => row.total_pnl <= 0);

  function toggleValue(list: string[], value: string) {
    return list.includes(value) ? list.filter((item) => item !== value) : [...list, value];
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Backtest Audit"
        title="Backtest curves remain visible as an audit layer, while the main proposal story stays anchored in analyst productivity and validated detection quality."
        description="Use the filters below to inspect backtest behaviour without letting a weak or sparse run dominate the product narrative."
        aside={
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard
              label="Visible series"
              value={formatNumber(chartSeries.length)}
              hint={`${scope[0].toUpperCase()}${scope.slice(1)} mode`}
              tone="neutral"
            />
            <StatCard
              label="Visible trades"
              value={formatNumber(weightedTrades)}
              hint={`${formatSignedCurrency(totalPnl)} combined PnL`}
              tone={pnlTone(totalPnl)}
            />
          </div>
        }
      />

      {allVisiblePairsNegative ? (
        <SectionCard title="Audit warning" kicker="Negative Snapshot">
          <p className="text-sm leading-6 text-slate-600">
            Every visible pair in this snapshot is negative on backtest. That does not invalidate the analyst-use-case,
            but it means profitability should be treated as an audit note, not as the primary headline.
          </p>
        </SectionCard>
      ) : null}

      {lowSupportHiddenPatterns.length > 0 ? (
        <SectionCard title="Support gate" kicker="Hidden Patterns">
          <p className="text-sm leading-6 text-slate-600">
            Hidden from the backtest audit by default because validation or test support is too low:{" "}
            {lowSupportHiddenPatterns.map((pattern) => humanizePattern(pattern)).join(", ")}.
          </p>
        </SectionCard>
      ) : null}

      <SectionCard
        title="Profitability controls"
        kicker="Filter The View"
        actions={
          <ToggleGroup
            options={[
              { label: "Champion", value: "champion" },
              { label: "Pairs", value: "pair" },
              { label: "Models", value: "model" },
              { label: "Patterns", value: "pattern" },
            ]}
            value={scope}
            onChange={setScope}
          />
        }
      >
        <div className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            <div>
              <p className="metric-label">Models</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {snapshot.meta.models.map((model) => (
                  <button
                    key={model}
                    type="button"
                    className={`rounded-full border px-3 py-2 text-sm font-semibold uppercase tracking-[0.14em] transition ${checkboxTone(
                      selectedModels.includes(model),
                    )}`}
                    onClick={() => setSelectedModels((current) => toggleValue(current, model))}
                  >
                    {model}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <p className="metric-label">Patterns</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {presentationPatterns.map((pattern) => (
                  <button
                    key={pattern}
                    type="button"
                    className={`rounded-full border px-3 py-2 text-sm font-semibold transition ${checkboxTone(
                      selectedPatterns.includes(pattern),
                    )}`}
                    onClick={() => setSelectedPatterns((current) => toggleValue(current, pattern))}
                  >
                    {humanizePattern(pattern)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Total PnL" value={formatSignedCurrency(totalPnl)} hint="Sum of visible pair-level backtests." tone={pnlTone(totalPnl)} />
            <StatCard label="Win rate" value={formatPercent(weightedWinRate)} hint="Trade-weighted across visible pairs." tone="neutral" />
            <StatCard label="Profit factor" value={formatNumber(averageProfitFactor)} hint="Average of visible pair summaries." tone="neutral" />
            <StatCard label="Worst drawdown" value={formatSignedCurrency(worstDrawdown)} hint="Most negative visible drawdown." tone="caution" />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Cumulative profitability curve" kicker="Step Equity Curve">
        <ProfitCurveChart series={chartSeries} />
      </SectionCard>

      <SectionCard title="Trade flow over time" kicker="Trade Timing">
        <TradeFlowChart series={chartSeries} />
      </SectionCard>

      <SectionCard title="Visible profitability rows" kicker="Current Selection">
        <div className="overflow-auto rounded-[1.5rem] border border-white/60">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-slate-950 text-white">
              <tr>
                <th className="px-4 py-3 font-medium">Pattern</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Trades</th>
                <th className="px-4 py-3 font-medium">PnL</th>
                <th className="px-4 py-3 font-medium">Win rate</th>
                <th className="px-4 py-3 font-medium">Profit factor</th>
                <th className="px-4 py-3 font-medium">Sharpe</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white/75">
              {visiblePairRows
                .slice()
                .sort((left, right) => right.total_pnl - left.total_pnl)
                .map((row: BacktestPairRow) => (
                  <tr key={`${row.pattern}-${row.model}`} className="hover:bg-white">
                    <td className="px-4 py-3 text-slate-700">{humanizePattern(row.pattern)}</td>
                    <td className="px-4 py-3 font-semibold uppercase tracking-[0.14em] text-slate-950">{row.model}</td>
                    <td className="px-4 py-3 text-slate-700">{formatNumber(row.trades)}</td>
                    <td className="px-4 py-3 font-semibold text-slate-950">{formatSignedCurrency(row.total_pnl)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatPercent(row.win_rate)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatNumber(row.profit_factor)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatNumber(row.sharpe)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
