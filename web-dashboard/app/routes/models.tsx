import { useState } from "react";
import { useOutletContext } from "react-router";

import type { Route } from "./+types/models";
import { PageHeader, Pill, SectionCard, StatCard, ToggleGroup } from "../components/ui";
import {
  averageByModel,
  formatPercent,
  formatNumber,
  hiddenPatterns,
  humanizePattern,
  metricLabels,
  type MetricKey,
  visiblePatterns,
  visibleSummaryRows,
} from "../lib/dashboard";
import type { DashboardOutletContext } from "../root";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Models · Candlestick Results" },
    {
      name: "description",
      content: "Classification performance matrix, confidence intervals, and per-model drilldowns for the supported subset of the latest run.",
    },
  ];
}

function heatStyle(value: number) {
  const hue = Math.round(145 - Math.max(0, Math.min(1, value)) * 100);
  const alpha = 0.15 + value * 0.4;
  return { backgroundColor: `hsla(${hue}, 72%, 52%, ${alpha})` };
}

export default function ModelsRoute() {
  const { snapshot } = useOutletContext<DashboardOutletContext>();
  const [metric, setMetric] = useState<MetricKey>("f1");

  if (!snapshot) {
    return null;
  }

  const presentationPatterns = visiblePatterns(snapshot);
  const lowSupportHiddenPatterns = hiddenPatterns(snapshot);
  const summaryRows = visibleSummaryRows(snapshot);
  const detailLookup = new Map(
    snapshot.classification.detail_rows.map((row) => [`${row.pattern}::${row.model}`, row]),
  );
  const modelAverages = averageByModel(summaryRows);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Classification Layer"
        title="Model selection stays anchored in validation discipline, then gets surfaced through the supported subset of patterns from the latest reproducible run."
        description="This page focuses the matrix on patterns that cleared the configured support floor. Lower-support rows remain in the raw artifacts, but they do not take space in the main comparison view."
        aside={
          <div className="grid gap-3 sm:grid-cols-2">
            {modelAverages.slice(0, 2).map((row) => (
              <StatCard
                key={row.model}
                label={`${row.model.toUpperCase()} mean F1`}
                value={formatPercent(row.f1)}
                hint={`${formatPercent(row.precision)} precision · ${formatPercent(row.recall)} recall`}
                tone="positive"
              />
            ))}
          </div>
        }
      />

      {lowSupportHiddenPatterns.length > 0 ? (
        <SectionCard title="Support floor" kicker="Hidden Patterns">
          <p className="text-sm leading-6 text-slate-600">
            Hidden from the main matrix because validation or test support stayed below the configured floor:{" "}
            {lowSupportHiddenPatterns.map((pattern) => humanizePattern(pattern)).join(", ")}.
          </p>
        </SectionCard>
      ) : null}

      <SectionCard
        title="Pattern × model matrix"
        kicker="At A Glance"
        actions={
          <ToggleGroup
            options={[
              { label: "F1", value: "f1" },
              { label: "Precision", value: "precision" },
              { label: "Recall", value: "recall" },
              { label: "PR AUC", value: "pr_auc" },
            ]}
            value={metric}
            onChange={setMetric}
          />
        }
      >
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {presentationPatterns.map((pattern) => (
            <article key={pattern} className="subtle-panel space-y-3 p-4">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-slate-950">{humanizePattern(pattern)}</h3>
                <Pill tone="neutral">{metricLabels[metric]}</Pill>
              </div>
              <div className="grid gap-2">
                {snapshot.meta.models.map((model) => {
                  const row = summaryRows.find((candidate) => candidate.pattern === pattern && candidate.model === model);
                  if (!row) {
                    return null;
                  }
                  const value = row[metric];
                  return (
                    <div
                      key={model}
                      className="flex items-center justify-between rounded-2xl px-3 py-3 text-sm text-slate-700"
                      style={heatStyle(value)}
                    >
                      <span className="font-semibold uppercase tracking-[0.16em] text-slate-950">{model}</span>
                      <span className="font-semibold text-slate-950">{formatPercent(value)}</span>
                    </div>
                  );
                })}
              </div>
            </article>
          ))}
        </div>
      </SectionCard>

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <SectionCard title="Model leaderboard" kicker="Mean performance across patterns">
          <div className="space-y-3">
            {modelAverages.map((row, index) => (
              <article key={row.model} className="leaderboard-row">
                <div className="flex items-center gap-4">
                  <div className="leaderboard-rank">{index + 1}</div>
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{row.model}</p>
                    <p className="text-sm text-slate-500">{row.count} visible model-pattern rows</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-sm">
                  <span className="text-slate-500">F1</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.f1)}</span>
                  <span className="text-slate-500">Precision</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.precision)}</span>
                  <span className="text-slate-500">Recall</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.recall)}</span>
                  <span className="text-slate-500">PR AUC</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.pr_auc)}</span>
                </div>
              </article>
            ))}
          </div>
        </SectionCard>

        <SectionCard title="Confidence interval inspection" kicker="Reliability">
          <div className="space-y-3">
            {summaryRows
              .slice()
              .sort((left, right) => right.f1 - left.f1)
              .slice(0, 8)
              .map((row) => {
                const detail = detailLookup.get(`${row.pattern}::${row.model}`);
                return (
                  <article key={`${row.pattern}-${row.model}`} className="subtle-panel p-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                          {humanizePattern(row.pattern)}
                        </p>
                        <h3 className="mt-1 text-lg font-semibold uppercase tracking-[0.12em] text-slate-950">
                          {row.model}
                        </h3>
                      </div>
                      <Pill tone="positive">Threshold {formatNumber(row.threshold)}</Pill>
                    </div>
                    <div className="mt-4 grid gap-3 sm:grid-cols-3">
                      <StatCard
                        label="F1 interval"
                        value={`${formatPercent(detail?.f1_ci_low)} – ${formatPercent(detail?.f1_ci_high)}`}
                        hint={`Point estimate ${formatPercent(row.f1)}`}
                        tone="neutral"
                      />
                      <StatCard
                        label="Precision interval"
                        value={`${formatPercent(detail?.precision_ci_low)} – ${formatPercent(detail?.precision_ci_high)}`}
                        hint={`Point estimate ${formatPercent(row.precision)}`}
                        tone="neutral"
                      />
                      <StatCard
                        label="Recall interval"
                        value={`${formatPercent(detail?.recall_ci_low)} – ${formatPercent(detail?.recall_ci_high)}`}
                        hint={`Point estimate ${formatPercent(row.recall)}`}
                        tone="neutral"
                      />
                    </div>
                  </article>
                );
              })}
          </div>
        </SectionCard>
      </section>

      <SectionCard title="Detailed comparison table" kicker="Full Scoreboard">
        <div className="overflow-auto rounded-[1.5rem] border border-white/60">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-slate-950 text-white">
              <tr>
                <th className="px-4 py-3 font-medium">Pattern</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Selection split</th>
                <th className="px-4 py-3 font-medium">F1</th>
                <th className="px-4 py-3 font-medium">Precision</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-4 py-3 font-medium">PR AUC</th>
                <th className="px-4 py-3 font-medium">Threshold</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white/75">
              {summaryRows
                .slice()
                .sort((left, right) => right.f1 - left.f1)
                .map((row) => (
                  <tr key={`${row.pattern}-${row.model}`} className="hover:bg-white">
                    <td className="px-4 py-3 text-slate-700">{humanizePattern(row.pattern)}</td>
                    <td className="px-4 py-3 font-semibold uppercase tracking-[0.14em] text-slate-950">{row.model}</td>
                    <td className="px-4 py-3 text-slate-700">{row.selection_split}</td>
                    <td className="px-4 py-3 text-slate-700">{formatPercent(row.f1)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatPercent(row.precision)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatPercent(row.recall)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatPercent(row.pr_auc)}</td>
                    <td className="px-4 py-3 text-slate-700">{formatNumber(row.threshold)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
