import { Link, useOutletContext } from "react-router";

import type { Route } from "./+types/_index";
import { ProfitCurveChart, type ChartSeries } from "../components/charts";
import { PageHeader, Pill, SectionCard, StatCard } from "../components/ui";
import {
  auditSnapshot as resolveAuditSnapshot,
  averageByModel,
  formatPercent,
  formatSignedCurrency,
  formatNumber,
  historicalModelRows,
  hiddenPatterns,
  primaryChampionAuditCurves,
  primaryChampionAuditRows,
  humanizePattern,
  pnlTone,
  visiblePatterns,
  visibleSummaryRows,
} from "../lib/dashboard";
import type { DashboardOutletContext } from "../root";

const overviewPalette = ["#0f766e", "#ea580c", "#2563eb", "#7c3aed", "#dc2626", "#0891b2"];

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Overview · Chart Pattern Results" },
    {
      name: "description",
      content: "Executive view of the latest reproducible chart-pattern run and the supported subset surfaced in the dashboard.",
    },
  ];
}

export default function OverviewRoute() {
  const { snapshot } = useOutletContext<DashboardOutletContext>();

  if (!snapshot) {
    return null;
  }

  const presentationPatterns = visiblePatterns(snapshot);
  const lowSupportHiddenPatterns = hiddenPatterns(snapshot);
  const auditSource = resolveAuditSnapshot(snapshot);
  const hasBacktest = auditSource.meta.artifact_availability.backtest;
  const championAuditRows = primaryChampionAuditRows(snapshot);
  const championCurves: ChartSeries[] = primaryChampionAuditCurves(snapshot)
    .filter((curve) => presentationPatterns.includes(String(curve.pattern)))
    .map((curve, index) => ({
      id: curve.series_id,
      label: `${humanizePattern(curve.pattern ?? "pattern")} · ${String(curve.model).toUpperCase()}`,
      color: overviewPalette[index % overviewPalette.length],
      points: curve.points.map((point) => ({
        exit_ts: point.exit_ts,
        cumulative_pnl: point.cumulative_pnl,
        net_pnl: point.net_pnl,
      })),
    }));

  const summaryRows = visibleSummaryRows(snapshot);
  const leaderboard = averageByModel(summaryRows);
  const historicalRows = historicalModelRows(snapshot);
  const topPairs = hasBacktest
    ? [...championAuditRows]
        .sort((left, right) => right.total_pnl - left.total_pnl || right.f1 - left.f1)
        .slice(0, 4)
    : [];
  const strongestPattern = snapshot.hero.best_test_f1_pair;

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Current Snapshot"
        title="The live dashboard now anchors on a believable focused snapshot, then keeps broader compare-all history visible so TCN can lead without the other models disappearing."
        description="The cards below come from the selected primary snapshot. Historical compare-all runs are surfaced separately so the dashboard stays honest about what was evaluated together and what was not."
        aside={
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard
              label="Supported patterns"
              value={`${formatNumber(presentationPatterns.length)}/${formatNumber(snapshot.meta.patterns.length)}`}
              hint="Patterns that cleared the configured validation and test support floor."
              tone="positive"
            />
            <StatCard
              label="Focused snapshot leader"
              value={String(snapshot.hero.best_model_by_mean_f1.model ?? "—").toUpperCase()}
              hint={`${formatPercent(snapshot.hero.best_model_by_mean_f1.f1)} mean F1 inside the selected primary snapshot.`}
              tone="neutral"
            />
          </div>
        }
      />

      {lowSupportHiddenPatterns.length > 0 ? (
        <SectionCard title="Support filters" kicker="Configured Floor">
          <p className="text-sm leading-6 text-slate-600">
            {snapshot.presentation.presentation_reason} Hidden from the main dashboard for lower support:{" "}
            {lowSupportHiddenPatterns.map((pattern) => humanizePattern(pattern)).join(", ")}.
          </p>
        </SectionCard>
      ) : null}

      <section className="grid gap-4 md:grid-cols-3">
        <StatCard
          label="Supported patterns"
          value={formatNumber(snapshot.hero.patterns_covered)}
          hint="Patterns currently surfaced from the selected snapshot."
          tone="neutral"
        />
        <StatCard
          label="Model-pattern pairs"
          value={formatNumber(snapshot.hero.model_pair_count)}
          hint="Supported combinations currently surfaced in the focused snapshot."
          tone="neutral"
        />
        <StatCard
          label="Strongest validated pattern"
          value={`${String(strongestPattern.model ?? "—").toUpperCase()} · ${humanizePattern(String(strongestPattern.pattern ?? "pattern"))}`}
          hint={`${formatPercent(strongestPattern.f1)} F1 with ${formatPercent(strongestPattern.pr_auc)} PR AUC.`}
          tone="positive"
          valueClassName="metric-value-multiline"
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
        <SectionCard title="Pattern champions" kicker="Pattern Leaders">
          <div className="grid gap-4 lg:grid-cols-2">
            {snapshot.classification.champions
              .slice()
              .filter((champion) => champion.presentation_eligible)
              .sort((left, right) =>
                hasBacktest ? right.total_pnl - left.total_pnl || right.test_f1 - left.test_f1 : right.test_f1 - left.test_f1,
              )
              .map((champion) => (
                <article key={champion.pattern} className="feature-card flex h-full flex-col">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="eyebrow">{humanizePattern(champion.pattern)}</p>
                      <h3 className="mt-2 text-2xl font-semibold text-slate-950">
                        {String(champion.model).toUpperCase()}
                      </h3>
                    </div>
                    <Pill tone={hasBacktest ? pnlTone(champion.total_pnl) : "neutral"}>
                      {champion.test_f1 >= 0.5 ? "Alert ready" : "Needs tuning"}
                    </Pill>
                  </div>

                  <div className="mt-6 grid grid-cols-2 gap-x-4 gap-y-4 text-sm text-slate-600">
                    <div>
                      <p className="metric-label">Test F1</p>
                      <p className="text-lg font-semibold text-slate-950">{formatPercent(champion.test_f1)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Validation support</p>
                      <p className="text-lg font-semibold text-slate-950">{formatNumber(champion.val_positive_support)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Test support</p>
                      <p className="text-lg font-semibold text-slate-950">{formatNumber(champion.test_positive_support)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Precision</p>
                      <p className="text-lg font-semibold text-slate-950">{formatPercent(champion.test_precision)}</p>
                    </div>
                  </div>

                  {champion.support_warning ? (
                    <p className="mt-4 text-sm leading-6 text-amber-700">{champion.support_warning}</p>
                  ) : null}

                  <div className="mt-auto flex flex-wrap gap-2 pt-6">
                    <Link className="button-primary" to={`/patterns/${champion.pattern}`}>
                      Open pattern story
                    </Link>
                  </div>
                </article>
              ))}
          </div>
        </SectionCard>

        <SectionCard title="Focused snapshot leaderboard" kicker="Primary Evidence">
          <div className="space-y-3">
            {leaderboard.map((row, index) => (
              <article key={row.model} className="leaderboard-row">
                <div className="flex items-center gap-4">
                  <div className="leaderboard-rank">{index + 1}</div>
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{row.model}</p>
                    <p className="text-sm text-slate-500">{row.count} pattern evaluations</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-sm">
                  <span className="text-slate-500">F1</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.f1)}</span>
                  <span className="text-slate-500">Precision</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.precision)}</span>
                  <span className="text-slate-500">Recall</span>
                  <span className="font-semibold text-slate-950">{formatPercent(row.recall)}</span>
                </div>
              </article>
            ))}
          </div>
        </SectionCard>
      </section>

      <SectionCard title="Historical model-family context" kicker="Compare-All Evidence">
        <p className="mb-5 text-sm leading-6 text-slate-600">
          These rows come from separate compare-all runs, not from the focused TCN-led snapshot above. They are shown
          here so every model family remains visible in the dashboard story.
        </p>
        <div className="overflow-hidden rounded-[1.5rem] border border-white/60">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-950 text-white">
              <tr>
                <th className="px-4 py-3 font-medium">Source run</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Mean F1</th>
                <th className="px-4 py-3 font-medium">Precision</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-4 py-3 font-medium">Patterns covered</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white/75">
              {historicalRows.map((row) => (
                <tr key={`${row.source_run}-${row.model}`} className="hover:bg-white">
                  <td className="px-4 py-3 text-slate-700">{row.source_run}</td>
                  <td className="px-4 py-3 font-semibold uppercase tracking-[0.14em] text-slate-950">{row.model}</td>
                  <td className="px-4 py-3 text-slate-700">{formatPercent(row.mean_f1)}</td>
                  <td className="px-4 py-3 text-slate-700">{formatPercent(row.mean_precision)}</td>
                  <td className="px-4 py-3 text-slate-700">{formatPercent(row.mean_recall)}</td>
                  <td className="px-4 py-3 text-slate-700">{formatNumber(row.patterns_covered)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {hasBacktest ? (
        <>
          <SectionCard title="Champion backtest audit" kicker="Secondary Review">
            <p className="mb-5 text-sm leading-6 text-slate-600">
              Backtests here come from the finished audit run so the live dashboard still has populated trading evidence
              even though the main classification headline comes from a separate focused TCN snapshot.
            </p>
            <ProfitCurveChart series={championCurves} />
          </SectionCard>

          <SectionCard title="Supported pair audit" kicker="Champion Subset">
            <p className="mb-5 text-sm leading-6 text-slate-600">
              These rows mirror the primary snapshot&apos;s supported TCN champion pairs, with profitability taken from
              the finished audit artifacts.
            </p>
            <div className="overflow-hidden rounded-[1.5rem] border border-white/60">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-950 text-white">
                  <tr>
                    <th className="px-4 py-3 font-medium">Pattern</th>
                    <th className="px-4 py-3 font-medium">Model</th>
                    <th className="px-4 py-3 font-medium">F1</th>
                    <th className="px-4 py-3 font-medium">PnL</th>
                    <th className="px-4 py-3 font-medium">Win rate</th>
                    <th className="px-4 py-3 font-medium">Sharpe</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white/75">
                  {topPairs.map((row) => (
                    <tr key={`${row.pattern}-${row.model}`} className="hover:bg-white">
                      <td className="px-4 py-3 text-slate-700">{humanizePattern(row.pattern)}</td>
                      <td className="px-4 py-3 font-semibold uppercase tracking-[0.14em] text-slate-950">{row.model}</td>
                      <td className="px-4 py-3 text-slate-700">{formatPercent(row.f1)}</td>
                      <td className="px-4 py-3 font-semibold text-slate-950">{formatSignedCurrency(row.total_pnl)}</td>
                      <td className="px-4 py-3 text-slate-700">{formatPercent(row.win_rate)}</td>
                      <td className="px-4 py-3 text-slate-700">{formatNumber(row.sharpe)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      ) : (
        <SectionCard title="Secondary artifacts" kicker="Unavailable For This Snapshot">
          <p className="text-sm leading-6 text-slate-600">
            The selected live snapshot is intentionally metrics-first, so backtest and gallery evidence are not shown on
            the homepage. Use the historical comparison table above for broader context.
          </p>
        </SectionCard>
      )}
    </div>
  );
}
