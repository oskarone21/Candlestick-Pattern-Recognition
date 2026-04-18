import { Link, useOutletContext } from "react-router";

import type { Route } from "./+types/_index";
import { ProfitCurveChart, type ChartSeries } from "../components/charts";
import { PageHeader, Pill, SectionCard, StatCard } from "../components/ui";
import {
  averageByModel,
  championSeriesIds,
  formatPercent,
  formatSignedCurrency,
  formatNumber,
  humanizePattern,
  pnlTone,
} from "../lib/dashboard";
import type { DashboardOutletContext } from "../root";

const overviewPalette = ["#0f766e", "#ea580c", "#2563eb", "#7c3aed", "#dc2626", "#0891b2"];

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Overview · Candlestick Results" },
    {
      name: "description",
      content: "Executive view of the latest finished candlestick experiment run and its profitability outlook.",
    },
  ];
}

export default function OverviewRoute() {
  const { snapshot } = useOutletContext<DashboardOutletContext>();

  if (!snapshot) {
    return null;
  }

  const championSet = championSeriesIds(snapshot);
  const championCurves: ChartSeries[] = snapshot.backtest.pair_curves
    .filter((curve) => championSet.has(curve.series_id))
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

  const leaderboard = averageByModel(snapshot.classification.summary_rows);
  const topPairs = [...snapshot.backtest.pair_rows]
    .sort((left, right) => right.total_pnl - left.total_pnl || right.f1 - left.f1)
    .slice(0, 4);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Current Snapshot"
        title="The latest completed run, organised for presentation: strongest patterns, clean profitability evidence, and the best-performing model families."
        description="This overview keeps the story tight. It highlights the current leaders, surfaces the equity curves behind them, and preserves the technical evidence without reading like presenter notes."
        aside={
          <div className="grid gap-3 sm:grid-cols-2">
            <StatCard
              label="Positive champion patterns"
              value={`${snapshot.hero.champion_positive_patterns}/${snapshot.hero.champion_total_patterns}`}
              hint="Patterns whose champion backtest finished above zero PnL."
              tone="positive"
            />
            <StatCard
              label="Best mean F1 model"
              value={String(snapshot.hero.best_model_by_mean_f1.model ?? "—").toUpperCase()}
              hint={`${formatPercent(snapshot.hero.best_model_by_mean_f1.f1)} average F1 across all patterns.`}
              tone="neutral"
            />
          </div>
        }
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Patterns covered"
          value={formatNumber(snapshot.hero.patterns_covered)}
          hint="Patterns with complete classification and business readouts."
          tone="neutral"
        />
        <StatCard
          label="Model-pattern pairs"
          value={formatNumber(snapshot.hero.model_pair_count)}
          hint="All evaluated combinations in the latest snapshot."
          tone="neutral"
        />
        <StatCard
          label="Best test F1 pair"
          value={`${String(snapshot.hero.best_test_f1_pair.model ?? "—").toUpperCase()} · ${humanizePattern(String(snapshot.hero.best_test_f1_pair.pattern ?? "pattern"))}`}
          hint={`${formatPercent(snapshot.hero.best_test_f1_pair.f1)} F1 with ${formatPercent(snapshot.hero.best_test_f1_pair.pr_auc)} PR AUC.`}
          tone="positive"
        />
        <StatCard
          label="Best backtest pair"
          value={`${String(snapshot.hero.best_backtest_pair.model ?? "—").toUpperCase()} · ${humanizePattern(String(snapshot.hero.best_backtest_pair.pattern ?? "pattern"))}`}
          hint={`${formatSignedCurrency(snapshot.hero.best_backtest_pair.total_pnl)} total PnL with ${formatPercent(snapshot.hero.best_backtest_pair.win_rate)} win rate.`}
          tone={pnlTone(Number(snapshot.hero.best_backtest_pair.total_pnl ?? 0))}
        />
      </section>

      <SectionCard
        title="Champion profitability curve"
        kicker="Profitability Snapshot"
        actions={<Pill tone="positive">Champion overlay</Pill>}
      >
        <ProfitCurveChart series={championCurves} />
      </SectionCard>

      <section className="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
        <SectionCard title="Pattern champions" kicker="Pattern Leaders">
          <div className="grid gap-4 lg:grid-cols-2">
            {snapshot.classification.champions
              .slice()
              .sort((left, right) => right.total_pnl - left.total_pnl)
              .map((champion) => (
                <article key={champion.pattern} className="feature-card flex h-full flex-col">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="eyebrow">{humanizePattern(champion.pattern)}</p>
                      <h3 className="mt-2 text-2xl font-semibold text-slate-950">
                        {String(champion.model).toUpperCase()}
                      </h3>
                    </div>
                    <Pill tone={pnlTone(champion.total_pnl)}>
                      {champion.total_pnl > 0 ? "Promising" : "Needs work"}
                    </Pill>
                  </div>

                  <div className="mt-6 grid grid-cols-2 gap-x-4 gap-y-4 text-sm text-slate-600">
                    <div>
                      <p className="metric-label">Test F1</p>
                      <p className="text-lg font-semibold text-slate-950">{formatPercent(champion.test_f1)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Total PnL</p>
                      <p className="text-lg font-semibold text-slate-950">{formatSignedCurrency(champion.total_pnl)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Win rate</p>
                      <p className="text-lg font-semibold text-slate-950">{formatPercent(champion.win_rate)}</p>
                    </div>
                    <div>
                      <p className="metric-label">Profit factor</p>
                      <p className="text-lg font-semibold text-slate-950">{formatNumber(champion.profit_factor)}</p>
                    </div>
                  </div>

                  <div className="mt-auto flex flex-wrap gap-2 pt-6">
                    <Link className="button-primary" to={`/patterns/${champion.pattern}`}>
                      Open pattern story
                    </Link>
                  </div>
                </article>
              ))}
          </div>
        </SectionCard>

        <SectionCard title="Model leaderboard" kicker="Across All Patterns">
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

      <SectionCard title="Top profitability candidates" kicker="Highest Total PnL">
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
    </div>
  );
}
