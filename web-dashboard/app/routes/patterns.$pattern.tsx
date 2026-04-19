import { useState } from "react";
import { useOutletContext, useParams } from "react-router";

import type { Route } from "./+types/patterns.$pattern";
import { ProfitCurveChart, patternStory, type ChartSeries } from "../components/charts";
import { EmptyState, PageHeader, Pill, SectionCard, StatCard } from "../components/ui";
import {
  buildGalleryAssetUrl,
  formatDateTime as formatCaptionDateTime,
  formatNumber,
  formatPercent,
  formatSignedCurrency,
  visiblePatterns,
  humanizePattern,
  pnlTone,
} from "../lib/dashboard";
import type { DashboardOutletContext } from "../root";

const patternPalette = ["#0f766e", "#ea580c", "#2563eb", "#7c3aed", "#dc2626"];

function GallerySampleFigure({
  bucket,
  index,
  pattern,
  sample,
}: {
  bucket: "tp" | "fp" | "fn";
  index: number;
  pattern: string;
  sample: {
    image_path: string;
    proba?: number | null;
    window_end_ts?: string | null;
  };
}) {
  const [isBroken, setIsBroken] = useState(false);

  return (
    <figure className="gallery-card">
      {!isBroken ? (
        <img
          src={buildGalleryAssetUrl(sample.image_path)}
          alt={`${humanizePattern(pattern)} ${bucket.toUpperCase()} sample ${index + 1}`}
          className="aspect-[4/3] w-full object-cover"
          onError={() => setIsBroken(true)}
        />
      ) : (
        <div className="gallery-fallback">
          <div className="space-y-2">
            <p className="eyebrow">Visual unavailable</p>
            <p className="text-sm leading-6 text-slate-600">
              This sample could not be rendered in the browser, but the underlying gallery asset is still tracked in
              the latest snapshot.
            </p>
          </div>
        </div>
      )}

      <figcaption className="space-y-1 px-4 py-3 text-sm text-slate-600">
        <p className="font-medium text-slate-950">{formatCaptionDateTime(sample.window_end_ts)}</p>
        <p>Probability {formatPercent(sample.proba ?? 0)}</p>
      </figcaption>
    </figure>
  );
}

export function meta({ params }: Route.MetaArgs) {
  const pattern = params.pattern ? humanizePattern(params.pattern) : "Pattern";
  return [
    { title: `${pattern} · Candlestick Results` },
    {
      name: "description",
      content: `Pattern-specific dashboard for ${pattern} covering champion metrics, profitability, and gallery evidence.`,
    },
  ];
}

export default function PatternRoute() {
  const { snapshot } = useOutletContext<DashboardOutletContext>();
  const params = useParams();
  const pattern = params.pattern ?? "";

  if (!snapshot) {
    return null;
  }

  if (!visiblePatterns(snapshot).includes(pattern)) {
    return (
      <EmptyState
        title="Pattern hidden from the supported snapshot"
        description="This pattern exists in the finished run artifacts, but it stayed below the configured support floor for the main dashboard."
      />
    );
  }

  const champion = snapshot.classification.champions.find((row) => row.pattern === pattern);
  const modelRows = snapshot.backtest.pair_rows
    .filter((row) => row.pattern === pattern)
    .sort((left, right) => right.total_pnl - left.total_pnl || right.f1 - left.f1);
  const chartSeries: ChartSeries[] = snapshot.backtest.pair_curves
    .filter((curve) => curve.pattern === pattern)
    .map((curve, index) => ({
      id: curve.series_id,
      label: String(curve.model).toUpperCase(),
      color: patternPalette[index % patternPalette.length],
      points: curve.points.map((point) => ({
        exit_ts: point.exit_ts,
        cumulative_pnl: point.cumulative_pnl,
        net_pnl: point.net_pnl,
      })),
    }));
  const samples = snapshot.gallery.samples[pattern] ?? {};

  if (!champion) {
    return (
      <EmptyState
        title="Champion summary missing"
        description="The latest run contains the pattern but did not produce a champion summary row for it."
      />
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Pattern Focus"
        title={`${humanizePattern(pattern)} in the latest completed run: classification strength, profitability curve, and concrete visual evidence.`}
        description={patternStory(pattern, champion.total_pnl, champion.win_rate)}
        aside={
          <div className="grid w-full max-w-[19rem] gap-3 xl:max-w-[24rem] 2xl:max-w-none 2xl:grid-cols-2">
            <StatCard
              label="Champion model"
              value={champion.model.toUpperCase()}
              hint={`Selected on ${champion.selection_split} with ${formatPercent(champion.selection_f1)} F1.`}
              tone="positive"
              valueClassName="text-[clamp(1.85rem,3vw,2.8rem)] leading-[0.92] break-words"
            />
            <StatCard
              label="Champion PnL"
              value={formatSignedCurrency(champion.total_pnl)}
              hint={`${formatPercent(champion.win_rate)} win rate · ${formatNumber(champion.profit_factor)} profit factor`}
              tone={pnlTone(champion.total_pnl)}
              valueClassName="text-[clamp(1.75rem,2.8vw,2.75rem)] leading-[0.92] break-words"
            />
          </div>
        }
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Test F1" value={formatPercent(champion.test_f1)} hint="Leakage-safe champion test metric." tone="neutral" />
        <StatCard label="Precision" value={formatPercent(champion.test_precision)} hint="False-positive discipline." tone="neutral" />
        <StatCard label="Recall" value={formatPercent(champion.test_recall)} hint="Coverage on positive examples." tone="neutral" />
        <StatCard label="Trades" value={formatNumber(champion.trades)} hint={`${formatSignedCurrency(champion.expectancy)} expectancy`} tone="neutral" />
      </section>

      <SectionCard title="Model profitability overlay" kicker="Pattern Performance">
        <ProfitCurveChart series={chartSeries} />
      </SectionCard>

      <SectionCard title="Model comparison" kicker="Latest Completed Run">
        <div className="space-y-3">
          {modelRows.map((row) => (
            <article key={`${row.pattern}-${row.model}`} className="leaderboard-row">
              <div className="flex items-center gap-3">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{row.model}</p>
                  <p className="text-sm text-slate-500">{formatPercent(row.f1)} F1</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-right text-sm">
                <span className="text-slate-500">PnL</span>
                <span className="font-semibold text-slate-950">{formatSignedCurrency(row.total_pnl)}</span>
                <span className="text-slate-500">Win rate</span>
                <span className="font-semibold text-slate-950">{formatPercent(row.win_rate)}</span>
                <span className="text-slate-500">Profit factor</span>
                <span className="font-semibold text-slate-950">{formatNumber(row.profit_factor)}</span>
              </div>
            </article>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="TP / FP / FN gallery" kicker="Visual Evidence">
        <div className="grid gap-5 xl:grid-cols-3">
          {(["tp", "fp", "fn"] as const).map((bucket) => (
            <article key={bucket} className="subtle-panel space-y-4 p-4">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-slate-950">{bucket.toUpperCase()}</h3>
                <Pill tone={bucket === "tp" ? "positive" : bucket === "fp" ? "caution" : "negative"}>
                  {(samples[bucket] ?? []).length} samples
                </Pill>
              </div>
              <div className="grid gap-4">
                {(samples[bucket] ?? []).length > 0 ? (
                  (samples[bucket] ?? []).map((sample, index) => (
                    <GallerySampleFigure
                      key={`${bucket}-${index}`}
                      bucket={bucket}
                      index={index}
                      pattern={pattern}
                      sample={sample}
                    />
                  ))
                ) : (
                  <div className="gallery-fallback min-h-56">
                    <div className="space-y-2">
                      <p className="eyebrow">No examples saved</p>
                      <p className="text-sm leading-6 text-slate-600">
                        The latest snapshot did not surface a sample for this bucket.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
