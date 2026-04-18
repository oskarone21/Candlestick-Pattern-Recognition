import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { motion, useReducedMotion } from "motion/react";

import { formatSignedCurrency, humanizePattern } from "../lib/dashboard";

export type ChartSeries = {
  id: string;
  label: string;
  color: string;
  points: Array<{
    exit_ts: string;
    cumulative_pnl: number;
    net_pnl?: number;
  }>;
};

const MARKER_SUFFIX = "__marker";
const TRADE_SUFFIX = "__trade";
const CURVE_SEED_OFFSET_MS = 60_000;

function formatChartTick(value: string | number | undefined) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "2-digit",
  }).format(date);
}

function formatTooltipDate(value: string | number | undefined) {
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
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function buildCurveRows(series: ChartSeries[]) {
  if (series.length === 0) {
    return [];
  }

  const preparedSeries = series.map((item) => ({
    id: item.id,
    points: item.points
      .map((point) => ({
        ...point,
        timestamp: new Date(point.exit_ts).getTime(),
      }))
      .filter((point) => Number.isFinite(point.timestamp))
      .sort((left, right) => left.timestamp - right.timestamp),
  }));

  const uniqueTimestamps = new Set<number>();
  for (const item of preparedSeries) {
    for (const point of item.points) {
      uniqueTimestamps.add(point.timestamp);
    }
  }

  const axis = [...uniqueTimestamps].sort((left, right) => left - right);
  if (axis.length === 0) {
    return [];
  }

  const seededAxis = [Math.max(axis[0] - CURVE_SEED_OFFSET_MS, 0), ...axis];
  const pointers = preparedSeries.map(() => 0);
  const lastValues = preparedSeries.map(() => 0);

  return seededAxis.map((timestamp) => {
    const row: Record<string, number | string | boolean | null> = {
      exit_ts: new Date(timestamp).toISOString(),
    };

    preparedSeries.forEach((item, index) => {
      let hasTradeMarker = false;
      let tradeValue: number | null = null;

      while (pointers[index] < item.points.length && item.points[pointers[index]].timestamp <= timestamp) {
        const point = item.points[pointers[index]];
        lastValues[index] = point.cumulative_pnl;
        if (point.timestamp === timestamp) {
          hasTradeMarker = true;
          tradeValue = point.net_pnl ?? null;
        }
        pointers[index] += 1;
      }

      row[item.id] = lastValues[index];
      row[`${item.id}${MARKER_SUFFIX}`] = hasTradeMarker;
      row[`${item.id}${TRADE_SUFFIX}`] = tradeValue;
    });

    return row;
  });
}

function buildTradeRows(series: ChartSeries[]) {
  const rows = new Map<string, { exit_ts: string; net_pnl: number }>();

  for (const item of series) {
    for (const point of item.points) {
      const existing = rows.get(point.exit_ts) ?? {
        exit_ts: point.exit_ts,
        net_pnl: 0,
      };
      existing.net_pnl += point.net_pnl ?? 0;
      rows.set(point.exit_ts, existing);
    }
  }

  return [...rows.values()].sort(
    (left, right) => new Date(left.exit_ts).getTime() - new Date(right.exit_ts).getTime(),
  );
}

function CurrencyTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{
    color?: string;
    dataKey?: string | number;
    name?: string;
    payload?: Record<string, unknown>;
    value?: number;
  }>;
}) {
  const actualPayload =
    payload?.filter((item) => {
      const markerKey = `${String(item.dataKey ?? "")}${MARKER_SUFFIX}`;
      return Boolean(item.payload?.[markerKey]);
    }) ?? [];
  const visiblePayload = actualPayload.length > 0 ? actualPayload : payload ?? [];

  if (!active || visiblePayload.length === 0) {
    return null;
  }

  const label = formatTooltipDate(String(visiblePayload[0].payload?.exit_ts ?? ""));

  return (
    <div className="rounded-3xl border border-white/60 bg-white/95 p-4 shadow-[0_20px_60px_-35px_rgba(15,23,42,0.7)] backdrop-blur">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <div className="mt-3 space-y-2">
        {visiblePayload.map((item) => {
          const tradeKey = `${String(item.dataKey ?? "")}${TRADE_SUFFIX}`;
          const tradeValue = item.payload?.[tradeKey];
          return (
          <div key={item.name} className="flex items-center justify-between gap-6 text-sm">
            <span className="flex items-center gap-2 text-slate-600">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="flex flex-col">
                  <span>{item.name?.replace("::", " · ") ?? "Series"}</span>
                  {typeof tradeValue === "number" ? (
                    <span className="text-xs text-slate-400">Trade {formatSignedCurrency(tradeValue)}</span>
                  ) : null}
                </span>
            </span>
            <span className="font-semibold text-slate-950">{formatSignedCurrency(item.value)}</span>
          </div>
          );
        })}
      </div>
    </div>
  );
}

function TradeMarker({
  active,
  color,
  cx,
  cy,
  dataKey,
  payload,
}: {
  active: boolean;
  color: string;
  cx?: number;
  cy?: number;
  dataKey: string;
  payload?: Record<string, unknown>;
}) {
  if (!active || cx == null || cy == null) {
    return null;
  }

  const markerKey = `${dataKey}${MARKER_SUFFIX}`;
  if (!payload?.[markerKey]) {
    return null;
  }

  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={color}
      stroke="rgba(255,255,255,0.95)"
      strokeWidth={2}
      style={{ filter: "drop-shadow(0 6px 14px rgba(15, 23, 42, 0.18))" }}
    />
  );
}

export function ProfitCurveChart({ series }: { series: ChartSeries[] }) {
  const prefersReducedMotion = useReducedMotion();
  const rows = buildCurveRows(series);

  if (series.length === 0 || rows.length === 0) {
    return <div className="chart-empty">No cumulative PnL curve is available for the current filters.</div>;
  }

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: "easeOut" }}
      className="chart-shell h-[380px] w-full"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
          <ReferenceLine y={0} stroke="rgba(15,23,42,0.18)" strokeDasharray="4 6" />
          <CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
          <XAxis
            dataKey="exit_ts"
            tickFormatter={formatChartTick}
            minTickGap={36}
            tick={{ fill: "#64748b", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tickFormatter={(value) => formatSignedCurrency(Number(value), true)}
            tick={{ fill: "#64748b", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={88}
          />
          <Tooltip content={<CurrencyTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} iconType="circle" />
          {series.map((item) => (
            <Line
              key={item.id}
              type="stepAfter"
              dataKey={item.id}
              name={item.label}
              stroke={item.color}
              strokeWidth={3}
              strokeLinecap="round"
              strokeLinejoin="round"
              dot={(props) => (
                <TradeMarker
                  active
                  color={item.color}
                  cx={props.cx}
                  cy={props.cy}
                  dataKey={item.id}
                  payload={props.payload as Record<string, unknown>}
                />
              )}
              activeDot={(props) => (
                <TradeMarker
                  active
                  color={item.color}
                  cx={props.cx}
                  cy={props.cy}
                  dataKey={item.id}
                  payload={props.payload as Record<string, unknown>}
                />
              )}
              isAnimationActive={!prefersReducedMotion}
              animationDuration={900}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

export function TradeFlowChart({ series }: { series: ChartSeries[] }) {
  const prefersReducedMotion = useReducedMotion();
  const rows = buildTradeRows(series);

  if (series.length === 0 || rows.length === 0) {
    return <div className="chart-empty">Trade-by-trade PnL will appear here once the filtered series includes trades.</div>;
  }

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: "easeOut", delay: 0.08 }}
      className="chart-shell h-[300px] w-full"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
          <ReferenceLine y={0} stroke="rgba(15,23,42,0.18)" strokeDasharray="4 6" />
          <CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
          <XAxis
            dataKey="exit_ts"
            tickFormatter={formatChartTick}
            minTickGap={36}
            tick={{ fill: "#64748b", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tickFormatter={(value) => formatSignedCurrency(Number(value), true)}
            tick={{ fill: "#64748b", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={88}
          />
          <Tooltip content={<CurrencyTooltip />} />
          <Bar
            dataKey="net_pnl"
            radius={[10, 10, 10, 10]}
            isAnimationActive={!prefersReducedMotion}
            animationDuration={700}
          >
            {rows.map((row) => (
              <Cell key={row.exit_ts} fill={row.net_pnl >= 0 ? "#0f766e" : "#b45309"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

export function patternStory(pattern: string, pnl: number, winRate: number) {
  const title = humanizePattern(pattern);
  if (pnl > 0 && winRate >= 0.6) {
    return `${title} combines positive profitability with a disciplined hit rate in the latest completed run.`;
  }
  if (pnl > 0) {
    return `${title} is profitable in this snapshot, although its execution profile still looks less stable than the top pattern leaders.`;
  }
  if (winRate >= 0.5) {
    return `${title} is directionally interesting, but the current entries and exits still dilute the realised PnL.`;
  }
  return `${title} trails the rest of the field in the latest run and currently reads as supporting evidence rather than a lead result.`;
}
