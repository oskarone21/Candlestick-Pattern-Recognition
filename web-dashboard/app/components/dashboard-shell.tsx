import type { ReactNode } from "react";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { NavLink, useLocation, useNavigation } from "react-router";

import { formatDateTime, formatNumber, humanizePattern, type DashboardSnapshot, type LatestFinishedRun } from "../lib/dashboard";
import { EmptyState, Pill } from "./ui";

export function DashboardShell({
  snapshot,
  latestRun,
  children,
}: {
  snapshot: DashboardSnapshot | null;
  latestRun: LatestFinishedRun | null;
  children: ReactNode;
}) {
  const navigation = useNavigation();
  const location = useLocation();
  const prefersReducedMotion = useReducedMotion();
  const navItems = [
    { label: "Overview", to: "/" },
    { label: "Models", to: "/models" },
    { label: "Profitability", to: "/profitability" },
    ...(snapshot?.meta.patterns ?? []).map((pattern) => ({
      label: humanizePattern(pattern),
      to: `/patterns/${pattern}`,
    })),
  ];

  return (
    <div className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top,rgba(13,148,136,0.11),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(245,158,11,0.14),transparent_24%),linear-gradient(180deg,#fffdf7_0%,#f4f7fb_52%,#eef2f7_100%)] text-slate-950">
      <div className="hero-orb hero-orb-a" />
      <div className="hero-orb hero-orb-b" />
      <div className="hero-orb hero-orb-c" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(15,23,42,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(15,23,42,0.03)_1px,transparent_1px)] bg-[size:72px_72px] opacity-40" />

      <header className="sticky top-0 z-40 border-b border-white/40 bg-white/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="space-y-1">
            <p className="eyebrow">Candlestick Pattern Intelligence</p>
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-lg font-semibold tracking-[0.02em] text-slate-950">Strategy Presentation Dashboard</h1>
              {snapshot ? <Pill tone="positive">{snapshot.meta.run_name}</Pill> : null}
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-2">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded-full px-3 py-2 text-sm font-medium transition ${
                    isActive
                      ? "bg-slate-950 text-white shadow-[0_20px_40px_-22px_rgba(15,23,42,0.8)]"
                      : "text-slate-600 hover:bg-white/90 hover:text-slate-950"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 pb-4 text-xs text-slate-500 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-center gap-2">
            <Pill tone="neutral">
              {snapshot ? `${formatNumber(snapshot.meta.models.length)} models` : "Waiting for a finished run"}
            </Pill>
            <Pill tone="neutral">
              {snapshot ? `${formatNumber(snapshot.meta.patterns.length)} patterns` : "No classification snapshot yet"}
            </Pill>
          </div>
          {latestRun ? (
            <span>Latest finished run updated {formatDateTime(latestRun.modifiedAt)}</span>
          ) : (
            <span>No completed run detected</span>
          )}
        </div>
        <div className="h-0.5 overflow-hidden bg-slate-200/60">
          <motion.div
            animate={{ width: navigation.state === "idle" ? "0%" : "100%" }}
            transition={{ duration: navigation.state === "idle" ? 0.2 : 0.6, ease: "easeOut" }}
            className="h-full bg-gradient-to-r from-teal-600 via-cyan-500 to-amber-400"
          />
        </div>
      </header>

      <main className="relative mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={snapshot ? location.pathname : "empty"}
            initial={prefersReducedMotion ? false : { opacity: 0, y: 24 }}
            animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
            exit={prefersReducedMotion ? undefined : { opacity: 0, y: -18 }}
            transition={{ duration: 0.42, ease: "easeOut" }}
            className="contents"
          >
            {snapshot ? (
              children
            ) : (
              <EmptyState
                title="No finished run is available yet"
                description="This dashboard only promotes completed experiment runs that have metrics, champion selections, backtest summaries, and gallery evidence. Once a full run is available, it will be picked up automatically."
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
