import type { ReactNode } from "react";

import { motion, useReducedMotion } from "motion/react";

type Tone = "positive" | "negative" | "neutral" | "caution";

export function PageHeader({
  eyebrow,
  title,
  description,
  aside,
}: {
  eyebrow: string;
  title: string;
  description: string;
  aside?: ReactNode;
}) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <motion.section
      initial={prefersReducedMotion ? false : { opacity: 0, y: 22, scale: 0.985 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.55, ease: "easeOut" }}
      className="hero-panel grid gap-6 overflow-hidden lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,0.7fr)]"
    >
      <div className="space-y-4">
        <p className="eyebrow">{eyebrow}</p>
        <div className="space-y-3">
          <h1 className="display-title max-w-3xl text-4xl sm:text-5xl">{title}</h1>
          <p className="max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">{description}</p>
        </div>
      </div>
      {aside ? <div className="hero-aside">{aside}</div> : null}
    </motion.section>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "neutral",
  className,
  valueClassName,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: Tone;
  className?: string;
  valueClassName?: string;
}) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <motion.article
      initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      whileHover={prefersReducedMotion ? undefined : { y: -6, scale: 1.01 }}
      transition={{ duration: 0.42, ease: "easeOut" }}
      className={`metric-card ${className ?? ""}`}
      data-tone={tone}
    >
      <p className="metric-label">{label}</p>
      <p className={`metric-value ${valueClassName ?? ""}`}>{value}</p>
      {hint ? <p className="metric-hint">{hint}</p> : null}
    </motion.article>
  );
}

export function SectionCard({
  title,
  kicker,
  actions,
  children,
}: {
  title: string;
  kicker?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <motion.section
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      whileHover={prefersReducedMotion ? undefined : { y: -4 }}
      transition={{ duration: 0.48, ease: "easeOut" }}
      className="glass-panel space-y-5 p-5 sm:p-6"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div className="space-y-1">
          {kicker ? <p className="eyebrow">{kicker}</p> : null}
          <h2 className="section-title">{title}</h2>
        </div>
        {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
      </div>
      {children}
    </motion.section>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: Tone;
}) {
  return (
    <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold" data-pill={tone}>
      {children}
    </span>
  );
}

export function ToggleGroup<T extends string>({
  options,
  value,
  onChange,
}: {
  options: Array<{ label: string; value: T }>;
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="inline-flex flex-wrap gap-2 rounded-full border border-white/50 bg-white/70 p-1 shadow-[0_20px_50px_-32px_rgba(15,23,42,0.55)]">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`rounded-full px-3 py-2 text-xs font-semibold transition sm:text-sm ${
            option.value === value
              ? "bg-slate-950 text-white shadow-[0_18px_32px_-20px_rgba(15,23,42,0.8)]"
              : "text-slate-600 hover:bg-slate-950/5 hover:text-slate-950"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <section className="glass-panel flex min-h-80 items-center justify-center p-8 text-center sm:p-12">
      <div className="max-w-xl space-y-4">
        <p className="eyebrow">Awaiting Results</p>
        <h2 className="display-title text-3xl">{title}</h2>
        <p className="text-sm leading-7 text-slate-600 sm:text-base">{description}</p>
      </div>
    </section>
  );
}
