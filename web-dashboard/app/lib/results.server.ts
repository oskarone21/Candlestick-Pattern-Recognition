import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import type { DashboardSnapshot, LatestFinishedRun } from "./dashboard";

const execFileAsync = promisify(execFile);

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(currentDir, "../../..");
const outputsRoot = path.join(repoRoot, "outputs");
const metricsRoot = path.join(outputsRoot, "metrics");
const backtestRoot = path.join(outputsRoot, "backtest");
const galleryRoot = path.join(outputsRoot, "gallery");
const generatedRoot = path.join(repoRoot, "web-dashboard", ".generated");
const builderScript = path.join(repoRoot, "scripts", "build_results_dashboard.py");
const defaultConfigPath = path.join(repoRoot, "configs", "config.yaml");
const dashboardConfigModule = "chart_patterns.config";
const builderDependencies = [
  builderScript,
  defaultConfigPath,
  path.join(repoRoot, "chart_patterns", "trading", "backtest.py"),
];

type RequiredSources = {
  metricsSummary: string;
  champions: string;
  backtestSummary: string;
  gallerySummary: string;
};

type DashboardRunCandidate = {
  runName: string;
  modifiedAt: string;
  sourcePaths: RequiredSources;
};

type DashboardRunConfig = {
  primaryRunName: string | null;
  artifactRunName: string | null;
  comparisonRunNames: string[];
};

function toPosixRelative(absolutePath: string) {
  return path.relative(repoRoot, absolutePath).split(path.sep).join("/");
}

function sourcePathsForRun(runName: string): RequiredSources {
  return {
    metricsSummary: path.join(metricsRoot, runName, "model_comparison_summary.csv"),
    champions: path.join(metricsRoot, runName, "champions.csv"),
    backtestSummary: path.join(backtestRoot, runName, "backtest_summary.csv"),
    gallerySummary: path.join(galleryRoot, runName, "gallery_summary.json"),
  };
}

async function pathExists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function latestSourceMtimeMs(paths: string[]) {
  const stats = await Promise.all(paths.map((filePath) => fs.stat(filePath)));
  return Math.max(...stats.map((stat) => stat.mtimeMs));
}

async function loadDashboardRunConfig(configPath = defaultConfigPath) {
  const { stdout } = await execFileAsync(
    "python",
    [
      "-c",
      [
        `from ${dashboardConfigModule} import load_config`,
        "import json",
        "import sys",
        "cfg = load_config(sys.argv[1])",
        "dashboard = cfg.get('dashboard', {})",
        "print(json.dumps({'primaryRunName': dashboard.get('primary_run_name'), 'artifactRunName': dashboard.get('artifact_run_name'), 'comparisonRunNames': dashboard.get('comparison_run_names', []) or []}))",
      ].join("; "),
      configPath,
    ],
    { cwd: repoRoot },
  );

  return JSON.parse(stdout.trim()) as DashboardRunConfig;
}

async function listFinishedRuns(): Promise<LatestFinishedRun[]> {
  if (!(await pathExists(metricsRoot))) {
    return [];
  }

  const entries = await fs.readdir(metricsRoot, { withFileTypes: true });
  const candidates: LatestFinishedRun[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const runName = entry.name;
    const sources = sourcePathsForRun(runName);
    const requiredFiles = Object.values(sources);
    const allPresent = await Promise.all(requiredFiles.map(pathExists));
    if (allPresent.some((present) => !present)) {
      continue;
    }

    const modifiedAt = new Date(await latestSourceMtimeMs(requiredFiles)).toISOString();
    candidates.push({
      runName,
      modifiedAt,
      sourcePaths: {
        metricsSummary: toPosixRelative(sources.metricsSummary),
        champions: toPosixRelative(sources.champions),
        backtestSummary: toPosixRelative(sources.backtestSummary),
        gallerySummary: toPosixRelative(sources.gallerySummary),
      },
    });
  }

  candidates.sort((left, right) => Date.parse(right.modifiedAt) - Date.parse(left.modifiedAt));
  return candidates;
}

async function listMetricRuns(): Promise<DashboardRunCandidate[]> {
  if (!(await pathExists(metricsRoot))) {
    return [];
  }

  const entries = await fs.readdir(metricsRoot, { withFileTypes: true });
  const candidates: DashboardRunCandidate[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const runName = entry.name;
    const sources = sourcePathsForRun(runName);
    const requiredFiles = [sources.metricsSummary, sources.champions];
    const allPresent = await Promise.all(requiredFiles.map(pathExists));
    if (allPresent.some((present) => !present)) {
      continue;
    }

    const optionalFiles = (
      await Promise.all([sources.backtestSummary, sources.gallerySummary].map(async (filePath) => (await pathExists(filePath) ? filePath : null)))
    ).filter((filePath): filePath is string => Boolean(filePath));

    const modifiedAt = new Date(await latestSourceMtimeMs([...requiredFiles, ...optionalFiles])).toISOString();
    candidates.push({
      runName,
      modifiedAt,
      sourcePaths: sources,
    });
  }

  candidates.sort((left, right) => Date.parse(right.modifiedAt) - Date.parse(left.modifiedAt));
  return candidates;
}

export async function findLatestFinishedRun(): Promise<LatestFinishedRun | null> {
  const runs = await listFinishedRuns();
  return runs[0] ?? null;
}

export async function ensureDashboardSnapshot(
  runName: string,
  configPath = defaultConfigPath,
  artifactRunName: string | null = null,
  comparisonRunNames: string[] = [],
) {
  const snapshotDir = path.join(generatedRoot, runName);
  const snapshotPath = path.join(snapshotDir, "data.json");
  const candidatePaths = [
    ...Object.values(sourcePathsForRun(runName)),
    ...(artifactRunName ? Object.values(sourcePathsForRun(artifactRunName)) : []),
    ...comparisonRunNames.flatMap((name) => Object.values(sourcePathsForRun(name))),
    ...builderDependencies,
    configPath,
  ];
  const sourcePathPresence = await Promise.all(candidatePaths.map(async (filePath) => ({ filePath, exists: await pathExists(filePath) })));
  const sourcePaths = sourcePathPresence.filter((item) => item.exists).map((item) => item.filePath);
  const snapshotExists = await pathExists(snapshotPath);
  const latestSourceMtime = await latestSourceMtimeMs(sourcePaths);

  let shouldBuild = !snapshotExists;
  if (!shouldBuild) {
    const snapshotStat = await fs.stat(snapshotPath);
    shouldBuild = snapshotStat.mtimeMs < latestSourceMtime;
  }

  if (shouldBuild) {
    await fs.mkdir(generatedRoot, { recursive: true });
    await execFileAsync(
      "python",
      [
        builderScript,
        "--config",
        configPath,
        "--run-name",
        runName,
        "--output-root",
        generatedRoot,
        "--data-only",
      ],
      { cwd: repoRoot },
    );
  }

  return snapshotPath;
}

export async function loadLatestDashboardSnapshot() {
  const finishedRuns = await listFinishedRuns();
  const metricRuns = await listMetricRuns();
  const latestRun = finishedRuns[0] ?? null;
  if (metricRuns.length === 0) {
    return { latestRun: null, snapshot: null as DashboardSnapshot | null };
  }

  const dashboardConfig = await loadDashboardRunConfig();
  const { primaryRunName, artifactRunName, comparisonRunNames } = dashboardConfig;
  if (primaryRunName) {
    const primaryRun = metricRuns.find((run) => run.runName === primaryRunName);
    if (primaryRun) {
      const snapshotPath = await ensureDashboardSnapshot(
        primaryRun.runName,
        defaultConfigPath,
        artifactRunName,
        comparisonRunNames,
      );
      const snapshot = JSON.parse(await fs.readFile(snapshotPath, "utf8")) as DashboardSnapshot;
      return { latestRun, snapshot };
    }
  }

  const candidates: Array<{ run: LatestFinishedRun; snapshot: DashboardSnapshot }> = [];
  for (const run of metricRuns) {
    const snapshotPath = await ensureDashboardSnapshot(
      run.runName,
      defaultConfigPath,
      artifactRunName,
      comparisonRunNames,
    );
    const snapshot = JSON.parse(await fs.readFile(snapshotPath, "utf8")) as DashboardSnapshot;
    candidates.push({ run, snapshot });
  }

  const eligible = candidates
    .filter(({ snapshot }) => snapshot.presentation?.presentation_eligible)
    .sort((left, right) => {
      const leftVisible = left.snapshot.presentation?.visible_patterns?.length ?? 0;
      const rightVisible = right.snapshot.presentation?.visible_patterns?.length ?? 0;
      if (rightVisible !== leftVisible) {
        return rightVisible - leftVisible;
      }

      const leftF1 = left.snapshot.presentation?.mean_visible_champion_f1 ?? 0;
      const rightF1 = right.snapshot.presentation?.mean_visible_champion_f1 ?? 0;
      if (rightF1 !== leftF1) {
        return rightF1 - leftF1;
      }

      const leftPrAuc = left.snapshot.presentation?.mean_visible_champion_pr_auc ?? 0;
      const rightPrAuc = right.snapshot.presentation?.mean_visible_champion_pr_auc ?? 0;
      if (rightPrAuc !== leftPrAuc) {
        return rightPrAuc - leftPrAuc;
      }

      return Date.parse(right.run.modifiedAt) - Date.parse(left.run.modifiedAt);
    });

  const selected = eligible[0] ?? candidates[0];
  return { latestRun, snapshot: selected.snapshot };
}

export async function readGalleryAsset(src: string) {
  const trimmed = src.trim().replace(/^\/+/, "");
  if (!trimmed) {
    return null;
  }

  const resolved = path.resolve(repoRoot, trimmed);
  const allowedRoots = [path.join(outputsRoot, "gallery"), generatedRoot];
  const isAllowed = allowedRoots.some(
    (allowedRoot) => resolved === allowedRoot || resolved.startsWith(`${allowedRoot}${path.sep}`),
  );

  if (!isAllowed || !(await pathExists(resolved))) {
    return null;
  }

  const body = await fs.readFile(resolved);
  const extension = path.extname(resolved).toLowerCase();
  const contentType =
    extension === ".png"
      ? "image/png"
      : extension === ".jpg" || extension === ".jpeg"
        ? "image/jpeg"
        : extension === ".webp"
          ? "image/webp"
          : "application/octet-stream";

  return {
    body,
    contentType,
  };
}
