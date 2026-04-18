import type { ReactNode } from "react";

import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useLoaderData,
  useRouteError,
} from "react-router";

import type { Route } from "./+types/root";
import appCssHref from "./app.css?url";
import { DashboardShell } from "./components/dashboard-shell";
import type { DashboardSnapshot, LatestFinishedRun } from "./lib/dashboard";
import { loadLatestDashboardSnapshot } from "./lib/results.server";

export async function loader({}: Route.LoaderArgs) {
  return await loadLatestDashboardSnapshot();
}

export function links() {
  return [{ rel: "stylesheet", href: appCssHref }];
}

export function meta({ data }: Route.MetaArgs) {
  const runName = data?.snapshot?.meta.run_name;
  return [
    { title: runName ? `${runName} · Candlestick Results` : "Candlestick Results Dashboard" },
    {
      name: "description",
      content:
        "Responsive experiment dashboard for candlestick model screening, profitability curves, and pattern evidence.",
    },
  ];
}

function Document({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  const { snapshot, latestRun } = useLoaderData<typeof loader>();

  return (
    <Document>
      <DashboardShell snapshot={snapshot} latestRun={latestRun}>
        <Outlet context={{ snapshot, latestRun }} />
      </DashboardShell>
    </Document>
  );
}

export function ErrorBoundary() {
  const error = useRouteError();
  const title = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : "Dashboard unavailable";
  const description = isRouteErrorResponse(error)
    ? error.data ?? "The requested dashboard page could not be rendered."
    : error instanceof Error
      ? error.message
      : "An unexpected error interrupted the dashboard.";

  return (
    <Document>
      <main className="mx-auto flex min-h-screen max-w-3xl items-center px-6 py-24">
        <section className="glass-panel w-full p-10 sm:p-14">
          <p className="eyebrow">Candlestick Results</p>
          <h1 className="display-title mt-4 text-4xl">{title}</h1>
          <p className="mt-4 max-w-2xl text-base text-slate-600">{description}</p>
        </section>
      </main>
    </Document>
  );
}

export type DashboardOutletContext = {
  snapshot: DashboardSnapshot | null;
  latestRun: LatestFinishedRun | null;
};
