import { index, route, type RouteConfig } from "@react-router/dev/routes";

export default [
  index("routes/_index.tsx"),
  route("models", "routes/models.tsx"),
  route("profitability", "routes/profitability.tsx"),
  route("patterns/:pattern", "routes/patterns.$pattern.tsx"),
  route("resources/gallery", "routes/resources.gallery.ts"),
] satisfies RouteConfig;
