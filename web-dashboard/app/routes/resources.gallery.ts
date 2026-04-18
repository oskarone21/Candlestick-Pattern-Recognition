import type { LoaderFunctionArgs } from "react-router";

import { readGalleryAsset } from "../lib/results.server";

export async function loader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const src = url.searchParams.get("src");

  if (!src) {
    throw new Response("Missing gallery asset path.", { status: 400 });
  }

  const asset = await readGalleryAsset(src);
  if (!asset) {
    throw new Response("Gallery asset not found.", { status: 404 });
  }

  return new Response(asset.body, {
    headers: {
      "Content-Type": asset.contentType,
      "Cache-Control": "public, max-age=3600",
    },
  });
}
