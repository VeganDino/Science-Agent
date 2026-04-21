// Streaming proxy for /api/chat. We do NOT use next.config.ts rewrites here
// because they buffer SSE — we need to pass the upstream ReadableStream
// through verbatim to preserve token-level streaming.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8787";

export async function POST(req: Request) {
  const body = await req.text();

  const upstream = await fetch(`${BACKEND}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type":
        req.headers.get("content-type") ?? "application/json",
    },
    body,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type":
        upstream.headers.get("content-type") ?? "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
      Connection: "keep-alive",
    },
  });
}
