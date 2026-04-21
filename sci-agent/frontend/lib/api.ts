import type { BfdtsTrace } from "./types";

export type StreamEvent =
  | { type: "thread"; thread_id: string }
  | { type: "subagent"; name: string }
  | { type: "token"; node: string; text: string }
  | { type: "reasoning_token"; node: string; text: string }
  | { type: "message_end"; node: string; final: boolean; text: string }
  | { type: "tool_call"; node: string; name: string; args: unknown }
  | { type: "tool_result"; node: string; name: string; content: string }
  | ({ type: "bfdts_trace" } & BfdtsTrace)
  | { type: "error"; message: string; traceback?: string }
  | { type: "done" };

// Same-origin: browser hits /api/* which Next.js proxies to the backend.
// See next.config.ts rewrites.
const API_BASE = "";

export interface UploadResult {
  filename: string;
  original_filename: string;
  path: string;
  size: number;
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function streamChat(
  query: string,
  threadId: string | null,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, thread_id: threadId }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const frame = parseSSE(raw);
      if (frame) onEvent(frame);
    }
  }
}

function parseSSE(raw: string): StreamEvent | null {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    const parsed = JSON.parse(data);
    return { type: event, ...parsed } as StreamEvent;
  } catch {
    return null;
  }
}
