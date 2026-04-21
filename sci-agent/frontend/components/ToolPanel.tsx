"use client";

import type { BfdtsTrace, StreamEventRecord } from "@/lib/types";
import { useStickyScroll } from "@/lib/useStickyScroll";
import CopyButton from "./CopyButton";
import { IconX } from "./icons";
import Markdown from "./Markdown";
import PlanTreeView from "./PlanTreeView";

function NodeBadge({ node }: { node?: string }) {
  if (!node || node === "agent" || node === "deep_agent") return null;
  return (
    <span className="inline-block rounded bg-bg-active px-1.5 py-0.5 text-[10px] font-medium text-text-secondary">
      {node}
    </span>
  );
}

function fmtArgs(args: unknown): string {
  if (args == null) return "";
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

function extractCode(args: unknown): string | null {
  if (args && typeof args === "object" && "code" in args) {
    const code = (args as Record<string, unknown>).code;
    if (typeof code === "string") return code;
  }
  return null;
}

function CodeBlock({
  code,
  language,
  maxHeight = 260,
}: {
  code: string;
  language?: string;
  maxHeight?: number;
}) {
  return (
    <div className="overflow-hidden rounded-md bg-black/60">
      <div className="flex items-center justify-between border-b border-bg-hover px-3 py-1 text-[11px] text-text-muted">
        <span>{language ?? "code"}</span>
        <CopyButton text={code} />
      </div>
      <pre
        className="scrollbar-thin overflow-auto px-3 py-2 font-mono text-[12px] leading-5 text-text-primary"
        style={{ maxHeight }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function looksLikeMarkdownTable(s: string): boolean {
  const lines = s.split("\n");
  for (let i = 0; i < lines.length - 1; i++) {
    const a = lines[i];
    const b = lines[i + 1];
    if (a.includes("|") && /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/.test(b)) {
      return true;
    }
  }
  return false;
}

const IMAGE_DATA_URI_RE = /^data:image\/[a-zA-Z0-9+.-]+;base64,[A-Za-z0-9+/=\s]+$/;

function renderResultContent(content: string) {
  const trimmed = content.trim();

  // Inline image (data URI)
  if (IMAGE_DATA_URI_RE.test(trimmed)) {
    // eslint-disable-next-line @next/next/no-img-element
    return (
      <img
        src={trimmed}
        alt="tool result"
        className="max-h-[320px] max-w-full rounded-md border border-bg-hover"
      />
    );
  }

  // Markdown table
  if (looksLikeMarkdownTable(trimmed)) {
    return <Markdown>{trimmed}</Markdown>;
  }

  // JSON → pretty-print as code
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      const parsed = JSON.parse(trimmed);
      const pretty = JSON.stringify(parsed, null, 2);
      return <CodeBlock code={pretty} language="json" maxHeight={200} />;
    } catch {
      // fall through
    }
  }

  const display = content.length > 1500 ? content.slice(0, 1500) + "…" : content;
  return (
    <pre className="scrollbar-thin max-h-[200px] overflow-auto whitespace-pre-wrap break-words rounded-md border border-bg-hover bg-bg-main/40 p-2 font-mono text-[11px] leading-5 text-text-secondary">
      <code>{display}</code>
    </pre>
  );
}

function ToolCall({ e }: { e: StreamEventRecord }) {
  const code = e.name === "run_python" ? extractCode(e.args) : null;
  const argsStr = fmtArgs(e.args);
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-text-primary">🔧 {e.name}</span>
        <NodeBadge node={e.node} />
      </div>
      {code ? (
        <CodeBlock code={code} language="python" />
      ) : argsStr && argsStr !== "{}" ? (
        <CodeBlock code={argsStr} language="json" maxHeight={160} />
      ) : null}
    </div>
  );
}

function ToolResult({ e }: { e: StreamEventRecord }) {
  const content = e.content ?? "";
  return (
    <div className="space-y-1 pl-4">
      <div className="flex items-center gap-2">
        <span className="text-[11px] uppercase tracking-wide text-text-muted">
          ↳ result
        </span>
        {e.name && (
          <span className="text-[11px] text-text-secondary">{e.name}</span>
        )}
      </div>
      {renderResultContent(content)}
    </div>
  );
}

export default function ToolPanel({
  events,
  trace,
  pending,
  onClose,
}: {
  events: StreamEventRecord[];
  trace?: BfdtsTrace;
  pending: boolean;
  onClose?: () => void;
}) {
  const toolEvents = events.filter(
    (e) => e.kind === "tool_call" || e.kind === "tool_result",
  );
  const totalChars = toolEvents.reduce(
    (s, e) => s + (e.content?.length ?? 0) + (e.name?.length ?? 0),
    0,
  );
  const scrollRef = useStickyScroll<HTMLDivElement>(
    `${toolEvents.length}:${totalChars}`,
  );

  return (
    <aside className="flex h-full w-[360px] flex-none flex-col border-l border-bg-hover bg-bg-sidebar">
      <div className="flex h-12 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-text-primary">
            Tool / Code
          </span>
          {pending && (
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-md text-text-muted hover:bg-bg-hover hover:text-text-primary"
            aria-label="Collapse tool panel"
            title="Hide tool panel"
          >
            <IconX width={16} height={16} />
          </button>
        )}
      </div>

      {/* Top half — tool call / result stream */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div
          ref={scrollRef}
          className="scrollbar-thin flex-1 space-y-3 overflow-y-auto px-4 pb-4"
        >
          {toolEvents.length === 0 ? (
            <p className="pt-4 text-xs text-text-muted">
              에이전트가 호출한 도구와 코드 실행 결과가 여기에 표시됩니다.
            </p>
          ) : (
            toolEvents.map((e) =>
              e.kind === "tool_call" ? (
                <ToolCall key={e.id} e={e} />
              ) : (
                <ToolResult key={e.id} e={e} />
              ),
            )
          )}
        </div>
      </div>

      {/* Bottom half — BFDTS planning tree view */}
      <div className="flex min-h-0 flex-1 flex-col border-t border-bg-hover">
        <PlanTreeView trace={trace} events={events} />
      </div>
    </aside>
  );
}
