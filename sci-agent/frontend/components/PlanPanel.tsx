"use client";

import type { StreamEventRecord } from "@/lib/types";
import { useStickyScroll } from "@/lib/useStickyScroll";
import { IconX } from "./icons";
import Markdown from "./Markdown";

function NodeBadge({ node }: { node?: string }) {
  if (!node || node === "agent" || node === "deep_agent") return null;
  return (
    <span className="inline-block rounded bg-bg-active px-1.5 py-0.5 text-[10px] font-medium text-text-secondary">
      {node}
    </span>
  );
}

function Step({ e }: { e: StreamEventRecord }) {
  if (e.kind === "subagent") {
    return (
      <div className="flex items-center gap-2 rounded-md bg-bg-hover/50 px-3 py-2">
        <span className="h-2 w-2 flex-none rounded-full bg-emerald-500" />
        <span className="text-xs text-text-secondary">Subagent activated</span>
        <NodeBadge node={e.node} />
      </div>
    );
  }
  if (e.kind === "reasoning" && e.text) {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wide text-text-muted">
            plan
          </span>
          <NodeBadge node={e.node} />
        </div>
        <div className="rounded-md border border-bg-hover bg-bg-main/40 px-3 py-2 text-[13px] text-text-secondary">
          <Markdown>{e.text}</Markdown>
        </div>
      </div>
    );
  }
  return null;
}

export default function PlanPanel({
  events,
  pending,
  onClose,
}: {
  events: StreamEventRecord[];
  pending: boolean;
  onClose?: () => void;
}) {
  const planEvents = events.filter(
    (e) => e.kind === "subagent" || e.kind === "reasoning",
  );
  // Sticky auto-scroll: retrigger on *any* text growth, not just event count.
  const totalChars = planEvents.reduce(
    (s, e) => s + (e.text?.length ?? 0),
    0,
  );
  const scrollRef = useStickyScroll<HTMLDivElement>(
    `${planEvents.length}:${totalChars}`,
  );

  return (
    <aside className="flex h-full w-[340px] flex-none flex-col border-l border-bg-hover bg-bg-sidebar">
      <div className="flex h-12 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-text-primary">Planning</span>
          {pending && (
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-md text-text-muted hover:bg-bg-hover hover:text-text-primary"
            aria-label="Collapse planning panel"
            title="Hide planning panel"
          >
            <IconX width={16} height={16} />
          </button>
        )}
      </div>

      <div
        ref={scrollRef}
        className="scrollbar-thin flex-1 space-y-3 overflow-y-auto px-4 pb-4"
      >
        {planEvents.length === 0 ? (
          <p className="pt-4 text-xs text-text-muted">
            쿼리를 보내면 에이전트의 계획과 추론 단계가 여기에 표시됩니다.
          </p>
        ) : (
          planEvents.map((e) => <Step key={e.id} e={e} />)
        )}
      </div>
    </aside>
  );
}
