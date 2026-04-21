"use client";

import { useMemo } from "react";

import type { BfdtsCandidate, BfdtsTrace, StreamEventRecord } from "@/lib/types";
import BfdtsGraph, { useStaggeredReveal } from "./BfdtsGraph";

const CHAIN_COLORS = [
  "text-emerald-300 bg-emerald-500/15",
  "text-blue-300 bg-blue-500/15",
  "text-pink-300 bg-pink-500/15",
  "text-amber-300 bg-amber-500/15",
  "text-violet-300 bg-violet-500/15",
];

function stepCountOf(c: BfdtsCandidate): number {
  if (typeof c.step_count === "number") return c.step_count;
  const raw = (c as unknown as { steps?: unknown }).steps;
  if (typeof raw === "number") return raw;
  if (Array.isArray(raw)) return raw.length;
  return 0;
}

function ChainCard({ c }: { c: BfdtsCandidate }) {
  const color = CHAIN_COLORS[(c.index - 1) % CHAIN_COLORS.length];
  const count = stepCountOf(c);
  return (
    <div className="rounded-md border border-bg-hover bg-bg-main/40 p-2 text-[12px]">
      <div className="mb-1 flex items-center gap-2">
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${color}`}>
          #{c.index}
        </span>
        <span className="text-text-muted">
          <code className="text-text-secondary">{c.from}</code>
          <span className="mx-1">→</span>
          <code className="text-text-secondary">{c.to}</code>
        </span>
        <span className="ml-auto text-[10px] text-text-muted">
          {count} step{count !== 1 ? "s" : ""}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-1">
        {c.tools.map((t, i) => (
          <span key={`${t}-${i}`} className="flex items-center gap-1">
            <span className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${color}`}>
              {t}
            </span>
            {i < c.tools.length - 1 && <span className="text-text-muted">→</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function PlanTreeView({
  trace,
  events,
}: {
  trace?: BfdtsTrace;
  events?: StreamEventRecord[];
}) {
  const toolCalls = (events ?? []).filter((e) => e.kind === "tool_call");
  const plannerCalled = toolCalls.some((e) => e.name === "make_science_plan");
  const agentSkippedPlanner = !trace && toolCalls.length > 0 && !plannerCalled;

  // Defensive: backend prior to the steps[] schema change returns
  // `steps` as a number and no per-hop type info. Use that fallback so the
  // UI still renders chains (without the graph) instead of crashing.
  const hasRichSteps = !!(
    trace &&
    trace.candidates.length > 0 &&
    Array.isArray((trace.candidates[0] as unknown as { steps?: unknown }).steps)
  );

  const totalEdges = useMemo(
    () =>
      trace && hasRichSteps
        ? trace.candidates.reduce((s, c) => s + c.steps.length, 0)
        : 0,
    [trace, hasRichSteps],
  );

  const visibleEdges = useStaggeredReveal(totalEdges, 220, trace?.receivedAt);

  if (!trace) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex h-9 items-center px-4 text-xs font-semibold text-text-primary">
          BFDTS Graph
        </div>
        <div className="scrollbar-thin flex-1 overflow-y-auto px-4 pb-3 text-xs text-text-muted">
          {agentSkippedPlanner ? (
            <div className="pt-2 space-y-2">
              <p className="rounded-md border border-yellow-500/30 bg-yellow-500/10 p-2 text-yellow-300">
                ⚠️ 에이전트가 `make_science_plan`을 호출하지 않고 바로 다른 도구로 넘어갔습니다.
              </p>
              <p>
                AGENTS.md의 지시(“FIRST tool call MUST be make_science_plan”)를 Nemotron이 무시한 경우입니다. 그래프는 계획 단계에서만 생성되므로 현재 쿼리엔 표시할 것이 없습니다.
              </p>
            </div>
          ) : (
            <p className="pt-2">
              `make_science_plan`이 호출되면 BFDTS가 탐색한 후보 체인이 그래프로 표시됩니다.
            </p>
          )}
        </div>
      </div>
    );
  }

  const done = visibleEdges >= totalEdges;
  const running = totalEdges > 0 && !done;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-9 items-center gap-2 px-4">
        <span className="text-xs font-semibold text-text-primary">BFDTS Graph</span>
        {running && (
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
        )}
        <span className="ml-auto text-[10px] text-text-muted">
          {trace.candidates.length} chain
          {trace.candidates.length > 1 ? "s" : ""} · {visibleEdges}/{totalEdges} hops
        </span>
      </div>

      <div className="scrollbar-thin flex-1 space-y-3 overflow-y-auto px-4 pb-3">
        {trace.candidates.length > 0 && (
          <div className="space-y-2">
            {trace.candidates.map((c) => (
              <ChainCard key={c.index} c={c} />
            ))}
          </div>
        )}

        {totalEdges > 0 ? (
          <div>
            <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-text-muted">
              <span>
                Graph · {trace.start} → {trace.end}
              </span>
              {done && <span className="text-emerald-400">complete</span>}
            </div>
            <div className="rounded-md bg-black/40 p-2">
              <BfdtsGraph
                candidates={trace.candidates}
                visibleEdges={visibleEdges}
              />
            </div>
          </div>
        ) : !hasRichSteps && trace.candidates.length > 0 ? (
          <div className="pt-2 text-xs text-yellow-300">
            ⚠️ 백엔드가 구버전 스키마로 응답했습니다 (steps 배열 없음). 그래프 시각화를 위해 백엔드를 재시작하세요.
          </div>
        ) : (
          <div className="pt-2 text-xs text-text-muted">
            이번 쿼리에선 BFDTS 체인 후보를 찾지 못했습니다.
          </div>
        )}
      </div>
    </div>
  );
}
