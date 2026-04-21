"use client";

import { useEffect, useMemo, useState } from "react";

import type { BfdtsCandidate } from "@/lib/types";

/**
 * Vertical DAG: types flow top → bottom, tool calls are labeled edges.
 * Same type referenced by multiple chains collapses to one node.
 * Parallel tools between the same type-pair fan out horizontally.
 */

interface NodePos {
  id: string;
  depth: number;
  x: number; // top-left of node rect
  y: number;
}

interface ToolEdge {
  from: string;
  to: string;
  tool: string;
  chainIndex: number;
  key: string;
}

interface Layout {
  nodes: Map<string, NodePos>;
  edges: ToolEdge[];
  width: number;
  height: number;
}

const CHAIN_COLORS = [
  "#34d399", // emerald
  "#60a5fa", // blue
  "#f472b6", // pink
  "#fbbf24", // amber
  "#a78bfa", // violet
];

// ----- dimensions -----
const NODE_W = 160;
const NODE_H = 44;
const LEVEL_GAP = 110; // vertical gap between depths
const SIBLING_GAP = 190; // horizontal spread when multiple types share a depth
const PAIR_FAN = 110; // horizontal fan for parallel tool edges
const TOP_PAD = 30;
const SIDE_PAD = 20;

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

const MAX_LABEL_W = 160;

function buildLayout(candidates: BfdtsCandidate[]): Layout {
  const depthOf = new Map<string, number>();
  const edges: ToolEdge[] = [];

  for (const c of candidates) {
    if (!depthOf.has(c.from)) depthOf.set(c.from, 0);
    c.steps.forEach((s, i) => {
      const inDepth = depthOf.get(s.input_type);
      if (inDepth === undefined) depthOf.set(s.input_type, i);
      const next = (depthOf.get(s.input_type) ?? 0) + 1;
      const prev = depthOf.get(s.output_type);
      if (prev === undefined || next < prev) depthOf.set(s.output_type, next);
      edges.push({
        from: s.input_type,
        to: s.output_type,
        tool: s.tool,
        chainIndex: c.index,
        key: `${c.index}:${i}:${s.tool}`,
      });
    });
  }

  // Parallel-edge count → bow magnitude. Labels on fanned edges extend
  // beyond node edges, so the SVG width must grow to fit them.
  const pairCounts = new Map<string, number>();
  for (const e of edges) {
    const k = `${e.from}|${e.to}`;
    pairCounts.set(k, (pairCounts.get(k) ?? 0) + 1);
  }
  let maxAbsBow = 0;
  for (const count of pairCounts.values()) {
    if (count > 1) {
      maxAbsBow = Math.max(maxAbsBow, ((count - 1) / 2) * PAIR_FAN);
    }
  }

  // How far a label can extend from a node's center.
  const labelHalfExtent = maxAbsBow + MAX_LABEL_W / 2;

  const cols = new Map<number, string[]>();
  for (const [id, d] of depthOf) {
    const arr = cols.get(d) ?? [];
    arr.push(id);
    cols.set(d, arr);
  }
  for (const arr of cols.values()) arr.sort();

  const depths = [...cols.keys()].sort((a, b) => a - b);
  const maxSiblings = Math.max(...[...cols.values()].map((c) => c.length));
  const siblingExtent =
    maxSiblings > 1 ? ((maxSiblings - 1) / 2) * SIBLING_GAP : 0;

  // Half-width from center must cover either the node OR the furthest label.
  const halfWidth =
    Math.max(NODE_W / 2, labelHalfExtent) + siblingExtent;
  const width = halfWidth * 2 + SIDE_PAD * 2;
  const centerX = width / 2;

  const nodes = new Map<string, NodePos>();
  for (const d of depths) {
    const items = cols.get(d)!;
    items.forEach((id, i) => {
      const offset = (i - (items.length - 1) / 2) * SIBLING_GAP;
      nodes.set(id, {
        id,
        depth: d,
        x: centerX + offset - NODE_W / 2,
        y: TOP_PAD + d * LEVEL_GAP,
      });
    });
  }

  const maxDepth = depths[depths.length - 1] ?? 0;
  const height = TOP_PAD * 2 + maxDepth * LEVEL_GAP + NODE_H;

  return { nodes, edges, width, height };
}

function TypeNode({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g transform={`translate(${x},${y})`}>
      {/* subtle glow */}
      <rect
        width={NODE_W}
        height={NODE_H}
        rx={NODE_H / 2}
        ry={NODE_H / 2}
        fill="#0f172a"
        stroke="#475569"
        strokeWidth={1.25}
      />
      <text
        x={NODE_W / 2}
        y={NODE_H / 2 + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#e2e8f0"
        fontSize={12.5}
        fontWeight={500}
        fontFamily="ui-sans-serif, system-ui"
      >
        {truncate(label, 22)}
      </text>
    </g>
  );
}

export default function BfdtsGraph({
  candidates,
  visibleEdges,
}: {
  candidates: BfdtsCandidate[];
  visibleEdges: number;
}) {
  const layout = useMemo(() => buildLayout(candidates), [candidates]);
  if (candidates.length === 0) return null;

  const { nodes, edges, width, height } = layout;
  const shown = edges.slice(0, visibleEdges);

  // Nodes to show: start types + any endpoints of shown edges.
  const shownNodeIds = new Set<string>();
  for (const c of candidates) shownNodeIds.add(c.from);
  for (const e of shown) {
    shownNodeIds.add(e.from);
    shownNodeIds.add(e.to);
  }

  // Parallel-edge fan-out accounting.
  const pairCounts = new Map<string, number>();
  for (const e of edges) {
    const k = `${e.from}|${e.to}`;
    pairCounts.set(k, (pairCounts.get(k) ?? 0) + 1);
  }
  const pairRunning = new Map<string, number>();

  return (
    <div className="scrollbar-thin flex justify-center overflow-x-auto">
      <svg width={width} height={height} className="block mx-auto">
        <defs>
          {CHAIN_COLORS.map((c, i) => (
            <marker
              key={i}
              id={`arrow-${i}`}
              viewBox="0 0 10 10"
              refX={8}
              refY={5}
              markerWidth={5}
              markerHeight={5}
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill={c} />
            </marker>
          ))}
        </defs>

        {/* Edges (under nodes) */}
        {shown.map((e) => {
          const from = nodes.get(e.from);
          const to = nodes.get(e.to);
          if (!from || !to) return null;
          const pk = `${e.from}|${e.to}`;
          const count = pairCounts.get(pk) ?? 1;
          const idx = pairRunning.get(pk) ?? 0;
          pairRunning.set(pk, idx + 1);
          const bow =
            count > 1 ? (idx - (count - 1) / 2) * PAIR_FAN : 0;

          const srcX = from.x + NODE_W / 2;
          const srcY = from.y + NODE_H;
          const tgtX = to.x + NODE_W / 2;
          const tgtY = to.y;

          const midY = (srcY + tgtY) / 2;
          const d = `M ${srcX} ${srcY} C ${srcX + bow} ${midY}, ${tgtX + bow} ${midY}, ${tgtX} ${tgtY}`;
          const colorIdx = (e.chainIndex - 1) % CHAIN_COLORS.length;
          const color = CHAIN_COLORS[colorIdx];

          // Place label at a t along the curve that differs per parallel edge,
          // so two edges at same (from,to) don't stack labels on top of each
          // other. Also separate horizontally via the bow.
          const t =
            count === 1 ? 0.5 : 0.32 + (idx / (count - 1)) * 0.36;
          const u = 1 - t;
          const cp1X = srcX + bow;
          const cp2X = tgtX + bow;
          const labelX =
            u * u * u * srcX +
            3 * u * u * t * cp1X +
            3 * u * t * t * cp2X +
            t * t * t * tgtX;
          const labelY =
            u * u * u * srcY +
            3 * u * u * t * midY +
            3 * u * t * t * midY +
            t * t * t * tgtY;

          const labelText = truncate(e.tool, 20);
          const labelW = Math.min(labelText.length * 6.5 + 12, 160);

          return (
            <g key={e.key} style={{ animation: "fadeIn 220ms ease-out both" }}>
              <title>{e.tool}</title>
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={2}
                strokeLinecap="round"
                markerEnd={`url(#arrow-${colorIdx})`}
                opacity={0.9}
              />
              <rect
                x={labelX - labelW / 2}
                y={labelY - 9}
                width={labelW}
                height={18}
                rx={9}
                fill="#0b0f1a"
                stroke={color}
                strokeWidth={0.9}
                opacity={0.98}
              />
              <text
                x={labelX}
                y={labelY + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={color}
                fontSize={10}
                fontFamily="ui-monospace, SFMono-Regular, monospace"
              >
                {labelText}
              </text>
            </g>
          );
        })}

        {/* Nodes (on top) */}
        {[...nodes.values()].map((n) =>
          shownNodeIds.has(n.id) ? (
            <TypeNode key={n.id} x={n.x} y={n.y} label={n.id} />
          ) : null,
        )}

        <style>{`@keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }`}</style>
      </svg>
    </div>
  );
}

/** Staggered reveal — pass `resetKey` (e.g. receivedAt) to re-animate. */
export function useStaggeredReveal(
  total: number,
  delayMs = 180,
  resetKey?: unknown,
): number {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
  }, [total, resetKey]);
  useEffect(() => {
    if (n >= total) return;
    const id = setTimeout(() => setN((x) => x + 1), delayMs);
    return () => clearTimeout(id);
  }, [n, total, delayMs]);
  return n;
}
