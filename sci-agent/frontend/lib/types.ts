export type Role = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

export type StreamEventKind =
  | "subagent"
  | "reasoning"
  | "tool_call"
  | "tool_result";

export interface StreamEventRecord {
  id: string;
  kind: StreamEventKind;
  node?: string;
  name?: string;
  text?: string;
  args?: unknown;
  content?: string;
  ts: number;
}

export interface BfdtsTreeNode {
  tool: string;
  depth: number;
  input_types: string[];
  output_types: string[];
  is_solution: boolean;
  children: BfdtsTreeNode[];
}

export interface BfdtsStep {
  tool: string;
  input_type: string;
  output_type: string;
}

export interface BfdtsCandidate {
  index: number;
  from: string;
  to: string;
  step_count: number;
  tools: string[];
  steps: BfdtsStep[];
}

export interface BfdtsTrace {
  goal: string;
  start?: string;
  end?: string;
  tree: BfdtsTreeNode;
  candidates: BfdtsCandidate[];
  /** Set by the frontend on receive — key for forcing re-animation. */
  receivedAt?: number;
}

export interface Chat {
  id: string;
  title: string;
  messages: ChatMessage[];
  events: StreamEventRecord[];
  trace?: BfdtsTrace;
  createdAt: number;
}
