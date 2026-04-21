"use client";

import type { ChatMessage } from "@/lib/types";
import Markdown from "./Markdown";

export default function ChatMessages({
  messages,
  pending,
}: {
  messages: ChatMessage[];
  pending: boolean;
}) {
  return (
    <div className="mx-auto w-full max-w-[760px] space-y-6 px-4 py-6">
      {messages.map((m, i) => {
        const isLastAssistant =
          m.role === "assistant" && i === messages.length - 1;
        const showPendingDot = isLastAssistant && pending && !m.content;
        return (
          <div key={m.id} className="flex gap-3">
            <div
              className={`flex h-7 w-7 flex-none items-center justify-center rounded-full text-[11px] font-semibold ${
                m.role === "user"
                  ? "bg-cyan-600 text-white"
                  : "bg-bg-active text-text-primary"
              }`}
            >
              {m.role === "user" ? "SU" : "AI"}
            </div>
            <div className="min-w-0 flex-1 pt-0.5 text-[15px] leading-7 text-text-primary">
              {m.role === "user" ? (
                <div className="whitespace-pre-wrap break-words">{m.content}</div>
              ) : showPendingDot ? (
                <div className="flex items-center gap-1 py-1 text-text-muted">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted" />
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted [animation-delay:150ms]" />
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-text-muted [animation-delay:300ms]" />
                </div>
              ) : (
                <Markdown>{m.content}</Markdown>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
