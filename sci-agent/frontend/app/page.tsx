"use client";

import { useCallback, useMemo, useState } from "react";

import MainArea from "@/components/MainArea";
import PlanPanel from "@/components/PlanPanel";
import Sidebar from "@/components/Sidebar";
import ToolPanel from "@/components/ToolPanel";
import { streamChat, uploadFile } from "@/lib/api";
import { getFileVisual } from "@/lib/fileIcon";
import { useLocalState } from "@/lib/persistence";
import type {
  Chat,
  ChatMessage,
  StreamEventKind,
  StreamEventRecord,
} from "@/lib/types";

const newId = () =>
  (globalThis.crypto?.randomUUID?.() ?? String(Math.random())).slice(0, 8);

export default function Page() {
  const [chats, setChats] = useLocalState<Chat[]>("sci-agent:chats:v1", []);
  const [threadIds, setThreadIds] = useLocalState<Record<string, string>>(
    "sci-agent:threads:v1",
    {},
  );
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showPlan, setShowPlan] = useState(true);
  const [showTool, setShowTool] = useState(true);

  const activeChat = useMemo(
    () => chats.find((c) => c.id === activeChatId) ?? null,
    [chats, activeChatId],
  );

  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
  }, []);

  const handleSelectChat = useCallback((id: string) => {
    setActiveChatId(id);
  }, []);

  const handleDeleteChat = useCallback(
    (id: string) => {
      setChats((prev) => prev.filter((c) => c.id !== id));
      setThreadIds((prev) => {
        const { [id]: _removed, ...rest } = prev;
        return rest;
      });
      setActiveChatId((curr) => (curr === id ? null : curr));
    },
    [setChats, setThreadIds],
  );

  const ensureChat = useCallback(
    (firstUserText: string): string => {
      if (activeChatId) return activeChatId;
      const id = newId();
      const title = firstUserText.slice(0, 32);
      const chat: Chat = {
        id,
        title,
        messages: [],
        events: [],
        createdAt: Date.now(),
      };
      setChats((prev) => [chat, ...prev]);
      setActiveChatId(id);
      return id;
    },
    [activeChatId],
  );

  const appendMessage = useCallback((chatId: string, msg: ChatMessage) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id === chatId ? { ...c, messages: [...c.messages, msg] } : c,
      ),
    );
  }, []);

  const setAssistantMessage = useCallback(
    (chatId: string, msgId: string, content: string) => {
      setChats((prev) =>
        prev.map((c) =>
          c.id !== chatId
            ? c
            : {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === msgId ? { ...m, content } : m,
                ),
              },
        ),
      );
    },
    [],
  );

  const appendEvent = useCallback(
    (
      chatId: string,
      kind: StreamEventKind,
      patch: Partial<StreamEventRecord>,
    ) => {
      const record: StreamEventRecord = {
        id: newId(),
        kind,
        ts: Date.now(),
        ...patch,
      };
      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId ? { ...c, events: [...c.events, record] } : c,
        ),
      );
    },
    [],
  );

  const handleSend = useCallback(
    async (text: string, files: File[] = []) => {
      const chatId = ensureChat(text || (files[0]?.name ?? "attachment"));

      // Clear per-query panels (Planning events + Tool events + BFDTS graph)
      // so each new user query starts from scratch. The main chat transcript
      // (messages) is preserved — only the right-side observability panels reset.
      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId ? { ...c, events: [], trace: undefined } : c,
        ),
      );

      setPending(true);

      // Upload any attached files first, then embed their paths in the query.
      let queryText = text;
      let userBubbleText = text;
      if (files.length > 0) {
        try {
          const uploads = await Promise.all(files.map((f) => uploadFile(f)));
          const lines = uploads
            .map((u) => `- ${u.path}   (original name: ${u.original_filename})`)
            .join("\n");
          const suffix =
            "\n\n---\n" +
            "The user has uploaded file(s) to the REAL server filesystem at the following absolute paths:\n\n" +
            lines +
            "\n\nThese are actual files on disk, NOT entries in your virtual filesystem. " +
            "DO NOT call `ls`, `glob`, or `read_file` on them — those tools only see the virtual FS and will return empty.\n" +
            "To process them, pass the absolute path directly to a tool that accepts a path, such as:\n" +
            "  • `ocr_image(image_path=\"/abs/path\")` for images and PDFs (returns extracted text)\n" +
            "  • `run_python(code=\"...\")` with `open('/abs/path')` or `pandas.read_csv('/abs/path')` for CSV/TSV/XLSX/text files\n";
          queryText = (text || "Please analyze the attached file(s).") + suffix;
          userBubbleText = (text ? text + "\n\n" : "") +
            files
              .map((f) => `${getFileVisual(f.name).emoji} ${f.name}`)
              .join("\n");
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          appendMessage(chatId, {
            id: newId(),
            role: "user",
            content: text || "(attachment)",
          });
          appendMessage(chatId, {
            id: newId(),
            role: "assistant",
            content: `⚠️ ${msg}`,
          });
          setPending(false);
          return;
        }
      }

      appendMessage(chatId, {
        id: newId(),
        role: "user",
        content: userBubbleText,
      });

      const assistantId = newId();
      appendMessage(chatId, { id: assistantId, role: "assistant", content: "" });
      // Per-LLM-call buffers — reset on every message_end.
      let contentBuffer = "";
      let reasoningId: string | null = null;

      const startOrAppendReasoning = (node: string, delta: string) => {
        if (!reasoningId) {
          const id = newId();
          reasoningId = id;
          const record: StreamEventRecord = {
            id,
            kind: "reasoning",
            node,
            text: delta,
            ts: Date.now(),
          };
          setChats((prev) =>
            prev.map((c) =>
              c.id === chatId ? { ...c, events: [...c.events, record] } : c,
            ),
          );
        } else {
          const id = reasoningId;
          setChats((prev) =>
            prev.map((c) =>
              c.id !== chatId
                ? c
                : {
                    ...c,
                    events: c.events.map((ev) =>
                      ev.id === id ? { ...ev, text: (ev.text ?? "") + delta } : ev,
                    ),
                  },
            ),
          );
        }
      };

      try {
        await streamChat(queryText, threadIds[chatId] ?? null, (e) => {
          switch (e.type) {
            case "thread":
              setThreadIds((prev) =>
                prev[chatId] ? prev : { ...prev, [chatId]: e.thread_id },
              );
              break;
            case "subagent":
              appendEvent(chatId, "subagent", { node: e.name });
              break;
            case "token":
              contentBuffer += e.text;
              setAssistantMessage(chatId, assistantId, contentBuffer);
              break;
            case "reasoning_token":
              startOrAppendReasoning(e.node, e.text);
              break;
            case "message_end": {
              // Reset reasoning target so the NEXT LLM call creates a fresh block.
              reasoningId = null;

              const finalText = contentBuffer || e.text || "";
              if (e.final) {
                // Final answer — lock into main chat.
                setAssistantMessage(chatId, assistantId, finalText);
              } else {
                // Non-final: whatever content was streamed was more planning text.
                // Save as an extra reasoning entry if non-trivial, then clear
                // main bubble so the next answer stream starts clean.
                if (finalText.trim()) {
                  appendEvent(chatId, "reasoning", {
                    node: e.node,
                    text: finalText,
                  });
                }
                setAssistantMessage(chatId, assistantId, "");
              }
              contentBuffer = "";
              break;
            }
            case "tool_call":
              appendEvent(chatId, "tool_call", {
                node: e.node,
                name: e.name,
                args: e.args,
              });
              break;
            case "tool_result":
              appendEvent(chatId, "tool_result", {
                node: e.node,
                name: e.name,
                content: e.content,
              });
              break;
            case "bfdts_trace": {
              const { type: _t, ...traceData } = e;
              const trace = { ...traceData, receivedAt: Date.now() };
              setChats((prev) =>
                prev.map((c) => (c.id === chatId ? { ...c, trace } : c)),
              );
              break;
            }
            case "error":
              setAssistantMessage(chatId, assistantId, `⚠️ ${e.message}`);
              break;
          }
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setAssistantMessage(chatId, assistantId, `⚠️ ${msg}`);
      } finally {
        setPending(false);
      }
    },
    [appendEvent, appendMessage, ensureChat, setAssistantMessage, threadIds],
  );

  const panelsVisible =
    activeChat !== null && (activeChat.events.length > 0 || pending);

  return (
    <div className="relative flex h-full w-full">
      {showSidebar ? (
        <Sidebar
          chats={chats}
          activeChatId={activeChatId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onDeleteChat={handleDeleteChat}
          onToggle={() => setShowSidebar(false)}
        />
      ) : null}
      <MainArea
        chat={activeChat}
        onSend={handleSend}
        pending={pending}
        onOpenSidebar={showSidebar ? undefined : () => setShowSidebar(true)}
      />
      {panelsVisible && showPlan && (
        <PlanPanel
          events={activeChat!.events}
          pending={pending}
          onClose={() => setShowPlan(false)}
        />
      )}
      {panelsVisible && showTool && (
        <ToolPanel
          events={activeChat!.events}
          trace={activeChat!.trace}
          pending={pending}
          onClose={() => setShowTool(false)}
        />
      )}
      {panelsVisible && (!showPlan || !showTool) && (
        <div className="flex flex-col gap-2 border-l border-bg-hover bg-bg-sidebar px-2 py-3">
          {!showPlan && (
            <button
              className="rounded px-2 py-1 text-[11px] text-text-secondary hover:bg-bg-hover"
              onClick={() => setShowPlan(true)}
              title="Show planning panel"
            >
              Plan
            </button>
          )}
          {!showTool && (
            <button
              className="rounded px-2 py-1 text-[11px] text-text-secondary hover:bg-bg-hover"
              onClick={() => setShowTool(true)}
              title="Show tool panel"
            >
              Tool
            </button>
          )}
        </div>
      )}
    </div>
  );
}
