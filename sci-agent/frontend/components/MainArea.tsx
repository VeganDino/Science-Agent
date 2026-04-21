"use client";

import type { Chat } from "@/lib/types";
import { useStickyScroll } from "@/lib/useStickyScroll";
import ChatInput from "./ChatInput";
import ChatMessages from "./ChatMessages";
import TopBar from "./TopBar";

type Props = {
  chat: Chat | null;
  onSend: (text: string, files: File[]) => void;
  pending: boolean;
  onOpenSidebar?: () => void;
};

function messagesSignature(chat: Chat | null): string {
  if (!chat) return "";
  const last = chat.messages[chat.messages.length - 1];
  return `${chat.id}:${chat.messages.length}:${last?.content.length ?? 0}`;
}

export default function MainArea({ chat, onSend, pending, onOpenSidebar }: Props) {
  const hasMessages = chat && chat.messages.length > 0;
  const scrollRef = useStickyScroll<HTMLDivElement>(messagesSignature(chat));

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col bg-bg-main">
      <TopBar onOpenSidebar={onOpenSidebar} />

      {hasMessages ? (
        <>
          <div ref={scrollRef} className="scrollbar-thin flex-1 overflow-y-auto">
            <ChatMessages messages={chat.messages} pending={pending} />
          </div>
          <div className="pb-2">
            <ChatInput onSend={onSend} disabled={pending} />
          </div>
        </>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center">
          <h1 className="mb-8 text-[28px] font-medium tracking-tight text-text-primary">
            무슨 작업을 하고 계세요?
          </h1>
          <div className="w-full">
            <ChatInput onSend={onSend} disabled={pending} />
          </div>
        </div>
      )}
    </main>
  );
}
