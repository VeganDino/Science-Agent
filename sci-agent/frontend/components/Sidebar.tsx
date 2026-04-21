"use client";

import type { Chat } from "@/lib/types";
import { IconNewChat, IconSidebar, IconTrash } from "./icons";

type Props = {
  chats: Chat[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
  onToggle?: () => void;
};

function NewChatRow({ onClick, active }: { onClick: () => void; active: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-text-primary transition-colors ${
        active ? "bg-bg-active" : "hover:bg-bg-hover"
      }`}
    >
      <span className="flex h-5 w-5 flex-none items-center justify-center text-text-secondary">
        <IconNewChat />
      </span>
      <span className="flex-1 text-left">새 채팅</span>
    </button>
  );
}

function ChatRow({
  chat,
  active,
  onSelect,
  onDelete,
}: {
  chat: Chat;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={`group relative flex items-center rounded-lg text-sm text-text-primary transition-colors ${
        active ? "bg-bg-active" : "hover:bg-bg-hover"
      }`}
    >
      <button
        type="button"
        onClick={onSelect}
        className="min-w-0 flex-1 truncate px-2 py-1.5 text-left"
      >
        {chat.title}
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          if (confirm(`"${chat.title}" 대화를 삭제할까요?`)) onDelete();
        }}
        className="mr-1 flex h-6 w-6 flex-none items-center justify-center rounded text-text-muted opacity-0 transition hover:bg-bg-active hover:text-red-400 group-hover:opacity-100 focus:opacity-100"
        aria-label="Delete chat"
        title="Delete chat"
      >
        <IconTrash width={14} height={14} />
      </button>
    </div>
  );
}

export default function Sidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onToggle,
}: Props) {
  return (
    <aside className="flex h-full w-[260px] flex-none flex-col bg-bg-sidebar">
      <div className="flex items-center justify-end px-3 pt-3 pb-2">
        <button
          onClick={onToggle}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-text-secondary hover:bg-bg-hover"
          aria-label="Collapse sidebar"
          title="Collapse sidebar"
        >
          <IconSidebar />
        </button>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto px-2 pb-2">
        <NewChatRow onClick={onNewChat} active={activeChatId === null} />

        {chats.length > 0 && (
          <>
            <div className="px-2 pt-4 pb-1 text-xs font-medium text-text-muted">
              최근
            </div>
            <div className="flex flex-col gap-0.5">
              {chats.map((c) => (
                <ChatRow
                  key={c.id}
                  chat={c}
                  active={c.id === activeChatId}
                  onSelect={() => onSelectChat(c.id)}
                  onDelete={() => onDeleteChat(c.id)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
