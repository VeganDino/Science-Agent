"use client";

import { IconSidebar } from "./icons";

type Props = {
  onOpenSidebar?: () => void;
};

export default function TopBar({ onOpenSidebar }: Props) {
  return (
    <header className="flex h-12 items-center gap-2 px-4">
      {onOpenSidebar && (
        <button
          onClick={onOpenSidebar}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-text-secondary hover:bg-bg-hover hover:text-text-primary"
          aria-label="Open sidebar"
          title="Open sidebar"
        >
          <IconSidebar />
        </button>
      )}
      <span className="px-1 text-base font-semibold text-text-primary">
        Sci-Agent
      </span>
    </header>
  );
}
