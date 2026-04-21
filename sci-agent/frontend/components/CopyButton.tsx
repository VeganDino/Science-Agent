"use client";

import { useState } from "react";

export default function CopyButton({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // clipboard may be unavailable in insecure contexts; ignore
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      className={`rounded px-2 py-0.5 text-[11px] text-text-secondary hover:bg-bg-hover ${className}`}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}
