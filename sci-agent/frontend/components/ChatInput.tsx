"use client";

import { useRef, useState, type ChangeEvent, type KeyboardEvent } from "react";
import { getFileVisual } from "@/lib/fileIcon";
import { IconPaperPlane, IconPlus, IconX } from "./icons";

type Props = {
  onSend: (text: string, files: File[]) => void;
  disabled?: boolean;
};

function prettySize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const canSend = !disabled && (value.trim().length > 0 || files.length > 0);

  const submit = () => {
    if (!canSend) return;
    onSend(value.trim(), files);
    setValue("");
    setFiles([]);
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const autoGrow = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list) return;
    setFiles((prev) => [...prev, ...Array.from(list)]);
    // Reset so selecting the same file again still fires onChange.
    e.target.value = "";
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="w-full">
      <div className="mx-auto max-w-[760px] px-4">
        <div className="rounded-[28px] bg-bg-input shadow-[0_0_0_1px_rgba(255,255,255,0.04)]">
          {files.length > 0 && (
            <div className="flex flex-wrap gap-2 px-3 pt-3">
              {files.map((f, i) => {
                const { Icon, color, label } = getFileVisual(f.name);
                return (
                  <div
                    key={`${f.name}-${i}`}
                    className="flex items-center gap-2 rounded-lg bg-bg-hover px-2 py-1.5"
                  >
                    <Icon width={20} height={20} className={color} />
                    <div className="flex min-w-0 flex-col">
                      <span className="max-w-[220px] truncate text-xs text-text-primary">
                        {f.name}
                      </span>
                      <span className="text-[10px] text-text-muted">
                        {label} · {prettySize(f.size)}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(i)}
                      className="rounded p-0.5 text-text-muted hover:bg-bg-active hover:text-text-primary"
                      aria-label="Remove file"
                    >
                      <IconX width={12} height={12} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          <div className="flex items-end gap-2 px-3 py-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="flex h-9 w-9 flex-none items-center justify-center rounded-full text-text-secondary hover:bg-bg-hover"
              aria-label="Attach file"
            >
              <IconPlus />
            </button>
            <input
              ref={fileRef}
              type="file"
              multiple
              onChange={onFileChange}
              className="hidden"
            />

            <textarea
              ref={taRef}
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                autoGrow(e.target);
              }}
              onKeyDown={onKeyDown}
              placeholder="무엇이든 물어보세요"
              rows={1}
              disabled={disabled}
              className="min-h-[36px] max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-6 text-text-primary placeholder:text-text-muted focus:outline-none disabled:opacity-60"
            />

            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-white text-black transition hover:bg-gray-200 disabled:cursor-not-allowed disabled:bg-bg-active disabled:text-text-muted disabled:hover:bg-bg-active"
              aria-label="Send"
            >
              <IconPaperPlane width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="pt-2 pb-3 text-center text-[11px] text-text-muted">
          Sci-Agent는 실험적입니다. 결과를 검증하세요.
        </div>
      </div>
    </div>
  );
}
