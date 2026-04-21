"use client";

import type { ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import CopyButton from "./CopyButton";

// react-markdown v9+ removed the `inline` prop. Inline `code` has no className;
// fenced blocks carry `language-xxx` and are wrapped in a `<pre>`.

interface HastNode {
  type?: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

function nodeToText(node: HastNode | undefined): string {
  if (!node) return "";
  if (node.type === "text") return node.value ?? "";
  if (!node.children) return "";
  return node.children.map(nodeToText).join("");
}

function getLanguage(className: unknown): string | undefined {
  const list = Array.isArray(className)
    ? (className as string[])
    : typeof className === "string"
      ? [className]
      : [];
  for (const c of list) {
    const m = /^language-(\w+)/.exec(c);
    if (m) return m[1];
  }
  return undefined;
}

type CodeProps = ComponentProps<"code">;
type PreProps = ComponentProps<"pre"> & { node?: HastNode };

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="my-2 leading-7">{children}</p>,
          h1: ({ children }) => (
            <h1 className="mt-4 mb-2 text-xl font-semibold">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mt-4 mb-2 text-lg font-semibold">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mt-3 mb-2 text-base font-semibold">{children}</h3>
          ),
          ul: ({ children }) => (
            <ul className="my-2 list-disc pl-6 leading-7">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2 list-decimal pl-6 leading-7">{children}</ol>
          ),
          li: ({ children }) => <li className="my-0.5">{children}</li>,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-cyan-400 underline underline-offset-2"
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-2 border-l-2 border-bg-hover pl-3 italic text-text-secondary">
              {children}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="scrollbar-thin my-3 overflow-auto rounded-md border border-bg-hover">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border-b border-bg-hover bg-bg-hover/50 px-3 py-1.5 text-left font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-bg-hover px-3 py-1.5 align-top">
              {children}
            </td>
          ),
          hr: () => <hr className="my-3 border-bg-hover" />,
          // Inline code only. Block code is rendered inside <pre> below.
          code: ({ className, children, ...rest }: CodeProps) => {
            const isFenced =
              typeof className === "string" && className.includes("language-");
            if (isFenced) {
              return (
                <code className={className} {...rest}>
                  {children}
                </code>
              );
            }
            return (
              <code
                className="rounded bg-bg-active px-1 py-0.5 font-mono text-[0.85em]"
                {...rest}
              >
                {children}
              </code>
            );
          },
          // Block code container.
          pre: ({ children, node }: PreProps) => {
            const codeNode = node?.children?.[0];
            const lang = getLanguage(codeNode?.properties?.className);
            const text = nodeToText(codeNode).replace(/\n$/, "");
            return (
              <div className="my-3 overflow-hidden rounded-md bg-black/60">
                <div className="flex items-center justify-between border-b border-bg-hover px-3 py-1 text-[11px] text-text-muted">
                  <span>{lang ?? "code"}</span>
                  <CopyButton text={text} />
                </div>
                <pre className="scrollbar-thin overflow-auto px-3 py-2 font-mono text-[12.5px] leading-5 text-text-primary">
                  {children}
                </pre>
              </div>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
