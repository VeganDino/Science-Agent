"use client";

import { useEffect, useRef } from "react";

/**
 * Robust stick-to-bottom auto-scroll.
 *
 * - Tracks whether the user is near the bottom (within 80px).
 * - Uses a ResizeObserver so ANY size change in the scroll container's
 *   children (e.g. a streaming message getting taller) triggers a re-scroll
 *   when the user is already at the bottom. Signal-driven useEffect alone
 *   misses cases where content grows between explicit signal updates.
 * - Wraps the scroll assignment in requestAnimationFrame so layout is
 *   settled before we read scrollHeight.
 *
 * If the user scrolls up, auto-scroll disengages until they return near
 * the bottom.
 */
export function useStickyScroll<T extends HTMLElement>(signal?: unknown) {
  const ref = useRef<T>(null);
  const wasAtBottom = useRef(true);

  const scrollToBottom = (el: HTMLElement) => {
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  };

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const updateAtBottom = () => {
      const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
      wasAtBottom.current = dist < 80;
    };

    // Initial measurement — before any scroll event fires.
    updateAtBottom();

    el.addEventListener("scroll", updateAtBottom, { passive: true });

    // Auto-follow content growth while at bottom.
    const ro = new ResizeObserver(() => {
      if (wasAtBottom.current) scrollToBottom(el);
    });
    // Observe the container itself (for clientHeight changes) AND each
    // direct child (for scrollHeight changes as content flows in).
    ro.observe(el);
    const childObserved = new Set<Element>();
    const observeChildren = () => {
      for (const child of Array.from(el.children)) {
        if (!childObserved.has(child)) {
          ro.observe(child);
          childObserved.add(child);
        }
      }
    };
    observeChildren();
    // New direct children may appear as the chat grows — watch for them.
    const mo = new MutationObserver(observeChildren);
    mo.observe(el, { childList: true });

    return () => {
      el.removeEventListener("scroll", updateAtBottom);
      ro.disconnect();
      mo.disconnect();
    };
  }, []);

  // Also react to explicit signal changes (chat switch, etc.) so we snap
  // to bottom on load even before any resize has happened.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (wasAtBottom.current) scrollToBottom(el);
  }, [signal]);

  return ref;
}
