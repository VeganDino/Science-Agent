"use client";

import { useEffect, useState } from "react";

/**
 * localStorage-backed state. Starts with `initial`, then on first client mount
 * replaces it with what's saved in localStorage (if any). Subsequent state
 * changes are written back. `hydrated` is false until the initial read is
 * done, so callers can avoid overwriting saved data with the pre-mount value.
 */
export function useLocalState<T>(
  key: string,
  initial: T,
): [T, React.Dispatch<React.SetStateAction<T>>, boolean] {
  const [value, setValue] = useState<T>(initial);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw !== null) {
        setValue(JSON.parse(raw) as T);
      }
    } catch {
      // corrupt / unavailable — ignore
    }
    setHydrated(true);
  }, [key]);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // quota exceeded or similar — drop silently
    }
  }, [key, value, hydrated]);

  return [value, setValue, hydrated];
}
