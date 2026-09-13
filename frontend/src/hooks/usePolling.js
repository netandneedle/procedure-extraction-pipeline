import { useEffect, useRef } from "react";

/**
 * Run an async callback on a fixed interval, safely.
 *
 * Carries the fixes the app's earlier pollers each learned separately:
 * - setTimeout-recursion instead of setInterval, so a slow callback can't
 *   stack parallel invocations (the App.jsx source-poller lesson);
 * - a cancelled flag, exposed to the callback as an `isCancelled()` getter,
 *   so an in-flight response resolving after unmount can't write state
 *   (the ExplorerView lesson);
 * - the latest callback is read through a ref, so a changing identity
 *   (e.g. a filter-dependent useCallback) neither restarts the timer nor
 *   fires an extra immediate call per keystroke — the bug that made every
 *   search keystroke hammer GET /feedback-patterns/captured.
 *
 * @param {(isCancelled: () => boolean) => Promise<void>|void} callback
 * @param {number} intervalMs
 * @param {{ immediate?: boolean }} [opts] fire once on mount (default true)
 */
export default function usePolling(callback, intervalMs, { immediate = true } = {}) {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const isCancelled = () => cancelled;

    const tick = async () => {
      try {
        await cbRef.current(isCancelled);
      } finally {
        if (!cancelled) timer = setTimeout(tick, intervalMs);
      }
    };

    if (immediate) tick();
    else timer = setTimeout(tick, intervalMs);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [intervalMs, immediate]);
}
