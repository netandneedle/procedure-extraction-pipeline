/**
 * useWebSocket — React hook for real-time pipeline status via WebSocket.
 *
 * Connects to ws://<host>/ws/pipeline/{threadId} and delivers JSON events
 * to a callback. Auto-reconnects with exponential backoff on disconnect.
 * Responds to server heartbeat pings with pong.
 *
 * Usage:
 *   const { connected } = useWebSocket(threadId, (event) => {
 *     // event: { type, thread_id, data }
 *   });
 *
 * The hook manages its own lifecycle: connects when threadId is truthy,
 * disconnects on unmount or when threadId changes to null.
 *
 * Lifecycle correctness — a generation counter is incremented on every
 * effect run and captured into each WebSocket's closure. Reconnect
 * callbacks and onclose handlers check the captured generation against
 * the current one and bail when stale. Without this, a threadId change
 * while a backoff timer is pending would have the old generation's
 * reconnect fire against the new (or absent) thread.
 */
import { useState, useEffect, useRef } from "react";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const RECONNECT_FACTOR = 2;
const NORMAL_CLOSE_CODE = 1000;

/**
 * Derive WebSocket URL from current page location. Vite proxies /ws to
 * the backend in dev, so loc.host works in both dev and prod.
 */
function getWsUrl(threadId) {
  const loc = window.location;
  const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${loc.host}/ws/pipeline/${threadId}`;
}

export default function useWebSocket(threadId, onEvent) {
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);
  const generationRef = useRef(0);

  // Keep callback ref current without re-triggering effect
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    if (!threadId) {
      setConnected(false);
      return undefined;
    }

    // Each effect run is its own generation. Any callback scheduled
    // inside the closure compares its captured generation against the
    // ref before doing work. Stale generations no-op.
    generationRef.current += 1;
    const myGen = generationRef.current;

    let ws = null;
    let retries = 0;
    let reconnectTimer = null;

    const isStale = () => generationRef.current !== myGen;

    const connect = () => {
      if (isStale()) return;
      const url = getWsUrl(threadId);
      ws = new WebSocket(url);

      ws.onopen = () => {
        if (isStale()) {
          ws.close(NORMAL_CLOSE_CODE);
          return;
        }
        setConnected(true);
        retries = 0;
      };

      ws.onmessage = (e) => {
        if (isStale()) return;
        try {
          const event = JSON.parse(e.data);
          if (event.type === "heartbeat") {
            ws.send("pong");
            return;
          }
          if (onEventRef.current) {
            onEventRef.current(event);
          }
        } catch (err) {
          console.warn("useWebSocket: failed to parse message", err);
        }
      };

      ws.onclose = (e) => {
        setConnected(false);
        // Cleanup-initiated close or stale generation: don't reconnect.
        if (e.code === NORMAL_CLOSE_CODE || isStale()) return;

        const delay = Math.min(
          RECONNECT_BASE_MS * Math.pow(RECONNECT_FACTOR, retries),
          RECONNECT_MAX_MS
        );
        retries += 1;
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // onclose will fire after onerror; reconnect logic lives there
      };
    };

    connect();

    return () => {
      // Invalidate the generation FIRST so any in-flight onclose bails
      // before scheduling another reconnect.
      generationRef.current += 1;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        // 1000 signals an intentional close so the server stops the heartbeat
        // and our own onclose handler short-circuits the reconnect path.
        ws.close(NORMAL_CLOSE_CODE);
        ws = null;
      }
      setConnected(false);
    };
  }, [threadId]);

  return { connected };
}
