/**
 * ConfirmModal — generic yes/no confirmation dialog.
 *
 * Controlled by the parent via `isOpen`. The parent supplies text + an
 * async `onConfirm`. The modal closes itself on confirm/cancel but the
 * parent is responsible for tearing down any optimistic state if onConfirm
 * throws.
 *
 * Optional `requireTypedConfirm`: when set to a non-empty string, the
 * confirm button stays disabled until the analyst types that exact
 * string (case-sensitive). Used for destructive actions like delete.
 */
import { useState, useEffect, useRef } from "react";

export default function ConfirmModal({
  isOpen,
  title = "Are you sure?",
  message = "",
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
  requireTypedConfirm = null,
  onConfirm,
  onClose,
}) {
  const [working, setWorking] = useState(false);
  const [typed, setTyped] = useState("");

  // Track mount so we don't call setWorking after a parent has unmounted
  // the modal during the await. React 18 silently ignores the call but
  // logs a dev warning, and StrictMode doubles it.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Reset typed value each time the modal opens so a prior session
  // doesn't auto-enable the confirm button.
  useEffect(() => {
    if (isOpen) setTyped("");
  }, [isOpen]);

  if (!isOpen) return null;

  const typedOk = !requireTypedConfirm || typed === requireTypedConfirm;

  async function handleConfirm() {
    if (!typedOk) return;
    setWorking(true);
    try {
      await onConfirm?.();
      onClose?.();
    } catch (err) {
      // Parent handled the error (e.g. toast / banner) and rethrew so
      // the modal stays open for retry. Suppress the rejection here so
      // React doesn't log an "unhandled promise rejection".
      console.warn("ConfirmModal: onConfirm rejected, staying open", err);
    } finally {
      if (mountedRef.current) setWorking(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-gb-bg0-h/75"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-gb-bg0-s border border-gb-bg2 rounded-xl p-6 w-[400px] max-w-[90vw]"
      >
        <h2 className="text-base font-semibold text-gb-fg0 mb-2">{title}</h2>
        {message && (
          <p className="text-[13px] text-gb-fg4 mb-5 leading-relaxed">{message}</p>
        )}
        {requireTypedConfirm && (
          <div className="mb-5">
            <p className="text-[12px] text-gb-fg4 mb-1.5">
              Type <span className="font-data text-gb-bright-orange">{requireTypedConfirm}</span> to confirm:
            </p>
            <input
              type="text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              disabled={working}
              autoFocus
              className="w-full px-2.5 py-1.5 text-[13px] bg-gb-bg0-h border border-gb-bg2 rounded-md text-gb-fg1 placeholder-gb-bg4 outline-none focus:border-gb-bright-orange font-data"
              placeholder={requireTypedConfirm}
            />
          </div>
        )}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={working}
            className="px-3.5 py-1.5 rounded-md text-[13px] border border-gb-bg2 text-gb-fg4 hover:bg-gb-bg1 hover:text-gb-fg1 transition-colors disabled:opacity-50"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={working || !typedOk}
            className={`px-3.5 py-1.5 rounded-md text-[13px] font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              danger
                ? "bg-gb-bright-red text-gb-bg0-h hover:bg-gb-red"
                : "bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green"
            }`}
          >
            {working ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
