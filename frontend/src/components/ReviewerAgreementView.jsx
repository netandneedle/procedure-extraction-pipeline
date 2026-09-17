/**
 * ReviewerAgreementView — how often the analyst takes the AI reviewer's advice.
 *
 * This is the instrument assist mode was built to produce. Every gate submit
 * writes a diff of what the reviewer recommended against what the analyst
 * actually did; this reads those back across every source and shows, per gate,
 * whether the advice is being taken — the only evidence that can say whether a
 * gate could ever run unattended.
 *
 * Two display rules, both load-bearing:
 *
 *   - A rate NEVER appears without the count it came from. 100% of 3 is not
 *     evidence, and a bare percentage is the easiest way for this screen to
 *     mislead the person reading it.
 *   - There is no "ready for autopilot" badge. The threshold is not a property
 *     of the number: a wrong entity removal is recoverable at the next gate, a
 *     wrong chunk drop deletes a procedure from the bundle with nothing
 *     downstream able to notice. One badge would launder a per-gate product
 *     decision into arithmetic.
 */
import { useCallback, useEffect, useState } from "react";

import { fetchAgreement } from "../api/reviewer";
import { GATES, GATE_STATUS_COLORS } from "../lib/gates";
import { CONFIDENCE_STYLES, rateLabel } from "../lib/reviewerSuggestions";

/** gates_enabled key -> the label and accent used on every other surface. */
const GATE_META = Object.fromEntries(
  GATES.map((g) => [g.enableKey, {
    title: g.title,
    color: GATE_STATUS_COLORS[g.status],
  }])
);

const TIERS = ["high", "medium", "low"];

/** A rate and its denominator, never one without the other. */
function Rate({ agreed, total, className = "" }) {
  const { pct, n } = rateLabel(agreed, total);
  if (pct === null) {
    return (
      <span className={`text-gb-fg4 ${className}`}>
        — <span className="text-[10px]">{n}</span>
      </span>
    );
  }
  return (
    <span className={className}>
      <span className="font-semibold">{pct}%</span>
      <span className="text-gb-fg4 text-[11px] ml-1">({n})</span>
    </span>
  );
}

/** Horizontal agreed/overridden bar. Width encodes the rate, not the volume. */
function Bar({ agreed, total }) {
  if (!total) return <div className="h-1.5 rounded bg-gb-bg2" />;
  const pct = (agreed / total) * 100;
  return (
    <div className="h-1.5 rounded bg-gb-bright-red/40 overflow-hidden">
      <div className="h-full bg-gb-bright-green" style={{ width: `${pct}%` }} />
    </div>
  );
}

function GateCard({ gate }) {
  const meta = GATE_META[gate.gate_key] || {};
  const kinds = Object.entries(gate.by_kind || {})
    .sort((a, b) => b[1].total - a[1].total);

  return (
    <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 p-4">
      <div className="flex items-baseline justify-between gap-3 flex-wrap mb-3">
        <h3 className={`text-[14px] font-semibold ${meta.color || "text-gb-fg1"}`}>
          {meta.title || gate.gate_key}
        </h3>
        <div className="text-[11px] font-data text-gb-fg4 flex items-center gap-2">
          <span>{gate.sources} source{gate.sources === 1 ? "" : "s"}</span>
          <span>·</span>
          <span>{gate.reviews} review{gate.reviews === 1 ? "" : "s"}</span>
          {gate.failures > 0 && (
            <>
              <span>·</span>
              <span
                className="text-gb-bright-red"
                title={
                  "The reviewer errored and the gate fell through to a human. " +
                  "A gate that fails runs is not a candidate for unattended " +
                  "operation whatever its agreement rate."
                }
              >
                {gate.failures} failed
              </span>
            </>
          )}
        </div>
      </div>

      <div className="mb-3">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-[11px] uppercase font-data text-gb-fg4">
            advice taken
          </span>
          <Rate agreed={gate.agreed} total={gate.total} className="text-[13px]" />
        </div>
        <Bar agreed={gate.agreed} total={gate.total} />
      </div>

      {/* The tier that actually matters. High-confidence is what the analyst's
          bulk-accept already takes without opening, so it is the claim the
          reviewer is making about itself; the overall rate blends in the low
          items the design never said were safe. */}
      <div className="mb-3">
        <p className="text-[11px] uppercase font-data text-gb-fg4 mb-1.5">
          by confidence
          <span
            className="ml-1.5 normal-case text-gb-gray cursor-help"
            title={
              "High-confidence recommendations are the ones the analyst can " +
              "bulk-accept without opening each. Their agreement rate is the " +
              "claim this reviewer is making about itself."
            }
          >
            — high is the one to read
          </span>
        </p>
        <div className="grid grid-cols-3 gap-2">
          {TIERS.map((tier) => {
            const t = gate.by_confidence?.[tier] || { agreed: 0, total: 0 };
            return (
              <div
                key={tier}
                className={`rounded border px-2 py-1.5 ${CONFIDENCE_STYLES[tier]}`}
              >
                <div className="text-[10px] font-data uppercase opacity-80">{tier}</div>
                <Rate agreed={t.agreed} total={t.total} className="text-[12px]" />
              </div>
            );
          })}
        </div>
      </div>

      {kinds.length > 0 && (
        <div className="mb-3">
          <p className="text-[11px] uppercase font-data text-gb-fg4 mb-1.5">
            by action
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {kinds.map(([kind, t]) => (
              <span key={kind} className="text-[11px] font-data text-gb-fg2">
                {kind} <Rate agreed={t.agreed} total={t.total} />
              </span>
            ))}
          </div>
        </div>
      )}

      {gate.unsupported_quotes > 0 && (
        <p className="text-[11px] text-gb-bright-yellow mb-3">
          ⚠ {gate.unsupported_quotes} recommendation
          {gate.unsupported_quotes === 1 ? "" : "s"} rested on a quote the report
          did not support
          <span className="text-gb-fg4">
            {" "}— a separate signal from agreement: the reviewer can be right
            for a reason it could not evidence.
          </span>
        </p>
      )}

      {/* The actionable half. The rate says a gate is not trusted; these say
          why — on a real source, with both sides attached. */}
      {gate.overrides?.length > 0 && (
        <details className="border-t border-gb-bg2 pt-2">
          <summary className="text-[11px] uppercase font-data text-gb-fg4 cursor-pointer hover:text-gb-fg2">
            {gate.overrides.length} override
            {gate.overrides.length === 1 ? "" : "s"} — where it was wrong
          </summary>
          <div className="mt-2 space-y-2">
            {gate.overrides.map((o, i) => (
              <div key={`${o.ref}-${i}`} className="text-[11px] leading-snug">
                <div className="font-data text-gb-fg2">
                  <span className="text-gb-fg4">{o.kind}</span>{" "}
                  <span className="text-gb-fg1">{o.ref}</span>
                  {" — "}
                  <span className="text-gb-bright-red">said {o.recommended}</span>
                  {", "}
                  <span className="text-gb-bright-green">chose {o.actual}</span>
                  <span
                    className={`ml-1.5 text-[9px] font-data px-1 py-0.5 rounded border ${CONFIDENCE_STYLES[o.confidence] || ""}`}
                  >
                    {o.confidence}
                  </span>
                </div>
                {o.rationale && (
                  <p className="text-gb-fg4 mt-0.5 pl-2 border-l-2 border-gb-bg2">
                    {o.rationale}
                  </p>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export default function ReviewerAgreementView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await fetchAgreement());
      setError(null);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const gates = data?.gates ?? [];

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-3 border-b border-gb-bg2 flex items-baseline justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-[14px] font-semibold text-gb-fg0">AI reviewer</h2>
          <p className="text-[11px] text-gb-fg4 mt-0.5 max-w-3xl">
            How often you took each gate&rsquo;s advice. Every gate you review in
            assist mode adds to this — an override is the valuable outcome, not
            the embarrassing one: it is a labeled disagreement on a real source
            with the correct answer attached.
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-data text-gb-fg4">
          <span>
            {loading ? "loading…" : `${data?.sources_reviewed ?? 0} sources reviewed`}
          </span>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="px-2 py-0.5 rounded border border-gb-bg2 hover:text-gb-fg1 hover:border-gb-fg4 transition-colors disabled:opacity-40"
          >
            refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="mx-6 mt-3 px-4 py-2.5 rounded-lg bg-gb-tag-gate-bg border border-gb-orange text-gb-bright-orange text-[13px]">
          {error}
        </div>
      )}

      <div className="flex-1 overflow-auto px-6 py-3">
        {!loading && !error && gates.length === 0 && (
          <p className="text-gb-gray text-sm py-8 text-center max-w-2xl mx-auto">
            No reviewed gates yet. Set a gate to{" "}
            <strong>AI Assisted Review</strong> when you add a source, then
            review it yourself — each decision you make records whether you
            agreed, and those are what fill this in.
            <br />
            <span className="text-[12px]">
              Approving everything without reading would measure the click, not
              your judgment, so it would tell you nothing here.
            </span>
          </p>
        )}

        <div className="grid gap-3 xl:grid-cols-2">
          {gates.map((g) => <GateCard key={g.gate_key} gate={g} />)}
        </div>
      </div>
    </div>
  );
}
