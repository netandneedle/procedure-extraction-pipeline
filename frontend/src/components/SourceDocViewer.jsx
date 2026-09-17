/**
 * SourceDocViewer — renders the original source document for a completed bundle.
 *
 * PDF files: rendered in an iframe via the /api/bundles/{id}/source endpoint.
 * Text-based files (HTML, Markdown, plaintext): fetched and rendered as
 * styled preformatted text with basic markdown-like formatting.
 *
 * Props:
 *   bundleId    — UUID of the completed bundle
 *   fileName    — original filename (for display)
 *   fileType    — MIME type string (application/pdf, text/html, etc.)
 */
import { useState, useEffect, useCallback } from "react";
import { getSourceFileUrl } from "../api/bundles";

// MIME types that get rendered as embedded documents (iframe)
const IFRAME_TYPES = new Set(["application/pdf"]);

// MIME types that get fetched and rendered as text
const TEXT_TYPES = new Set([
  "text/html",
  "text/markdown",
  "text/plain",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);

export default function SourceDocViewer({ bundleId, fileName, fileType }) {
  const [textContent, setTextContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState("rendered"); // "rendered" | "raw"

  const sourceUrl = bundleId ? getSourceFileUrl(bundleId) : null;
  const isIframe = IFRAME_TYPES.has(fileType);
  const isText = TEXT_TYPES.has(fileType) || (!isIframe && fileType?.startsWith("text/"));

  // Fetch text content for non-PDF sources
  useEffect(() => {
    if (!bundleId || !isText) {
      setTextContent(null);
      return;
    }
    let canceled = false;
    setLoading(true);
    fetch(sourceUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (!canceled) setTextContent(text);
      })
      .catch((err) => {
        console.error("SourceDocViewer: fetch failed", err);
        if (!canceled) setTextContent("[Failed to load source content]");
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });
    return () => { canceled = true; };
  }, [bundleId, isText, sourceUrl]);

  const handleOpenExternal = useCallback(() => {
    if (sourceUrl) window.open(sourceUrl, "_blank");
  }, [sourceUrl]);

  if (!bundleId) {
    return (
      <div className="flex-1 flex items-center justify-center text-gb-bg4 text-sm">
        Select a bundle to view its source
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Pane header */}
      <div className="flex items-center justify-between px-3.5 py-2 bg-gb-bg0 border-b border-gb-bg1 shrink-0">
        <span className="text-[11px] font-semibold text-gb-fg3 uppercase tracking-wider flex items-center gap-1.5">
          <span className="text-[13px]">📄</span>
          Source Document
          {fileName && (
            <span className="font-normal text-gb-fg4 font-data text-[10px] ml-1 truncate max-w-[200px]">
              {fileName}
            </span>
          )}
        </span>
        <div className="flex gap-1">
          {isText && (
            <>
              <button
                onClick={() => setViewMode("rendered")}
                className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
                  viewMode === "rendered"
                    ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                    : "text-gb-fg4 border-gb-bg2 hover:text-gb-fg1"
                }`}
              >
                Rendered
              </button>
              <button
                onClick={() => setViewMode("raw")}
                className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
                  viewMode === "raw"
                    ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                    : "text-gb-fg4 border-gb-bg2 hover:text-gb-fg1"
                }`}
              >
                Raw
              </button>
            </>
          )}
          <button
            onClick={handleOpenExternal}
            className="px-2 py-0.5 rounded text-[10px] font-data text-gb-fg4 border border-gb-bg2 hover:text-gb-fg1 transition-colors"
          >
            ↗ Open
          </button>
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-auto bg-gb-bg0-h">
        {isIframe && (
          <iframe
            src={sourceUrl}
            className="w-full h-full border-0"
            title="Source document"
          />
        )}

        {isText && loading && (
          <div className="flex items-center justify-center h-full text-gb-gray text-sm">
            Loading source...
          </div>
        )}

        {isText && !loading && textContent !== null && (
          <div className="p-6 max-w-[750px] mx-auto">
            {viewMode === "raw" ? (
              <pre className="text-[11px] font-data text-gb-fg2 whitespace-pre-wrap leading-relaxed">
                {textContent}
              </pre>
            ) : (
              <div className="text-[13px] text-gb-fg2 leading-relaxed space-y-3">
                {renderTextContent(textContent, fileType)}
              </div>
            )}
          </div>
        )}

        {!isIframe && !isText && (
          <div className="flex flex-col items-center justify-center h-full text-gb-bg4 text-sm gap-2">
            <span className="text-2xl opacity-40">📎</span>
            <span>Preview not available for {fileType || "this file type"}</span>
            <button
              onClick={handleOpenExternal}
              className="mt-2 px-3 py-1 rounded text-[11px] font-data text-gb-bright-blue border border-gb-bright-blue hover:bg-gb-bright-blue-dim transition-colors"
            >
              Download file
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


/**
 * Basic text renderer: splits into paragraphs, applies minimal formatting.
 * For HTML sources, strips tags and renders as paragraphs.
 * For markdown, renders headings and code blocks with styling.
 */
function renderTextContent(text, mimeType) {
  if (mimeType === "text/html") {
    // Strip HTML tags for rendered view, preserve paragraph structure
    const stripped = text
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/<\/p>/gi, "\n\n")
      .replace(/<\/div>/gi, "\n")
      .replace(/<\/h[1-6]>/gi, "\n\n")
      .replace(/<[^>]+>/g, "")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&nbsp;/g, " ");
    return stripped
      .split(/\n{2,}/)
      .filter((p) => p.trim())
      .map((p, i) => <p key={i}>{p.trim()}</p>);
  }

  // Markdown / plaintext: basic paragraph + heading rendering
  return text
    .split(/\n{2,}/)
    .filter((p) => p.trim())
    .map((block, i) => {
      const trimmed = block.trim();
      // Headings
      if (trimmed.startsWith("# "))
        return <h2 key={i} className="text-[16px] font-semibold text-gb-bright-orange mt-4 mb-1">{trimmed.slice(2)}</h2>;
      if (trimmed.startsWith("## "))
        return <h3 key={i} className="text-[14px] font-semibold text-gb-bright-orange mt-3 mb-1">{trimmed.slice(3)}</h3>;
      if (trimmed.startsWith("### "))
        return <h4 key={i} className="text-[13px] font-semibold text-gb-fg1 mt-2 mb-1">{trimmed.slice(4)}</h4>;
      // Code blocks
      if (trimmed.startsWith("```"))
        return (
          <pre key={i} className="bg-gb-bg1 border border-gb-bg2 rounded px-3 py-2 text-[11px] font-data text-gb-bright-aqua whitespace-pre-wrap">
            {trimmed.replace(/^```\w*\n?/, "").replace(/\n?```$/, "")}
          </pre>
        );
      // Regular paragraph
      return <p key={i}>{trimmed}</p>;
    });
}
