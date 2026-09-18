"""extract_entities node: Stage 2a of the extraction pipeline.

LLM node. Takes parsed text and extracts structured entities using
Claude tool_use for guaranteed typed output.

WHAT THIS NODE EXTRACTS:
    - Threat actors (APT groups, threat group names)
    - Malware families and variants
    - Tools (offensive tools, utilities, LOLBins)
    - Campaigns (named campaigns or operations)
    - IOCs: hashes, IPs, domains, URLs, emails, file paths, registry keys
    - Victim info: sectors, geographies
    - Infrastructure: C2 servers, staging servers, etc.

WHAT THIS NODE ALSO DOES:
    - Detects and preserves verbatim detection rules from the source
      (Sigma, YARA, KQL, CQL, etc.) as DetectionRule objects
    - Does NOT generate detection rules, only preserves them

DESIGN DECISIONS:
    - tool_use forces Claude to return structured JSON matching our schema
    - tool_choice is forced (type: "tool") so Claude always uses the tool
    - Temperature 0.0 for deterministic extraction
    - Metadata context (author, campaign, malware family) is injected into
      the system prompt so Claude has priors for disambiguation
    - Entity IDs are generated here (uuid4), not by the LLM

READS: parsed_text, metadata
WRITES: entities, detection_rules, status, current_node
"""

from __future__ import annotations

import logging
import re
import uuid

from app.services.stix_schema import industry_sector_vocab
from app.graph.state import (
    DetectionRuleType,
    EntityType,
    PipelineState,
    PipelineStatus,
    SectionClassification,
)
from app.nodes.llm.llm_adapter import call_llm
from app.nodes.llm.tool_models import ExtractEntitiesOutput
from app.services.feedback_examples import relevant_examples_cached
from app.services.feedback_patterns import (
    denylist_match_value,
    load_denylist,
    relevant_addendum_cached,
)


# Feedback categories that are operationally relevant to entity extraction.
# These are the LLM's chronic failure modes for this stage:
#   defender_ioc — letterhead / CERT contact info miscoded as IoC
#   brand_as_malware — technique-pattern brand names (ClickFix, MFA fatigue)
#                       miscoded as malware/tool
#   false_positive_entity — other entity-level false positives
#   mis_attribution — cluster vs campaign vs intrusion-set confusion
#   other — the synthesizer's catch-all, AND the silent fallback when it omits
#           a category entirely (feedback_synthesis: `category =
#           (raw.get("category") or "other")`). Every node reads it, because a
#           category no node reads makes a real analyst correction structurally
#           unreachable forever with nothing logged — which is what happened to
#           its two patterns, one of them promoted_to_prompt. Eligibility is not
#           a dump: retrieval is relevance-first and still applies the top-N cut.
#           Pinned by tests/test_contracts.py::TestFeedbackCategoriesAreReadable.
_ENTITY_FEEDBACK_CATEGORIES = (
    "defender_ioc",
    "brand_as_malware",
    "false_positive_entity",
    "mis_attribution",
    "other",
)


async def _fetch_feedback_addendum(state) -> str:
    """Analyst feedback patterns RELEVANT TO THIS SOURCE, as a prompt addendum.
    Source-relative hybrid retrieval, TTL-cached — see relevant_addendum_cached.
    Best-effort: returns "" on any failure."""
    return await relevant_addendum_cached(
        state, categories=_ENTITY_FEEDBACK_CATEGORIES,
        node="extract_entities", limit=15,
    )


async def _fetch_feedback_examples(state) -> str:
    """Past analyst corrections most similar to this source, as demonstrations.

    A separate channel from `_fetch_feedback_addendum` on purpose: the rules
    are LLM-written generalizations and the examples are records, they fail in
    different ways, and keeping the fetches apart is what lets an ablation arm
    vary one without the other. Best-effort: "" on any failure.
    """
    return await relevant_examples_cached(
        state, areas=("entities",), node="extract_entities",
    )


async def _apply_entity_denylist(entities: list[dict]) -> int:
    """Tag entities whose value is on the analyst denylist (deterministic
    guardrail behind a promoted_to_denylist pattern).

    Tagged entities carry ``denylisted=True`` + ``denylist_pattern_id`` +
    ``denylist_reason``; gate_0 turns the tag into a gate_action=remove that
    the analyst can override at the gate. Best-effort: load_denylist returns
    empty maps on any failure, so this no-ops rather than blocking the run.
    Returns the count tagged (for logging).
    """
    denylist = await load_denylist()
    if not denylist.get("values"):
        return 0
    tagged = 0
    for e in entities:
        info = denylist_match_value(
            e.get("value", ""), denylist, entity_type=e.get("entity_type"),
        )
        if info:
            e["denylisted"] = True
            e["denylist_pattern_id"] = info["pattern_id"]
            e["denylist_reason"] = (
                f"Matches analyst denylist (pattern {info['pattern_id']}): "
                f"{(info['pattern'] or '')[:160]}"
            )
            tagged += 1
    return tagged


logger = logging.getLogger(__name__)


# =============================================================================
# Tool definitions for Claude tool_use
# =============================================================================

# The entity types Claude can assign. Mirrors EntityType enum.
_ENTITY_TYPE_VALUES = [e.value for e in EntityType]

# Detection rule types. Mirrors DetectionRuleType enum.
# Generated from the bundled STIX schema rather than hand-maintained.
# The prose list this replaces had drifted from the real vocabulary —
# it offered `defence`, `maritime`, `media`, `pharmaceutical` and
# `real-estate`, none of which are STIX values, and `sectors` is not
# enum-checked at validation so they would have shipped.
_SECTOR_VOCAB_LINE = ", ".join(industry_sector_vocab())

_RULE_TYPE_VALUES = [r.value for r in DetectionRuleType]

EXTRACT_ENTITIES_TOOL = {
    "name": "extract_entities",
    "description": (
        "Extract ALL structured entities from the threat intelligence text. "
        "Be thorough: extract every intrusion set, threat actor, malware, tool, "
        "campaign, vulnerability, organization, location, IOC, victim sector, "
        "infrastructure, software, and user account mention. "
        "Also extract any detection rules found VERBATIM in the text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": (
                    "Every entity found in the source. Include ALL mentions, "
                    "even if repeated. Deduplication happens downstream."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": (
                                "The entity value. Naming rules depend on type: "
                                "For intrusion_set: use EXACTLY the name the source uses. "
                                "Never rename across vendor nomenclatures. "
                                "For malware/tool: prefer MITRE ATT&CK canonical name, "
                                "fallback to Malpedia name, then source name. "
                                "For IOCs: raw indicator value as-is. "
                                "For vulnerability: CVE ID (e.g., CVE-2023-46604). "
                                "For everything else: use the name as it appears in the source."
                            ),
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": _ENTITY_TYPE_VALUES,
                            "description": "Entity classification.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Extraction confidence. 1.0 = explicitly named, "
                                "0.7 = strongly implied, 0.4 = inferred from context."
                            ),
                        },
                        "context_snippet": {
                            "type": "string",
                            "description": (
                                "The 1-2 sentence excerpt from the source where "
                                "this entity was mentioned. Used for provenance."
                            ),
                        },
                        "organization_role": {
                            "type": "string",
                            "enum": ["victim", "sponsor", "publisher", "author", "other"],
                            "description": (
                                "Only for entity_type=organization. The role this "
                                "organization plays in the report context. "
                                "'author' = the analyst team that wrote the report "
                                "(e.g., Mandiant, Unit42). 'publisher' = the legal "
                                "entity that distributes it (e.g., Google, Palo Alto "
                                "Networks). When the same org is both, prefer "
                                "'author' (analyst attribution is more specific)."
                            ),
                        },
                        "location_role": {
                            "type": "string",
                            "enum": ["victim", "origin", "context"],
                            "description": (
                                "Only for entity_type=location. 'victim' = location "
                                "where targets were attacked. 'origin' = location "
                                "attributed to the attacker (attacker geography). "
                                "'context' = mentioned as background, neither target "
                                "nor attacker origin. Required for every location."
                            ),
                        },
                    },
                    "required": ["value", "entity_type", "confidence"],
                },
            },
            "detection_rules": {
                "type": "array",
                "description": (
                    "Detection rules found VERBATIM in the source text. "
                    "Only include rules that are quoted directly from the source. "
                    "Do NOT generate or invent rules. If the source contains "
                    "no detection rules, return an empty array."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_type": {
                            "type": "string",
                            "enum": _RULE_TYPE_VALUES,
                            "description": "Type of detection rule.",
                        },
                        "rule_content": {
                            "type": "string",
                            "description": (
                                "The complete rule text, verbatim from the source. "
                                "Include the full rule body, not just a snippet."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "The source author's description of the rule, if provided.",
                        },
                    },
                    "required": ["rule_type", "rule_content"],
                },
            },
            "is_sequential": {
                "type": "boolean",
                "description": (
                    "Whether the source describes a chronologically ordered "
                    "sequence of attacker actions (true) or catalogs procedures "
                    "without intrinsic ordering between them (false). Classify "
                    "the SOURCE AS A WHOLE — see SOURCE STRUCTURE in the system "
                    "prompt for criteria. When unsure, prefer false."
                ),
            },
            "sequentiality_rationale": {
                "type": "string",
                "description": (
                    "One-sentence explanation citing source phrasing that "
                    "drove the is_sequential classification. Surfaces in the "
                    "Gate 0 review chip so analysts can sanity-check or "
                    "override. Example: 'Source narrates a single intrusion "
                    "from initial access through exfiltration with explicit "
                    "time ordering.'"
                ),
            },
        },
        "required": ["entities", "detection_rules", "is_sequential", "sequentiality_rationale"],
    },
}


# =============================================================================
# System prompt
# =============================================================================

_SYSTEM_PROMPT_TEMPLATE = """You are an expert Cyber Threat Intelligence (CTI) analyst performing entity extraction on a threat intelligence source.

YOUR TASK:
Extract every structured entity from the provided text using the extract_entities tool. Be exhaustive.

ENTITY TYPES AND GUIDANCE:

SDO-producing types (become STIX Domain Objects):
- intrusion_set: Activity clusters tracked by threat intel vendors. APT designations (APT29, APT41), vendor-specific clusters (UNC4899, SCATTERED SPIDER, Storm-0558, DEV-0537). CRITICAL: use EXACTLY the name the source uses. Never rename or merge across vendor nomenclatures. If the source says "UNC4899", extract "UNC4899", NOT "Lazarus Group". Different vendors cluster differently.
- threat_actor: The real-world actor/organization BEHIND intrusion sets. State sponsors (SVR, GRU, MSS, IRGC), criminal organizations. NOT the intrusion set cluster name. Example: "APT29" = intrusion_set, "SVR" = threat_actor. These are linked via attributed-to relationships downstream.
- malware: Malware families, variants, backdoors, RATs, implants. Prefer the MITRE ATT&CK canonical name if one exists (check S#### entries), then Malpedia name, then source name. Not tools or utilities.
- tool: Offensive tools (Cobalt Strike, Mimikatz), LOLBins (certutil, BITSAdmin), or dual-use utilities. Prefer ATT&CK canonical name, then Malpedia, then source name. Distinguish from malware: tools are commercially available or dual-use, malware is purpose-built.

ATTRIBUTION DECISION TREE (Malware vs Tool classification):
  - Automated behavior (drops files, beacons to C2, self-replicates, injects into processes): classify as malware.
  - Operator using a legitimate/dual-use tool hands-on-keyboard (certutil, PsExec, BITSAdmin, PowerShell): classify as tool.
  - Commercial offensive software (Cobalt Strike, Brute Ratel, Sliver): classify as tool.
  - Malware-as-a-Service (MaaS) / commodity malware sold on underground markets (StealC, Lumma, RedLine, Raccoon): classify as malware. CRITICAL: MaaS malware is operated by many unrelated actors. Do NOT imply a specific intrusion_set link. The downstream pipeline will NOT create an attributed-to SRO from MaaS malware to any intrusion set unless the source explicitly attributes the campaign to a tracked group.
  - Custom implant or backdoor built by a specific group: classify as malware.
  - When uncertain: prefer malware for purpose-built code, tool for anything with legitimate uses.

NOT MALWARE, NOT TOOLS — social-engineering / technique patterns:
Some named phenomena in CTI describe a TECHNIQUE PATTERN, not a piece of code. They have no binary, no hash, no C2 — they describe a class of operator behavior. Examples: ClickFix, Browser-in-the-Browser (BitB), MFA fatigue / MFA bombing, EvilProxy-style adversary-in-the-middle phishing, drive-by compromise, watering hole, pretexting. These are NOT malware and NOT tools. They map to ATT&CK techniques downstream (e.g., ClickFix → T1204.004 User Execution: Malicious Copy and Paste). Do NOT extract them as malware or tool entities even if the source uses sentences like "the actor deployed ClickFix" or "they used a ClickFix lure."
- If the source uses the pattern name as a campaign descriptor ("a ClickFix campaign," "the BitB campaign targeting banks"), apply the UNNAMED CAMPAIGN rule under `campaign` and extract the descriptive phrase as a Campaign entity with confidence 0.5-0.6.
- Otherwise, simply do not extract the pattern name as an entity — the chunker will surface the behavior in its own narrative and technique-mapping handles the ATT&CK ID assignment.
- Quick disambiguator: if there is no plausible binary, hash, or executable to point at — it's a technique pattern, not malware/tool.
- campaign: Named operations or campaigns (e.g., "Operation Aurora", "SolarWinds compromise"). Not the actor name. UNNAMED CAMPAIGNS: When a source uses "campaign" descriptively without a formal name (e.g., "a ClickFix campaign," "a phishing campaign targeting healthcare"), extract the descriptive phrase as the value (e.g., "ClickFix campaign") with confidence 0.5-0.6. These become Campaign SDOs downstream but are flagged as unnamed for analyst review at Gate 0. Do NOT fabricate a campaign name; use the source's phrasing.
- vulnerability: CVE identifiers. Extract the full CVE ID (e.g., "CVE-2023-46604"). If the source describes a vulnerability without a CVE, extract the description as the value.
- organization: Named organizations mentioned in the source. Set organization_role to indicate the role:
  - "victim" — targeted companies/agencies (e.g., "Acme Corp" → victim).
  - "sponsor" — state sponsors when described as an org rather than an intrusion set (e.g., "IRGC" → sponsor).
  - "author" — the analyst TEAM that wrote the report (e.g., "Mandiant", "GTIG", "Unit42", "Talos"). The byline / "by ..." attribution.
  - "publisher" — the legal entity that DISTRIBUTES the report (e.g., "Google" for Mandiant/GTIG reports, "Palo Alto Networks" for Unit42, "Cisco" for Talos). The footer / copyright line.
  - "other" — anything else (mentioned third parties, vendors named in passing, etc.).
  When the same org is both author and publisher (small vendors that publish their own work), prefer "author" — analyst attribution is more specific. When unsure, prefer "publisher" for top-level corporate names and "author" for security-team brands.
- location: Countries, regions, cities. Use standard country names ("United States", "South Korea", "Turkey"). For multi-country regions use the CANONICAL SIX and nothing else: EMEA (Europe, Middle East, Africa), APAC (Asia-Pacific), LATAM (Latin America), NA (North America), CIS (Commonwealth of Independent States), ANZ (Australia and New Zealand). These are standard and must NOT be decomposed into constituent countries. A NON-standard coinage — "AMEA", "EMEIA", "Asia and Africa" — is not one of the six: emit the canonical region it corresponds to, or the individual countries the source actually names. A narrower geographic region the source itself uses ("Eastern Europe", "South Asia") stays as written — do not widen it to a canonical acronym, since that would broaden a targeting claim the source did not make. ALWAYS set location_role: "victim" for locations where the attack targeted entities (e.g., "U.S. healthcare providers" -> victim), "origin" for locations attributed to the attacker (e.g., "Russia-based actors" -> origin), "context" for geography mentioned as background with no target/attacker role (e.g., "the 2022 Eastern European conflict" -> context). When unclear, prefer "context" to avoid false attribution.
- victim_sector: Industry sectors targeted. Use ONLY these STIX industry-sector-ov values: {_SECTOR_VOCAB_LINE}. If the source names a sector with no value here (e.g. "legal & professional services"), pick the closest listed value rather than inventing one — do NOT emit a sector the source never states.
PLACEHOLDERS AND REDACTIONS ARE NOT INDICATORS:
Reports print template placeholders where a real value would go: "<organization>.enrollms[.]com", "[COMPANY NAME] DATA BREACH", "[Unique Session ID]", "[pseudorandom_alphanumeric_string]@gmail.com", "C:\\Users\\<REDACTED_USER_1>\\...".
- Never emit a value containing <angle>, [square] or {curly} placeholder text as an IOC. It is not resolvable and pollutes any blocklist built from the bundle.
- When the placeholder is only a subdomain label and the rest is real, extract the REGISTRABLE DOMAIN instead: "<organization>.enrollms[.]com" -> "enrollms.com". That is the actual indicator; the pattern belongs in the description.

VICTIM-SIDE IDENTIFIERS ARE NOT INDICATORS:
Sample logs and case studies contain the VICTIM's identifiers — mailbox owners, UserId fields in victim telemetry, internal hostnames, "victim.user@organization.com". These are context, never adversary IOCs. Extracting them is wrong twice over: they identify the wrong side of the conflict, and on a report with genuine unredacted victim data, publishing them in a bundle meant for sharing would leak it. Extract the ATTACKER's infrastructure and accounts only.

DEFENSIVE PRODUCTS ARE NOT ADVERSARY TOOLING:
Reports end with remediation advice and detection guidance naming DEFENDER products — "enable Microsoft Defender Credential Protection", "leverage SmartScreen", "Google SecOps rule packs", "implement FIDO2 keys". These are recommendations to the reader, not anything the adversary used or targeted.
- Never extract a product as `tool` or `software` because it appears in a remediation, hardening, mitigation or detection section.
- `software` means a product the adversary TARGETED or ABUSED (Okta, SharePoint, the exploited appliance). `tool` means something the adversary USED.
- The test: did the actor touch it, or is the report telling the reader to deploy it?

DO NOT INVENT WHAT THE SOURCE DOES NOT STATE:
If the report never names a victim sector, emit no victim_sector. If it never names a location, emit no location. A low-confidence guess is worse than an absence: the bundle presents it with the same weight as an observed fact.

- infrastructure: attacker-OPERATED assets (C2 servers, staging infrastructure, redirectors, VPN exit nodes, bulletproof hosting) AND the venues and intermediaries that exist to serve adversary purposes (criminal forums, marketplaces, leak/negotiation sites, hop-through hosts). Use when the source describes infrastructure conceptually, not as a raw IOC. NOT a neutral utility the actor merely rides on — a CDN, registrar, nameserver, hosting provider or mail provider being ABUSED is an `organization`, because typing it here asserts adversary control the source does not support. The test: does the service exist to serve adversary purposes, or is it a general-purpose utility that happens to have an adversary among its users?

SCO-producing types (become STIX Cyber Observables attached to procedures):
- ioc_hash: MD5, SHA-1, SHA-256, SHA-512 file hashes. Extract the raw hash string.
- ioc_ip: IPv4 or IPv6 addresses. Include port if mentioned. Both v4 and v6 use this type.
- ioc_domain: Domain names used for C2, staging, or exfiltration.
- ioc_url: Full URLs (with path). Use ioc_domain if only a bare domain is mentioned.
- ioc_email: Email addresses used in phishing, registration, or communication.
- ioc_file_path: File paths on disk (e.g., C:\\Windows\\Temp\\payload.exe).
- ioc_registry_key: Windows registry keys modified by malware.
- ioc_mutex: Mutex names or named pipes created by malware for synchronization.
- ioc_command_line: Full or near-full command-line strings cited VERBATIM from the source. Include every argument, flag, switch, redirection, and environment variable shown. Examples: "powershell.exe -ExecutionPolicy Bypass -enc <base64>", "wmic process call create 'cmd.exe /c ...'", "wevtutil cl Security", "schtasks /create /tn Updater /tr ...". DO NOT paraphrase, summarize, abbreviate, normalize whitespace, or fabricate. If the source only describes a command in prose ("the operator ran a powershell encoded payload") WITHOUT showing the literal string, DO NOT extract — that's a procedure description, not a command-line entity. Only extract when the source shows the actual string. Confidence: 1.0 for verbatim quotes; never below 0.8 — if you're tempted to use a lower confidence, you're inventing instead of extracting.
- ioc_process_name: Concrete executable filenames cited in command lines, process trees, or prose, typically with extension (.exe, .dll, .so, .ps1). Examples: "powershell.exe", "certutil.exe", "rundll32.exe", "evil.dll", "cmd.exe", "mshta.exe", "regsvr32.exe". Distinct from `tool` (which is the abstract capability label, e.g., "certutil" without extension treated as a LOLBin). When the source mentions both forms — e.g., "the operator used certutil" AND "certutil.exe loaded the cert" — extract `tool=certutil` AND `ioc_process_name=certutil.exe` (the dedup key includes type, so both survive). Do NOT extract bare "powershell" without the extension as a process_name; that's the `tool` form.
- software: Targeted software products WITH version when available. Examples: "Apache ActiveMQ 5.15.0", "Microsoft Exchange Server 2019", "Confluence 8.5.0". Not malware, not tools. The software being attacked or exploited.
- user_account: Compromised user accounts, service accounts, or credentials. Examples: "admin@target.com", "svc_backup", "DOMAIN\\admin".

NAMING RULES (CRITICAL):
1. INTRUSION SETS: ALWAYS preserve the source's exact name. Do not rename, merge, or substitute alias names. UNC4899 stays UNC4899. SCATTERED SPIDER stays SCATTERED SPIDER.
2. MALWARE & TOOLS: Prefer MITRE ATT&CK name (e.g., ATT&CK calls it "Cobalt Strike" not "CS" or "cobaltstrike"). If not in ATT&CK, prefer Malpedia canonical name. Last resort: use the source's name.
3. IOCs: Extract the raw value exactly as written. Never modify IP addresses, hashes, or domain names.
4. VULNERABILITY: Always include the CVE ID when available.
5. COMMAND LINES & PROCESS NAMES: Extract command lines verbatim — never paraphrase, abbreviate, or reconstruct. Process names are distinct from tool entities: extract both when both apply (e.g., `tool=certutil` for the LOLBin label AND `ioc_process_name=certutil.exe` for the on-disk binary). The dedup key includes entity_type, so the same string with two types co-exists. Bare process names without an extension belong to `tool`, not `ioc_process_name`.

CONFIDENCE SCORING:
- 1.0: Explicitly named with clear attribution ("APT29 deployed Cobalt Strike")
- 0.8-0.9: Named but context is less definitive ("the group used a tool similar to Mimikatz")
- 0.5-0.7: Implied or requires inference ("the Eastern European threat group" -> inferred location/actor)
- 0.3-0.5: Weak reference or ambiguous

DETECTION RULES:
- ONLY extract rules that appear VERBATIM in the source text
- Do NOT generate, synthesize, or infer detection rules
- If the source has a Sigma rule in a code block, extract the FULL rule body
- If no detection rules exist in the source, return an empty array

SOURCE STRUCTURE (for the is_sequential field):

Classify the SOURCE AS A WHOLE, not individual entities. Two shapes:

SEQUENTIAL (is_sequential=true) — the source narrates execution of one or more intrusions with implied or explicit chronological ordering. Tell-tale phrasing:
- "After gaining initial access, the actor then..."
- "The campaign began on [date] with..."
- "Following execution of X, Y was used to..."
- A timeline, kill-chain walkthrough, or IR writeup.
Examples: incident reports, intrusion writeups, single-campaign analyses, sandbox detonation traces.

NON-SEQUENTIAL (is_sequential=false) — the source CATALOGS procedures, tools, or behaviors without claiming ordering. Tell-tale phrasing:
- "The actor has been observed using X, Y, and Z..."
- Bulleted procedure inventories
- Threat-actor profiles listing TTPs
- "Common procedures include..."
Examples: vendor threat-actor profiles, capability inventories, technique-trend reports, quarterly threat-landscape briefings, TTP inventories.

WHEN UNSURE, prefer false. A false answer keeps the pipeline from inventing ordering that isn't in the source. The downstream chunker can still emit precedes edges when the source EXPLICITLY describes ordering — is_sequential controls only whether the pipeline fills in ordering when the source doesn't.

Edge cases:
- Mixed source (incident writeup with a "background context" catalog section): classify by the DOMINANT shape. If the bulk of procedural content is sequential, return true.
- Same procedure described twice (once in narrative, once in a summary list): true if the narrative is the primary content.
- Single-paragraph alerts naming one technique without context: false — no sequence to extract.

DEDUPLICATION:
- Extract each UNIQUE entity once. If "Cobalt Strike" appears 5 times, extract it once.
- Exception: if the same value has different entity_types (e.g., "certutil" as both tool and file path), extract both."""

# Injected rather than f-stringed: the prompt contains literal JSON
# braces that an f-string would require escaping throughout.
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace(
    "{_SECTOR_VOCAB_LINE}", _SECTOR_VOCAB_LINE,
)



def _build_metadata_context(metadata: dict) -> str:
    """Build a metadata context block for the system prompt.

    Gives Claude priors about the source so it can better disambiguate
    entities. For example, knowing the report is about APT29 helps
    Claude assign the right actor even when the text says "the group."
    """
    parts = []
    if metadata.get("author"):
        parts.append(f"Source author: {metadata['author']}")
    if metadata.get("threat_actor"):
        parts.append(f"Known threat actor: {metadata['threat_actor']}")
    if metadata.get("campaign"):
        parts.append(f"Campaign: {metadata['campaign']}")
    if metadata.get("malware_family"):
        parts.append(f"Malware family: {metadata['malware_family']}")
    if metadata.get("publication_date"):
        parts.append(f"Published: {metadata['publication_date']}")
    if metadata.get("source_url"):
        parts.append(f"Source URL: {metadata['source_url']}")

    if parts:
        return "\n\nSOURCE CONTEXT (from ingestion metadata):\n" + "\n".join(parts)
    return ""


# =============================================================================
# Node function
# =============================================================================

async def extract_entities(state: PipelineState) -> dict:
    """Stage 2a: Extract structured entities from parsed text.

    LangGraph node function. Calls Claude with tool_use to extract
    entities and detection rules from the parsed source text.

    Async because the prompt-build step fetches FeedbackPattern rows from
    postgres (the feedback flywheel) and the LLM call itself is awaited.

    Args:
        state: Current pipeline state. Must contain:
            - parsed_text: The text to extract from
            - metadata: Source metadata for context priors

    Returns:
        Dict with entities, detection_rules, status, current_node.
        On failure: also sets error field.
    """
    parsed_text = state.get("parsed_text", "")
    metadata = state.get("metadata", {})
    sequentiality = state.get("sequentiality", "auto") or "auto"

    logger.info(
        "extract_entities: starting, text_length=%d, sequentiality=%s",
        len(parsed_text), sequentiality,
    )

    update: dict = {
        "status": PipelineStatus.EXTRACTING_ENTITIES.value,
        "current_node": "extract_entities",
    }

    if not parsed_text:
        logger.error("extract_entities: no parsed_text in state")
        update["error"] = "No parsed text available for entity extraction"
        update["status"] = PipelineStatus.FAILED.value
        update["entities"] = []
        update["detection_rules"] = []
        # Resolve sequentiality even on the empty-text failure path so
        # downstream nodes always see a concrete bool. Defaults to True
        # (preserves prior pipeline behavior of emitting precedes edges).
        update["is_sequential"] = sequentiality != "no"
        update["sequentiality_rationale"] = (
            "No parsed text — defaulted from sequentiality setting"
        )
        return update

    try:
        # Build the prompt with metadata context + recent analyst feedback
        # patterns (the flywheel). Fetch is best-effort: if the DB query
        # fails, we proceed without feedback rather than failing the run.
        feedback_addendum = await _fetch_feedback_addendum(state)
        feedback_examples = await _fetch_feedback_examples(state)
        system = (SYSTEM_PROMPT + _build_metadata_context(metadata)
                  + feedback_addendum + feedback_examples)

        # Call Claude with forced tool_use
        response = await call_llm(
            system=system,
            messages=[{
                "role": "user",
                "content": (
                    "Extract all entities and detection rules from the "
                    "following threat intelligence text:\n\n"
                    f"---\n{_text_for_entity_extraction(state)}\n---"
                ),
            }],
            tools=[EXTRACT_ENTITIES_TOOL],
            tool_choice={"type": "tool", "name": "extract_entities"},
            temperature=0.0,
            output_model=ExtractEntitiesOutput,
        )

        # The adapter drops a malformed list item rather than failing the
        # whole response (one of 181 entities once cost every entity in the
        # report). Say so here, next to the count, so a short list can be
        # read against what was thrown away.
        if response.dropped_items:
            logger.warning(
                "extract_entities: adapter dropped %d malformed item(s) from %s "
                "-- the rest of the list was kept",
                len(response.dropped_items),
                sorted({d["field"] for d in response.dropped_items}),
            )

        # Process entities from tool output
        raw_entities = response.tool_output.get("entities", [])
        entities = _process_entities(raw_entities)

        # Deterministic guardrail: tag any entity matching the analyst
        # denylist so gate_0 auto-removes it (analyst can override at the
        # gate). Best-effort — never blocks extraction.
        tagged = await _apply_entity_denylist(entities)
        if tagged:
            logger.info(
                "extract_entities: %d entit(ies) flagged by analyst denylist",
                tagged,
            )

        # Process detection rules
        raw_rules = response.tool_output.get("detection_rules", [])
        detection_rules = _process_detection_rules(raw_rules)

        # Resolve is_sequential. Analyst's pre-flight choice wins when set;
        # only consult the LLM's classification on "auto". The unused field
        # in the override paths is cheap (~30 tokens) and keeps the schema
        # uniformly required.
        if sequentiality == "yes":
            is_sequential = True
            rationale = "Analyst override at upload: yes"
        elif sequentiality == "no":
            is_sequential = False
            rationale = "Analyst override at upload: no"
        else:
            is_sequential = bool(response.tool_output.get("is_sequential", False))
            rationale = (response.tool_output.get("sequentiality_rationale") or "").strip()

        logger.info(
            "extract_entities: extracted %d entities, %d detection rules, "
            "is_sequential=%s. tokens: in=%d, out=%d",
            len(entities), len(detection_rules), is_sequential,
            response.input_tokens, response.output_tokens,
        )

        update["entities"] = entities
        update["detection_rules"] = detection_rules
        update["is_sequential"] = is_sequential
        update["sequentiality_rationale"] = rationale

    except Exception as e:
        logger.exception("extract_entities: failed")
        update["error"] = f"Entity extraction failed: {type(e).__name__}: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["entities"] = []
        update["detection_rules"] = []
        # Same default as the empty-text path — preserve prior PRECEDES
        # behavior unless the analyst explicitly said "no".
        update["is_sequential"] = sequentiality != "no"
        update["sequentiality_rationale"] = (
            "Entity extraction failed — defaulted from sequentiality setting"
        )

    return update


# =============================================================================
# Post-processing
# =============================================================================

_REFANG_TYPES = {
    EntityType.IOC_DOMAIN.value,
    EntityType.IOC_IP.value,
    EntityType.IOC_URL.value,
    # Email + hash entity types may appear defanged in some reports.
    # Hash defang is rare in practice, but emails frequently use [at] / [.].
    # Not refanging IOC_HASH since hash values shouldn't contain `.` or `@`.
}


# --- Extraction-quality backstops ------------------------------------------

# Template placeholders the report itself uses to stand in for a real value:
# "<organization>.enrollms[.]com", "[COMPANY NAME]", "{victim}".
_PLACEHOLDER_RE = re.compile(r"<[^<>]{1,40}>|\[[^\[\]]{1,40}\]|\{[^{}]{1,40}\}")

# Leading placeholder on a hostname — the registrable domain after it is the
# real indicator, so strip rather than drop.
_LEADING_PLACEHOLDER_HOST_RE = re.compile(
    r"^(?:<[^<>]+>|\[[^\[\]]+\]|\{[^{}]+\})[.\-_]+",
)

# Redacted / sample identifiers that stand in for a victim.
_VICTIM_PLACEHOLDER_TOKENS = (
    "victim.user", "victim_user", "redacted", "<user", "[user",
    "example.com", "organization.com", "company.com", "yourcompany",
    "pseudorandom", "unique alphanumeric",
)

# Entity types where a placeholder makes the value unusable as an indicator.
_IOC_TYPES_FOR_PLACEHOLDER = frozenset({
    EntityType.IOC_DOMAIN.value, EntityType.IOC_URL.value,
    EntityType.IOC_EMAIL.value, EntityType.IOC_IP.value,
    EntityType.IOC_HASH.value, EntityType.IOC_FILE_PATH.value,
    EntityType.IOC_PROCESS_NAME.value, EntityType.IOC_REGISTRY_KEY.value,
    EntityType.IOC_COMMAND_LINE.value, EntityType.IOC_MUTEX.value,
})

# Below this, the model is signaling it is guessing. The audit found a
# fabricated `victim_sector: financial-services` at 0.3 on a report that
# names no sector anywhere.
LOW_CONFIDENCE_FLOOR = 0.4


def _normalize_placeholder_value(value: str, entity_type: str) -> str | None:
    """Strip or reject template placeholders in an IOC value.

    The report prints its credential-harvesting domains as
    `<organization>.enrollms[.]com` — a naming PATTERN, not a resolvable
    name. Emitted literally they become domain-name SCOs containing
    "<organization>", unusable in any blocklist or hunt, while the genuinely
    actionable indicator (the registrable base domain) is lost. Returns the
    cleaned value, or None when nothing usable remains.
    """
    if entity_type not in _IOC_TYPES_FOR_PLACEHOLDER:
        return value
    if not _PLACEHOLDER_RE.search(value):
        return value

    stripped = _LEADING_PLACEHOLDER_HOST_RE.sub("", value).strip()
    if stripped and not _PLACEHOLDER_RE.search(stripped):
        return stripped
    return None


def _is_victim_side_identifier(value: str, entity_type: str) -> bool:
    """True for identifiers belonging to the VICTIM, not the adversary.

    `victim.user@organization.com` was extracted as an ioc_email at
    confidence 1.0 from a redacted sample audit log. Wrong twice over: it is
    a placeholder, and a victim's mailbox is never an adversary indicator.
    On a report with genuine unredacted victim data this would publish it.
    """
    if entity_type not in (
        EntityType.IOC_EMAIL.value, EntityType.USER_ACCOUNT.value,
    ):
        return False
    lowered = value.lower()
    return any(token in lowered for token in _VICTIM_PLACEHOLDER_TOKENS)


def _process_entities(raw_entities: list[dict]) -> list[dict]:
    """Convert raw LLM output to Entity-shaped dicts.

    - Assigns unique entity_id (uuid4)
    - Validates entity_type against EntityType enum
    - Clamps confidence to 0.0-1.0
    - Refangs defanged IOC values (domains, IPs, URLs) to canonical form
      so STIX SCOs ship with proper values and downstream IoC-linking can
      match against chunk artifacts. Refang is type-gated: actor names and
      tool names are NEVER refanged (they may legitimately contain brackets).
    - Deduplicates by (value, entity_type), AFTER refanging so
      `domain[.]com` and `domain.com` from the same report collapse.
    """
    from app.utils.refang import refang

    seen: set[tuple[str, str]] = set()
    entities: list[dict] = []

    valid_types = {e.value for e in EntityType}

    for raw in raw_entities:
        value = raw.get("value", "").strip()
        entity_type = raw.get("entity_type", "")
        confidence = raw.get("confidence", 0.5)

        # Skip empty values
        if not value:
            continue

        # Validate entity type
        if entity_type not in valid_types:
            logger.warning(
                "extract_entities: skipping entity with invalid type '%s': %s",
                entity_type, value,
            )
            continue

        # Refang IOCs before dedup so defanged + fanged variants collapse.
        if entity_type in _REFANG_TYPES:
            refanged = refang(value)
            if refanged != value:
                logger.debug(
                    "extract_entities: refanged %s '%s' -> '%s'",
                    entity_type, value, refanged,
                )
                value = refanged

        # Strip or reject template placeholders.
        cleaned = _normalize_placeholder_value(value, entity_type)
        if cleaned is None:
            logger.info(
                "extract_entities: dropping %s %r — template placeholder, "
                "not a resolvable indicator", entity_type, value,
            )
            continue
        if cleaned != value:
            logger.info(
                "extract_entities: %s %r -> %r (stripped placeholder)",
                entity_type, value, cleaned,
            )
            value = cleaned

        # Victim-side identifiers are context, never adversary IOCs.
        if _is_victim_side_identifier(value, entity_type):
            logger.info(
                "extract_entities: dropping %s %r — victim-side identifier",
                entity_type, value,
            )
            continue

        # Dedup by (value_lower, type)
        dedup_key = (value.lower(), entity_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Clamp confidence
        confidence = max(0.0, min(1.0, float(confidence)))

        entity_dict = {
            "entity_id": f"ent-{uuid.uuid4().hex[:8]}",
            "entity_type": entity_type,
            "value": value,
            "confidence": confidence,
            "source_location": {},  # Populated if we add offset tracking later
            "gate_action": None,
            "edited_value": None,
            "edited_type": None,
            "edit_rationale": None,
        }

        # Flag sub-threshold entities so gate_0 can default them to
        # remove (the denylist pattern) rather than shipping a guess. The
        # audit found `victim_sector: financial-services` at 0.3 on a report
        # that names no sector at all — the model was signaling doubt and
        # nothing downstream acted on it.
        if confidence < LOW_CONFIDENCE_FLOOR:
            entity_dict["low_confidence"] = True
            entity_dict["low_confidence_reason"] = (
                f"confidence {confidence:.2f} is below the {LOW_CONFIDENCE_FLOOR} "
                f"floor; the model signaled it was unsure"
            )

        # Preserve organization_role for organization entities. The "author"
        # value distinguishes the analyst team (e.g. Mandiant) from the
        # parent publisher (e.g. Google); a real-source walkthrough flagged
        # conflation between the two.
        if entity_type == EntityType.ORGANIZATION.value:
            org_role = raw.get("organization_role", "other")
            if org_role in ("victim", "sponsor", "publisher", "author", "other"):
                entity_dict["organization_role"] = org_role
            else:
                entity_dict["organization_role"] = "other"

        # Preserve location_role for location entities.
        # Default "context" to avoid false-attribution if LLM omits the field.
        if entity_type == EntityType.LOCATION.value:
            loc_role = raw.get("location_role", "context")
            if loc_role in ("victim", "origin", "context"):
                entity_dict["location_role"] = loc_role
            else:
                entity_dict["location_role"] = "context"

        # Preserve context_snippet for provenance
        snippet = raw.get("context_snippet", "")
        if snippet:
            entity_dict["context_snippet"] = snippet

        entities.append(entity_dict)

    return entities


def _looks_like_a_rule_body(content: str) -> bool:
    """True when `content` is plausibly a detection rule, not just its name.

    Vendor reports routinely list the *names* of the rules that catch an
    actor ("Okta Admin Console Access Failure", "O365 SharePoint High Volume
    File Access Events"). The extractor was accepting those as `rule_content`,
    and serialization then emitted them as STIX Indicators with
    `pattern_type: "sigma"` — a rule title presented to consumers as a Sigma
    pattern, which nothing can parse.

    A real rule body is multi-line or carries syntax markers. A short,
    title-cased phrase with no such markers is a name.
    """
    if "\n" in content:
        return True
    # Symbolic markers only. Matching the English words "and"/"or"/"not"
    # misfires on rule names that read as prose — "O365 SharePoint Bulk File
    # Access or Download via PowerShell" is a title, not a boolean expression.
    # Real rule bodies carry syntax: Sigma YAML has ":", KQL and SPL have "|"
    # and "=", YARA has "{".
    markers = (":", "{", "}", "|", "=", "(", ")", "$", "\\", "==", "!=", "->")
    if any(m in content for m in markers):
        return True
    # No structure and short enough to be a label.
    return len(content) > 120


# Sections whose content is not about what the adversary did. Detection
# logic and technique tables were already out of scope conceptually; the
# addition here is that they can finally be EXCLUDED, because
# classify_sections now runs before this node rather than inside the chunker
# downstream of it.
_ENTITY_EXCLUDED_SECTIONS = frozenset({
    # Vendor detection rules and mitigation guidance — the source of the
    # defensive products (Defender, SmartScreen, SecOps) that were being
    # typed as adversary tooling.
    SectionClassification.DETECTION_LOGIC.value,
    # ATT&CK mapping tables: technique IDs, not entities.
    SectionClassification.TECHNIQUE_REFERENCE.value,
    # NOT excluded: `metadata`. It looks like boilerplate, but the footer and
    # header carry the report's provenance — the publishing entity, the
    # authoring team, publication dates — which are legitimate CTI entities.
    # Excluding it dropped "Google" as publisher on one report, caught in
    # end-to-end validation after the unit tests were green.
})


def _text_for_entity_extraction(state: PipelineState) -> str:
    """The portion of the source worth mining for entities.

    Reading raw `parsed_text` — the whole document, including "Remediation
    and Hardening" advice and vendor detection-rule listings — produced six
    DEFENSIVE products on one campaign source (Microsoft Defender,
    SmartScreen, Google Workspace, Password Alert, Google SecOps, Google Safe
    Browsing) typed as adversary tools and targeted software,
    indistinguishable in the bundle from Okta and SharePoint.

    Falls back to the full text whenever filtering would leave nothing —
    an empty string here hard-fails the source, and a missing or unhelpful
    classification must not cost us the whole run.
    """
    parsed_text = state.get("parsed_text", "") or ""
    sections = state.get("classified_sections") or []
    if not sections:
        return parsed_text

    kept = [
        sec.get("text", "")
        for sec in sections
        if sec.get("classification") not in _ENTITY_EXCLUDED_SECTIONS
    ]
    filtered = "\n\n".join(t for t in kept if t.strip())
    if not filtered.strip():
        logger.warning(
            "extract_entities: section filter left no text; using full document",
        )
        return parsed_text

    dropped = len(sections) - len([t for t in kept if t.strip()])
    if dropped:
        logger.info(
            "extract_entities: skipping %d non-behavioral section(s) "
            "(%d -> %d chars)",
            dropped, len(parsed_text), len(filtered),
        )
    return filtered


def _process_detection_rules(raw_rules: list[dict]) -> list[dict]:
    """Convert raw LLM output to DetectionRule-shaped dicts.

    - Assigns unique rule_id (uuid4)
    - Validates rule_type against DetectionRuleType enum
    - Skips rules with empty content
    - Skips rule NAMES masquerading as rule bodies (see _looks_like_a_rule_body)
    """
    valid_types = {r.value for r in DetectionRuleType}
    rules: list[dict] = []

    for raw in raw_rules:
        rule_type = raw.get("rule_type", "")
        rule_content = raw.get("rule_content", "").strip()
        description = raw.get("description", "")

        if not rule_content:
            continue

        if not _looks_like_a_rule_body(rule_content):
            logger.info(
                "extract_entities: skipping detection rule whose content is a "
                "NAME rather than a rule body: %r",
                rule_content[:80],
            )
            continue

        if rule_type not in valid_types:
            logger.warning(
                "extract_entities: skipping rule with invalid type '%s'",
                rule_type,
            )
            continue

        rules.append({
            "rule_id": f"rule-{uuid.uuid4().hex[:8]}",
            "rule_type": rule_type,
            "rule_content": rule_content,
            "description": description,
            "source_location": {},
        })

    return rules
