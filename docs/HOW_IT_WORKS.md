# How it works

This is the backend, end to end, for someone who has not opened the code.
It follows one report — the CISA advisory that ships in `data/samples/` — from
upload to a finished STIX bundle, and at every stage says what goes in, what
comes out, whether a model was involved, and what the analyst sees.

The reference version of this document, with file paths, state fields and
routing tables, is [ARCHITECTURE.md](ARCHITECTURE.md). How the system learns
from corrections is its own document, [FEEDBACK_FLYWHEEL.md](FEEDBACK_FLYWHEEL.md).

## 1. The object

Threat intelligence has a good vocabulary for *what* adversaries do. MITRE
ATT&CK's technique T1059.001 says "adversaries abuse PowerShell". That is true
of thousands of unrelated intrusions, which is why it is nearly useless to a
defender on its own. What a defender needs is the *how*: a named behaviour, the
actual command line, the parent process, the platform, the tactic it served,
and a citation to the report that described it.

That "how" is what this pipeline extracts. It is modelled as a STIX 2.1
extension object called `x-procedure`, and it is the only thing in the output
that is not already a standard STIX type. Every bundle embeds the
`extension-definition` that declares it, so a STIX consumer can tell what the
object is without reading this repository.

A procedure has three parts that all have to be present for the object to be
well-formed:

- **Techniques** — the ATT&CK technique IDs it implements. Usually one, often
  two or three, because a real procedure routinely spans tactics: "deploy and
  operate a remote-access tool" is command-and-control *and* persistence.
- **Log sources** — where a defender would see it.
- **Components** — the observables that constitute it: a Process carrying the
  command line, the File it dropped, the registry key it wrote. Ordered,
  because sequence carries meaning.

Two design rules follow from that and shape everything downstream.

**Every procedure is an instance, not a template.** The same behaviour reported
by two vendors becomes two objects with the same name, different IDs and
different source references. Deduplication happens at query time, through a
computed fingerprint (a hash of the sorted techniques, platforms and tactics),
never at write time. The pipeline never destroys "who said this, and when" to
save a row.

**No fabricated detail.** A thin source produces a low-confidence procedure,
not an invented command line. Fabricated detail is worse than missing detail
because it looks like evidence. Confidence here means "how completely have we
decomposed this", not "how strongly does the model feel about it".

> **A real one, from the sample run.** `Automate Ransomware Deployment via Batch Scripts` — techniques T1059.003 Windows Command Shell, T1136.001 Local Account, T1484.001 Group Policy Modification, T1112 Modify Registry, T1685.005 Clear Windows Event Logs, T1027.013 Encrypted/Encoded File; platforms Windows; confidence 78 (source reliability 80 blended with the model's behavioural confidence and a still-empty component set, which is why it is not higher). Its description opens: "Automate ransomware deployment and execution while establishing a backup admin account and erasing event logs to evade detection and hinder incident response. R…"


## 2. The shape of a run

The pipeline is one state machine, built with LangGraph. Sixteen nodes run in
order; each reads a single shared state object and returns the fields it
changed. After every node the whole state is written to Postgres as a
checkpoint.

That checkpoint is why the human gates work the way they do. The graph is
compiled to *pause before* each of the four gate nodes. When it pauses, the
state is already durable: the API process can restart, the browser tab can be
closed, and review can take five minutes or five days. When the analyst
submits, the API writes the decisions into the checkpoint and streams the graph
again from exactly where it stopped. The gate node then runs, applies the
decisions, and the pipeline continues.

Seven of the sixteen nodes call a language model. The rest are ordinary Python,
and the dividing line is deliberate: deterministic wherever possible, a model
only where judgement is genuinely needed. Several things people assume are AI
are not — technique ID validation, the brand map, fingerprinting, denylist
enforcement and every piece of STIX validation are plain code.

Every model call goes through one adapter that is provider-neutral. The
default is Anthropic; any OpenAI-compatible endpoint works too. Responses are
cached by a hash of the full request, so re-running a stage with the same
inputs is free, and prompt edits automatically invalidate the cache because
the prompt is part of the key.

## 3. Stage by stage

The stages below are in pipeline order. Gates are numbered as the UI numbers
them (0 to 3); the backend's own node names differ for historical reasons and
are given in brackets.

### Parse

**In:** the uploaded file. **Out:** the document as Markdown, plus a stash of
every figure rendered to an image. **Model:** none.

PDFs, HTML and DOCX go through Docling, which produces clean Markdown with
heading structure intact and leaves an `<!-- image -->` placeholder where each
figure was. The figures are rendered once here and written to a per-source
stash, because the next stage needs them and converting a PDF twice costs a
minute or more.

*What can go wrong:* a scanned PDF with no text layer produces almost nothing;
the run fails here with a parse error rather than proceeding on an empty
document.

### Read the figures

**In:** the figure stash. **Out:** the Markdown with each placeholder replaced
by the figure's transcribed content. **Model:** yes, a vision call per figure.

Vendor reports put their best material in pictures: attack-chain diagrams,
command-line screenshots, process trees. Text extraction drops all of it. This
stage sends each figure to the model with a small tool schema, gets back a
classification (diagram, screenshot, decorative, other) and a transcription,
and splices the transcription into the document inside `[FIGURE …]` markers so
later stages can tell figure-derived text from prose. Decorative figures are
removed; a figure the model could not read keeps its placeholder as a visible
signal. The stage can be switched off per source.

*What can go wrong:* a dense screenshot can exhaust the per-figure token
budget; the figure is then marked failed and the run continues without it.

### Classify sections

**In:** the Markdown. **Out:** each section labelled by function — behavioural
narrative, indicator list, mitigation advice, boilerplate. **Model:** yes.

The advisory's "Technical Details" section is where the attack is described;
its "Mitigations" section is advice to defenders and would poison extraction
if treated as adversary behaviour. This stage is a recall tool: when unsure it
labels a section as narrative and lets the chunker decide.

### Extract entities

**In:** the classified document. **Out:** a typed list of named things —
intrusion sets, malware, tools, organisations, locations, vulnerabilities —
plus one judgement about the whole document. **Model:** yes.

The document-level judgement is whether the source is *sequential*: an incident
report describes events in order; a threat-actor profile is a catalogue with no
order at all. That single boolean decides, three stages later, whether the
pipeline is allowed to build a kill chain. The model is told to prefer "not
sequential" when unsure, because a false yes invents structure that looks
real, while a false no leaves disconnected chunks the analyst spots at once.
The analyst can override it at upload.

Two guardrails run here without a model. Terms an analyst has previously
confirmed as never wanted (a vendor's own contact address, say) are marked for
removal. And brand names for social-engineering techniques — "ClickFix", "MFA
fatigue" — are explicitly *not* extracted as malware or tools; they come back
later as technique candidates instead.

### Gate 0 — entities

The analyst sees every entity with its type and, for organisations and
locations, its role: author, publisher, sponsor, victim, origin. They can edit,
remove, add, or correct a role. Denylisted entities arrive pre-marked for
removal with a badge and can be flipped back for this source only. Nothing
downstream has run yet, so this is cheap to get right.

### Chunk into procedures

**In:** the document and the entities. **Out:** a list of chunks, each a
candidate procedure, with a predecessor graph between them. **Model:** yes, and
this is the hardest call in the pipeline.

The question the chunker answers is not "is this a different action" but "is
this a different *objective*". Several actions that serve one goal are one
chunk; the moment the goal changes, a new chunk starts. Each chunk carries a
verbatim excerpt from the source and the byte offsets where that excerpt lives,
so the review UI can highlight exactly what the chunk was built from. Each
chunk also names its predecessors, which becomes the kill chain.

The whole document goes to the model in one call up to a generous size limit.
An earlier design split long documents into windows and stitched the numbering
afterwards; the model could not tell whether it was numbering globally or
locally, so sequences collided and predecessor references dangled. Seeing the
whole kill chain at once fixed all of it. The windowed path was removed: a
document too large for one call fails loudly rather than mis-sequencing.

When the source was judged non-sequential, the model is told to leave the
predecessor links empty unless the text explicitly states an order, and the
backstop that links orphaned chunks to their neighbour is switched off. A
catalogue of twenty unrelated procedures gets no invented kill chain.

*What can go wrong:* zero chunks is a hard failure, not an empty success. The
run stops here rather than flowing into a gate that would auto-approve an
empty list.

### Gate 1 — procedures

A three-pane canvas: the source text on the left with every chunk's excerpt
highlighted, the chunk graph in the middle, a field editor on the right. Click
a highlight and the graph centres on that chunk. Drag between nodes to add an
ordering edge; select an edge and press Delete to remove it. The analyst can
edit a chunk's text or excerpt, drop it, add one the model missed, and set
branch and convergence flags where the attack forks or rejoins.

Rejecting sends the run back to the chunker with the analyst's reason injected
into the next prompt as highest-priority guidance, so the rerun actually
produces something different.

### Map techniques

**In:** each chunk. **Out:** for each chunk, the ATT&CK techniques it
implements, each with a confidence bucket and a verbatim quote. **Model:** yes,
twice per chunk, with deterministic steps between.

This stage exists because asking a model for a technique ID directly returns
real, retired and imaginary IDs with nothing to distinguish them. So three
independent sources produce *candidates*, and the catalogue decides what is
real:

1. **Propose.** The model names techniques from memory, with no catalogue in
   front of it. This is what catches a technique added after the retriever's
   embedding was built.
2. **Retrieve.** A retriever ranks the real ATT&CK catalogue against the chunk
   text — semantically by default, using a security-domain embedding model,
   or lexically if that model is not installed.
3. **Brand map.** A small table maps marketing names to technique IDs, for
   reports that say "deployed a ClickFix lure" without ever describing the
   mechanism.

Everything proposed is validated against the catalogue: hallucinated IDs are
dropped, revoked ones redirected to their replacement, deprecated ones
removed. What survives becomes one pool, and it is a union, not a
replacement: the model widens the search, the catalogue decides what exists.

Then a second model call **picks** from the pool. Every pick carries a bucket
(definite, probable, possible), a five-to-thirty-word quote copied from the
chunk, and a rationale. If the quote is not literally present in the chunk
text, the pick is automatically demoted to *possible* and its confidence
capped. The model does not get to grade its own evidence. Definite and probable
picks go into the bundle; possible picks go to a review lane where the analyst
can promote them.

### Draft procedures

**In:** the chunks with their techniques. **Out:** a draft `x-procedure` per
chunk — name in the form *[Verb] [Object] via [Tool]*, a description that leads
with the objective, platforms, the command lines the source actually contains,
and the observables. **Model:** yes.

Names carry no actor: attribution belongs on graph edges, and the moment an
actor's name is in the string, the same behaviour from a different actor
becomes a different object.

### Gate 2 — techniques

Each draft is a card with its picked techniques and a "review for inclusion"
lane holding the possible-bucket picks. The verdicts say what they do, because
the analyst needs to know the blast radius before clicking: **Approve**,
**Discard procedure**, **Re-map techniques** (re-runs the mapping stage), and
**Re-chunk** (re-runs chunking, because if the boundary was wrong the mapping
was always going to be wrong too). Within a card the analyst edits the
technique list or promotes a possible pick with one click. Every correction is
recorded durably; it is the primary thing the feedback loop learns from.

### Normalize

**In:** the approved drafts. **Out:** the same drafts with STIX identifiers,
fingerprints, a blended confidence, the resolved sequence, and a preview of
every relationship the bundle will contain. **Model:** none.

Confidence is a weighted blend: how much the analyst trusts the publisher, how
confident the model was in the behaviour, and how complete the component set
is. Sequencing is resolved here from the chunk predecessor graph into
`PRECEDES` links, and where the graph forks or rejoins, Attack Flow operator
objects (AND, OR, XOR) are materialised. Where the source described a runtime
check ("if domain-joined, Kerberoast; otherwise NTLM relay"), that becomes an
Attack Flow condition.

### Gate 3 — bundle

The last look before the bundle ships, as a graph. The Flow tab shows only the
procedures and their order, left to right: the kill chain at a glance. The
Bundle tab shows every object and every relationship, one procedure in focus
at a time, with layers that can be hidden. The analyst can change a
relationship's type, remove one, or draw a new one. Rejecting re-runs
normalization.

### Serialize

**In:** everything approved. **Out:** a STIX 2.1 bundle. **Model:** none.

The bundle is self-contained: the procedures, the ATT&CK techniques they
reference, the actors, malware, tools and victims from the entity list, the
observables, a report object that lists everything, and the relationships
between them (no TLP marking is attached today; see the known gaps in
[X_PROCEDURE.md](X_PROCEDURE.md)). `PRECEDES` links and the Attack Flow object
that names the chain's starting points are emitted only when the source was
judged sequential.

### Validate

**In:** the bundle. **Out:** the bundle, possibly corrected, or a failure.
**Model:** none.

Validation is all-or-nothing: the schema for every object type, reference
integrity (no relationship may point at an object that is not there), and
Attack Flow integrity all pass, or the run fails. Recoverable problems —
a reversed relationship, a duplicate, a malformed enum — are repaired
automatically, and every repair is written to a list the analyst sees on the
source card, because silent repair is just a slower bug. Real violations mark
the source Failed and nothing is distributed.

### Distribute

**In:** a valid bundle. **Out:** the bundle stored in Postgres, and, if graph
writes are enabled, the procedures and their relationships written to Neo4j.
**Model:** none.

Graph writes are off by default. When on, the pipeline never writes an ATT&CK
object or an edge between two ATT&CK objects: the catalogue belongs to its
loader, and a pipeline run must never overwrite MITRE's own dates and names.
Every node the pipeline does write carries a marker, and a report-to-object
provenance edge records what each run contributed. That edge is also the undo:
one script removes a single report's contribution and nothing else.

### Learn

**In:** every decision the analyst made at the four gates. **Out:** rules and
examples for the next run. **Model:** yes, one call.

This is the feedback loop, and it has its own document:
[FEEDBACK_FLYWHEEL.md](FEEDBACK_FLYWHEEL.md). The short version: corrections
are turned into reusable rules, only the rules relevant to the next report
are injected into its prompts, and each rule is then scored on whether it
actually prevented the correction it was meant to prevent.

## 4. The gates

Four gates, four questions:

| Gate | Question | A reject re-runs |
|---|---|---|
| 0 | Are these the right named things, with the right roles? | nothing (edits apply in place) |
| 1 | Are these the right procedures, in the right order? | chunking |
| 2 | Are these the right techniques for each procedure? | mapping, or chunking |
| 3 | Is this the right bundle? | normalization |

Rejection is never a dead end. The analyst's reason travels into the rerun's
prompt as the first thing the model reads. And when a gate can send work back
to more than one place, it goes to the stage that actually caused the problem:
a bad chunk boundary at Gate 2 re-runs chunking, not mapping, because
re-mapping a wrong chunk is wasted work.

Each gate can be switched off per source, in which case it auto-approves,
switched to **assist** mode, or set to **auto**, where the AI reviewer's
recommendation is applied unattended (see SECURITY.md before using that on a
source you do not trust). In assist mode an AI reviewer reads the report
once, carries its own reasoning from gate to gate, and posts recommendations
next to each item. Two rules keep that honest. Every quote in a recommendation
is checked against the source text, and an unsupported quote forces the
recommendation to low confidence, which keeps it out of bulk-accept. And every
submit records what the reviewer recommended against what the analyst
actually did, so the Reviewer tab can show, per gate and per confidence tier,
how often the advice was taken. That number is the only evidence that can say
whether a gate is ever safe to leave unattended, and nothing in the product
says it for you.

## 5. The discipline

Five rules explain most of the design decisions above.

- **Deterministic wherever possible.** A model call is a cost, a latency and a
  source of variance; it is spent only where judgement is needed.
- **Every model claim is anchored in the source.** Chunks carry their excerpt
  and its offsets; technique picks carry a quote that must be literally
  present. Evidence the model cannot point at is evidence the pipeline does not
  trust.
- **No fabricated command lines.** If the report does not contain it, the
  bundle does not either.
- **Fail loud.** Zero chunks is a failure. A validation violation stops the
  run. Repairs are logged where the analyst sees them.
- **The catalogue is read-only.** ATT&CK objects are re-emitted in bundles so
  each bundle stands alone, but they are never written to the graph, and the
  procedure-to-technique edges the pipeline does write get their own
  relationship type so they never blur into ATT&CK's own.
