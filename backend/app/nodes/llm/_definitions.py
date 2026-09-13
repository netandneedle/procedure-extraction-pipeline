"""Shared definitions for the LLM-stage nodes.

The procedure definition is the foundational concept the chunker, technique
extractor (propose + pick steps), and procedure drafter all anchor on. Keeping
it in one place means any revision propagates everywhere instead of drifting
across module-private constants.

This module deliberately holds *only* prompt-injectable prose, not Python
behavior. If you need to update the definition, edit the constant here.
"""

from __future__ import annotations


PROCEDURE_DEFINITION = """PROCEDURE — definition for this pipeline:
A procedure is a discrete, repeatable technical implementation that
integrates one or more techniques, often spanning multiple tactics, to
fulfill a specific adversarial objective as an atomic event within an
attack sequence.

Unpacking the parts:
- DISCRETE + ATOMIC: irreducible. If you can split a procedure into two
  parts that each serve a different objective, it is two procedures, not
  one. Atomicity is judged at the OBJECTIVE level, not at the action or
  tactic level.
- REPEATABLE: a procedure is a recipe / pattern, not a single
  observation. The same procedure executed five times against five hosts
  is ONE procedure with multiple sightings, not five procedures.
- TECHNICAL IMPLEMENTATION: the HOW. A procedure has structure — commands,
  observables, sequence of operations — not just a verb-event description.
- INTEGRATES ONE OR MORE TECHNIQUES, SPANNING MULTIPLE TACTICS: a single
  procedure routinely composes techniques across tactic boundaries (e.g.,
  a download-via-LOLBin spans command-and-control + defense-evasion;
  a copy-paste lure spans initial-access + execution).
- SPECIFIC ADVERSARIAL OBJECTIVE: the north-star that scopes the
  procedure. Techniques BELONG in the procedure if they serve the
  objective. Techniques DRIFT OUT if they serve a different objective —
  even if they appear in the same source paragraph.

(Note: ATT&CK's "procedure examples" are illustrations of techniques,
which is a different concept. Our procedure is a first-class object
with its own identity, objective, and observables.)
"""
