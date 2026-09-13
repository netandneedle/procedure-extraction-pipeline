"""Remove one report's contribution from the Neo4j graph.

WHY THIS EXISTS
Without provenance a write to the graph is irreversible: if nothing records
which report contributed which node, "undo that ingest" has exactly one answer
— wipe the database and reload the ATT&CK catalogue, losing every other report
with it.

`distribute` writes `(:Report)-[:DESCRIBES]->(object)` for everything a
bundle contributed, which makes the delete set a traversal.

TWO GUARDS, BOTH LOAD-BEARING

  x_ingested_by = 'pipeline'
      A report legitimately DESCRIBES the ATT&CK techniques it covers, and
      those nodes are the catalogue's. Deleting by DESCRIBES alone would take
      a technique out of ATT&CK because one report happened to be the only one
      citing it. Only nodes this pipeline created are ever deleted.

  described by exactly one report
      Two reports can contribute the same SCO — the same file hash, the same
      domain. Removing one report must not delete a node the other still
      needs, so a node shared with another report is kept and only its
      DESCRIBES edge is cut.

USAGE (inside the api container)
    python -m scripts.undo_graph_source --list
    python -m scripts.undo_graph_source --report-id report--<uuid> --dry-run
    python -m scripts.undo_graph_source --report-id report--<uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import settings
from app.services.neo4j import run_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("undo_graph_source")

LIST_CYPHER = """
MATCH (r:Report)
OPTIONAL MATCH (r)-[:DESCRIBES]->(n)
RETURN r.stix_id AS stix_id, r.name AS name, count(n) AS describes
ORDER BY r.name
"""

# Split deliberately: `owned` is what would be deleted, `shared` is what is
# kept because another report also points at it, and `catalogue` is what is
# never touched. Reporting the three separately is the difference between an
# undo you can check and one you have to trust.
PREVIEW_CYPHER = """
MATCH (r:Report {stix_id: $report_id})-[:DESCRIBES]->(n)
WITH n,
     // coalesce, not a bare comparison: x_ingested_by is NULL on catalogue
     // nodes, `NULL = 'pipeline'` is NULL rather than false, and
     // `CASE WHEN NULL` counts nothing — so the first version of this preview
     // reported 0 catalogue nodes while correctly protecting 241 of them.
     coalesce(n.x_ingested_by, '') = 'pipeline' AS ours,
     size([(n)<-[:DESCRIBES]-(x:Report) | x]) AS reports
RETURN
  count(CASE WHEN ours AND reports = 1 THEN 1 END) AS owned,
  count(CASE WHEN ours AND reports > 1 THEN 1 END) AS shared,
  count(CASE WHEN NOT ours THEN 1 END) AS catalogue
"""

DELETE_CYPHER = """
MATCH (r:Report {stix_id: $report_id})-[:DESCRIBES]->(n)
WHERE n.x_ingested_by = 'pipeline'
  AND size([(n)<-[:DESCRIBES]-(x:Report) | x]) = 1
DETACH DELETE n
RETURN count(*) AS deleted
"""

# The Report itself is not in its own object_refs, so it survives the sweep
# above and has to be removed explicitly — after its contents, or the
# single-report test on those contents stops being true.
DELETE_REPORT_CYPHER = """
MATCH (r:Report {stix_id: $report_id})
WHERE r.x_ingested_by = 'pipeline'
DETACH DELETE r
RETURN count(*) AS deleted
"""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-id", help="stix_id of the Report to remove")
    ap.add_argument("--list", action="store_true", help="list ingested reports")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, delete nothing")
    args = ap.parse_args()

    if args.list:
        rows = await run_query(LIST_CYPHER)
        if not rows:
            logger.info("no reports in the graph")
        for r in rows:
            logger.info("%s  %-46s  %d objects",
                        r["stix_id"], (r["name"] or "")[:46], r["describes"])
        return 0

    if not args.report_id:
        logger.error("--report-id is required (or --list)")
        return 2

    preview = (await run_query(PREVIEW_CYPHER, {"report_id": args.report_id}))
    if not preview or not any(preview[0].values()):
        logger.error("no report %s in the graph, or it describes nothing",
                     args.report_id)
        return 1
    p = preview[0]
    logger.info("would delete %d node(s) this pipeline created and no other "
                "report describes", p["owned"])
    logger.info("would keep   %d node(s) shared with another report", p["shared"])
    logger.info("would keep   %d catalogue-owned node(s) (ATT&CK's, not ours)",
                p["catalogue"])

    if args.dry_run:
        logger.info("DRY RUN: nothing deleted")
        return 0

    if not settings.neo4j_writes_enabled:
        logger.error("NEO4J_WRITES_ENABLED is false; re-run with it set, or "
                     "use --dry-run")
        return 1

    deleted = (await run_query(DELETE_CYPHER, {"report_id": args.report_id}))[0]
    report = (await run_query(DELETE_REPORT_CYPHER, {"report_id": args.report_id}))[0]
    logger.info("deleted %d object(s) + %d report node",
                deleted["deleted"], report["deleted"])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
