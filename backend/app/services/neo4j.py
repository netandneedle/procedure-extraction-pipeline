"""Neo4j connection and query helpers.

All Cypher queries go through this module. Nodes never construct
Cypher inline. This keeps Neo4j access testable and centralized.
"""

from neo4j import AsyncGraphDatabase

from app.config import settings

_driver = None


async def get_driver():
    """Get or create the Neo4j async driver."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def run_query(query: str, parameters: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return results as dicts."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        return [record.data() async for record in result]


async def write_transaction(queries: list[dict]) -> int:
    """Execute a batch of Cypher write queries in a single transaction.

    Each query dict must have 'query' (Cypher string) and 'params' (dict).
    All queries succeed or all are rolled back.

    Returns the number of queries executed.
    """
    if not queries:
        return 0

    driver = await get_driver()
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            for q in queries:
                await tx.run(q["query"], q["params"])
            await tx.commit()
            return len(queries)
        except Exception:
            await tx.rollback()
            raise


async def write_transaction_counted(queries: list[dict]) -> list[int]:
    """Like `write_transaction`, but returns the row count each query produced.

    A Cypher `MATCH a, b MERGE (a)-[r]->(b)` whose endpoints do not exist
    returns zero rows and does nothing. No error, no warning, the transaction
    commits. That is how a relationship can vanish between a validated bundle
    and the graph with nothing anywhere recording it. Give those queries a
    `RETURN` and count the rows, and the silence becomes a number.

    Kept separate from `write_transaction` rather than changing its signature,
    because scripts/one_off_distribute.py calls that one.
    """
    if not queries:
        return []

    driver = await get_driver()
    counts: list[int] = []
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            for q in queries:
                result = await tx.run(q["query"], q["params"])
                counts.append(len([r async for r in result]))
            await tx.commit()
            return counts
        except Exception:
            await tx.rollback()
            raise


async def close():
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
