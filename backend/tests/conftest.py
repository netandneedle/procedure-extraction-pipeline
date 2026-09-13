"""Test environment setup for the backend's internal pytest suite.

Mirrors the repo-root tests/conftest.py setup for the path/retriever overrides.
The backend pytest config has its own testpaths (backend/tests/), so it doesn't
pick up the repo-root conftest.
"""

from pathlib import Path

# Point Settings.attack_stix_path at the repo's local data dir so tests can
# load the ATT&CK catalogue. The default is /data/attack/... (the docker
# bind-mount path) which doesn't exist on the host running pytest.
_LOCAL_ATTACK_PATH = (
    Path(__file__).parent.parent.parent / "data" / "attack" / "enterprise-attack-19.2.json"
)
if _LOCAL_ATTACK_PATH.exists():
    from app.config import settings
    settings.attack_stix_path = str(_LOCAL_ATTACK_PATH)
    from app.services import attack_data as _attack_data_mod
    _attack_data_mod._instance = None
    from app.services import procedure_matcher as _matcher_mod
    _matcher_mod._index_cache = None

# Use the lightweight token_overlap retriever for tests. The default
# "embedding" retriever requires sentence-transformers + a ~570 MB
# SecureBERT 2.0 download, which is overkill for unit/integration tests.
from app.config import settings as _settings
_settings.technique_retriever = "token_overlap"

# Never let the unit suite reach Neo4j. `distribute` is gated on
# settings.neo4j_writes_enabled, which pydantic reads from .env — so the day
# that flag was set to true for real use, the suite started opening a live
# bolt connection and attempting real writes. It surfaced as an event-loop
# error rather than as pollution, which is luck: on a different ordering the
# write succeeds and `malware--001` and friends land in the graph.
#
# Deliberately not a fixture. distribute() reads the flag at call time, and
# module-level constants are built at import; forcing it here means no test
# can opt back in by accident.
_settings.neo4j_writes_enabled = False
