#!/usr/bin/env bash
# Download the ATT&CK Enterprise STIX bundle the pipeline is pinned to.
#
# The version must match `attack_stix_path` in backend/app/config.py AND the
# catalogue loaded into Neo4j by scripts/load_attack.py: the pipeline maps
# techniques from this file and writes edges to the graph's nodes, so the two
# have to agree. Re-run both after bumping the version.
set -euo pipefail

VERSION="${ATTACK_VERSION:-19.2}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/attack"
DEST="$DEST_DIR/enterprise-attack-${VERSION}.json"
# master, not a tag: attack-stix-data publishes no per-version tag for this
# path (ATT&CK-v19.2 returns 404). The file is versioned by its name instead.
URL="https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack-${VERSION}.json"

mkdir -p "$DEST_DIR"
if [ -s "$DEST" ]; then
  echo "already present: $DEST"
else
  echo "downloading ATT&CK Enterprise v${VERSION} (~54 MB) ..."
  curl -fL --progress-bar -o "$DEST.part" "$URL"
  mv "$DEST.part" "$DEST"
fi
python3 - "$DEST" <<'PY'
import json, sys
bundle = json.load(open(sys.argv[1]))
n = sum(1 for o in bundle.get("objects", []) if o.get("type") == "attack-pattern")
print(f"ok: {sys.argv[1]} ({n} attack-pattern objects)")
PY
echo
echo "Next, load it into Neo4j (needs the neo4j Python driver on the host):"
echo "  python3 -m pip install neo4j"
echo "  python3 scripts/load_attack.py --bundle \"$DEST\" --uri bolt://localhost:7687 --user neo4j --password \"\$NEO4J_PASSWORD\""
