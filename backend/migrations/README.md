# Migrations

There is no migration runner. A fresh database gets its complete schema at API
startup: `Base.metadata.create_all` creates every table, and a small loop in
`backend/app/main.py` issues `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for any
model column an existing table lacks. That covers every schema change a fresh
install or a normal upgrade needs.

The SQL files that used to live here were one-off data backfills for databases
that predated a change. They are not needed on a new install and are not
shipped in this repository.
