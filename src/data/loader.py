"""Load raw JSON tickets into DuckDB for dbt consumption."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src import config


def find_raw_json(explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw JSON not found: {path}")
        return path

    for candidate in config.RAW_JSON_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No raw ticket JSON found. Expected one of: "
        + ", ".join(str(path) for path in config.RAW_JSON_CANDIDATES)
    )


def load_json_to_duckdb(raw_json_path: Path, duckdb_path: Path) -> int:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("DROP TABLE IF EXISTS raw_tickets")
        con.execute(
            """
            CREATE TABLE raw_tickets AS
            SELECT * FROM read_json_auto(?);
            """,
            [str(raw_json_path)],
        )
        row_count = con.execute("SELECT COUNT(*) FROM raw_tickets").fetchone()[0]
        return int(row_count)
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load raw JSON tickets into DuckDB")
    parser.add_argument("--input", type=str, default=None, help="Path to tickets JSON")
    parser.add_argument(
        "--output-db",
        type=str,
        default=str(config.DUCKDB_PATH),
        help="Path to output DuckDB file",
    )
    args = parser.parse_args()

    config.ensure_directories()
    raw_json = find_raw_json(args.input)
    row_count = load_json_to_duckdb(raw_json, Path(args.output_db))

    print(f"Loaded {row_count} rows")
    print(f"Raw source: {raw_json}")
    print(f"DuckDB: {args.output_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
