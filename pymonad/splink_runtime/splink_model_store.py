"""Splink runtime model persistence helpers."""
# Splink's model/load metadata requires linker internals; no public alternative for these fields.
# pylint: disable=protected-access
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from splink import DuckDBAPI, Linker

from .splink_tables import input_table_value
from .splink_types import PredictionInputTableNames


def _splink_model_paths(splink_key: Any) -> tuple[Path, Path]:
    key_str = str(splink_key).replace("/", "_")
    base_dir = Path("splink_models")
    model_path = base_dir / f"splink_model_{key_str}.json"
    meta_path = base_dir / f"splink_model_{key_str}.meta.json"
    return model_path, meta_path


def save_splink_model(
    splink_key: Any,
    linker: Linker,
    input_table: PredictionInputTableNames,
) -> None:
    """Persist a trained Splink model and metadata for later reload."""
    if splink_key is None:
        return
    model_path, meta_path = _splink_model_paths(splink_key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    linker.misc.save_model_to_json(str(model_path), overwrite=True)
    actual_input_table: str | list[str] = input_table_value(input_table)
    try:
        input_tables = list(linker._input_tables_dict.values())
        if len(input_tables) == 1:
            actual_input_table = input_tables[0].physical_name
        elif len(input_tables) > 1:
            actual_input_table = [df.physical_name for df in input_tables]
    except Exception:  # pylint: disable=W0718
        actual_input_table = input_table_value(input_table)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"input_table": actual_input_table}, f, indent=2)


def load_splink_model(
    splink_key: Any,
    con: duckdb.DuckDBPyConnection,
) -> Linker | None:
    """Load a previously saved Splink model from disk when available."""
    if splink_key is None:
        return None
    model_path, meta_path = _splink_model_paths(splink_key)
    if not model_path.is_file() or not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        input_table = meta.get("input_table")
        if input_table is None:
            return None
        db_api = DuckDBAPI(connection=con)
        return Linker(input_table, str(model_path), db_api=db_api)
    except Exception:  # pylint: disable=W0718
        return None
