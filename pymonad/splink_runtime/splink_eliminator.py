"""Splink runtime eliminator."""
# The Run eliminator intentionally evaluates Run internals (_step/_perform) while interpreting intents.
# pylint: disable=protected-access
from __future__ import annotations

from typing import Any, cast

from ..environment import DbBackend, Environment
from ..run import ErrorPayload, HasSplinkLinker, Run, ask, get_splink_linker, put_splink_linker, throw
from .splink_engine import run_splink_dedupe_monadic
from .splink_model_store import load_splink_model, save_splink_model
from .splink_types import SplinkDedupeJob, SplinkVisualizeJob
from .splink_visualize import run_splink_visualize

A = Any


def run_splink(prog: Run[A]) -> Run[A]:
    def step(self_run: Run[Any]) -> A:
        parent = self_run._perform

        def perform(intent: Any, current: Run[Any]) -> Any:
            match intent:
                case SplinkDedupeJob():
                    env = cast(Environment, ask()._step(current))
                    con = env["connections"].get(DbBackend.DUCKDB)
                    if con is None:
                        return throw(ErrorPayload("Splink requires a DuckDB connection."))._step(current)
                    out = run_splink_dedupe_monadic(intent)._step(current)
                    put_splink_linker(intent.splink_key, out[0])._step(current)
                    save_splink_model(intent.splink_key, out[0], intent.input_table)
                    return out
                case SplinkVisualizeJob():
                    linker = get_splink_linker(intent.splink_key)._step(current)
                    if linker is None:
                        return throw(ErrorPayload("No Splink linker stored for visualization."))._step(current)
                    return run_splink_visualize(linker, intent)
                case HasSplinkLinker(key):
                    linker = get_splink_linker(key)._step(current)
                    if linker is not None:
                        return True
                    env = cast(Environment, ask()._step(current))
                    con = env["connections"].get(DbBackend.DUCKDB)
                    if con is None:
                        return False
                    reloaded = load_splink_model(key, con)
                    if reloaded is None:
                        return False
                    put_splink_linker(key, reloaded)._step(current)
                    return True
                case _:
                    return parent(intent, current)

        inner = Run(prog._step, perform)
        return inner._step(inner)

    return Run(step, lambda i, c: c._perform(i, c))


__all__ = ["run_splink"]
