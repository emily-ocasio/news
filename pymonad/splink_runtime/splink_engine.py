"""Splink runtime orchestration engine."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import cast

import duckdb
from splink import DuckDBAPI, Linker

from ..array import Array
from ..maybe import Just
from ..monad import Unit, unit
from ..run import ErrorPayload, Run, ask, put_splink_context, pure, throw
from ..runsql import with_duckdb
from ..environment import DbBackend, Environment

from .splink_cluster import run_unique_matching_and_cluster_from_ctx
from .splink_context import (
    context_replace,
    tables_get_optional,
    tables_get_required,
    update_splink_context,
    with_splink_context,
    with_splink_context_linker,
)
from .splink_predict import splink_predict_pairs_from_ctx
from .splink_tables import (
    add_all_tables,
    add_dedupe_tables,
    add_link_type_tables,
    add_unique_matching_tables,
    validate_splink_dedupe_input_tables,
)
from .splink_types import (
    ResultClustersTableName,
    ResultPairsTableName,
    SplinkContext,
    SplinkDedupeJob,
    SplinkLinkType,
    SplinkPhase,
    SplinkPredictResult,
    SplinkTableNames,
    UniqueIdColumnName,
)


def _init_splink_dedupe_context(job: SplinkDedupeJob) -> Run[Unit]:
    unique_id_col = UniqueIdColumnName(cast(str, job.settings.get("unique_id_column_name", "unique_id")))
    prediction_rules = Array.make(tuple(job.settings.get("blocking_rules_to_generate_predictions", [])))
    training_rules = Array.make(tuple(job.training_blocking_rules if job.training_blocking_rules else prediction_rules))
    link_type = SplinkLinkType.from_settings(job.settings)
    capture_blocked_edges = (
        link_type == SplinkLinkType.DEDUPE_ONLY
        and job.capture_blocked_edges
    )

    def _build_tables() -> SplinkTableNames:
        tables = add_all_tables(SplinkTableNames.empty(), job.pairs_out, job.input_table)
        tables = add_dedupe_tables(
            tables,
            link_type,
            job.clusters_out,
            job.do_not_link_table,
            job.blocked_pairs_out,
            job.pairs_out,
        )
        tables = add_unique_matching_tables(tables, job.unique_matching, job.unique_pairs_table)
        tables = add_link_type_tables(tables, link_type, job.pairs_out)
        return tables

    def _finish(tables: SplinkTableNames) -> Run[Unit]:
        ctx = SplinkContext(
            phase=SplinkPhase.INIT,
            tables=tables,
            unique_id_col=unique_id_col,
            prediction_rules=prediction_rules,
            training_rules=training_rules,
            training_block_level_map=job.training_block_level_map,
            settings=job.settings,
            predict_threshold=job.predict_threshold,
            cluster_threshold=job.cluster_threshold,
            deterministic_rules=job.deterministic_rules,
            deterministic_recall=job.deterministic_recall,
            train_first=job.train_first,
            skip_u_estimation=job.skip_u_estimation,
            visualize=job.visualize,
            unique_matching=job.unique_matching,
            em_max_runs=job.em_max_runs,
            em_min_runs=job.em_min_runs,
            em_stop_delta=job.em_stop_delta,
            capture_blocked_edges=capture_blocked_edges,
            do_not_link_left_col=job.do_not_link_left_col,
            do_not_link_right_col=job.do_not_link_right_col,
        )
        return put_splink_context(ctx)

    return validate_splink_dedupe_input_tables(
        input_tables=job.input_table,
        link_type=link_type,
        clusters_out=job.clusters_out,
        unique_matching=job.unique_matching,
        unique_pairs_table=job.unique_pairs_table,
        blocked_pairs_out=job.blocked_pairs_out,
        do_not_link_table=job.do_not_link_table,
    ) ^ _finish(_build_tables())


def _configure_splink_logger_step() -> Run[Unit]:
    def _configure(name: str) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(15)
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(15)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False

    def _configure_all(_: Unit) -> Unit:
        _configure("splink")
        _configure("splink.internals")
        return unit

    return _configure_all & pure(unit)


def _load_splink_duckdb_step() -> Run[Unit]:
    def _with_env(env: Environment) -> Run[Unit]:
        con = env["connections"].get(DbBackend.DUCKDB)
        if con is None:
            return throw(ErrorPayload("Splink requires a DuckDB connection."))

        def _update(ctx: SplinkContext) -> SplinkContext:
            return replace(
                ctx,
                phase=SplinkPhase.PREPARE,
                db_api=Just(DuckDBAPI(connection=cast(duckdb.DuckDBPyConnection, con))),
            )

        return update_splink_context(_update)

    return ask() >> _with_env


def _splink_dedupe_predict_pairs() -> Run[Unit]:
    def _store_result(result: SplinkPredictResult) -> Run[Unit]:
        return context_replace(linker=Just(result.linker), phase=SplinkPhase.PREDICT)

    return with_splink_context(splink_predict_pairs_from_ctx) >> _store_result


def _splink_dedupe_finalize(
    ctx: SplinkContext,
    linker: Linker,
) -> Run[tuple[Linker, str, str]]:
    def _with_pairs(pairs_table: ResultPairsTableName) -> Run[tuple[Linker, str, str]]:
        clusters_table = tables_get_optional(ctx.tables, ResultClustersTableName)
        return context_replace(phase=SplinkPhase.DONE) ^ pure((linker, str(pairs_table), str(clusters_table)))

    return tables_get_required(ctx.tables, ResultPairsTableName) >> _with_pairs


def run_splink_dedupe_monadic(job: SplinkDedupeJob) -> Run[tuple[Linker, str, str]]:
    chain = (
        _init_splink_dedupe_context(job)
        ^ _configure_splink_logger_step()
        ^ _load_splink_duckdb_step()
        ^ _splink_dedupe_predict_pairs()
        ^ with_splink_context(run_unique_matching_and_cluster_from_ctx)
        ^ with_splink_context_linker(_splink_dedupe_finalize)
    )
    return with_duckdb(chain)
