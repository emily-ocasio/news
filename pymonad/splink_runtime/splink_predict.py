"""Splink runtime prediction and exclusion-list helpers."""
# Splink prediction rule mutation requires internal settings objects; no stable public accessor exists.
# pylint: disable=protected-access
from __future__ import annotations

from typing import Any, Callable, cast

from splink import DuckDBAPI, Linker
from splink.internals.blocking import BlockingRule
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.blocking_rule_creator_utils import to_blocking_rule_creator
from splink.internals.blocking_rule_library import And, CustomRule

from ..array import Array
from ..hashset import HashSet
from ..maybe import Just
from ..monad import Unit, unit
from ..run import Run, fold_run, pure
from ..runsql import SQL, sql_exec
from ..tuple import Tuple
from .splink_cluster import diagnostic_cluster_blocked_edges_run, resolve_pair_id_cols_from_table_step
from .splink_context import (
    _with_splink_context_api_plan,
    _with_splink_context_linker_plan,
    context_replace,
    maybe_get_required,
    predict_result_from_ctx,
    _require_db_api,
    _require_linker,
    _validate_predict_plan,
    validate_predict_tables,
    tables_get_optional,
    tables_get_required,
    with_splink_context,
)
from .splink_tables import input_table_value
from .splink_training import print_prediction_counts, train_linker_for_prediction
from .splink_types import (
    BlockedIdLeftColumnName,
    BlockedIdRightColumnName,
    BlockingRuleLike,
    BlockingRuleLikes,
    ClusterPairsTableName,
    DoNotLinkTableName,
    ExclusionInputTableName,
    PairsCaptureTableName,
    PairsSourceTableName,
    PairsTableName,
    PairLeftIdColumnName,
    PairRightIdColumnName,
    PredictPlan,
    PredictionInputTableName,
    PredictionInputTableNames,
    RetainColumnName,
    RetainColumnNames,
    CustomStringBlockingRule,
    SplinkContext,
    SplinkLinkType,
    UniqueIdColumnName,
    _concat_blocking_rule_likes,
)


def _build_linker_for_prediction(
    *,
    settings: dict[str, Any],
    db_api: DuckDBAPI,
    prediction_rules: BlockingRuleLikes,
    input_table: str | list[str],
    extra_columns_to_retain: RetainColumnNames,
) -> Linker:
    settings = dict(settings)
    if extra_columns_to_retain:
        existing = list(settings.get("additional_columns_to_retain", []))
        merged = list(dict.fromkeys(existing + list(map(str, extra_columns_to_retain))))
        settings["additional_columns_to_retain"] = merged
    return Linker(
        input_table,
        settings
        | {
            "retain_intermediate_calculation_columns": True,
            "max_iterations": 100,
            "blocking_rules_to_generate_predictions": list(prediction_rules),
        },
        db_api=db_api,
    )


def _set_diagnostic_plan(ctx: SplinkContext) -> Run[Unit]:
    input_table = ctx.tables.get_required(PredictionInputTableNames)
    plan = PredictPlan(
        prediction_rules=ctx.prediction_rules,
        training_rules=ctx.training_rules,
        input_table_for_prediction=input_table,
        extra_columns_to_retain=Array.empty(),
    )
    return context_replace(predict_plan=Just(plan))


def _append_clause_to_rules(rules: BlockingRuleLikes, clause: str) -> Array[BlockingRuleLike]:
    updated: list[BlockingRuleLike] = []
    for rule in rules:
        if isinstance(rule, BlockingRuleCreator):
            updated.append(And(rule, CustomRule(clause)))
        else:
            updated.append(CustomStringBlockingRule(f"({rule}) AND {clause}"))
    return Array.make(tuple(updated))


def _set_final_plan(ctx: SplinkContext) -> Run[Unit]:
    input_table = ctx.tables.get_required(PredictionInputTableNames)
    if ctx.capture_blocked_edges:
        exclusion_clause = (
            f"NOT (list_contains(l.exclusion_ids, r.{ctx.unique_id_col}) "
            f"OR list_contains(r.exclusion_ids, l.{ctx.unique_id_col}))"
        )
        plan = PredictPlan(
            prediction_rules=_append_clause_to_rules(ctx.prediction_rules, exclusion_clause),
            training_rules=_append_clause_to_rules(ctx.training_rules, exclusion_clause),
            input_table_for_prediction=PredictionInputTableNames(f"{input_table}_exc"),
            extra_columns_to_retain=Array.pure(RetainColumnName("exclusion_ids")),
        )
    else:
        plan = PredictPlan(
            prediction_rules=ctx.prediction_rules,
            training_rules=ctx.training_rules,
            input_table_for_prediction=input_table,
            extra_columns_to_retain=Array.empty(),
        )
    return context_replace(predict_plan=Just(plan))


def _build_capture_rules(
    training_rules: BlockingRuleLikes,
    prediction_rules: BlockingRuleLikes,
) -> Array[BlockingRuleLike]:
    capture_rules: list[BlockingRuleLike] = []
    seen_rules: set[str] = set()
    for rule in _concat_blocking_rule_likes(training_rules, prediction_rules):
        if isinstance(rule, BlockingRuleCreator):
            capture_rules.append(rule)
            continue
        if rule in seen_rules:
            continue
        capture_rules.append(rule)
        seen_rules.add(rule)
    return Array.make(tuple(capture_rules))


def _capture_blocked_edges_validate_tables(ctx: SplinkContext) -> Run[Unit]:
    return (
        tables_get_required(ctx.tables, PredictionInputTableNames)
        ^ tables_get_required(ctx.tables, PairsTableName)
        ^ pure(unit)
    )


def _capture_blocked_edges_validate(_: SplinkContext) -> Run[Unit]:
    return (
        with_splink_context(_require_db_api)
        ^ with_splink_context(_validate_predict_plan)
        ^ with_splink_context(_require_linker)
        ^ with_splink_context(_capture_blocked_edges_validate_tables)
    )


def _with_temp_blocking_rules_on_linker(
    linker: Linker,
    rules: Array[BlockingRuleLike],
    action: Callable[[], Any],
) -> Run[Any]:
    def _with_preceding_rules(dialected: Array[BlockingRule]) -> Run[Array[BlockingRule]]:
        def _step(acc: Array[BlockingRule], rule: BlockingRule) -> Run[Array[BlockingRule]]:
            rule.add_preceding_rules(list(acc.a))
            return pure(Array.snoc(acc, rule))

        return fold_run(dialected, Array.empty(), _step)

    def _run() -> Run[Any]:
        settings_obj = linker._settings_obj
        old_rules = settings_obj._blocking_rules_to_generate_predictions
        sql_dialect = cast(str, linker._db_api.sql_dialect.sql_dialect_str)
        dialected = rules.map(lambda rule: to_blocking_rule_creator(rule).get_blocking_rule(sql_dialect))

        def _with_rules(blocking_rules_dialected: Array[BlockingRule]) -> Run[Any]:
            settings_obj._blocking_rules_to_generate_predictions = list(blocking_rules_dialected)
            try:
                return pure(action())
            finally:
                settings_obj._blocking_rules_to_generate_predictions = old_rules

        return _with_preceding_rules(dialected) >> _with_rules

    return _run()


def _capture_blocked_edges_run(ctx: SplinkContext, linker: Linker, plan: PredictPlan) -> Run[Unit]:
    pairs_out = ctx.tables.get_required(PairsTableName)
    capture_pairs_table = PairsCaptureTableName(f"{pairs_out}_capture")

    def _with_pairs_source(pairs_source: PairsSourceTableName) -> Run[Unit]:
        return sql_exec(SQL(
            f"CREATE OR REPLACE TABLE {capture_pairs_table} AS SELECT * FROM {pairs_source}"
        ))

    return (
        _with_temp_blocking_rules_on_linker(
            linker,
            _build_capture_rules(plan.training_rules, plan.prediction_rules),
            lambda: PairsSourceTableName(
                linker.inference.predict(threshold_match_probability=ctx.predict_threshold).physical_name
            ),
        )
        >> _with_pairs_source
        >> (lambda _: context_replace(cluster_pairs_table=ClusterPairsTableName(str(capture_pairs_table))))
    )


def _capture_blocked_edges(ctx: SplinkContext) -> Run[Unit]:
    if not ctx.capture_blocked_edges:
        return pure(unit)
    return (
        with_splink_context(_capture_blocked_edges_validate)
        ^ _with_splink_context_linker_plan(_capture_blocked_edges_run)
    )


def _diagnostic_cluster_blocked_edges_validate_tables(ctx: SplinkContext) -> Run[Unit]:
    return (
        tables_get_required(ctx.tables, PredictionInputTableNames)
        ^ tables_get_required(ctx.tables, PairsCaptureTableName)
        ^ tables_get_required(ctx.tables, DoNotLinkTableName)
        ^ pure(unit)
    )


def _diagnostic_cluster_blocked_edges_validate(_: SplinkContext) -> Run[Unit]:
    return (
        with_splink_context(_require_db_api)
        ^ with_splink_context(_require_linker)
        ^ with_splink_context(_diagnostic_cluster_blocked_edges_validate_tables)
    )


def _diagnostic_cluster_blocked_edges(ctx: SplinkContext) -> Run[Unit]:
    if not ctx.capture_blocked_edges:
        return pure(unit)
    return (
        with_splink_context(_diagnostic_cluster_blocked_edges_validate)
        ^ with_splink_context(diagnostic_cluster_blocked_edges_run)
    )


def _create_exclusion_list_table_step(
    *,
    input_table: PredictionInputTableName,
    output_table: ExclusionInputTableName,
    do_not_link_table: DoNotLinkTableName,
    unique_id_column_name: UniqueIdColumnName,
    bl_id_left_col: BlockedIdLeftColumnName,
    bl_id_right_col: BlockedIdRightColumnName,
) -> Run[Unit]:
    return sql_exec(SQL(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH pairs AS (
          SELECT CAST({bl_id_left_col} AS VARCHAR) AS victim_id,
                 CAST({bl_id_right_col} AS VARCHAR) AS other_id
          FROM {do_not_link_table}
          UNION ALL
          SELECT CAST({bl_id_right_col} AS VARCHAR) AS victim_id,
                 CAST({bl_id_left_col} AS VARCHAR) AS other_id
          FROM {do_not_link_table}
        ),
        agg AS (
          SELECT victim_id, array_agg(DISTINCT other_id) AS exclusion_ids
          FROM pairs GROUP BY victim_id
        )
        SELECT v.*, COALESCE(a.exclusion_ids, []::VARCHAR[]) AS exclusion_ids
        FROM {input_table} v
        LEFT JOIN agg a ON CAST(v.{unique_id_column_name} AS VARCHAR) = a.victim_id
    """))


def _prepare_exclusion_list_from_ctx(ctx: SplinkContext) -> Run[Unit]:
    def _with_plan(plan: PredictPlan) -> Run[Unit]:
        def _with_inputs(input_tables: PredictionInputTableNames) -> Run[Unit]:
            input_table = input_tables.left().value_or("")
            do_not_link_table = cast(DoNotLinkTableName, ctx.tables.get(DoNotLinkTableName))
            if ctx.capture_blocked_edges:
                return _create_exclusion_list_table_step(
                    input_table=PredictionInputTableName(input_table),
                    output_table=ExclusionInputTableName(f"{input_table}_exc"),
                    do_not_link_table=do_not_link_table,
                    unique_id_column_name=ctx.unique_id_col,
                    bl_id_left_col=ctx.do_not_link_left_col,
                    bl_id_right_col=ctx.do_not_link_right_col,
                )
            return pure(unit)

        _ = plan
        return tables_get_required(ctx.tables, PredictionInputTableNames) >> _with_inputs

    return maybe_get_required(ctx.predict_plan, label="Predict plan") >> _with_plan


def _prepare_exclusion_list(_: SplinkContext) -> Run[Unit]:
    return (
        with_splink_context(_validate_predict_plan)
        ^ with_splink_context(_prepare_exclusion_list_from_ctx)
    )


def _build_linker_from_ctx(ctx: SplinkContext, db_api: DuckDBAPI, plan: PredictPlan) -> Run[Unit]:
    return context_replace(
        linker=Just(_build_linker_for_prediction(
            settings=ctx.settings,
            db_api=db_api,
            prediction_rules=plan.prediction_rules,
            input_table=input_table_value(plan.input_table_for_prediction),
            extra_columns_to_retain=plan.extra_columns_to_retain,
        ))
    )


def _filter_pairs_table_do_not_link_step(ctx: SplinkContext) -> Run[Unit]:
    pairs_table = ctx.tables.get_required(PairsTableName)
    do_not_link_table = ctx.tables.get_required(DoNotLinkTableName)

    def _with_cols(
        resolved: Tuple[HashSet, Tuple[PairLeftIdColumnName, PairRightIdColumnName]]
    ) -> Run[Unit]:
        id_cols = resolved.snd
        left_id_col = id_cols.fst
        right_id_col = id_cols.snd
        return sql_exec(SQL(f"""
            CREATE OR REPLACE TABLE {pairs_table} AS
            SELECT p.* FROM {pairs_table} p
            WHERE NOT EXISTS (
              SELECT 1 FROM {do_not_link_table} d
              WHERE (
                CAST(d.{ctx.do_not_link_left_col} AS VARCHAR) = CAST(p.{left_id_col} AS VARCHAR)
                AND CAST(d.{ctx.do_not_link_right_col} AS VARCHAR) = CAST(p.{right_id_col} AS VARCHAR)
              ) OR (
                CAST(d.{ctx.do_not_link_left_col} AS VARCHAR) = CAST(p.{right_id_col} AS VARCHAR)
                AND CAST(d.{ctx.do_not_link_right_col} AS VARCHAR) = CAST(p.{left_id_col} AS VARCHAR)
              )
            )
        """))

    return resolve_pair_id_cols_from_table_step(pairs_table, ctx.unique_id_col) >> _with_cols


def _predict_pairs_from_ctx(ctx: SplinkContext, linker: Linker, _: PredictPlan) -> Run[Unit]:
    def _with_pairs(pairs_out: PairsTableName) -> Run[Unit]:
        do_not_link_table = tables_get_optional(ctx.tables, DoNotLinkTableName)
        link_type = SplinkLinkType.from_settings(ctx.settings)
        return (
            sql_exec(SQL(
                f"CREATE OR REPLACE TABLE {pairs_out} AS "
                f"SELECT * FROM {PairsSourceTableName(linker.inference.predict(threshold_match_probability=ctx.predict_threshold).physical_name)}"
            ))
            ^ (with_splink_context(_filter_pairs_table_do_not_link_step) if do_not_link_table.is_present() else pure(unit))
            ^ (
                sql_exec(SQL(f"""
                    CREATE OR REPLACE TABLE {pairs_out}_top1 AS
                    WITH ranked AS (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY unique_id_r
                            ORDER BY match_probability DESC,
                                     COALESCE(match_weight, 0) DESC,
                                     CAST(unique_id_l AS VARCHAR)
                        ) AS rn
                        FROM {pairs_out}
                    )
                    SELECT * EXCLUDE (rn)
                    FROM ranked
                    WHERE rn = 1;
                """))
                if link_type == SplinkLinkType.LINK_ONLY
                else pure(unit)
            )
        )

    return tables_get_required(ctx.tables, PairsTableName) >> _with_pairs


def _predict_pairs_step(_: SplinkContext) -> Run[Unit]:
    return (
        with_splink_context(_validate_predict_plan)
        ^ _with_splink_context_linker_plan(_predict_pairs_from_ctx)
    )


def splink_predict_pairs_from_ctx(_: SplinkContext):
    """Run the full prediction pipeline and return the resulting pair metadata."""
    def _with_ctx(ctx: SplinkContext):
        if ctx.capture_blocked_edges:
            return (
                with_splink_context(_set_diagnostic_plan)
                ^ _with_splink_context_api_plan(_build_linker_from_ctx)
                ^ with_splink_context(print_prediction_counts)
                ^ train_linker_for_prediction()
                ^ with_splink_context(_capture_blocked_edges)
                ^ with_splink_context(_diagnostic_cluster_blocked_edges)
                ^ with_splink_context(_prepare_exclusion_list)
                ^ with_splink_context(_set_final_plan)
                ^ _with_splink_context_api_plan(_build_linker_from_ctx)
                ^ with_splink_context(print_prediction_counts)
                ^ train_linker_for_prediction()
                ^ with_splink_context(_predict_pairs_step)
                ^ _with_splink_context_linker_plan(predict_result_from_ctx)
            )
        return (
            with_splink_context(_set_final_plan)
            ^ _with_splink_context_api_plan(_build_linker_from_ctx)
            ^ with_splink_context(print_prediction_counts)
            ^ train_linker_for_prediction()
            ^ with_splink_context(_predict_pairs_step)
            ^ _with_splink_context_linker_plan(predict_result_from_ctx)
        )

    return with_splink_context(validate_predict_tables) ^ with_splink_context(_with_ctx)
