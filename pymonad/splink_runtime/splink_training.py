"""Splink runtime model training and EM helpers."""
# Splink EM diagnostics and lambda tuning depend on internals exposed only via protected members.
# pylint: disable=protected-access
from __future__ import annotations

import math
from typing import Any, cast

from splink import DuckDBAPI, Linker, blocking_analysis
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.blocking_rule_creator_utils import to_blocking_rule_creator
from splink.internals.comparison import Comparison as SplinkComparison
from splink.internals.misc import bayes_factor_to_prob, prob_to_bayes_factor
from splink.internals.parse_sql import get_columns_used_from_sql

from ..array import Array
from ..hashmap import HashMap, flatten_map
from ..maybe import Just, Maybe, Nothing
from ..monad import Unit, bind_first, unit
from ..run import Run, fold_run, pure, put_line
from .splink_context import _with_splink_context_linker_plan, maybe_get_required, with_splink_context_linker
from .splink_tables import input_table_value
from .splink_types import (
    BlockingRuleLike,
    ComparisonLevelKey,
    EmParam,
    EmParams,
    PredictPlan,
    PredictionInputTableNames,
    SplinkContext,
)


def _put_line_unit(line: str) -> Run[Unit]:
    return put_line(line) ^ pure(unit)


def _put_lines(lines: list[str]) -> Run[Unit]:
    run = pure(unit)
    for line in lines:
        run = run ^ _put_line_unit(line)
    return run


def print_prediction_counts(ctx: SplinkContext) -> Run[Unit]:
    """Print estimated comparison counts implied by current blocking rules."""
    match ctx.predict_plan:
        case Just(plan):
            rules = plan.prediction_rules
            table = input_table_value(plan.input_table_for_prediction)
        case _:
            rules = ctx.prediction_rules
            table = input_table_value(ctx.tables.get_required(PredictionInputTableNames))
    if not rules:
        return pure(unit)

    def _run(db_api: DuckDBAPI) -> Run[Unit]:
        try:
            tables = table if isinstance(table, list) else [table]
            counts_df = blocking_analysis.cumulative_comparisons_to_be_scored_from_blocking_rules_data(
                table_or_tables=tables,
                blocking_rules=rules,
                link_type=ctx.settings.get("link_type", "dedupe_only"),
                db_api=db_api,
                unique_id_column_name=ctx.settings.get("unique_id_column_name", "unique_id"),
                source_dataset_column_name=ctx.settings.get("source_dataset_column_name"),
            )
            total_comparisons = int(counts_df["row_count"].sum())
            return _put_line_unit(
                f"Total comparisons to be scored (pre-threshold): {total_comparisons}"
            )
        except Exception as exc:  # pylint: disable=W0718
            return _put_line_unit(f"Skipping blocking count analysis: {exc}")

    return maybe_get_required(ctx.db_api, label="Splink DuckDB API") >> _run


def _train_linker_setup(ctx: SplinkContext, linker: Linker, _: PredictPlan) -> Run[Unit]:
    def _run() -> Run[Unit]:
        linker.training.estimate_probability_two_random_records_match(
            list(ctx.deterministic_rules), recall=ctx.deterministic_recall
        )
        if not ctx.skip_u_estimation:
            linker.training.estimate_u_using_random_sampling(1e8)
        return pure(unit)

    return _run()


def _maybe_adjust_lambda_for_training_rule(
    training_rule: BlockingRuleLike,
    ctx: SplinkContext,
    linker: Linker,
) -> Run[Unit]:
    if len(ctx.training_block_level_map) == 0:
        return pure(unit)

    def _normalize_rule_key(rule: BlockingRuleLike) -> str:
        if isinstance(rule, BlockingRuleCreator):
            sql_dialect = cast(str, linker._db_api.sql_dialect.sql_dialect_str)
            return to_blocking_rule_creator(rule).get_blocking_rule(sql_dialect).blocking_rule_sql
        return str(rule)

    level_keys = ctx.training_block_level_map.get(training_rule)
    if level_keys is None:
        training_rule_str = _normalize_rule_key(training_rule)
        for rule_key, mapped_levels in ctx.training_block_level_map.items():
            rule_key_str = _normalize_rule_key(rule_key)
            if rule_key_str == training_rule_str or (rule_key_str and rule_key_str in training_rule_str):
                level_keys = mapped_levels
                break
    if level_keys is None:
        return pure(unit)

    settings_obj = linker._settings_obj
    total_m = 0.0
    total_u = 0.0
    matched_levels: list[ComparisonLevelKey] = []
    for level_key in level_keys:
        found_level = None
        for comp in settings_obj.comparisons:
            name = f"{comp.output_column_name}_{comp.comparison_description}"
            if name != level_key.comparison_name:
                continue
            for level in comp.comparison_levels:
                if level.label_for_charts == level_key.level_name:
                    found_level = level
                    break
            if found_level is not None:
                break
        if found_level is None:
            continue
        m = found_level.m_probability
        u = found_level.u_probability
        if m is None or u is None or u == 0:
            continue
        total_m += m
        total_u += u
        matched_levels.append(level_key)

    if not matched_levels or total_u == 0:
        return pure(unit)

    bayes_factor = total_m / total_u
    original_lambda = linker._settings_obj._probability_two_random_records_match
    adjusted_lambda = bayes_factor_to_prob(prob_to_bayes_factor(original_lambda) * bayes_factor)
    linker._settings_obj._probability_two_random_records_match = adjusted_lambda
    return _put_lines([
        "Manual lambda adjustment for training block:",
        f"  training_rule={training_rule}",
        f"  overall_bayes_factor=sum(m)/sum(u)={bayes_factor:.6f}",
        f"  lambda_pre={original_lambda:.6f}",
        f"  lambda_post={adjusted_lambda:.6f}",
    ])


def _log_training_comparisons(
    training_rule: BlockingRuleLike,
    ctx: SplinkContext,
    linker: Linker,
) -> Run[Unit]:
    try:
        blocking_rule = to_blocking_rule_creator(training_rule).get_blocking_rule(linker._sql_dialect_str)
        sqlglot_dialect = linker._db_api.sql_dialect.sqlglot_dialect
        br_cols = get_columns_used_from_sql(blocking_rule.blocking_rule_sql, sqlglot_dialect=sqlglot_dialect)
    except Exception as exc:  # pylint: disable=W0718
        return _put_line_unit(f"EM training comparisons unavailable ({exc})")

    comparisons = linker._settings_obj.comparisons
    included: list[SplinkComparison] = []
    excluded: list[SplinkComparison] = []
    br_cols_set = set(br_cols)
    for comp in comparisons:
        comp_cols = [c.input_name for c in comp._input_columns_used_by_case_statement]
        if br_cols_set.intersection(comp_cols):
            excluded.append(comp)
        else:
            included.append(comp)

    lines = [
        f"EM training comparisons for rule: {blocking_rule.blocking_rule_sql}",
        f"  blocking_columns: {', '.join(sorted(br_cols_set)) or 'none'}",
        f"  included ({len(included)})",
        f"  excluded ({len(excluded)})",
    ]
    try:
        input_tables = list(linker._input_tables_dict.values())
        table_or_tables: str | list[str]
        if len(input_tables) == 1:
            table_or_tables = input_tables[0].physical_name
        else:
            table_or_tables = [df.physical_name for df in input_tables]
        counts = blocking_analysis.count_comparisons_from_blocking_rule(
            table_or_tables=table_or_tables,
            blocking_rule=blocking_rule.blocking_rule_sql,
            link_type=ctx.settings.get("link_type", "dedupe_only"),
            db_api=linker._db_api,
            unique_id_column_name=ctx.settings.get("unique_id_column_name", "unique_id"),
            source_dataset_column_name=ctx.settings.get("source_dataset_column_name"),
            compute_post_filter_count=True,
        )
        lines.append(
            f"  training_pairs_post_filter: {counts.get('number_of_comparisons_to_be_scored_post_filter_conditions')}"
        )
    except Exception as exc:  # pylint: disable=W0718
        lines.append(f"  training_pairs_count_skipped: {exc}")
    return _put_lines(lines)


def _train_linker_em_runs(ctx: SplinkContext, linker: Linker, plan: PredictPlan) -> Run[Unit]:
    if not ctx.train_first:
        return pure(unit)

    def _max_delta_from_params(prev_params: EmParams, current_params: EmParams) -> tuple[float, ComparisonLevelKey | None]:
        deltas: list[float] = []
        max_key: ComparisonLevelKey | None = None
        max_delta_local = -1.0
        for key, now in current_params.items():
            if key not in prev_params:
                continue
            prev = prev_params[key]
            match now.m, now.u, prev.m, prev.u:
                case Just(m_now), Just(u_now), Just(m_prev), Just(u_prev):
                    if u_now != 0 and u_prev != 0:
                        delta = abs(math.log2(m_now / u_now) - math.log2(m_prev / u_prev))
                        deltas.append(delta)
                        if delta > max_delta_local:
                            max_key = key
                            max_delta_local = delta
        return (max(deltas) if deltas else 1.0), max_key

    def _train_block(prev_block_params: EmParams, training_rule: BlockingRuleLike, run_idx: int) -> Run[EmParams]:
        def _run_em() -> Run[EmParams]:
            linker.training.estimate_parameters_using_expectation_maximisation(
                blocking_rule=training_rule,
                fix_probability_two_random_records_match=False,
            )
            current_settings = linker.misc.save_model_to_json(out_path=None)
            current_params = _extract_em_params(current_settings)
            if len(prev_block_params) != 0:
                max_delta, _ = _max_delta_from_params(prev_block_params, current_params)
                return _put_lines([
                    f"EM run {run_idx + 1}/{ctx.em_max_runs} block drift: max_delta={max_delta:.6f}"
                ]) ^ pure(current_params)
            return pure(current_params)

        def _restore_lambda(original_lambda: float, params: EmParams) -> Run[EmParams]:
            linker._settings_obj._probability_two_random_records_match = original_lambda
            return pure(params)

        original_lambda = linker._settings_obj._probability_two_random_records_match
        return (
            with_splink_context_linker(bind_first(_maybe_adjust_lambda_for_training_rule, training_rule))
            ^ with_splink_context_linker(bind_first(_log_training_comparisons, training_rule))
            >> (lambda _: _run_em())
            >> (lambda params: _restore_lambda(original_lambda, params))
        )

    def _train_run(state: tuple[EmParams, bool], run_idx: int) -> Run[tuple[EmParams, bool]]:
        prev_params, stop = state
        if stop:
            return pure(state)

        def _train_rule(prev_block_params: EmParams, training_rule: BlockingRuleLike) -> Run[EmParams]:
            return _train_block(prev_block_params, training_rule, run_idx)

        def _with_current(current_params: EmParams) -> Run[tuple[EmParams, bool]]:
            if len(prev_params) == 0:
                return pure((current_params, False))
            max_delta, _ = _max_delta_from_params(prev_params, current_params)
            stop_now = run_idx + 1 >= ctx.em_min_runs and max_delta < ctx.em_stop_delta
            return _put_lines([
                f"EM run {run_idx + 1}/{ctx.em_max_runs}: max_delta={max_delta:.6f}"
            ]) ^ pure((current_params, stop_now))

        init: EmParams = HashMap.empty()
        rules: Array[BlockingRuleLike] = Array.make(plan.training_rules.a)
        return fold_run(rules, init, _train_rule) >> _with_current

    run_indices: Array[int] = Array.make(tuple(range(ctx.em_max_runs)))
    init: EmParams = HashMap.empty()
    return fold_run(run_indices, (init, False), _train_run) ^ pure(unit)


def train_linker_for_prediction() -> Run[Unit]:
    """Run linker setup and optional EM training before prediction."""
    return (
        _with_splink_context_linker_plan(_train_linker_setup)
        ^ _with_splink_context_linker_plan(_train_linker_em_runs)
    )


def _extract_em_params(settings_dict: dict[str, Any]) -> EmParams:
    def _level_entry(level: dict[str, Any]):
        raw_name = level.get("label_for_charts", "")
        level_name = str(raw_name).strip()
        maybe_key = Just(level_name) if level_name else Nothing
        return (
            maybe_key,
            EmParam(
                m=_maybe_float(level.get("m_probability")),
                u=_maybe_float(level.get("u_probability")),
            ),
        )

    def _comparison_map(comparison: dict[str, Any]) -> tuple[Maybe[str], HashMap[str, EmParam]]:
        output_name = str(comparison.get("output_column_name", ""))
        description = str(comparison.get("comparison_description", ""))
        comp_name = f"{output_name}_{description}"
        level_map = HashMap.from_iterable(comparison.get("comparison_levels", []), _level_entry)
        return Just(comp_name), level_map

    nested = HashMap.from_iterable(settings_dict.get("comparisons", []), _comparison_map)
    return flatten_map(
        nested,
        lambda comp_name, level_name: ComparisonLevelKey(
            comparison_name=comp_name,
            level_name=level_name,
        ),
    )


def _maybe_float(value: Any) -> Maybe[float]:
    if value is None:
        return Nothing
    return Just(value)
