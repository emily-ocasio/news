"""Splink runtime context access and validation helpers."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, TypeVar

from splink import DuckDBAPI, Linker

from ..maybe import Just, Maybe
from ..monad import Unit, unit
from ..run import ErrorPayload, Run, get_splink_context, put_splink_context, pure, throw
from .splink_types import (
    PredictPlan,
    PairsTableName,
    PredictionInputTableNames,
    SplinkContext,
    SplinkPredictResult,
    SplinkTableNames,
    TTableName,
    DoNotLinkTableName,
    BlockedPairsTableName,
    UniquePairsTableName,
    ClustersTableName,
)

A = TypeVar("A")


def _require_splink_context() -> Run[SplinkContext]:
    return get_splink_context() >> (
        lambda ctx: throw(ErrorPayload("Splink context is not initialized."))
        if ctx is None else pure(ctx)
    )


def maybe_get_required(value: Maybe[A], *, label: str) -> Run[A]:
    """Extract a required Maybe value or raise a labeled runtime error."""
    match value:
        case Just(v):
            return pure(v)
        case _:
            return throw(ErrorPayload(f"{label} is not initialized."))


def with_splink_context(f: Callable[[SplinkContext], Run[A]]) -> Run[A]:
    """Run a function with the initialized Splink context."""
    return _require_splink_context() >> f


def _with_splink_context_api_plan(
    f: Callable[[SplinkContext, DuckDBAPI, PredictPlan], Run[A]]) -> Run[A]:
    def _with_ctx(ctx: SplinkContext) -> Run[A]:
        def _with_api(db_api: DuckDBAPI) -> Run[A]:
            def _with_plan(plan: PredictPlan) -> Run[A]:
                return f(ctx, db_api, plan)
            return maybe_get_required(ctx.predict_plan, label="Predict plan") >> _with_plan
        return maybe_get_required(ctx.db_api, label="Splink DuckDB API") >> _with_api
    return with_splink_context(_with_ctx)


def _with_splink_context_linker_plan(
    f: Callable[[SplinkContext, Linker, PredictPlan], Run[A]]) -> Run[A]:
    def _with_ctx(ctx: SplinkContext) -> Run[A]:
        def _with_linker(linker: Linker) -> Run[A]:
            def _with_plan(plan: PredictPlan) -> Run[A]:
                return f(ctx, linker, plan)
            return maybe_get_required(ctx.predict_plan, label="Predict plan") >> _with_plan
        return maybe_get_required(ctx.linker, label="Splink linker") >> _with_linker
    return with_splink_context(_with_ctx)


def with_splink_context_linker(f: Callable[[SplinkContext, Linker], Run[A]]) -> Run[A]:
    """Run a function with both current Splink context and active linker."""
    def _with_ctx(ctx: SplinkContext) -> Run[A]:
        def _with_linker(linker: Linker) -> Run[A]:
            return f(ctx, linker)
        return maybe_get_required(ctx.linker, label="Splink linker") >> _with_linker
    return with_splink_context(_with_ctx)


def update_splink_context(update_fn) -> Run[Unit]:
    """Update the stored Splink context via a pure transformation function."""
    return _require_splink_context() >> (lambda ctx: put_splink_context(update_fn(ctx)))


def context_replace(**kwargs: Any) -> Run[Unit]:
    """Replace selected fields on the current Splink context dataclass."""
    def _with_ctx(ctx: SplinkContext) -> Run[Unit]:
        return put_splink_context(replace(ctx, **kwargs))
    return _require_splink_context() >> _with_ctx


def _require_db_api(ctx: SplinkContext) -> Run[Unit]:
    match ctx.db_api:
        case Just(_):
            return pure(unit)
        case _:
            return throw(ErrorPayload("Splink DuckDB API is not initialized."))


def tables_get_required(tables: SplinkTableNames, key: type[TTableName]) -> Run[TTableName]:
    """Fetch a required typed table reference from the table registry."""
    value = tables.get(key)
    if value is None:
        return throw(ErrorPayload(f"Missing table name for {key.__name__}."))
    if not isinstance(value, key):
        return throw(ErrorPayload(f"Expected {key.__name__}, got {type(value).__name__}."))
    return pure(value)


def tables_get_optional(tables: SplinkTableNames, key: type[TTableName]) -> TTableName:
    """Fetch an optional typed table reference or return an empty default."""
    value = tables.get(key)
    if value is None:
        return key()  # type: ignore[call-arg]
    return value


def _validate_predict_plan(ctx: SplinkContext) -> Run[Unit]:
    match ctx.predict_plan:
        case Just(_):
            return pure(unit)
        case _:
            return throw(ErrorPayload("Predict plan is not initialized."))


def _require_linker(ctx: SplinkContext) -> Run[Unit]:
    match ctx.linker:
        case Just(_):
            return pure(unit)
        case _:
            return throw(ErrorPayload("Splink linker was not created."))


def predict_result_from_ctx(
    ctx: SplinkContext,
    linker: Linker,
    plan: PredictPlan,
) -> Run[SplinkPredictResult]:
    """Build the typed prediction result payload from current context state."""
    def _with_pairs(pairs_out: PairsTableName) -> Run[SplinkPredictResult]:
        clusters_out = tables_get_optional(ctx.tables, ClustersTableName)
        blocked_pairs_out = tables_get_optional(ctx.tables, BlockedPairsTableName)
        unique_pairs_table = tables_get_optional(ctx.tables, UniquePairsTableName)
        do_not_link_table = tables_get_optional(ctx.tables, DoNotLinkTableName)
        return pure(
            SplinkPredictResult(
                linker=linker,
                input_table_for_prediction=plan.input_table_for_prediction,
                unique_id_col=ctx.unique_id_col,
                pairs_out=pairs_out,
                clusters_out=clusters_out,
                do_not_link_table=do_not_link_table if do_not_link_table.is_present() else None,
                do_not_link_left_col=ctx.do_not_link_left_col,
                do_not_link_right_col=ctx.do_not_link_right_col,
                blocked_pairs_out=blocked_pairs_out,
                unique_pairs_table=unique_pairs_table,
            )
        )

    return tables_get_required(ctx.tables, PairsTableName) >> _with_pairs


def validate_predict_tables(ctx: SplinkContext) -> Run[Unit]:
    """Validate required tables exist before running pair prediction."""
    return (
        tables_get_required(ctx.tables, PairsTableName)
        ^ tables_get_required(ctx.tables, PredictionInputTableNames)
        ^ pure(unit)
    )
