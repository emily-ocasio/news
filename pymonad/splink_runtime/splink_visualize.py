"""Splink runtime visualization helpers."""
# Splink chart/debug paths rely on linker/db internal members with no equivalent public API.
# pylint: disable=protected-access
from __future__ import annotations

from dataclasses import dataclass
import html
import re
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import altair as alt
import numpy as np
import pandas as pd
from splink import Linker
from splink.internals.blocking_rule_creator_utils import to_blocking_rule_creator
from splink.internals.cache_dict_with_logging import CacheDictWithLogging
from splink.internals.charts import save_offline_chart, unlinkables_chart
from splink.internals.pipeline import CTEPipeline

from calculations import distance_km
from ..array import Array
from ..maybe import Just, Maybe, Nothing
from .splink_cluster import drop_all_splink_tables
from .splink_types import (
    BlockingRuleLike,
    CustomStringBlockingRule,
    SplinkChartType,
    SplinkLinkType,
    SplinkVisualizeJob,
)


_FLOAT_RE = re.compile(
    r"""
    ^\[\s*
    [-+]?  # sign
    (?:\d+\.\d*|\.\d+|\d+)  # int/float
    (?:[eE][-+]?\d+)?       # exponent
    (?:\s+
        [-+]?
        (?:\d+\.\d*|\.\d+|\d+)
        (?:[eE][-+]?\d+)?
    )*
    \s*\]$
    """,
    re.VERBOSE,
)


@dataclass
class VisualizeContext:
    """Mutable runtime context for visualization processing."""

    settings: dict[str, Any]
    link_type: SplinkLinkType
    unique_id_col: str
    left_id_col: str | None
    right_id_col: str | None
    df_pairs: Any = None
    pd_pairs: pd.DataFrame | None = None
    inspect_df: pd.DataFrame | None = None
    inspect_ids: set[Any] | None = None
    df_clustered: Any = None
    clustered_df: pd.DataFrame | None = None


def _with_temp_blocking_rules_on_linker_sync(
    linker: Linker,
    rules: Array[BlockingRuleLike],
    action: Callable[[], Any],
) -> Any:
    settings_obj = linker._settings_obj
    old_rules = settings_obj._blocking_rules_to_generate_predictions
    try:
        sql_dialect = cast(str, linker._db_api.sql_dialect.sql_dialect_str)
        dialected = rules.map(
            lambda rule: to_blocking_rule_creator(rule).get_blocking_rule(sql_dialect)
        )
        acc = Array.empty()
        for rule in dialected:
            rule.add_preceding_rules(list(acc.a))
            acc = Array.snoc(acc, rule)
        settings_obj._blocking_rules_to_generate_predictions = list(acc)
        return action()
    finally:
        settings_obj._blocking_rules_to_generate_predictions = old_rules


def _make_visualize_context(linker: Linker) -> VisualizeContext:
    """Initialize visualization context from linker settings."""
    settings = linker.misc.save_model_to_json(out_path=None)
    link_type = SplinkLinkType.from_settings(settings)
    unique_id_col = settings.get("unique_id_column_name", "unique_id")
    return VisualizeContext(
        settings=settings,
        link_type=link_type,
        unique_id_col=unique_id_col,
        left_id_col=f"{unique_id_col}_l",
        right_id_col=f"{unique_id_col}_r",
        inspect_ids=set(),
    )


def _midpoint_blocking_rule(
    left_midpoints: Sequence[int] | None,
    right_midpoints: Sequence[int] | None,
) -> CustomStringBlockingRule | None:
    """Build midpoint-only blocking SQL when midpoint filters are provided."""
    clauses: list[str] = []
    if left_midpoints:
        left_vals = ",".join(str(int(v)) for v in left_midpoints)
        clauses.append(f"l.midpoint_day in ({left_vals})")
    if right_midpoints:
        right_vals = ",".join(str(int(v)) for v in right_midpoints)
        clauses.append(f"r.midpoint_day in ({right_vals})")
    if not clauses:
        return None
    return CustomStringBlockingRule(" and ".join(clauses))


def _sql_literal(value: Any) -> str:
    """Quote a SQL literal value for inline filter clauses."""
    return "'" + str(value).replace("'", "''") + "'"


def _distinct_sources(linker: Linker, self_link_table: str, source_col: str) -> list[Any]:
    """Return distinct source datasets from a self-link table."""
    source_col_sql = f'"{source_col}"'
    suffix = abs(hash((self_link_table, source_col))) % 1000000
    pipeline = CTEPipeline()
    sql = (
        f"select distinct {source_col_sql} as source_dataset "
        f"from {self_link_table} where {source_col_sql} is not null"
    )
    pipeline.enqueue_sql(sql, f"__splink__unlinkable_sources_{suffix}")
    data = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)
    rows = data.as_record_dict()
    data.drop_table_from_database_and_remove_from_cache()
    return [row["source_dataset"] for row in rows]


def _unlinkables_records(
    linker: Linker,
    self_link_table: str,
    where_clause: str | None = None,
) -> list[dict[str, Any]]:
    """Compute unlinkables chart records from a self-link table."""
    suffix = abs(hash((self_link_table, where_clause))) % 1000000
    round_table = f"__splink__df_round_self_link_{suffix}"
    prop_table = f"__splink__df_unlinkables_proportions_{suffix}"
    cum_table = f"__splink__df_unlinkables_proportions_cumulative_{suffix}"
    pipeline = CTEPipeline()
    where_sql = f"where {where_clause}" if where_clause else ""
    sql = f"""
        select
        round(match_weight, 2) as match_weight,
        round(match_probability, 5) as match_probability
        from {self_link_table}
        {where_sql}
    """
    pipeline.enqueue_sql(sql, round_table)
    sql = f"""
        select
        max(match_weight) as match_weight,
        match_probability,
        count(*) / cast( sum(count(*)) over () as float) as prop
        from {round_table}
        group by match_probability
        order by match_probability
    """
    pipeline.enqueue_sql(sql, prop_table)
    sql = f"""
        select *,
        sum(prop) over(order by match_probability) as cum_prop
        from {prop_table}
        where match_probability < 1
    """
    pipeline.enqueue_sql(sql, cum_table)
    data = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)
    records = data.as_record_dict()
    data.drop_table_from_database_and_remove_from_cache()
    return records


def _unlinkables_records_for_source(
    linker: Linker,
    self_link_table: str,
    source_col: str,
    source_value: Any,
) -> list[dict[str, Any]]:
    """Compute unlinkables records scoped to one source dataset value."""
    source_col_sql = f'"{source_col}"'
    source_literal = _sql_literal(source_value)
    where_clause = f"{source_col_sql} = {source_literal}"
    return _unlinkables_records(linker, self_link_table, where_clause)


def _with_cache_uid(linker: Linker, new_uid: str, action: Callable[[], Any]) -> Any:
    """Run an action with a temporary Splink cache UID."""
    old_uid = linker._db_api._cache_uid
    linker._db_api._cache_uid = new_uid
    try:
        return action()
    finally:
        linker._db_api._cache_uid = old_uid


def _with_isolated_cache(linker: Linker, action: Callable[[CacheDictWithLogging], Any]) -> Any:
    """Run an action with an isolated intermediate table cache."""
    db_api = linker._db_api
    old_cache = db_api._intermediate_table_cache
    new_cache = CacheDictWithLogging()
    db_api._intermediate_table_cache = new_cache
    try:
        return action(new_cache)
    finally:
        db_api._intermediate_table_cache = old_cache


def _vectors_to_similarity(left: str, right: str) -> tuple[str, str]:
    """Convert vector-string pairs to cosine similarity display values when possible."""

    def to_maybe_vec(s: str) -> Maybe[np.ndarray]:
        return (
            Just(np.fromstring(s.strip()[1:-1], sep=" "))
            if _FLOAT_RE.match(s.strip())
            else Nothing
        )

    match to_maybe_vec(left), to_maybe_vec(right):
        case Just(x), Just(y) if len(x) == len(y):
            dot_product = np.dot(x, y)
            norm_vec1 = np.linalg.norm(x)
            norm_vec2 = np.linalg.norm(y)
            similarity = (
                0
                if norm_vec1 == 0 or norm_vec2 == 0
                else dot_product / (norm_vec1 * norm_vec2)
            )
            return "cosine similarity:", f"{similarity:.6f}"
        case _:
            return left, right


def _locations_to_distance(left: str, right: str) -> tuple[str, str]:
    """Convert location-like strings to distance display values when possible."""
    min_lat, max_lat = 38.7, 39.0
    min_lon, max_lon = -77.2, -76.9

    def extract_lat_lon(parts: list[str]) -> tuple[float, float] | None:
        if len(parts) < 4:
            return None

        lat: float | None = None
        lon: float | None = None
        for part in parts:
            try:
                value = float(part)
            except ValueError:
                continue
            if lat is None and min_lat <= value <= max_lat:
                lat = value
            elif lon is None and min_lon <= value <= max_lon:
                lon = value
            if lat is not None and lon is not None:
                return lat, lon
        return None

    left_vals = left.split(",")
    right_vals = right.split(",")
    left_location = extract_lat_lon(left_vals)
    right_location = extract_lat_lon(right_vals)

    if left_location is None or right_location is None:
        return left, right

    left_lat, left_lon = left_location
    right_lat, right_lon = right_location

    return f"{left_vals[0]} dist (km):", f"{right_vals[0]} {distance_km((left_lat, left_lon), (right_lat, right_lon)):.3f}"


def _truncate_long_chart_values(chart, max_length=500, disp=10) -> alt.Chart:
    """Truncate very long chart values and map vectors/locations to compact displays."""
    spec = chart.to_dict()
    spec.pop("$schema", None)
    spec.pop("config", None)

    def must_truncate(key, value) -> bool:
        return (
            key in ("value_l", "value_r")
            and isinstance(value, str)
            and len(value) > max_length
        )

    def truncate_value(value):
        return value[:disp] + "..." + value[-disp:]

    for _, rows in spec.get("datasets", {}).items():
        for row in rows:
            if (
                "value_l" in row
                and "value_r" in row
                and isinstance(row["value_l"], str)
                and isinstance(row["value_r"], str)
            ):
                row["value_l"], row["value_r"] = _vectors_to_similarity(
                    row["value_l"], row["value_r"]
                )
                row["value_l"], row["value_r"] = _locations_to_distance(
                    row["value_l"], row["value_r"]
                )
            for key, value in row.items():
                if must_truncate(key, value):
                    row[key] = truncate_value(value)

    return alt.Chart.from_dict(spec)


def _predict_pairs_for_visualization(
    linker: Linker,
    job: SplinkVisualizeJob,
    ctx: VisualizeContext,
):
    """Predict pairwise records for visualization, applying midpoint blocking when requested."""
    use_midpoint_blocking = job.chart_type in (
        SplinkChartType.WATERFALL,
        SplinkChartType.CLUSTER,
    ) and (job.left_midpoints or job.right_midpoints)
    if use_midpoint_blocking:
        rule = _midpoint_blocking_rule(job.left_midpoints, job.right_midpoints)
        if rule:
            print("Using midpoint-only blocking for prediction.")
            return _with_temp_blocking_rules_on_linker_sync(
                linker,
                Array.pure(rule),
                lambda: linker.inference.predict(threshold_match_probability=0),
            )
    return linker.inference.predict(threshold_match_probability=0)


def _handle_model_charts(linker: Linker, job: SplinkVisualizeJob) -> bool:
    """Render model-level charts when requested."""
    if job.chart_type != SplinkChartType.MODEL:
        return False
    chart = linker.visualisations.match_weights_chart()
    chart.show()  # type: ignore
    chart = linker.visualisations.m_u_parameters_chart()
    chart.show()  # type: ignore
    return True


def _handle_parameter_estimate_comparisons(linker: Linker, job: SplinkVisualizeJob) -> bool:
    """Render parameter estimate comparison chart and diagnostics when requested."""
    if job.chart_type != SplinkChartType.PARAMETER_ESTIMATE_COMPARISONS:
        return False

    records = linker._settings_obj._parameter_estimates_as_records
    not_observed_text = "level not observed in training dataset"
    null_log_odds = [
        r for r in records if r.get("estimated_probability_as_log_odds") is None
    ]
    if null_log_odds:
        print(
            "Parameter estimate comparisons: levels with no plotted points "
            "(null log-odds)."
        )
        print(f"Total records: {len(records)}; null log-odds: {len(null_log_odds)}")
        print("Examples (comparison, level, m/u, estimate):")
        for rec in null_log_odds[:12]:
            estimate = rec.get("estimated_probability")
            if estimate == not_observed_text:
                estimate = not_observed_text
            print(
                f"  - {rec.get('comparison_name')} | "
                f"{rec.get('comparison_level_label')} | "
                f"{rec.get('m_or_u')} | {estimate}"
            )
    chart = linker.visualisations.parameter_estimate_comparisons_chart()
    chart = chart.encode(  # type: ignore
        color=alt.Color(
            "estimate_description:N",
            legend=alt.Legend(
                labelLimit=800,
                titleLimit=800,
                columns=1,
            ),
        )
    )
    chart.show()  # type: ignore
    return True


def _handle_unlinkables_chart(
    linker: Linker,
    job: SplinkVisualizeJob,
    ctx: VisualizeContext,
) -> bool:
    """Render unlinkables charts for single or multiple source datasets."""
    if job.chart_type != SplinkChartType.UNLINKABLES:
        return False

    print("Generating unlinkables chart…")
    if ctx.link_type == SplinkLinkType.LINK_ONLY.value and len(linker._input_tables_dict) == 2:
        self_link_df = _with_cache_uid(
            linker,
            f"unlinkables_{uuid.uuid4().hex[:8]}",
            linker._self_link,
        )
        col_names = [col.name for col in self_link_df.columns]
        source_col = next(
            (
                col
                for col in ("source_dataset_l", "source_dataset", "source_dataset_r")
                if col in col_names
            ),
            None,
        )
        if source_col:
            sources = _distinct_sources(linker, self_link_df.physical_name, source_col)
            if sources:
                for source in sources:
                    print(f"Generating unlinkables chart for {source}…")
                    records = _unlinkables_records_for_source(
                        linker,
                        self_link_df.physical_name,
                        source_col,
                        source,
                    )
                    chart = unlinkables_chart(records, "match_weight", str(source))
                    chart.show()  # type: ignore
                self_link_df.drop_table_from_database_and_remove_from_cache()
                return True
        print("Source dataset column not found; using combined unlinkables chart.")
        for _, df in linker._input_tables_dict.items():
            label = df.physical_name
            print(f"Generating unlinkables chart for {label}…")
            single_settings = dict(ctx.settings)
            single_settings["link_type"] = "dedupe_only"
            single_settings.pop("source_dataset_column_name", None)
            single_settings["linker_uid"] = f"{label}_{uuid.uuid4().hex[:8]}"
            temp_linker = Linker(df.physical_name, single_settings, db_api=linker._db_api)

            def _run_self_link(cache):
                temp_linker._intermediate_table_cache = cache  # pylint: disable=W0640
                return _with_cache_uid(
                    linker,
                    f"unlinkables_{uuid.uuid4().hex[:8]}",
                    temp_linker._self_link,  # pylint: disable=W0640
                )

            self_link_df = _with_isolated_cache(linker, _run_self_link)
            records = _unlinkables_records(linker, self_link_df.physical_name)
            chart = unlinkables_chart(records, "match_weight", str(label))
            chart.show()  # type: ignore
            self_link_df.drop_table_from_database_and_remove_from_cache()
        return True

    chart = linker.evaluation.unlinkables_chart()
    chart.show()  # type: ignore
    return True


def _handle_comparison_chart(linker: Linker, job: SplinkVisualizeJob, df_pairs: Any) -> bool:
    """Render comparison dashboard when requested."""
    if job.chart_type != SplinkChartType.COMPARISON:
        return False
    print("\nGenerating comparison viewer dashboard…")
    linker.visualisations.comparison_viewer_dashboard(
        df_pairs,
        "comparison_viewer.html",
        overwrite=True,
        num_example_rows=5,
    )
    print("Comparison viewer dashboard written to comparison_viewer.html")
    return True


def _prepare_inspection_subset(
    linker: Linker,
    job: SplinkVisualizeJob,
    ctx: VisualizeContext,
) -> pd.DataFrame | None:
    """Prepare filtered inspection dataframe and print member summaries for waterfall/cluster charts."""
    if ctx.pd_pairs is None:
        return None

    inspect_df = ctx.pd_pairs
    inspect_ids: set[Any] = set()

    if job.chart_type not in (SplinkChartType.WATERFALL, SplinkChartType.CLUSTER):
        ctx.inspect_df = inspect_df
        ctx.inspect_ids = inspect_ids
        return None

    if job.left_midpoints:
        if "midpoint_day_l" in inspect_df.columns:
            inspect_df = inspect_df[inspect_df["midpoint_day_l"].isin(job.left_midpoints)]
        else:
            print("midpoint_day_l column not found; skipping left midpoint filter.")
    if job.right_midpoints:
        if "midpoint_day_r" in inspect_df.columns:
            inspect_df = inspect_df[inspect_df["midpoint_day_r"].isin(job.right_midpoints)]
        else:
            print("midpoint_day_r column not found; skipping right midpoint filter.")

    print(f"Total predictions within threshold: {len(ctx.pd_pairs)}")
    print(f"Number of records in selection: {len(inspect_df)}\n")
    if job.chart_type == SplinkChartType.WATERFALL:
        print("Waterfall chart members:")
    else:
        print("Cluster chart members:")

    if ctx.left_id_col not in inspect_df.columns or ctx.right_id_col not in inspect_df.columns:
        if "unique_id_l" in inspect_df.columns and "unique_id_r" in inspect_df.columns:
            ctx.left_id_col, ctx.right_id_col = "unique_id_l", "unique_id_r"
        else:
            ctx.left_id_col, ctx.right_id_col = None, None

    if ctx.left_id_col and ctx.left_id_col in inspect_df.columns:
        inspect_ids.update(inspect_df[ctx.left_id_col].dropna().tolist())
    if ctx.right_id_col and ctx.right_id_col in inspect_df.columns:
        inspect_ids.update(inspect_df[ctx.right_id_col].dropna().tolist())

    if job.chart_type == SplinkChartType.CLUSTER and ctx.left_id_col and ctx.df_pairs is not None:
        try:
            if ctx.df_clustered is None:
                ctx.df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
                    ctx.df_pairs, 0.01
                )
            ctx.clustered_df = ctx.df_clustered.as_pandas_dataframe()
            clustered_df_local = cast(pd.DataFrame, ctx.clustered_df)
            cluster_id_map = dict(
                zip(clustered_df_local[ctx.unique_id_col], clustered_df_local["cluster_id"])
            )
            cluster_id_series = None
            if ctx.left_id_col in inspect_df.columns:
                cluster_id_series = inspect_df[ctx.left_id_col].map(cluster_id_map)
            if ctx.right_id_col and ctx.right_id_col in inspect_df.columns:
                right_cluster = inspect_df[ctx.right_id_col].map(cluster_id_map)
                cluster_id_series = right_cluster if cluster_id_series is None else cluster_id_series.combine_first(right_cluster)
            if cluster_id_series is not None:
                inspect_df = inspect_df.assign(cluster_id=cluster_id_series)
        except Exception as exc:  # pylint: disable=W0718
            print(f"Unable to attach cluster_id to members table: {exc}")

    display_cols = [
        col
        for col in [
            "cluster_id",
            "match_probability",
            ctx.left_id_col,
            ctx.right_id_col,
            "midpoint_day_l",
            "midpoint_day_r",
        ]
        if col and col in inspect_df.columns
    ]
    if display_cols:
        print_df = inspect_df[display_cols]
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            None,
        ):
            print(print_df)
    else:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            None,
        ):
            print(inspect_df.reset_index())
        print_df = None

    ctx.inspect_df = inspect_df
    ctx.inspect_ids = inspect_ids
    return print_df


def _to_record_number(value: Any) -> int | None:
    """Parse record_number-like values to int where possible."""
    try:
        if pd.isna(value):
            return None
    except Exception:  # pylint: disable=W0718
        pass
    try:
        return int(value)
    except Exception:  # pylint: disable=W0718
        try:
            return int(float(str(value)))
        except Exception:  # pylint: disable=W0718
            return None


def _write_waterfall_html_with_table(
    chart_path: Path,
    display_df: pd.DataFrame,
    waterfall_spec: dict[str, Any],
) -> None:
    """Write waterfall chart HTML with synchronized member table section."""
    save_offline_chart(
        waterfall_spec,
        filename=str(chart_path),
        overwrite=True,
        print_msg=False,
    )

    header_cells = "".join(f"<th>{html.escape(str(col))}</th>" for col in display_df.columns)
    table_rows = []
    for _, row in display_df.iterrows():
        record_number = _to_record_number(row.get("record_number"))
        row_cells = "".join(
            f"<td>{html.escape('' if pd.isna(val) else str(val))}</td>" for val in row
        )
        if record_number is None:
            row_attrs = 'class="waterfall-row" data-record-number=""'
        else:
            row_attrs = (
                f'class="waterfall-row" id="waterfall-row-{record_number}" '
                f'data-record-number="{record_number}"'
            )
        table_rows.append(f"<tr {row_attrs}>{row_cells}</tr>")

    table_html = (
        "\n<section style=\"margin:24px 12px;\">"
        "\n  <h3 style=\"margin:0 0 8px 0; font-family:sans-serif;\">Waterfall chart members</h3>"
        "\n  <div style=\"overflow-x:auto;\">"
        "\n    <table style=\"border-collapse:collapse; font-family:Menlo,Consolas,monospace; font-size:9px; min-width:720px; line-height:1.1;\">"
        f"\n      <thead><tr>{header_cells}</tr></thead>"
        f"\n      <tbody>{''.join(table_rows)}</tbody>"
        "\n    </table>"
        "\n  </div>"
        "\n  <style>"
        "\n    table th, table td { border:1px solid #ddd; padding:1px 4px; text-align:left; white-space:nowrap; }"
        "\n    table thead th { background:#f4f4f4; font-weight:600; }"
        "\n    table tbody tr:nth-child(even) { background:#fafafa; }"
        "\n    .waterfall-row { cursor:pointer; }"
        "\n    .waterfall-row:hover { background:#eef5ff !important; }"
        "\n    .waterfall-row.is-active { background:#dfefff !important; }"
        "\n  </style>"
        "\n</section>\n"
    )

    html_text = chart_path.read_text(encoding="utf-8")
    if "</body>" in html_text:
        html_prefix, html_suffix = html_text.rsplit("</body>", 1)
        html_text = f"{html_prefix}{table_html}</body>{html_suffix}"
    else:
        html_text += table_html

    embed_prefix = "vegaEmbed('#mychart', "
    embed_suffix = ").catch(console.error);"
    embed_start = html_text.find(embed_prefix)
    embed_end = html_text.find(embed_suffix, embed_start)
    if embed_start != -1 and embed_end != -1:
        spec_expr = html_text[embed_start + len(embed_prefix):embed_end]
        bridge_script = (
            "const waterfallSpec = "
            + spec_expr
            + ";\n"
            "vegaEmbed('#mychart', waterfallSpec)\n"
            "  .then((result) => {\n"
            "    const view = result.view;\n"
            "    window.waterfallView = view;\n"
            "    const highlightRow = (recordNumber) => {\n"
            "      const target = String(recordNumber);\n"
            "      const rows = document.querySelectorAll('.waterfall-row');\n"
            "      rows.forEach((row) => {\n"
            "        row.classList.toggle('is-active', row.dataset.recordNumber === target);\n"
            "      });\n"
            "    };\n"
            "    if (!window.__waterfallRowClickBound) {\n"
            "      document.addEventListener('click', (event) => {\n"
            "        const target = event.target;\n"
            "        if (!(target instanceof Element)) {\n"
            "          return;\n"
            "        }\n"
            "        const row = target.closest('.waterfall-row');\n"
            "        if (!row) {\n"
            "          return;\n"
            "        }\n"
            "        const value = Number(row.getAttribute('data-record-number'));\n"
            "        if (!Number.isFinite(value)) {\n"
            "          console.warn('Invalid record_number in table row:', row.getAttribute('data-record-number'));\n"
            "          return;\n"
            "        }\n"
            "        view.signal('record_number', value).runAsync().catch((err) => {\n"
            "          console.warn('Unable to set record_number signal:', err);\n"
            "        });\n"
            "      });\n"
            "      window.__waterfallRowClickBound = true;\n"
            "    }\n"
            "    try {\n"
            "      view.addSignalListener('record_number', (_name, value) => {\n"
            "        highlightRow(value);\n"
            "      });\n"
            "      window.requestAnimationFrame(() => {\n"
            "        highlightRow(view.signal('record_number'));\n"
            "      });\n"
            "    } catch (err) {\n"
            "      console.warn('Signal synchronization unavailable:', err);\n"
            "    }\n"
            "  })\n"
            "  .catch(console.error);"
        )
        html_text = html_text[:embed_start] + bridge_script + html_text[embed_end + len(embed_suffix):]
    else:
        print("Unable to locate vegaEmbed call for table-chart synchronization.")

    chart_path.write_text(html_text, encoding="utf-8")


def _handle_waterfall_chart(
    linker: Linker,
    job: SplinkVisualizeJob,
    ctx: VisualizeContext,
    print_df: pd.DataFrame | None,
) -> bool:
    """Render waterfall chart and augmented HTML table output when requested."""
    if job.chart_type != SplinkChartType.WATERFALL:
        return False

    if ctx.inspect_df is None or len(ctx.inspect_df) == 0:
        print("No records match the requested midpoint filters; skipping waterfall chart.")
        return True

    if print_df is None:
        print_df = ctx.inspect_df.copy()

    inspect_dict = cast(list[dict[str, Any]], ctx.inspect_df.to_dict(orient="records"))
    waterfall = linker.visualisations.waterfall_chart(inspect_dict, filter_nulls=False)
    waterfall_no_summary = _truncate_long_chart_values(waterfall)

    print_df = print_df.copy()
    if "record_number" not in print_df.columns:
        print_df = print_df.reset_index(drop=False).rename(columns={"index": "record_number"})

    _fields_to_show = [
        "record_number",
        "cluster_id",
        "match_probability",
        ctx.left_id_col,
        ctx.right_id_col,
        "midpoint_day_l",
        "midpoint_day_r",
    ]
    _fields_to_show = [c for c in _fields_to_show if c in print_df.columns]

    display_columns = [
        c
        for c in [
            "record_number",
            "cluster_id",
            "match_probability",
            ctx.left_id_col,
            ctx.right_id_col,
            "midpoint_day_l",
            "midpoint_day_r",
        ]
        if c and c in print_df.columns
    ]
    display_df = print_df[display_columns].copy() if display_columns else print_df.copy()
    if "match_probability" in display_df.columns:
        display_df["match_probability"] = display_df["match_probability"].map(
            lambda x: f"{x:.6f}" if pd.notna(x) else ""
        )

    waterfall_spec = cast(dict[str, Any], waterfall_no_summary.to_dict())
    chart_path = Path("waterfall_chart_with_table.html")
    _write_waterfall_html_with_table(chart_path, display_df, waterfall_spec)
    print(f"Waterfall chart written to {chart_path.resolve()}")
    try:
        webbrowser.open(chart_path.resolve().as_uri())
    except Exception as exc:  # pylint: disable=W0718
        print(f"Unable to auto-open chart in browser: {exc}")
    return True


def _handle_cluster_chart(linker: Linker, job: SplinkVisualizeJob, ctx: VisualizeContext) -> bool:
    """Render cluster studio dashboard for selected rows when requested."""
    if job.chart_type != SplinkChartType.CLUSTER:
        return False

    if ctx.link_type != SplinkLinkType.DEDUPE_ONLY:
        print("Cluster studio charts are only available for dedupe models.")
        return True

    if ctx.df_pairs is None:
        return True

    if ctx.df_clustered is None:
        print("\nGenerating unconstrained clusters for charting…")
        ctx.df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
            ctx.df_pairs, 0.01
        )
    else:
        print("\nUsing cached unconstrained clusters for charting…")

    print("\nGenerating Cluster Studio dashboard…")
    try:
        if ctx.clustered_df is None:
            ctx.clustered_df = ctx.df_clustered.as_pandas_dataframe()
        clustered_df_local = cast(pd.DataFrame, ctx.clustered_df)
        inspect_ids = ctx.inspect_ids or set()
        if inspect_ids:
            filtered = clustered_df_local[clustered_df_local[ctx.unique_id_col].isin(inspect_ids)]
        else:
            filtered = clustered_df_local.iloc[0:0]
        cluster_ids = sorted(filtered["cluster_id"].dropna().unique().tolist())
    except Exception as exc:  # pylint: disable=W0718
        print(f"Unable to compute cluster_ids for dashboard: {exc}")
        return True

    if not cluster_ids:
        print("No clusters found for the selected rows; skipping dashboard.")
        return True

    try:
        pairs_df = ctx.df_pairs.as_pandas_dataframe()
        mask = pd.Series(False, index=pairs_df.index)
        if ctx.left_id_col and ctx.left_id_col in pairs_df.columns:
            mask |= pairs_df[ctx.left_id_col].isin(ctx.inspect_ids or set())
        if ctx.right_id_col and ctx.right_id_col in pairs_df.columns:
            mask |= pairs_df[ctx.right_id_col].isin(ctx.inspect_ids or set())
        filtered_pairs = pairs_df.loc[mask]
        clustered_df_local = cast(pd.DataFrame, ctx.clustered_df)
        filtered_clusters = clustered_df_local[
            clustered_df_local[ctx.unique_id_col].isin(ctx.inspect_ids or set())
        ]
        df_pairs_filtered = linker._db_api.register_table(
            filtered_pairs,
            f"__splink__df_pairs_inspect_{id(filtered_pairs)}",
        )
        df_clustered_filtered = linker._db_api.register_table(
            filtered_clusters,
            f"__splink__df_clustered_inspect_{id(filtered_clusters)}",
        )
    except Exception as exc:  # pylint: disable=W0718
        print(f"Unable to filter dashboard inputs: {exc}")
        return True

    linker.visualisations.cluster_studio_dashboard(
        df_pairs_filtered,
        df_clustered_filtered,
        "cluster_studio.html",
        cluster_ids=cluster_ids,
        overwrite=True,
    )
    print("Cluster studio dashboard written to cluster_studio.html")
    return True


def _dispatch_chart(
    linker: Linker,
    job: SplinkVisualizeJob,
    ctx: VisualizeContext,
    print_df: pd.DataFrame | None,
) -> None:
    """Dispatch chart-specific rendering after prediction context is prepared."""
    if _handle_waterfall_chart(linker, job, ctx, print_df):
        return
    if _handle_comparison_chart(linker, job, ctx.df_pairs):
        return
    _handle_cluster_chart(linker, job, ctx)


def run_splink_visualize(linker: Linker, job: SplinkVisualizeJob) -> None:
    """Render the requested Splink visualization for the active linker/model."""
    alt.renderers.enable("browser")
    ctx = _make_visualize_context(linker)

    if _handle_model_charts(linker, job):
        return
    if _handle_parameter_estimate_comparisons(linker, job):
        return
    if _handle_unlinkables_chart(linker, job, ctx):
        return

    if job.chart_type == SplinkChartType.COMPARISON:

        def _query_dicts(sql: str) -> list[dict]:
            rel = linker._db_api._execute_sql_against_backend(sql)
            cols = [d[0] for d in (rel.description or [])]
            rows = rel.fetchall()
            return [dict(zip(cols, row)) for row in rows]

        try:
            linker.table_management.invalidate_cache()
            drop_all_splink_tables(linker._db_api._execute_sql_against_backend, _query_dicts)
        except Exception:  # pylint: disable=W0718
            pass

    ctx.df_pairs = _predict_pairs_for_visualization(linker, job, ctx)
    ctx.pd_pairs = ctx.df_pairs.as_pandas_dataframe()
    ctx.inspect_df = ctx.pd_pairs
    ctx.inspect_ids = set()

    print_df = _prepare_inspection_subset(linker, job, ctx)
    _dispatch_chart(linker, job, ctx, print_df)
