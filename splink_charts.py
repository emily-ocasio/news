"""
Visualize the latest Splink dedupe/linkage run.
"""

from pymonad import (
    Run,
    InputPrompt,
    get_line,
    put_line,
    pure,
    splink_visualize_job,
    unit,
)
from pymonad import runsplink
from menuprompts import NextStep


def _parse_midpoint_list(raw: str) -> tuple[list[int] | None, list[str]]:
    raw = raw.strip()
    if not raw:
        return None, []
    tokens = [t.strip() for t in raw.split(",")]
    values: list[int] = []
    invalid: list[str] = []
    for token in tokens:
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            invalid.append(token)
    return (values if values else None), invalid


def _run_visualization(linker, left_raw: str, right_raw: str) -> Run[NextStep]:
    left_midpoints, left_invalid = _parse_midpoint_list(left_raw)
    right_midpoints, right_invalid = _parse_midpoint_list(right_raw)
    warnings: list[str] = []
    if left_invalid:
        warnings.append(f"Left midpoint days ignored: {', '.join(left_invalid)}")
    if right_invalid:
        warnings.append(f"Right midpoint days ignored: {', '.join(right_invalid)}")

    warn_run = put_line("; ".join(warnings)) ^ pure(unit) if warnings else pure(unit)
    return (
        warn_run
        ^ splink_visualize_job(
            linker=linker,
            left_midpoints=left_midpoints,
            right_midpoints=right_midpoints,
        )
        ^ pure(NextStep.CONTINUE)
    )


def _visualize_with_linker(linker) -> Run[NextStep]:
    if linker is None:
        return put_line(
            "[C] No latest Splink linker found. Run a Splink job first."
        ) ^ pure(NextStep.CONTINUE)
    return (
        put_line("For waterfall charts, provide the desired midpoint days for the records:")
        ^ get_line(InputPrompt("Left midpoint days> "))
        >> (
            lambda left_raw: get_line(InputPrompt("Right midpoint days> "))
            >> (lambda right_raw: _run_visualization(linker, str(left_raw), str(right_raw)))
        )
    )


def splink_charts() -> Run[NextStep]:
    """
    Entry point for controller to visualize the latest Splink run.
    """
    return pure(runsplink.get_latest_splink_linker()) >> _visualize_with_linker
