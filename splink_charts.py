"""
Visualize the latest Splink dedupe/linkage run.
"""

from enum import Enum
from pymonad import (
    Run,
    InputPrompt,
    get_line,
    put_line,
    pure,
    splink_visualize_job,
    unit,
    has_splink_linker,
)
from splink_types import SplinkType
from menuprompts import NextStep, MenuChoice, MenuPrompts, input_from_menu


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


def _run_visualization(
    splink_key: SplinkType,
    left_raw: str,
    right_raw: str,
) -> Run[NextStep]:
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
            splink_key=splink_key,
            left_midpoints=left_midpoints,
            right_midpoints=right_midpoints,
        )
        ^ pure(NextStep.CONTINUE)
    )


SPLINK_PROMPTS = (
    "[D]edup",
    "[O]rphan",
    "[S]HR",
)


class SplinkChoice(Enum):
    """
    Menu options for Splink visualization.
    """
    DEDUP = MenuChoice("D")
    ORPHAN = MenuChoice("O")
    SHR = MenuChoice("S")
    QUIT = MenuChoice("Q")


def _choice_to_type(choice: SplinkChoice) -> SplinkType:
    match choice:
        case SplinkChoice.DEDUP:
            return SplinkType.DEDUP
        case SplinkChoice.ORPHAN:
            return SplinkType.ORPHAN
        case SplinkChoice.SHR:
            return SplinkType.SHR
        case _:
            return SplinkType.DEDUP


def _visualize_for_type(splink_key: SplinkType) -> Run[NextStep]:
    return (
        put_line("For waterfall charts, provide the desired midpoint days for the records:")
        ^ get_line(InputPrompt("Left midpoint days> "))
        >> (
            lambda left_raw: get_line(InputPrompt("Right midpoint days> "))
            >> (lambda right_raw: _run_visualization(splink_key, str(left_raw), str(right_raw)))
        )
    )


def _check_and_visualize(splink_key: SplinkType) -> Run[NextStep]:
    return has_splink_linker(splink_key) >> (
        lambda exists: (
            _visualize_for_type(splink_key)
            if exists
            else put_line("No splink data available for visualization")
            ^ pure(NextStep.CONTINUE)
        )
    )


def splink_charts() -> Run[NextStep]:
    """
    Entry point for controller to visualize the latest Splink run.
    """
    return (
        put_line("Select Splink run type:")
        ^ input_from_menu(MenuPrompts(SPLINK_PROMPTS))
        >> (lambda choice: pure(SplinkChoice(choice)))
        >> (
            lambda choice: pure(NextStep.CONTINUE)
            if choice == SplinkChoice.QUIT
            else _check_and_visualize(_choice_to_type(choice))
        )
    )
