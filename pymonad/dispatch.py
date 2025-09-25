"""
Defines functions with side effects and maps them to intents
"""
from dataclasses import dataclass
from typing import Callable

import duckdb
from splink import Linker, DuckDBAPI
from .string import String

class InputPrompt(String):
    """
    Represents a prompt for user input.
    """
@dataclass(frozen=True)
class PutLine:
    """Base I/O: output a line."""
    s: str
    end: str = '\n'

@dataclass(frozen=True)
class GetLine:
    """Base I/O: input a line with prompt."""
    prompt: InputPrompt

@dataclass(frozen=True)
class SplinkDedupeJob:
    """
    Splink deduplication intent
    """
    duckdb_path: str
    input_table: str
    settings: dict
    predict_threshold: float
    cluster_threshold: float
    pairs_out: str
    clusters_out: str
    train_first: bool = False

REAL_DISPATCH: dict[type, Callable] = {}

def intentdef(intent: type) -> Callable[[Callable], Callable]:
    """
    Decorator for intent functions
    Registers the function in the REAL_DISPATCH dictionary
    """
    def decorator(func: Callable) -> Callable:
        REAL_DISPATCH[intent] = func
        return func
    return decorator

@intentdef(PutLine)
def _putline(x: PutLine) -> None:
    """
    Print a message to the console
    """
    print(x.s, end=x.end)

@intentdef(GetLine)
def _getline(x: GetLine) -> String:
    """
    Get a line of input from the user
    """
    return String(input(x.prompt if x.prompt[-1] == ' ' else x.prompt + ' '))

@intentdef(SplinkDedupeJob)
def _splink_dedupe(job: SplinkDedupeJob) -> tuple[str, str]:
    db_api = DuckDBAPI(connection=duckdb.connect(job.duckdb_path))
    linker = Linker(job.input_table, job.settings, db_api=db_api)

    if job.train_first:
        # Use your prediction blocking rules for training too
        training_rules = job.settings.get(
            "blocking_rules_to_generate_predictions", [])
        # Splink v4: British spelling + explicit training rules param
        linker.training.estimate_parameters_using_expectation_maximisation(
            blocking_rule=training_rules
        )

    df_pairs = linker.inference.predict(
        threshold_match_probability=job.predict_threshold
    )
    df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_pairs,
        threshold_match_probability=job.cluster_threshold,
    )
    # 3) Persist outputs into stable tables in the same DB
    con = duckdb.connect(job.duckdb_path)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {job.pairs_out} AS "
                    f"SELECT * FROM {df_pairs.physical_name}")
        con.execute(f"CREATE OR REPLACE TABLE {job.clusters_out} AS "
                    f"SELECT * FROM {df_clusters.physical_name}")
    finally:
        con.close()

    return (job.pairs_out, job.clusters_out)
