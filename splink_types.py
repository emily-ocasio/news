"""
Application-defined keys for Splink state registry.
"""
from enum import Enum


class SplinkType(str, Enum):
    """
    Tags for storing Splink linkers in monadic state.
    """
    DEDUP = "dedup"
    ORPHAN = "orphan"
    ORPHAN_ADJ_SCORE = "orphan_adj_score"
    POSTADJ_ORPHAN_CLUSTER = "postadj_orphan_cluster"
    SHR = "shr"
