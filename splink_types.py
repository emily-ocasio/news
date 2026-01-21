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
    SHR = "shr"
