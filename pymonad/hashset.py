"""Implements a purescript-like HashSet type in Python."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from .array import Array

A = TypeVar("A")


@dataclass(frozen=True)
class HashSet[A]:
    """
    Represents an immutable HashSet backed by a Python set.
    """

    data: set[A]

    def __iter__(self):
        """Iterates over the elements of the HashSet."""
        return iter(self.data)

    def __len__(self) -> int:
        """Returns the number of elements in the HashSet."""
        return len(self.data)

    def __contains__(self, item: A) -> bool:
        """Allows use of `item in my_hashset`."""
        return item in self.data

    def member(self, item: A) -> bool:
        """Returns True if the item exists in the HashSet."""
        return item in self.data

    @classmethod
    def empty(cls) -> "HashSet[A]":
        """Creates an empty HashSet."""
        return cls(set())

    @classmethod
    def fromArray(cls, arr: Array[A]) -> "HashSet[A]":
        """Creates a HashSet from a purescript-like Array."""
        return cls(set(arr))
