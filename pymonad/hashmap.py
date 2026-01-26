"""Implements a purescript-like HashMap type in Python."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Iterator, ItemsView, KeysView, ValuesView, TypeVar, Self

from .functor import Functor, map  # pylint: disable=redefined-builtin
from .maybe import Just, Maybe

K = TypeVar("K")
A = TypeVar("A")
B = TypeVar("B")
K0 = TypeVar("K0")
V0 = TypeVar("V0")
T = TypeVar("T")
K1 = TypeVar("K1")
K2 = TypeVar("K2")


@dataclass(frozen=True)
class HashMap[K, A](Functor[A]):
    """
    Represents an immutable HashMap backed by a Python dict.
    Functor maps over values while preserving keys.
    """

    data: dict[K, A]

    def __iter__(self) -> Iterator[K]:
        """Iterates over keys."""
        return iter(self.data)

    def __len__(self) -> int:
        """Returns the number of entries in the HashMap."""
        return len(self.data)

    def __getitem__(self, key: K) -> A:
        """Returns the value associated with the given key."""
        return self.data[key]

    def __contains__(self, key: K) -> bool:
        """Returns True if the key exists in the HashMap."""
        return key in self.data

    def items(self) -> ItemsView[K, A]:
        """Returns a view of the HashMap's items."""
        return self.data.items()

    def keys(self) -> KeysView[K]:
        """Returns a view of the HashMap's keys."""
        return self.data.keys()

    def values(self) -> ValuesView[A]:
        """Returns a view of the HashMap's values."""
        return self.data.values()

    def get(self, key: K, default: A | None = None) -> A | None:
        """Returns the value for key if present, otherwise default."""
        return self.data.get(key, default)

    @classmethod
    def make(cls, data: dict[K, A]) -> HashMap[K, A]:
        """Creates a new instance of HashMap with a copy of the dict."""
        return cls(dict(data))

    @classmethod
    def empty(cls) -> HashMap[K, A]:
        """Creates an empty HashMap."""
        return cls({})

    @classmethod
    def from_mappable(
        cls,
        mappable: Mapping[K0, V0],
        key_map: Callable[[K0], Maybe[K]],
        value_map: Callable[[V0], A],
    ) -> HashMap[K, A]:
        """
        Builds a HashMap from a mapping by transforming keys and values.
        Only entries whose key_map returns Just(k) are included.
        """
        data: dict[K, A] = {}
        for in_key, in_value in mappable.items():
            match key_map(in_key):
                case Just(key):
                    data[key] = value_map(in_value)
        return cls(data)

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[T],
        key_value_map: Callable[[T], tuple[Maybe[K], A]],
    ) -> HashMap[K, A]:
        """
        Builds a HashMap from an iterable by mapping each item to a Maybe key
        and a value. Only entries whose key is Just(k) are included.
        """
        data: dict[K, A] = {}
        for item in iterable:
            maybe_key, value = key_value_map(item)
            match maybe_key:
                case Just(key):
                    data[key] = value
        return cls(data)

    def union(self: Self, other: HashMap[K, A]) -> HashMap[K, A]:
        """Left-biased union of two HashMaps."""
        if not other.data:
            return self
        if not self.data:
            return other
        data = dict(other.data)
        data.update(self.data)
        return self.__class__(data)

    def map_keys(self, f: Callable[[K], K1]) -> HashMap[K1, A]:
        """Maps a function over keys while preserving values."""
        return HashMap({f(k): v for k, v in self.data.items()})

    def set(self: Self, key: K, value: A) -> Self:
        """Returns a new HashMap with the key set to the given value."""
        new_data = dict(self.data)
        new_data[key] = value
        return self.__class__(new_data)

    def delete(self: Self, key: K) -> Self:
        """Returns a new HashMap with the key removed (if present)."""
        if key not in self.data:
            return self
        new_data = dict(self.data)
        del new_data[key]
        return self.__class__(new_data)

    def __rand__(self, other: Callable[[A], B]) -> HashMap[K, B]:
        """Defines the right-hand side of the map operation."""
        return map(other, self)

    def map(self: Self, f: Callable[[A], B]) -> HashMap[K, B]:
        """Maps a function over values while preserving keys."""
        return HashMap({k: f(v) for k, v in self.data.items()})

    def fold_with_index(self, f: Callable[[K, B, A], B], acc: B) -> B:
        """
        Folds over the HashMap with access to each key (index).
        The function f takes (key, acc, value) and returns the next acc.
        """
        for key, value in self.data.items():
            acc = f(key, acc, value)
        return acc

    def __repr__(self) -> str:
        """String representation of the HashMap."""
        return f"HashMap({self.data})"

    def __eq__(self, other) -> bool:
        """Equality check for HashMap."""
        return isinstance(other, HashMap) and self.data == other.data


def flatten_map(
    nested: HashMap[K0, HashMap[K1, A]],
    normalize: Callable[[K0, K1], K2],
) -> HashMap[K2, A]:
    """
    Flattens a nested HashMap by normalizing outer and inner keys into one key.
    Uses left-biased union when normalized keys collide.
    """
    def step(outer_key: K0, acc: HashMap[K2, A], inner_map: HashMap[K1, A]) -> HashMap[K2, A]:
        normalized = inner_map.map_keys(lambda inner_key: normalize(outer_key, inner_key))
        return acc.union(normalized)

    return nested.fold_with_index(step, HashMap.empty())
