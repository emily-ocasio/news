"""
This module provides a base class for Semigroup instances.
"""

from typing import Protocol, Self


class Semigroup(Protocol):
    """Base class for Semigroup instances.

    To implement a semigroup instance, create a sub-class of Semigroup and
    override the addition_operation method ensuring that the closure and
    associativity laws hold.

    """

    # def __or__(self , other: Self) -> Self:
    #     if not isinstance(other, self.__class__):
    #         raise ValueError("Incompatible Semigroup addition")
    #     return self.append(other)

    # def __eq__(self, other: object) -> bool:
    #     if not isinstance(other, self.__class__):
    #         return False
    #     return self.value == other.value

    def append(self, other: Self) -> Self:
        """Combines two Semigroup instances."""
        ...
