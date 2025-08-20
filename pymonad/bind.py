from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic


F = TypeVar("F")
A = TypeVar("A")
B = TypeVar("B")

class Bind[F, A](ABC):
    """
    Abstract base class for Bind instances.
    Extends Apply with the bind operation.
    """
    ...

    @abstractmethod
    def bind(self, m: Callable[[A], "Bind[F, B]"]) -> "Bind[F, B]":
        """
        Chains computations by passing the value inside the Bind to function f,
        which returns a new Bind instance.
        """
        ...
