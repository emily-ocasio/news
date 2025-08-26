""" Concrete String implementation as a Monoid wrapper of str """
from typing import Self, NamedTuple

from .array import Array
from .maybe import Just, Maybe, Nothing
from .monoid import Monoid

class Char(str):
    """Wraps a single character value as a subtype of str."""
    def __new__(cls, value: str):
        if len(value) != 1:
            raise ValueError("Char must be a single character")
        return str.__new__(cls, value)

    def __repr__(self):
        return f'"{self}"'

    def __eq__(self, other) -> bool:
        return str.__eq__(self, other) if isinstance(other, Char) else False

    def isdigit(self) -> bool:
        return str.isdigit(self)

    def isalpha(self) -> bool:
        return str.isalpha(self)

class String(str, Monoid):
    """Wraps a string value as a subtype of str and matches Monoid Protocol."""
    def __new__(cls, value: str):
        return str.__new__(cls, value)

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError("Incompatible Semigroup addition")
        return self.append(other)

    def __repr__(self):
        return f'"{self}"'

    def __eq__(self, other) -> bool:
        return str.__eq__(self, other) if isinstance(other, String) else False

    def __hash__(self):
        return str.__hash__(self)

    def append(self: Self, other: Self) -> Self:
        return type(self)(str.__add__(self, other))

    @classmethod
    def mempty(cls) -> Self:
        """Returns the identity element for the String monoid."""
        return cls("")

    def uncons(self) -> Maybe['HeadTail']:
        """Returns the first character and the rest of the string as a tuple."""
        match self:
            case "":
                return Nothing
            case _:
                head = Char(self[0])
                tail = String(self[1:])
                return Just(HeadTail(head, tail))

class HeadTail(NamedTuple):
    """ Result of uncons operation on String"""
    head: Char
    tail: String

def from_char_array(chars: Array[Char]) -> String:
    """Converts an array of Char to a String."""
    return String("".join(chars))

def from_string(s: String) -> Maybe[int]:
    """Converts a String to an integer."""
    return Just(int(s)) if s.isdigit() else Nothing
