""" Concrete String implementation as a Monoid wrapper of str """
from dataclasses import dataclass
from typing import Self, NamedTuple

from .array import Array
from .maybe import Just, Maybe, Nothing
from .monoid import Monoid
from .semigroup import Semigroup

@dataclass(frozen=True)
class Char:
    """Wraps a single character value."""
    c: str

    def __post_init__(self):
        if len(self.c) != 1:
            raise ValueError("Char must be a single character")

    def __repr__(self):
        """String representation of the Char."""
        return f'"{self.c}"'

    def __eq__(self, other) -> bool:
        return self.c == other.c if isinstance(other, Char) else False
    
    def isdigit(self) -> bool:
        """Checks if the character is a digit."""
        return self.c.isdigit()
    
    def isalpha(self) -> bool:
        """Checks if the character is an alphabetic character."""
        return self.c.isalpha()


@dataclass(frozen=True)
class String(Monoid):
    """Wraps a string value while implicitly matching Monoid Protocol."""
    s: str

    def __or__(self , other: Self) -> Self:
        if not isinstance(other, self.__class__):
            raise ValueError("Incompatible Semigroup addition")
        return self.append(other)
    
    def __repr__(self):
        """String representation of the String."""
        return f'"{self.s}"'
    
    def __eq__(self, other) -> bool: 
        return self.s == other.s if isinstance(other, String) else False

    def append(self: Self, other: Self) -> Self:
        return type(self)(self.s + other.s)

    @classmethod
    def mempty(cls) -> Self:
        """Returns the identity element for the String monoid."""
        return cls("")

    def uncons(self) -> Maybe['HeadTail']:
        """Returns the first character and the rest of the string as a tuple."""
        match self.s:
            case "":
                return Nothing
            case _:
                head = Char(self.s[0])
                tail = String(self.s[1:])
                return Just(HeadTail(head, tail))


class HeadTail(NamedTuple):
    head: Char
    tail: String

def fromCharArray(chars: Array[Char]) -> String:
    """Converts an array of Char to a String."""
    return String("".join(c.c for c in chars))

def fromString(s: String) -> Maybe[int]:
    """Converts a String to an integer."""
    return Just(int(s.s)) if s.s.isdigit() else Nothing