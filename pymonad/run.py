"""
# run_effects_menu_all.py — Python "Run" (à la PureScript)
#   with Reader + State + Except in the controller
"""

# pylint: disable=W0212
# pylint: disable=E1101

from dataclasses import dataclass
from typing import Any, Callable, Dict, TypeVar

from .either import Either, Left, Right
from .functor import Functor
A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")


# ===== Intents (data-only) =====
@dataclass(frozen=True)
class Ask:
    """Reader intent."""


@dataclass(frozen=True)
class Get:
    """State.get intent."""


@dataclass(frozen=True)
class Put:
    """State.put intent."""
    s: Any


@dataclass(frozen=True)
class Throw:
    """Except.throw intent (payload is user data)."""
    e: Any


@dataclass(frozen=True)
class Rethrow:
    """Lift pure Either into Except."""
    res: Either


@dataclass(frozen=True)
class PutLine:
    """Base I/O: output a line."""
    s: str


@dataclass(frozen=True)
class GetLine:
    """Base I/O: input a line with prompt."""
    prompt: str

# Private control-flow sentinel for non-local exit of Throw/Rethrow(Left)


class _Thrown(Exception):
    __slots__ = ("payload",)

    def __init__(self, payload: Any):
        self.payload = payload

# ===== Run carrier =====


@dataclass(slots=True, frozen=True)
class Run[A](Functor[A]):
    """
    Carrier for effectful computations, parameterized by type A.
    Encapsulates a computation step and a performer for handling intents.
    """
    _step: Callable[["Run[Any]"], A]
    _perform: Callable[[Any, "Run[Any]"], Any]

    def __init__(self, step: Callable[["Run[Any]"], A],
                 perform: Callable[[Any, "Run[Any]"], Any]):
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "_perform", perform)

    def __rand__(self, f: Callable[[A], B]) -> "Run[B]":
        """
        Enables using the & operator for mapping a function
        over the Run.
        """
        return self.map(f)

    def map(self, f: Callable[[A], B]) -> "Run[B]":
        """
        Functor map: transforms the result of the computation using function f.
        """
        return Run(lambda self_run: f(self._step(self_run)), self._perform)  # pylint: disable=no-member

    def bind(self, f: Callable[[A], "Run[B]"]) -> "Run[B]":
        """
        Monad bind: chains computations by passing the result to function f,
          which returns a new Run.
        """
        return Run(lambda self_run: f(self._step(self_run))._step(self_run),
                   self._perform)

def _unhandled(intent: Any, *_: Any) -> Any:
    raise RuntimeError(f"Unhandled intent: {type(intent).__name__}")


def pure(x: A) -> Run[A]:
    """
    Lift a pure value into the Run monad.
    """
    return Run(lambda _self: x, _unhandled)

# ===== Smart constructors =====


def ask() -> Run[dict]:
    """Create a Run action to request the environment (Reader effect)."""
    return Run(lambda self: self._perform(Ask(), self), _unhandled)


def get() -> Run[Any]:
    """Create a Run action to get the current state (State effect)."""
    return Run(lambda self: self._perform(Get(), self), _unhandled)


def put(s: Any) -> Run[None]:
    """Create a Run action to set the state (State effect)."""
    return Run(lambda self: self._perform(Put(s), self), _unhandled)


def throw(e: Any) -> Run[Any]:
    """Create a Run action to throw an error (Except effect)."""
    return Run(lambda self: self._perform(Throw(e), self), _unhandled)


def rethrow(res: Either) -> Run[Any]:
    """Create a Run action to lift an Either into Except effect."""
    return Run(lambda self: self._perform(Rethrow(res), self), _unhandled)


def put_line(s: str) -> Run[None]:
    """Create a Run action to output a line (base effect)."""
    return Run(lambda self: self._perform(PutLine(s), self), _unhandled)


def get_line(prompt: str) -> Run[str]:
    """Create a Run action to input a line with a prompt (base effect)."""
    return Run(lambda self: self._perform(GetLine(prompt), self), _unhandled)

# ===== Eliminators =====


def run_reader(env: dict, prog: Run[A]) -> Run[A]:
    """
    Interprets Reader intents by providing the environment to computations.
    """
    def step(self_run: Run[Any]) -> A:
        parent = self_run._perform

        def perform(intent, current):
            match intent:
                case Ask(): return env
                case _: return parent(intent, current)
        inner = Run(prog._step, perform)
        return inner._step(inner)
    return Run(step, lambda i, c: c._perform(i, c))


def run_state(initial: Any, prog: Run[A]) -> Run[tuple[Any, A]]:
    """
    Interprets State intents by managing and updating the state
        throughout the computation.
    Returns a Run containing a tuple of the final state and the result.
    """
    def step(self_run: Run[Any]) -> tuple[Any, A]:
        parent = self_run._perform
        box = {"s": initial}

        def perform(intent, current):
            match intent:
                case Get():
                    return box["s"]
                case Put(s):
                    box["s"] = s
                    return None
                case _:
                    return parent(intent, current)
        inner = Run(prog._step, perform)
        v = inner._step(inner)
        return (box["s"], v)
    return Run(step, lambda i, c: c._perform(i, c))


def run_except(prog: Run[A]) -> Run[Either]:
    """
    Interpret Throw/Rethrow into Either:
      - Throw(e), Rethrow(Left e) → Left(e)
      - Rethrow(Right a) → a
    Unanticipated Python exceptions are also returned as 
        Left(ex) here (by design).
    """

    def step(self_run: Run[Any]) -> Either:
        parent = self_run._perform

        def perform(intent, current):
            match intent:
                case Throw(e): raise _Thrown(e)
                case Rethrow(Left(e)): raise _Thrown(e)
                case Rethrow(Right(a)): return a
                case _: return parent(intent, current)
        inner = Run(prog._step, perform)
        try:
            return Right(inner._step(inner))
        except _Thrown as te:
            return Left(te.payload)
        except Exception as ex: # pylint: disable=broad-except
            return Left(ex)
    return Run(step, lambda i, c: c._perform(i, c))


def run_base_effect(dispatch: Dict[type, Callable[[Any], Any]],
                    prog: Run[A]) -> A:
    """
    Executes the base effects by dispatching intents
    to their corresponding performer functions.
    """
    def perform_base(intent, _):
        fn = dispatch.get(type(intent))
        if fn is None:
            raise RuntimeError(f"No performer for {type(intent).__name__}")
        return fn(intent)
    seeded = Run(prog._step, perform_base)
    return seeded._step(seeded)

# ===== Controller (uses Reader + State + Except) =====

# pylint: disable=C0301
# def menu() -> Run[None]:
#     """
#     Options:
#       1) Increment counter (State)
#       2) Throw an app error (Except)
#       3) Show current counter (State) and env-based greeting (Reader via ask)
#       4) Quit
#     """
#     def loop() -> Run[None]:
#         return (
#             ask().bind(lambda env:
#                        putLine(
#                           f"[{env.get('app', 'app')}] 1) Inc  2) Throw  3) Show  4) Quit")
#                        .bind(lambda _:
#                              getLine(f"{env.get('prompt', '> ')}")
#                              .bind(lambda choice:
#                                    get().bind(lambda n:
#                                               # Branches
#                                               (  # 1) increment counter
#                                                   put(n + 1)
#                                                   .bind(lambda _:
#                                                         putLine(f"Count was {n}; now {n+1}"))
#                                                   .bind(lambda _:
#                                                         loop())
#                                               ) if choice == "1" else
#                                               (  # 2) throw an error via Except
#                                                   throw(
#                                                       {"code": "Boom", "msg": "Something went wrong"})
#                                               ) if choice == "2" else
#                                               (  # 3) show state + env greeting
#                                                   putLine(
#                                                       f"Hello from {env.get('app', 'app')}! Counter = {n}")
#                                                   .bind(lambda _:
#                                                         loop())
#                                               ) if choice == "3" else
#                                               (  # 4) quit
#                                                   pure(None)
#                                               ) if choice == "4" else
#                                               (  # invalid
#                                                   putLine("Invalid choice").bind(lambda _:
#                                                                                  loop())
#                                               )
#                                               ))))
#         )
#     return loop()


# # ===== Real base performers (console) =====
# REAL_DISPATCH: Dict[type, Callable[[Any], Any]] = {
#     PutLine: lambda i: print(i.s),
#     GetLine: lambda i: input(i.prompt),
# }

# # ===== Wiring: choose effect order (semantics differ) =====
#     # A) rollback-on-error (Except inside State): state updates discarded if an error occurs
#     # prog = run_reader(env, runState(0, run_except(menu())))

#     # B) preserve-state-on-error (State inside Except): capture final state even on failure
#     prog = runReader(env, run_except(runState(0, menu())))

#     result = runBaseEffect(REAL_DISPATCH, prog)
#     print("Program result:", result)
