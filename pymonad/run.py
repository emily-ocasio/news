"""
# run_effects_menu_all.py — Python "Run" (à la PureScript)
#   with Reader + State + Except in the controller
"""

# pylint: disable=W0212
# pylint: disable=E1101

from collections.abc import Callable
from dataclasses import dataclass
import traceback
from typing import cast, Any, TypeVar

from .array import Array
from .dispatch import GetLine, PutLine, InputPrompt
from .either import Either, Left, Right
from .environment import Environment, Namespace, PromptKey, AllPrompts, \
    all_prompts
from .functor import Functor
from .monad import ap
from .monoid import Monoid
from .string import String
from .tuple import Tuple

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
E = TypeVar("E")
M = TypeVar("M", bound=Monoid)

Err = TypeVar("Err")   # application-specific error type

class ErrorPayload[Err](String):
    """Application-specific error payload.

    Usage:
      - ErrorPayload[str]("message")
      - ErrorPayload[MyError]( "message", MyError.SomeCase )

    The instance behaves like the existing String (constructed with the message)
    and carries an optional `app` attribute of type Optional[Err].
    """
    app: Err | None = None
    def __new__(cls, s: str, app: Err | None = None):
        obj = super().__new__(cls, s)
        object.__setattr__(obj, "app", app)
        return obj


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
    e: ErrorPayload[Any]

@dataclass(frozen=True)
class Rethrow:
    """Lift pure Either into Except."""
    res: Either[ErrorPayload[Any], Any]


# Private control-flow sentinel for non-local exit of Throw/Rethrow(Left)


class _Thrown(Exception):
    __slots__ = ("payload",)

    def __init__(self, payload: ErrorPayload[Any]):
        self.payload: ErrorPayload[Any] = payload

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

    def __rshift__(self, f: Callable[[A], "Run[B]"]) -> "Run[B]":
        """
        Enables using the >> operator for chaining computations
        over the Run.
        """
        return self._bind(f)

    def __mul__(self: "Run[Callable[[B], C]]", other: "Run[B]") -> "Run[C]":
        """
        Enables using the * operator for applying a function
        in the Run to a value in the Run.
        """
        return self._apply(other)

    def __xor__(self, other: "Run[B]") -> "Run[B]":
        """
        Enables using the ^ operator for applying a function
        in the Run to a value in the Run.
        """
        return self.apply_second(other)

    def map(self, f: Callable[[A], B]) -> "Run[B]":
        """
        Functor map: transforms the result of the computation using function f.
        """
        return Run(lambda self_run: f(self._step(self_run)), self._perform)  # pylint: disable=no-member

    def _apply(self: "Run[Callable[[B], C]]", other: "Run[B]") -> "Run[C]":
        """
        Applies a function in the context of the
        Run monad to a value in the context.
        """
        return ap(self, other, Run)

    def _bind(self, f: Callable[[A], "Run[B]"]) -> "Run[B]":
        """
        Monad bind: chains computations by passing the result to function f,
          which returns a new Run.
        """
        return Run(lambda self_run: f(self._step(self_run))._step(self_run),
                   self._perform)

    def apply_second(self, other: "Run[B]") -> "Run[B]":
        """
        Applies the second Run to the first Run,
        discarding the result of the first Run.
        """
        return self >> (lambda _: other)

def _unhandled(intent: Any, *_: Any) -> Any:
    raise RuntimeError(f"Unhandled intent: {type(intent).__name__}")


def pure(x: A) -> Run[A]:
    """
    Lift a pure value into the Run monad.
    """
    return Run(lambda _self: x, _unhandled)

# ===== Smart constructors =====

def ask() -> Run[Environment]:
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


def get_line(prompt: InputPrompt) -> Run[String]:
    """Create a Run action to input a line with a prompt (base effect)."""
    return Run(lambda self: self._perform(GetLine(prompt), self), _unhandled)

# ===== Eliminators =====

def run_reader(env: Environment, prog: Run[A]) -> Run[A]:
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

def local(modify: Callable[[Environment], Environment], prog: "Run[A]")\
    -> "Run[A]":
    """
    Run sub program with a modified reader environment.
    """
    return \
        ask() >> (lambda env:
        run_reader(modify(env), prog)
        )

def run_state(initial: M, prog: Run[A]) -> Run[Tuple[M, A]]:
    """
    Interprets State intents by managing and updating the state
        throughout the computation.
    Returns a Run containing a tuple of the final state and the result.
    """
    def step(self_run: Run[Any]) -> Tuple[M, A]:
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
        return Tuple(box["s"], v)
    return Run(step, lambda i, c: c._perform(i, c))


def run_except(prog: Run[A]) -> Run[Either[ErrorPayload, A]]:
    """
    Interpret Throw/Rethrow into Either:
      - Throw(e), Rethrow(Left e) → Left(e)
      - Rethrow(Right a) → a
    Unanticipated Python exceptions are also returned as 
        Left(ex) here (by design).
    """

    def step(self_run: Run[Any]) -> Either[ErrorPayload, A]:
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
            tb_exc = traceback.TracebackException.from_exception(ex)
            tb_str = ''.join(tb_exc.format())
            # Default ErrorPayload carries only a diagnostic message;
            # applications can create ErrorPayload[MyError](msg, app_err)
            # where needed.
            return Left(ErrorPayload(f"Unhandled exception: {ex}\n{tb_str}"))
    return Run(step, lambda i, c: c._perform(i, c))


def run_base_effect(dispatch: dict[type, Callable[[Any], Any]],
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


S = TypeVar("S")   # stop marker

def foldm_either_loop_bind(
    items: Array[A],
    init: B,
    step: Callable[[B, A], "Run[Either[S, B]]"],   # (acc, a) -> Run[Either S B]
) -> "Run[Either[S, B]]":
    """
    Loop-based, short-circuiting foldM:
      - Each iteration uses Run.bind to build the next step.
      - Immediately _step() to force effects and observe Either.
      - Break on Left; otherwise continue with Right acc'.
    No long bind chain is kept in memory.
    """
    def run(self_run: "Run[object]") -> "Either[S, B]":

        # Start as Right(init) inside Run
        acc_r: "Run[Either[S, B]]" = pure(Right(init))
        for a in items:
            def iteration(acc: Either[S, B], a=a) -> "Run[Either[S, B]]":
                if isinstance(acc, Left):
                    return pure(acc)
                return step(acc.r, a)
            # One monadic link: acc_r >>= \e -> case e of
            # Left s -> pure(Left s) ; Right acc -> step acc a
            acc_r = acc_r._bind(iteration)
            # Force now, so we can short-circuit via host break
            res = acc_r._step(self_run)
            if isinstance(res, Left):
                return res              # hard short-circuit: no more steps run
            # res = Right(acc') ; reset acc_r to a pure Right for the next link
            acc_r = pure(res)

        # Finished all items; force the final Right acc once
        return acc_r._step(self_run)

    # Pass-through performer
    return Run(run, lambda intent, current: current._perform(intent, current))

def with_namespace(
    ns: Namespace,
    prompts: AllPrompts,
    subprog: "Run[A]"
) -> "Run[A]":
    """
    Run subprog with a with specific prompts injected into the environment
    prompts_ns will determine the new Namespace
    """
    def mod(env: Environment) -> Environment:
        named_prompts = env["prompts_by_ns"]
        new_prompts_for_ns = all_prompts(env, ns) | prompts
        new_prompts_by_ns = named_prompts | { ns: new_prompts_for_ns }
        return cast(Environment, env | {
            "prompt_ns": ns,
            "prompts_by_ns": new_prompts_by_ns,
        })

    return local(mod, subprog)

def input_with_prompt(key_or_literal: PromptKey | InputPrompt,
                      *,
                      fallback_global: bool = True) -> "Run[String]":
    """
    Look up prompt in the current namespace only (strict).
    If fallback_global=True, try global "" before using the literal.
    """
    def resolve(env: Environment) -> InputPrompt:
        prompt_key = PromptKey(key_or_literal)
        literal = InputPrompt(key_or_literal)
        ns_input_prompts = all_prompts(env)['input_prompts']
        if fallback_global:
            global_input_prompts = all_prompts(
                env, Namespace(""))['input_prompts']
            return ns_input_prompts.get(prompt_key,
                    global_input_prompts.get(prompt_key,
                    literal))
        return ns_input_prompts.get(prompt_key, literal)
    return \
        ask() >> (lambda env:
        get_line(resolve(env)))

def input_number(key: PromptKey,
                 min_value: int = 0,
                 max_value: int = 10000000000) -> Run[int]:
    """
    Get a number input from the user within the specified range.
    Repeats until a valid number is entered.
    """
    def loop(repeat_prompt: PromptKey | InputPrompt = InputPrompt('> '))\
        -> Run[int]:
        return \
            input_with_prompt(repeat_prompt) >> (lambda s:
            _process_input(str(s))
        )

    def _process_input(line: str) -> Run[int]:
        try:
            value = int(line)
            if min_value <= value <= max_value:
                return pure(value)
            return loop()
        except ValueError:
            return loop()

    return loop(key)
