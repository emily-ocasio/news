"""
Additional types and functions for validation in Run context
"""
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from typing import TypeVar

from .array import Array
from .run import Run, ErrorPayload, run_except, foldm_either_loop_bind, UserAbort
from .semigroup import Semigroup
from .either import Either, Left, Right
from .string import String
from .monad import Unit, unit
from .traverse import array_traverse_run
from .validation import V, Valid, Invalid

S = TypeVar('S')
T = TypeVar('T')
E = TypeVar('E', bound=Semigroup)

class FailureType(Enum):
    """
    Enumeration of validation error types
    """

@dataclass(frozen=True)
class FailureDetail[S]:
    """
    Represents a single validation error
    """
    type: FailureType
    s: String

type FailureDetails[S] = Array[FailureDetail[S]] # All failures for one item

@dataclass(frozen=True)
class ItemFailures[S]:
    """
    Represents an item with all its validation errors
    """
    item: S
    details: FailureDetails[S]

type ItemsFailures[S] = Array[ItemFailures[S]] # All failures for many items

# Each validator produces an Array[FailureDetail] in a Run[V] context
# Even if only one failure, it is in an array to allow accumulation
# If no failures, validator returns Valid(Unit) in Run context
# DEPRECATED: use PureValidator with validate_item_pure/validate_all_pure.
type Validator[S] = Callable[[S], Run[V[FailureDetails[S], Unit]]]
type PureValidator[S] = Callable[[S], V[FailureDetails[S], Unit]]
type PureValidators[S] = Array[PureValidator[S]]

@dataclass(frozen=True)
class ValidationAcc[S]:
    """
    Pure-validation accumulator across all items.
    """
    valid_items: Array[S]
    failures: ItemsFailures[S]
    processed: int

@dataclass(frozen=True)
class ProcessAllAcc[S, T]:
    """
    Accumulator for process_all fold.
    """
    results: Array[T]
    failures: ItemsFailures[S]
    processed: int

@dataclass(frozen=True)
class StopProcessing[S, T]:
    """
    Marker for short-circuiting process_all.
    """
    acc: ProcessAllAcc[S, T]
    reason: ErrorPayload

@dataclass(frozen=True)
class ProcessAcc[S, T]:
    """
    Processing accumulator for monadic happy-path execution.
    """
    results: Array[T]
    failures: ItemsFailures[S]
    processed: int

@dataclass(frozen=True)
class StopRun[S, T]:
    """
    Marker for short-circuiting process_items.
    """
    acc: ProcessAcc[S, T]
    reason: ErrorPayload

def no_op(_: ErrorPayload) -> Run[Unit]:
    """
    No-op error handler
    """
    return Run.pure(unit)

def validate_item_pure(validators: PureValidators[S], item: S) \
    -> V[ItemsFailures[S], S]:
    """
    Apply pure validators to an item and accumulate errors applicatively.
    """
    if validators.length == 0:
        return V.pure(item)

    validations = (lambda v: v(item)) & validators
    combined: V[FailureDetails[S], Unit] = validations.foldl(
        lambda a, b: a ^ b,
        V.pure(unit)
    )
    match combined.validity:
        case Invalid(details):
            return V.invalid(Array((ItemFailures(item, details),)))
        case Valid(_):
            return V.pure(item)

def validate_all_pure(validators: PureValidators[S], items: Array[S]) \
    -> ValidationAcc[S]:
    """
    Validate all items using pure applicative validation only.
    """
    def step(acc: ValidationAcc[S], item: S) -> ValidationAcc[S]:
        v_item = validate_item_pure(validators, item)
        match v_item.validity:
            case Invalid(failures):
                return ValidationAcc(
                    acc.valid_items,
                    acc.failures.append(failures),
                    acc.processed + 1
                )
            case Valid(valid_item):
                return ValidationAcc(
                    Array.snoc(acc.valid_items, valid_item),
                    acc.failures,
                    acc.processed + 1
                )
    init = ValidationAcc(Array.mempty(), Array.mempty(), 0)
    return items.foldl(step, init)

def process_items(
    happy: Callable[[S], Run[T]],
    items: Array[S],
    render: Callable[[ErrorPayload], FailureDetails[S]],
    unhappy: Callable[[ErrorPayload], Run[Unit]] = no_op
) -> Run[Either[StopRun[S, T], ProcessAcc[S, T]]]:
    """
    Process items monadically without running validation.
    """
    def acc_init() -> ProcessAcc[S, T]:
        return ProcessAcc(Array.mempty(), Array.mempty(), 0)

    def acc_with_result(acc: ProcessAcc[S, T], result: T) -> ProcessAcc[S, T]:
        return ProcessAcc(
            Array.snoc(acc.results, result),
            acc.failures,
            acc.processed + 1
        )

    def acc_with_run_failure(acc: ProcessAcc[S, T],
                             item: S,
                             err: ErrorPayload) -> ProcessAcc[S, T]:
        return ProcessAcc(
            acc.results,
            Array.snoc(acc.failures, ItemFailures(item, render(err))),
            acc.processed + 1
        )

    def handle_run_result(
        acc: ProcessAcc[S, T],
        item: S,
        result: Either[ErrorPayload, T]
    ) -> Run[Either[StopRun[S, T], ProcessAcc[S, T]]]:
        match result:
            case Right(r):
                return Run.pure(Right.pure(acc_with_result(acc, r)))
            case Left(err):
                if isinstance(err.app, UserAbort):
                    return Run.pure(Left(StopRun(acc, err)))
                return unhappy(err) ^ Run.pure(
                    Right.pure(acc_with_run_failure(acc, item, err))
                )

    def process_one(acc: ProcessAcc[S, T], item: S) \
        -> Run[Either[StopRun[S, T], ProcessAcc[S, T]]]:
        return run_except(happy(item)) >> \
            (lambda ei: handle_run_result(acc, item, ei))

    if items.length == 0:
        return Run.pure(Right.pure(acc_init()))

    return foldm_either_loop_bind(items, acc_init(), process_one)

def validate_item(validators: Array[Validator[S]], item: S) \
    -> Run[V[ItemsFailures[S], S]]:
    """
    DEPRECATED: use validate_item_pure for pure validation concerns.
    Apply a list of effectul validators to an item, accumulating errors.
    """
    def traverse_validators() -> Run[Array[V[FailureDetails, Unit]]]:
        def validate(val: Validator[S]) -> Run[V[FailureDetails, Unit]]:
            # Run a single validator on the item
            # The item is captured in the closure
            return val(item)
        return array_traverse_run(validators, validate) if validators.length > 0 \
            else Run.pure(Array((V.pure(unit),))) # No validators, return empty
    def append_validation(a: V[FailureDetails, Unit],
                          b: V[FailureDetails, Unit]) \
                            -> V[FailureDetails, Unit]:
        # Accumulate validation results, discarding successes
        return a ^ b
    def to_items_failures(v_combined: V[FailureDetails, Unit]) -> \
        Run[V[ItemsFailures[S], S]]:
        # Lift back up to Run
        # If any failures, return the ItemFailures with item reference
        # Otherwise return the original item
        match v_combined.validity:
            case Invalid(details):
                return Run.pure(V.invalid(
                    Array((ItemFailures(item, details),))))
            case Valid(_):
                return Run.pure(V.pure(item))
    def combine_validations(vs: Array[V[FailureDetails, Unit]]) \
        -> Run[V[ItemsFailures[S], S]]:
        # Accumulate all the validation results in array into single V
        return to_items_failures(vs.foldl(append_validation, V.pure(unit)))

    # First traverse the array of validators and apply them to the item,
    # within a Run context, producing vs ~ Array[V[FailureDetails, Unit]]
    # then combine themdsf into a single V[FailureDetails, Unit],
    # and finally map to return the original item on success
    return \
        traverse_validators() >> combine_validations


def process_all(validators: Array[Validator[S]],
                render: Callable[[ErrorPayload], FailureDetails],
                happy: Callable[[S], Run[T]],
                items: Array[S],
                unhappy: Callable[[ErrorPayload], Run[Unit]] = no_op ) \
                -> Run[Either[StopProcessing[S, T],
                              V[ItemsFailures[S], Array[T]]]]:
    """
    DEPRECATED: legacy hybrid API that interleaves validation and processing.
    Prefer validate_all_pure + process_items.

    Given array of items, perform effectful validation on each,
    and if validations pass, perform a happy path action.
    Keep trying each subsequent item even if some fail validation or 
    final action.
    Collect all failures (validation or happy-path) using applicative V
    Short-circuit with StopProcessing when a user-initiated abort is detected.
    """

    def acc_init() -> ProcessAllAcc[S, T]:
        return ProcessAllAcc(Array.mempty(), Array.mempty(), 0)

    def acc_with_result(acc: ProcessAllAcc[S, T], result: T) \
        -> ProcessAllAcc[S, T]:
        return ProcessAllAcc(
            Array.snoc(acc.results, result),
            acc.failures,
            acc.processed + 1
        )

    def acc_with_failures(acc: ProcessAllAcc[S, T],
                          failures: ItemsFailures[S]) \
        -> ProcessAllAcc[S, T]:
        return ProcessAllAcc(
            acc.results,
            acc.failures.append(failures),
            acc.processed + 1
        )

    def acc_with_run_failure(acc: ProcessAllAcc[S, T],
                             item: S,
                             err: ErrorPayload) \
        -> ProcessAllAcc[S, T]:
        return ProcessAllAcc(
            acc.results,
            Array.snoc(acc.failures, ItemFailures(item, render(err))),
            acc.processed + 1
        )

    def handle_run_result(acc: ProcessAllAcc[S, T],
                          item: S,
                          result: Either[ErrorPayload, T]) \
        -> Run[Either[StopProcessing[S, T], ProcessAllAcc[S, T]]]:
        match result:
            case Right(r):
                return Run.pure(Right(acc_with_result(acc, r)))
            case Left(err):
                if isinstance(err.app, UserAbort):
                    return Run.pure(Left(StopProcessing(acc, err)))
                return \
                    unhappy(err) ^ \
                    Run.pure(Right.pure(acc_with_run_failure(acc, item, err)))

    def handle_validation(acc: ProcessAllAcc[S, T],
                          item: S,
                          v_item: V[ItemsFailures[S], S]) \
        -> Run[Either[StopProcessing[S, T], ProcessAllAcc[S, T]]]:
        match v_item.validity:
            case Invalid(failures):
                return Run.pure(Right(acc_with_failures(acc, failures)))
            case Valid(valid_item):
                return run_except(happy(valid_item)) >> \
                    (lambda ei: handle_run_result(acc, item, ei))

    def process_one(acc: ProcessAllAcc[S, T],
                    item: S) -> Run[Either[StopProcessing[S, T],
                                          ProcessAllAcc[S, T]]]:
        return validate_item(validators, item) >> \
            (lambda v_item: handle_validation(acc, item, v_item))

    def to_v(acc: ProcessAllAcc[S, T]) \
        -> V[ItemsFailures[S], Array[T]]:
        if acc.failures.length > 0:
            return V.invalid(acc.failures)
        return V.pure(acc.results)

    if items.length == 0:
        return Run.pure(Right(V.pure(Array(()))))

    return \
        foldm_either_loop_bind(items, acc_init(), process_one) >> \
        (lambda ei: Run.pure(
            Left(ei.l) if isinstance(ei, Left) else Right(to_v(ei.r))
        ))
