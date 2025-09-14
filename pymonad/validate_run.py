"""
Additional types and functions for validation in Run context
"""
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from typing import TypeVar

from .array import Array
from .run import Run, ErrorPayload, run_except
from .semigroup import Semigroup
from .either import Either, Left, Right
from .string import String
from .monad import Unit, unit
from .traverse import array_traverse, array_sequence
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
type Validator[S] = Callable[[S], Run[V[FailureDetails[S], Unit]]]

def validate_item(validators: Array[Validator[S]], item: S) \
    -> Run[V[ItemsFailures[S], S]]:
    """
    Apply a list of effectul validators to an item, accumulating errors.
    """
    def traverse_validators() -> Run[Array[V[FailureDetails, Unit]]]:
        def validate(val: Validator[S]) -> Run[V[FailureDetails, Unit]]:
            # Run a single validator on the item
            # The item is captured in the closure
            return val(item)
        return array_traverse(validators, validate) if validators.length > 0 \
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
                items: Array[S]) \
                -> Run[V[ItemsFailures[S], Array[T]]]:
    """
    Given array of items, perform effectful validation on each,
    and if validations pass, perform a happy path action.
    Keep trying each subsequent item even if some fail validation or 
    final action.
    Collect all failures (validation or happy-path) using applicative V
    """

    def run_happy_catching(item: S) -> Run[V[ItemsFailures[S], T]]:
        def catch_run_exceptions(result: Either[ErrorPayload, T]) \
            -> Run[V[ItemsFailures, T]]:
            match result:
                case Right(r):
                    # No run exception
                    return Run.pure(V.pure(r))
                case Left(err):
                    # Re-render run exception into item failure
                    return Run.pure(
                        V.invalid(Array((ItemFailures(item, render(err)),))))
        return run_except(happy(item)) >> catch_run_exceptions
    def process_validation(v_item: V[ItemsFailures[S], S]) \
        -> Run[V[ItemsFailures[S], T]]:
        # Run the happy path if validations pass
        match v_item.validity:
            case Invalid(failures):
                return Run.pure(V.invalid(failures))
            case Valid(item):
                return run_happy_catching(item)
    def process_one(item: S) -> Run[V[ItemsFailures[S], T]]:
        return \
            validate_item(validators, item) >> process_validation
    def combine_results(vs: Array[V[ItemsFailures[S], T]]) \
        -> Run[V[ItemsFailures[S], Array[T]]]:
        # Combine all the results into a single array
        return Run.pure(array_sequence(vs))
    return array_traverse(items, process_one) >> combine_results
