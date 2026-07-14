"""Resolve the immutable publication session before opening resources."""

import argparse
from dataclasses import dataclass

from publication_profiles import (
    Availability,
    PublicationKey,
    PublicationProfile,
    PUBLICATION_PROFILES,
    resolve_publication_profile,
)
from pymonad import Just, Maybe, Nothing, String


class PublicationArgument(String):
    """Raw publication value supplied through the startup interface."""


class PublicationSelectionPrompt(String):
    """Prompt displayed when no publication argument was supplied."""


@dataclass(frozen=True)
class StartupArguments:
    """Strongly typed startup arguments."""

    publication: Maybe[PublicationKey]


def _publication_options() -> String:
    """Render registered publication keys without a native collection."""
    def append_option(
        key: PublicationKey, current: String, _: PublicationProfile
    ) -> String:
        separator = String("/") if current else String.mempty()
        return current | separator | String(key)

    return PUBLICATION_PROFILES.fold_with_index(
        append_option, String.mempty()
    )


def _parse_arguments() -> StartupArguments:
    """Parse the optional command-line selector into a Maybe value."""
    parser = argparse.ArgumentParser(
        description="Run the news analysis application for one publication."
    )
    parser.add_argument(
        "--publication",
        default="",
        metavar="KEY",
        help=f"publication profile key ({_publication_options()})",
    )
    raw = PublicationArgument(parser.parse_args().publication.strip().lower())
    publication: Maybe[PublicationKey] = (
        Nothing if not raw else Just(PublicationKey(raw))
    )
    return StartupArguments(publication)


def _prompt_for_publication() -> PublicationKey:
    """Prompt once for a publication when the CLI argument is absent."""
    prompt = PublicationSelectionPrompt(
        f"Select publication [{_publication_options()}] > "
    )
    raw = input(prompt).strip().lower()
    return PublicationKey(raw)


def select_publication_profile() -> PublicationProfile:
    """Select and validate one operational profile before resource setup."""
    arguments = _parse_arguments()
    match arguments.publication:
        case Just(key):
            selected_key = key
        case _:
            selected_key = _prompt_for_publication()

    match resolve_publication_profile(selected_key):
        case Just(profile):
            if profile.session_availability is Availability.UNAVAILABLE:
                raise SystemExit(
                    f"Publication '{selected_key}' is recognized but unavailable."
                )
            return profile
        case _:
            raise SystemExit(
                f"Invalid publication '{selected_key}'. Expected one of: "
                f"{_publication_options()}."
            )
