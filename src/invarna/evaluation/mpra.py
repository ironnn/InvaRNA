"""MPRA rank-correlation summaries."""

from .multi_species import summarize


def summarize_mpra(frame, truth, prediction, library="library"):
    return summarize(frame, truth=truth, prediction=prediction, species=library).rename(
        columns={"species": "library"}
    )
