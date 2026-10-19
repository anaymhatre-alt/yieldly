"""Fuzzy partitions built from triangular sets.

Every input is covered by a Ruspini partition: neighbouring triangles overlap so
memberships always sum to 1, and the two outer sets are shoulders that extend to
the ends of the number line. That removes the "dead zones" and silent fallbacks
of fixed-width triangular sets: every finite input activates at least one rule.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def partition_memberships(x, centers: Sequence[float]) -> np.ndarray:
    """Return an (n, k) array of membership degrees for k sets at ``centers``.

    Set i peaks at ``centers[i]`` and falls to zero at its neighbours' centres.
    Values beyond the first or last centre belong fully to the outer set.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size < 2 or np.any(np.diff(c) <= 0):
        raise ValueError("centers must be a strictly increasing 1-D sequence of length >= 2")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite numbers")

    xc = np.clip(x, c[0], c[-1])
    idx = np.clip(np.searchsorted(c, xc, side="right") - 1, 0, c.size - 2)
    t = (xc - c[idx]) / (c[idx + 1] - c[idx])
    mu = np.zeros((x.size, c.size))
    rows = np.arange(x.size)
    mu[rows, idx] = 1.0 - t
    mu[rows, idx + 1] = t
    return mu
