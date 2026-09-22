"""Exact critical-value enumeration of the CoCurve interaction path."""
from __future__ import annotations

import multiprocessing as mp
from typing import Sequence

import numpy as np


def solve_with_breakpoint(
    h: np.ndarray,
    costs: np.ndarray,
    ratio: float,
    interaction_strength: float,
    unit_layers: Sequence[int] | None,
    layer_cap: float | None,
    protected_layers: set[int] | None,
) -> tuple[list[int], float, float]:
    """Solve once and return the first larger strength that changes a greedy step."""
    size = h.shape[0]
    costs = np.asarray(costs, dtype=np.float64)
    layers = None if unit_layers is None else np.asarray(unit_layers, dtype=np.int64)
    protected = np.zeros(size, dtype=bool)
    layer_costs = layer_pruned = None
    if layers is not None:
        count = int(layers.max()) + 1
        layer_costs = np.bincount(layers, weights=costs, minlength=count)
        layer_pruned = np.zeros(count, dtype=np.float64)
        if protected_layers:
            protected = np.isin(layers, np.asarray(sorted(protected_layers)))

    target = float(costs.sum()) * ratio
    selected = np.zeros(size, dtype=bool)
    accumulated = np.zeros(size, dtype=np.float64)
    spent = 0.0
    next_breakpoint = np.inf

    while spent < target:
        eligible = ~selected & ~protected
        if layers is not None and layer_cap is not None:
            eligible &= layer_pruned[layers] + costs <= layer_costs[layers] * layer_cap
        candidates = np.flatnonzero(eligible)
        if not candidates.size:
            break
        intercept = 0.5 * h[candidates, candidates] / costs[candidates]
        slope = accumulated[candidates] / costs[candidates]
        score = intercept + interaction_strength * slope
        winner = int(np.argmin(score))

        slope_delta = slope - slope[winner]
        intercept_delta = intercept - intercept[winner]
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = np.where(
                np.abs(slope_delta) > 1e-300,
                -intercept_delta / slope_delta,
                np.inf,
            )
        crossings = crossings[
            np.isfinite(crossings) & (crossings > interaction_strength + 1e-12)
        ]
        if crossings.size:
            next_breakpoint = min(next_breakpoint, float(crossings.min()))

        unit = int(candidates[winner])
        selected[unit] = True
        accumulated += h[:, unit]
        spent += float(costs[unit])
        if layers is not None:
            layer_pruned[layers[unit]] += float(costs[unit])

    return np.flatnonzero(selected).tolist(), spent / float(costs.sum()), next_breakpoint


def _enumerate_range(state: tuple, low: float, high: float, limit: int) -> list[dict]:
    h, costs, ratio, layers, cap, protected = state
    records = []
    strength = float(low)
    while strength <= high and len(records) < limit:
        pruned, actual, following = solve_with_breakpoint(
            h, costs, ratio, strength, layers, cap, protected
        )
        end = min(following, high) if np.isfinite(following) else high
        records.append({
            "lambda_lo": strength,
            "lambda_hi": float(end),
            "pruned_units": [int(unit) for unit in pruned],
            "actual_prune_ratio": float(actual),
        })
        if not np.isfinite(following) or following >= high:
            break
        strength = float(following) + 1e-9
    if not records or records[-1]["lambda_hi"] < high - 1e-9:
        raise RuntimeError(f"interaction path did not cover [{low}, {high}]")
    return records


_WORKER_STATE: tuple | None = None


def _init_worker(state: tuple) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _worker(job: tuple[int, float, float, int]) -> tuple[int, list[dict]]:
    index, low, high, limit = job
    assert _WORKER_STATE is not None
    return index, _enumerate_range(_WORKER_STATE, low, high, limit)


def enumerate_path(
    h: np.ndarray,
    costs: np.ndarray,
    ratio: float,
    unit_layers: Sequence[int] | None,
    layer_cap: float | None,
    protected_layers: set[int] | None,
    *,
    maximum: float = 1.0,
    workers: int = 1,
    chunks: int | None = None,
    max_solutions: int = 400_000,
) -> list[dict]:
    """Enumerate every distinct greedy solution on ``lambda in [0, maximum]``.

    This walks analytic breakpoints rather than sampling a grid.  Adjacent
    intervals with the same final mask are merged; recurring non-adjacent masks
    retain all of their intervals.
    """
    state = (
        np.asarray(h, dtype=np.float64),
        np.asarray(costs, dtype=np.float64),
        float(ratio),
        None if unit_layers is None else list(map(int, unit_layers)),
        layer_cap,
        protected_layers,
    )
    workers = max(1, int(workers))
    if workers == 1:
        raw = _enumerate_range(state, 0.0, float(maximum), max_solutions)
    else:
        chunks = int(chunks or 4 * workers)
        bounds = np.linspace(0.0, maximum, chunks + 1)
        jobs = [(i, float(bounds[i]), float(bounds[i + 1]), max_solutions)
                for i in range(chunks)]
        pieces = {}
        with mp.get_context("spawn").Pool(
            workers, initializer=_init_worker, initargs=(state,)
        ) as pool:
            for index, records in pool.imap_unordered(_worker, jobs):
                pieces[index] = records
        raw = [record for index in range(chunks) for record in pieces[index]]

    merged: list[dict] = []
    for record in raw:
        key = frozenset(record["pruned_units"])
        if merged and frozenset(merged[-1]["pruned_units"]) == key:
            merged[-1]["lambda_hi"] = record["lambda_hi"]
        else:
            merged.append(record)

    unique: list[dict] = []
    index: dict[frozenset[int], int] = {}
    for record in merged:
        key = frozenset(record["pruned_units"])
        interval = [record["lambda_lo"], record["lambda_hi"]]
        if key in index:
            unique[index[key]]["intervals"].append(interval)
        else:
            record["intervals"] = [interval]
            index[key] = len(unique)
            unique.append(record)
    return unique
