"""Frozen-policy population runs.

Each round: draw a uniformly random perfect matching (with random orientation), compute every agent's
p_i = P(A | its history) from the frozen policy, sample all actions jointly with the chosen sampler,
pay each pair, then update every memory synchronously.

Random streams are separate and seeded by the run seed, so treatments are paired: the same seed gives
the same matchings, the same initial state and common sampling randomness across samplers. Histories
then diverge as a consequence of treatment, which is intended (total effect).
"""

import math
from dataclasses import dataclass

import numpy as np

from .game import HistorySpace, initial_codes, payoff
from .samplers import Sampler


@dataclass
class RunResult:
    seed: int
    N: int
    sampler: str
    entered: bool
    first_entry: int | None  # 1-indexed round of first x_t >= threshold
    rmst: float  # restricted mean first-entry time contribution: first_entry if entered, else T
    occupancy: float  # fraction of rounds with x_t >= threshold
    final_in: bool
    final_x: float
    exits: int  # times x_t dropped back below the threshold after being at or above it
    mean_reward: float  # mean per-agent pairwise payoff over all rounds
    mean_x_late: float  # mean share playing A over the second half of the horizon
    reached_all_A: bool  # x_t == 1 in some round (unanimity is exponentially rare under independence: diagnostic only)
    mean_alpha: float
    infeasible_frac: float  # rounds where the variance-restoring alpha had to be clipped
    shared_frac: float


def run_population(
    pA: np.ndarray,
    space: HistorySpace,
    N: int,
    T: int,
    sampler: Sampler,
    init: str,
    seed: int,
    threshold: float = 0.75,
    keep_trace: bool = False,
    game: str = "coord",
):
    if N % 2:
        raise ValueError("N must be even for perfect matching")
    match_rng = np.random.default_rng([seed, 0])
    sample_rng = np.random.default_rng([seed, 1])
    init_rng = np.random.default_rng([seed, 2])

    codes = initial_codes(space, init, N, init_rng)
    xs, rewards, alphas = np.empty(T), np.empty(T), np.empty(T)
    infeasible = shared = 0
    partner = np.empty(N, dtype=np.int64)

    for t in range(T):
        pairs = match_rng.permutation(N).reshape(-1, 2)
        partner[pairs[:, 0]], partner[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
        p = pA[codes]
        a, info = sampler(p, pairs, sample_rng)
        ap = a[partner]
        xs[t] = a.mean()
        rewards[t] = payoff(a, ap, game).mean()
        alphas[t] = info["alpha"]
        infeasible += not info["feasible"]
        shared += info["shared"]
        codes = space.next[codes, a, ap]

    inside = xs >= threshold
    entered = bool(inside.any())
    first = int(np.argmax(inside)) + 1 if entered else None
    res = RunResult(
        seed=seed,
        N=N,
        sampler=sampler.name,
        entered=entered,
        first_entry=first,
        rmst=float(first if entered else T),
        occupancy=float(inside.mean()),
        final_in=bool(inside[-1]),
        final_x=float(xs[-1]),
        exits=int(np.sum(inside[:-1] & ~inside[1:])),
        mean_reward=float(rewards.mean()),
        mean_x_late=float(xs[T // 2 :].mean()),
        reached_all_A=bool((xs == 1).any()),
        mean_alpha=float(alphas.mean()),
        infeasible_frac=infeasible / T,
        shared_frac=shared / T,
    )
    return (res, {"x": xs, "reward": rewards}) if keep_trace else res


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(results: list[RunResult]) -> list[dict]:
    """Per (N, sampler) cell. The unit is a whole population run, never an agent or interaction."""
    cells: dict[tuple, list[RunResult]] = {}
    for r in results:
        cells.setdefault((r.N, r.sampler), []).append(r)
    out = []
    for (N, s), rs in sorted(cells.items()):
        n, k = len(rs), sum(r.entered for r in rs)
        lo, hi = wilson(k, n)
        out.append(
            {
                "N": N,
                "sampler": s,
                "runs": n,
                "entry_prob": k / n,
                "entry_ci_lo": lo,
                "entry_ci_hi": hi,
                "rmst": float(np.mean([r.rmst for r in rs])),
                "occupancy": float(np.mean([r.occupancy for r in rs])),
                "final_in_prob": float(np.mean([r.final_in for r in rs])),
                "mean_reward": float(np.mean([r.mean_reward for r in rs])),
                "mean_x_late": float(np.mean([r.mean_x_late for r in rs])),
                "all_A_prob": float(np.mean([r.reached_all_A for r in rs])),
                "mean_exits": float(np.mean([r.exits for r in rs])),
                "mean_alpha": float(np.mean([r.mean_alpha for r in rs])),
                "infeasible_frac": float(np.mean([r.infeasible_frac for r in rs])),
            }
        )
    return out


def paired_contrasts(results: list[RunResult], baseline: str = "independent") -> list[dict]:
    """Entry-probability difference vs the baseline sampler at each N, paired by seed."""
    by = {(r.N, r.sampler, r.seed): r for r in results}
    Ns = sorted({r.N for r in results})
    samplers = sorted({r.sampler for r in results} - {baseline})
    out = []
    for N in Ns:
        for s in samplers:
            seeds = [sd for (n, smp, sd) in by if n == N and smp == s and (N, baseline, sd) in by]
            if not seeds:
                continue
            d = np.array([by[(N, s, sd)].entered - by[(N, baseline, sd)].entered for sd in seeds], float)
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan
            out.append({"N": N, "sampler": s, "vs": baseline, "runs": len(d), "diff": d.mean(),
                        "ci_lo": d.mean() - 1.96 * se, "ci_hi": d.mean() + 1.96 * se})
    return out
