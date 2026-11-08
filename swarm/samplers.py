"""Joint action samplers. All preserve each agent's conditional P(A | its history) = p_i.

Every sampler takes p (length N), pairs (an N/2 x 2 array of agent indices, column 0 = position 0,
column 1 = position 1, orientation random and never shown to models) and an rng. It returns
(actions as an int8 array, info dict). Samplers never see rewards, which label is better, or the future.

  independent      one uniform per agent.
  pair_shared      PRIMARY intervention. With probability alpha, all position-0 agents share one uniform U
                   and all position-1 agents share an independent V; otherwise draws are independent. Each
                   pair always gets two independent uniforms, so within-pair action probabilities and
                   current expected reward match the independent sampler. Only dependence across pairs changes.
  global_shared    SECONDARY. With probability alpha, every agent shares one uniform. This also changes
                   within-pair coordination, so it is not the clean test.
  block            COMPARATOR. Pairs are split at random into n_blocks blocks, each with its own shared U and V
                   (no mixture). With 2 blocks and homogeneous p this matches the N0 = 4 reference variance
                   but has different large-excursion tails from the pair_shared mixture.

alpha can be fixed, or set each round to restore the conditional variance of the action share x' to the
independent N0-agent reference: Var = S / (N * N0), where S = sum_i p_i (1 - p_i).
"""

from dataclasses import dataclass, field

import numpy as np


def cov_sum_shared(p: np.ndarray) -> float:
    """sum over i<j of [min(p_i, p_j) - p_i p_j], the covariance two agents gain from sharing one uniform."""
    s = np.sort(np.asarray(p, dtype=float))
    m = s.size
    if m < 2:
        return 0.0
    sum_min = float(np.sum(s * (m - 1 - np.arange(m))))
    sum_prod = float((s.sum() ** 2 - np.sum(s**2)) / 2)
    return sum_min - sum_prod


def restore_alpha(S: float, C: float, N: int, N0: int) -> tuple[float, bool]:
    """alpha such that (S + 2 alpha C) / N^2 = S / (N N0). Returns (alpha clipped to [0, 1], feasible)."""
    if N <= N0 or S <= 0:
        return 0.0, True
    if C <= 0:
        return 0.0, False
    a = S * (N / N0 - 1) / (2 * C)
    return float(min(max(a, 0.0), 1.0)), bool(0.0 <= a <= 1.0)


def alpha_pair_homogeneous(N: int, N0: int = 4) -> float:
    """Closed-form restoring alpha for pair_shared at homogeneous p."""
    return 0.0 if N <= N0 else (1 / N0 - 1 / N) / (1 / 2 - 1 / N)


def alpha_global_homogeneous(N: int, N0: int = 4) -> float:
    return 0.0 if N <= N0 else (1 / N0 - 1 / N) / (1 - 1 / N)


@dataclass
class Sampler:
    """A configured sampler. kind: independent | pair_shared | global_shared | block."""

    kind: str
    alpha: float | None = None  # None with pair_shared/global_shared means "restore to N0 variance"
    N0: int = 4
    n_blocks: int = 2
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            if self.kind in ("pair_shared", "global_shared"):
                tag = "restore" if self.alpha is None else f"{self.alpha:g}"
                self.name = f"{self.kind}:{tag}"
            elif self.kind == "block":
                self.name = f"block:{self.n_blocks}"
            else:
                self.name = self.kind

    def __call__(self, p: np.ndarray, pairs: np.ndarray, rng: np.random.Generator):
        N = p.size
        if self.kind == "independent":
            return (rng.random(N) < p).astype(np.int8), {"alpha": 0.0, "feasible": True, "shared": False}

        if self.kind == "pair_shared":
            if self.alpha is None:
                S = float(np.sum(p * (1 - p)))
                C = cov_sum_shared(p[pairs[:, 0]]) + cov_sum_shared(p[pairs[:, 1]])
                alpha, feasible = restore_alpha(S, C, N, self.N0)
            else:
                alpha, feasible = self.alpha, True
            shared = rng.random() < alpha
            if shared:
                u = np.empty(N)
                u[pairs[:, 0]] = rng.random()
                u[pairs[:, 1]] = rng.random()
            else:
                u = rng.random(N)
            return (u < p).astype(np.int8), {"alpha": alpha, "feasible": feasible, "shared": shared}

        if self.kind == "global_shared":
            if self.alpha is None:
                S = float(np.sum(p * (1 - p)))
                alpha, feasible = restore_alpha(S, cov_sum_shared(p), N, self.N0)
            else:
                alpha, feasible = self.alpha, True
            shared = rng.random() < alpha
            u = np.full(N, rng.random()) if shared else rng.random(N)
            return (u < p).astype(np.int8), {"alpha": alpha, "feasible": feasible, "shared": shared}

        if self.kind == "block":
            n_pairs = pairs.shape[0]
            block_of_pair = rng.permutation(n_pairs) % self.n_blocks  # balanced, independent of p
            u = np.empty(N)
            for b in range(self.n_blocks):
                rows = pairs[block_of_pair == b]
                u[rows[:, 0]] = rng.random()
                u[rows[:, 1]] = rng.random()
            return (u < p).astype(np.int8), {"alpha": 1.0, "feasible": True, "shared": True}

        raise ValueError(f"unknown sampler kind {self.kind!r}")


def parse_sampler(spec: str, N0: int = 4) -> Sampler:
    """'independent', 'pair_shared:restore', 'pair_shared:0.3', 'global_shared:restore', 'block:2'."""
    kind, _, arg = spec.partition(":")
    if kind == "independent":
        return Sampler("independent", N0=N0)
    if kind in ("pair_shared", "global_shared"):
        alpha = None if arg in ("", "restore") else float(arg)
        return Sampler(kind, alpha=alpha, N0=N0)
    if kind == "block":
        return Sampler("block", n_blocks=int(arg or 2), N0=N0)
    raise ValueError(f"unknown sampler spec {spec!r}")
