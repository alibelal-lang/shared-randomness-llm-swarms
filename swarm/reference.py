"""Exact finite-state reference check from the research plan.

Synthetic population-count process: every agent plays A with p(x) = 0.02 + 0.96 sigmoid(10 (x - 0.5)),
where x is the previous round's share playing A. Start at x = 0.25; discovery is entry into x >= 0.75.
This validates the measurement design and sampler moments only. It is not LLM evidence.
"""

from math import comb, exp

import numpy as np

from .samplers import alpha_pair_homogeneous


def ref_p(x: float) -> float:
    return 0.02 + 0.96 / (1 + exp(-10 * (x - 0.5)))


def binom_pmf(N: int, p: float) -> np.ndarray:
    return np.array([comb(N, k) * p**k * (1 - p) ** (N - k) for k in range(N + 1)])


def transition_matrix(N: int, kind: str = "independent", N0: int = 4) -> np.ndarray:
    """Count-chain kernel. kind: 'independent' or 'pair_restore' (pair_shared at the restoring alpha)."""
    alpha = 0.0 if kind == "independent" else alpha_pair_homogeneous(N, N0)
    P = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        p = ref_p(k / N)
        shared = np.zeros(N + 1)  # position-0 half moves together, position-1 half moves together
        shared[0] += (1 - p) ** 2
        shared[N // 2] += 2 * p * (1 - p)
        shared[N] += p**2
        P[k] = (1 - alpha) * binom_pmf(N, p) + alpha * shared
    return P


def first_entry_prob(P: np.ndarray, start: int, target: np.ndarray, T: int) -> float:
    Q = P.copy()
    Q[target] = 0.0
    Q[target, target] = 1.0  # make the target absorbing
    dist = np.zeros(P.shape[0])
    dist[start] = 1.0
    for _ in range(T):
        dist = dist @ Q
    return float(dist[target].sum())


def terminal_prob(P: np.ndarray, start: int, target: np.ndarray, T: int) -> float:
    dist = np.zeros(P.shape[0])
    dist[start] = 1.0
    for _ in range(T):
        dist = dist @ P
    return float(dist[target].sum())


def reference_table(Ns=(4, 8, 16, 32), T: int = 100, x0: float = 0.25, threshold: float = 0.75):
    rows = []
    for N in Ns:
        start = round(x0 * N)
        target = np.array([k for k in range(N + 1) if k / N >= threshold])
        Pi, Pp = transition_matrix(N, "independent"), transition_matrix(N, "pair_restore")
        rows.append(
            {
                "N": N,
                "independent_first_entry": first_entry_prob(Pi, start, target, T),
                "pair_restore_first_entry": first_entry_prob(Pp, start, target, T),
                "pair_restore_terminal": terminal_prob(Pp, start, target, T),
            }
        )
    return rows
