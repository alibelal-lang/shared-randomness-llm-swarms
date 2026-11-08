"""Repeated binary pairwise game with finite local memory.

Action 1 = A, the better convention. Action 0 = B, the inferior incumbent. Payoffs are in GAMES
(default "coord": A-A pays 2 each, B-B 1 each, a mismatch 0; "stag": a stag hunt in which B is safe). The letters a model sees are set in prompts.py, so
"A"/"B" here are internal names only.

An agent's state is its last H interactions, oldest first, each stored as (own action, partner action).
Payoffs follow deterministically from those pairs, so they are not stored.
"""

from itertools import product

import numpy as np

A, B = 1, 0
NAMES = {A: "A", B: "B"}


# GAMES[name][own, partner] = payoff to "own". Index 1 = A (better convention), 0 = B (incumbent).
#   coord  pure coordination: A-A 2, B-B 1, mismatch 0. A is payoff- AND risk-dominant (A is a best reply once
#          more than 1/3 of partners play A), so staying at B is only history-following.
#   stag   stag hunt: A-A 4, B-B 3, B is safe (pays 3 whatever the partner does), A alone pays 0. A is
#          payoff-dominant but B is risk-dominant (A is a best reply only once more than 3/4 play A).
GAMES = {
    "coord": np.array([[1, 0], [0, 2]]),
    "stag": np.array([[3, 3], [0, 4]]),
}


def payoff(own, partner, game: str = "coord"):
    """Payoff to one member of a pair; works elementwise on numpy arrays."""
    return GAMES[game][np.asarray(own), np.asarray(partner)]


class HistorySpace:
    """Every ordered history of length 0..H, each with an integer code and a transition table."""

    def __init__(self, H: int = 2):
        self.H = H
        self.histories: list[tuple] = [()]
        for length in range(1, H + 1):
            self.histories += list(product(product((A, B), repeat=2), repeat=length))
        self.index = {h: i for i, h in enumerate(self.histories)}
        # next[code, own, partner] -> code after appending (own, partner) and keeping the last H rounds
        self.next = np.zeros((len(self.histories), 2, 2), dtype=np.int64)
        for h, i in self.index.items():
            for own in (A, B):
                for partner in (A, B):
                    self.next[i, own, partner] = self.index[(h + ((own, partner),))[-H:] if H else ()]

    def __len__(self) -> int:
        return len(self.histories)

    @staticmethod
    def key(h: tuple) -> str:
        """Readable key, own action then partner action per round, oldest first: 'BB.AB'. Empty is '-'."""
        return ".".join(NAMES[o] + NAMES[p] for o, p in h) or "-"

    def keys(self) -> list[str]:
        return [self.key(h) for h in self.histories]


INITIAL_CONDITIONS = ("incumbent_B", "incumbent_A", "empty", "split")


def initial_codes(space: HistorySpace, name: str, N: int, rng: np.random.Generator) -> np.ndarray:
    """Starting history code for each of N agents.

    incumbent_B: every agent remembers H rounds of B-B (the controlled incumbent convention).
    incumbent_A: every agent remembers H rounds of A-A.
    empty:       no memory.
    split:       a random half remembers A-A rounds, the other half B-B rounds.
    """
    bb = space.index[((B, B),) * space.H]
    aa = space.index[((A, A),) * space.H]
    if name == "incumbent_B":
        return np.full(N, bb)
    if name == "incumbent_A":
        return np.full(N, aa)
    if name == "empty":
        return np.full(N, space.index[()])
    if name == "split":
        codes = np.full(N, bb)
        codes[rng.permutation(N)[: N // 2]] = aa
        return codes
    raise ValueError(f"unknown initial condition {name!r}; choose from {INITIAL_CONDITIONS}")
