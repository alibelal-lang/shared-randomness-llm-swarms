"""Live population validation: every decision is a stateless Adaption call (independent arm only).

Matchings and initial states use the same seeded streams as simulate.run_population, so a simulated comparator
with the extracted policy sees identical matchings. Invalid answers are retried up to 3 times. If still invalid,
the action is drawn uniformly at random from a separate seeded stream and logged; this rule is the same everywhere.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .extract import AdaptionSampleBackend, clean_answer
from .game import HistorySpace, initial_codes
from .prompts import LABEL_MAPPINGS, build_prompt


class LiveAgentCaller:
    def __init__(self, model: str, temperature: float = 1.0, reasoning_effort="none", pool_size: int = 64):
        self.be = AdaptionSampleBackend(model, samples=1, temperature=temperature, reasoning_effort=reasoning_effort)
        self.pool = ThreadPoolExecutor(max_workers=pool_size)
        self.calls = 0
        self.cost = 0.0

    def decide(self, prompt: str, mapping: int):
        """Returns (action or None, attempts, cost)."""
        lab = LABEL_MAPPINGS[mapping]
        letter_to_action = {v: k for k, v in lab.items()}
        cost = 0.0
        for attempt in range(1, 4):
            text, err, _, _, c = self.be._one(prompt)
            cost += c
            ans = clean_answer(text) if not err else None
            if ans is not None:
                return letter_to_action[ans], attempt, cost
        return None, 3, cost


def run_live(caller: LiveAgentCaller, space: HistorySpace, framing: int, mapping: int, N: int, T: int, seed: int,
             init: str = "incumbent_B"):
    match_rng = np.random.default_rng([seed, 0])
    init_rng = np.random.default_rng([seed, 2])
    fallback_rng = np.random.default_rng([seed, 3])
    codes = initial_codes(space, init, N, init_rng)
    partner = np.empty(N, dtype=np.int64)
    xs, decisions = [], []
    for t in range(T):
        pairs = match_rng.permutation(N).reshape(-1, 2)
        partner[pairs[:, 0]], partner[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
        prompts = [build_prompt(space.histories[c], framing, mapping) for c in codes]
        results = list(caller.pool.map(lambda p: caller.decide(p, mapping), prompts))
        a = np.empty(N, dtype=np.int8)
        for i, (act, attempts, cost) in enumerate(results):
            fallback = act is None
            a[i] = int(fallback_rng.integers(2)) if fallback else act
            decisions.append({"seed": seed, "round": t + 1, "agent": i, "history": space.key(space.histories[codes[i]]),
                              "action": int(a[i]), "attempts": attempts, "fallback": fallback, "cost": cost})
        ap = a[partner]
        xs.append(float(a.mean()))
        codes = space.next[codes, a, ap]
    return np.array(xs), decisions


def live_campaign(model: str, variants, Ns, runs: int, T: int, init="incumbent_B", run_threads: int = 16,
                  pool_size: int = 96, seed_offset: int = 0):
    caller = LiveAgentCaller(model, pool_size=pool_size)
    space = HistorySpace(2)
    jobs = [(f, m, N, s) for (f, m) in variants for N in Ns for s in range(seed_offset, seed_offset + runs)]
    t0 = time.time()

    def one(job):
        f, m, N, s = job
        xs, dec = run_live(caller, space, f, m, N, T, s, init)
        return job, xs, dec

    with ThreadPoolExecutor(max_workers=run_threads) as ex:
        out = list(ex.map(one, jobs))
    caller.pool.shutdown()
    return out, time.time() - t0
