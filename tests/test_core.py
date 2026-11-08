"""Checks for the game, samplers, reference table and policy loading. Run: .venv/bin/python -m pytest -q"""

import itertools
import json
import math

import numpy as np
import pytest

from swarm.extract import clean_answer, label_masses, load_policy
from swarm.game import A, B, HistorySpace, payoff
from swarm.prompts import build_prompt
from swarm.reference import reference_table, transition_matrix, ref_p, alpha_pair_homogeneous
from swarm.samplers import Sampler, cov_sum_shared
from swarm.simulate import run_population

# ---------- game ----------


def test_history_counts():
    assert len(HistorySpace(2)) == 21
    assert len(HistorySpace(3)) == 85


def test_transitions_keep_last_H():
    s = HistorySpace(2)
    h = ((B, B), (A, B))
    nxt = s.histories[s.next[s.index[h], A, A]]
    assert nxt == ((A, B), (A, A))
    assert s.histories[s.next[s.index[()], B, A]] == ((B, A),)


def test_payoffs():
    assert payoff(A, A) == 2 and payoff(B, B) == 1 and payoff(A, B) == 0 and payoff(B, A) == 0


def test_prompts_counterbalance_and_hide_population():
    h = ((B, B), (B, B))
    p0, p1 = build_prompt(h, 0, 0), build_prompt(h, 0, 1)
    assert "Both choose J: you each get 2 points" in p0  # mapping 0: J is the better option
    assert "Both choose J: you each get 1 point" in p1  # mapping 1: J is the inferior option
    assert "You chose K, your partner chose K" in p0 and "You chose J, your partner chose J" in p1
    for f, m in itertools.product((0, 1), (0, 1)):
        text = build_prompt(h, f, m).lower()
        assert "population" not in text and "better" not in text


# ---------- reference table from the research plan ----------

DOC = {  # N: (independent first entry, pair-restore first entry, pair-restore terminal)
    4: (0.3911, 0.3911, 0.3085),
    8: (0.0255, 0.6308, 0.4153),
    16: (0.00016, 0.6941, 0.4295),
    32: (None, 0.7157, 0.4314),
}


def test_reference_table_matches_plan():
    for r in reference_table():
        ind, pf, pt = DOC[r["N"]]
        if ind is None:
            assert r["independent_first_entry"] < 1e-5
        else:
            assert r["independent_first_entry"] == pytest.approx(ind, abs=0.5 * 10 ** -(len(str(ind)) - 2))
        assert r["pair_restore_first_entry"] == pytest.approx(pf, abs=5e-5)
        assert r["pair_restore_terminal"] == pytest.approx(pt, abs=5e-5)


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_reference_kernel_moments(N):
    P = transition_matrix(N, "pair_restore")
    k = np.arange(N + 1)
    for row, start in zip(P, range(N + 1)):
        p = ref_p(start / N)
        mean = row @ k / N
        var = row @ (k / N) ** 2 - mean**2
        assert mean == pytest.approx(p, abs=1e-12)
        assert var == pytest.approx(p * (1 - p) / min(N, 4), abs=1e-12)


# ---------- samplers ----------


def test_cov_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    p = rng.random(9)
    brute = sum(min(p[i], p[j]) - p[i] * p[j] for i in range(9) for j in range(i + 1, 9))
    assert cov_sum_shared(p) == pytest.approx(brute)


def _draws(sampler, p, pairs, n, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([sampler(p, pairs, rng)[0] for _ in range(n)], dtype=float)


@pytest.mark.parametrize("spec", [Sampler("independent"), Sampler("pair_shared", alpha=0.6),
                                  Sampler("pair_shared"), Sampler("global_shared", alpha=0.6),
                                  Sampler("block", n_blocks=2)])
def test_individual_marginals_preserved(spec):
    rng = np.random.default_rng(42)
    N = 8
    p = rng.uniform(0.05, 0.95, N)
    pairs = rng.permutation(N).reshape(-1, 2)
    d = _draws(spec, p, pairs, 40000)
    se = np.sqrt(p * (1 - p) / len(d))
    assert np.all(np.abs(d.mean(0) - p) < 4.5 * se), spec.name


@pytest.mark.parametrize("spec", [Sampler("pair_shared", alpha=1.0), Sampler("pair_shared"),
                                  Sampler("block", n_blocks=2)])
def test_within_pair_distribution_preserved(spec):
    """Pair-preserving samplers keep every pair's joint action distribution independent."""
    rng = np.random.default_rng(7)
    N = 8
    p = rng.uniform(0.1, 0.9, N)
    pairs = rng.permutation(N).reshape(-1, 2)
    d = _draws(spec, p, pairs, 40000)
    for i, j in pairs:
        both = np.mean(d[:, i] * d[:, j])
        se = math.sqrt(p[i] * p[j] * (1 - p[i] * p[j]) / len(d))
        assert abs(both - p[i] * p[j]) < 4.5 * se


def test_global_shared_changes_within_pair_coordination():
    """Sanity check that the test above can fail: the secondary sampler does change pair coordination."""
    p = np.full(8, 0.5)
    pairs = np.arange(8).reshape(-1, 2)
    d = _draws(Sampler("global_shared", alpha=1.0), p, pairs, 4000)
    assert np.mean(d[:, 0] * d[:, 1]) > 0.45  # vs 0.25 under independence


def test_pair_restore_hits_reference_variance_heterogeneous():
    rng = np.random.default_rng(3)
    N, N0 = 16, 4
    p = rng.uniform(0.2, 0.8, N)
    pairs = rng.permutation(N).reshape(-1, 2)
    smp = Sampler("pair_shared", N0=N0)
    _, info = smp(p, pairs, np.random.default_rng(0))
    assert info["feasible"]
    x = _draws(smp, p, pairs, 60000).mean(1)
    target = np.sum(p * (1 - p)) / (N * N0)
    assert x.var() == pytest.approx(target, rel=0.03)
    assert x.mean() == pytest.approx(p.mean(), abs=0.003)


def test_homogeneous_alpha_example_from_plan():
    assert alpha_pair_homogeneous(16) == pytest.approx(3 / 7)


# ---------- simulation ----------


def test_simulation_is_paired_and_deterministic():
    space = HistorySpace(2)
    pA = np.full(len(space), 0.3)
    r1 = run_population(pA, space, 8, 50, Sampler("independent"), "incumbent_B", seed=5)
    r2 = run_population(pA, space, 8, 50, Sampler("independent"), "incumbent_B", seed=5)
    assert r1 == r2


def test_deterministic_policy_stays_put():
    space = HistorySpace(2)
    pA = np.zeros(len(space))
    r = run_population(pA, space, 16, 30, Sampler("pair_shared"), "incumbent_B", seed=0)
    assert not r.entered and r.mean_reward == 1.0


# ---------- extraction ----------


def test_label_masses_sum_space_variants_and_flag_missing():
    top = [("J", math.log(0.6)), (" J", math.log(0.1)), ("<think>", math.log(0.05))]
    m = label_masses(top)
    assert m["mass"]["J"] == pytest.approx(0.7)
    assert not m["found"]["K"] and m["min_listed"] == pytest.approx(0.05)


def test_clean_answer():
    assert clean_answer(" J") == "J" and clean_answer("**K**") == "K" and clean_answer("J.") == "J"
    assert clean_answer("I choose J") is None and clean_answer("") is None and clean_answer("j") is None


def test_load_policy_temperature_and_mapping(tmp_path):
    space = HistorySpace(2)
    recs = []
    for m, (mA, mB) in {0: (0.6, 0.3), 1: (0.2, 0.6)}.items():
        for h in space.histories:
            recs.append({"history": space.key(h), "framing": 0, "mapping": m, "repeat": 0, "mass_A": mA,
                         "mass_B": mB, "found_A": True, "found_B": True, "min_listed": 0.01})
    path = tmp_path / "pol.json"
    path.write_text(json.dumps({"meta": {"H": 2, "model": "fake"}, "records": recs}))
    p1, _, _ = load_policy(str(path), 0, 0, temperature=1.0)
    p2, _, _ = load_policy(str(path), 0, 0, temperature=2.0)
    assert p1[0] == pytest.approx(2 / 3)
    assert p2[0] == pytest.approx(1 / (1 + math.exp(-math.log(2) / 2)))
    p3, _, _ = load_policy(str(path), 0, 1)
    assert p3[0] == pytest.approx(0.25)


# ---- Amendment 3 additions ----

def test_stag_payoffs():
    from swarm.game import payoff as pay
    assert pay(A, A, "stag") == 4 and pay(B, B, "stag") == 3
    assert pay(B, A, "stag") == 3 and pay(A, B, "stag") == 0  # B is safe, A alone gets nothing


def test_original_prompts_unchanged():
    from swarm.prompts import prompt_set_hash
    assert prompt_set_hash(HistorySpace(2).histories, (0, 1), (0, 1)) == "0adadb7b1d5603ae"


def test_new_mappings_cross_letter_and_order():
    from swarm.prompts import FIRST_LETTER, LABEL_MAPPINGS
    combos = {(LABEL_MAPPINGS[m][A], FIRST_LETTER[m] == LABEL_MAPPINGS[m][A]) for m in range(4)}
    assert combos == {("J", True), ("J", False), ("K", True), ("K", False)}


def test_all_framings_render_both_games():
    from swarm.prompts import ALL_FRAMINGS
    h = ((B, B), (A, B))
    for game in ("coord", "stag"):
        for f in ALL_FRAMINGS:
            for m in range(4):
                text = build_prompt(h, f, m, game)
                assert "random" in text.lower() and "J" in text and "K" in text
                if game == "stag":
                    assert "4" in text and "3" in text


def test_stag_reward_in_simulation():
    from swarm.simulate import run_population
    from swarm.samplers import parse_sampler
    space = HistorySpace(2)
    allB = np.zeros(len(space))
    r = run_population(allB, space, 8, 10, parse_sampler("independent"), "incumbent_B", 0, game="stag")
    assert r.mean_x_late == 0 and abs(r.mean_reward - 3) < 1e-12
