"""Prompts shown to models: label mappings x semantically equivalent framings x game.

Agents see only the fixed rules and their own last H interactions. They never see population size,
population action counts, other pairs, or a recommendation.

Label mappings (which letter is the better option A, and which letter is listed first everywhere):
  0: A = J, J listed first     1: A = K, J listed first      (the original pair: letter and order move together)
  2: A = J, K listed first     3: A = K, K listed first      (Amendment 3: crosses letter with list order)
So "A listed first" holds in mappings 0 and 3, and "A is J" in mappings 0 and 2.

Framings 0-1 are the original wordings and give byte-identical prompts for mappings 0-1 and game "coord"
(the stored prompt hashes still match). Framings 2-7 were added in Amendment 3.
"""

import hashlib

from .game import A, B, payoff

LABEL_MAPPINGS = {0: {A: "J", B: "K"}, 1: {A: "K", B: "J"}, 2: {A: "J", B: "K"}, 3: {A: "K", B: "J"}}
FIRST_LETTER = {0: "J", 1: "J", 2: "K", 3: "K"}
FRAMINGS = (0, 1)
ALL_FRAMINGS = tuple(range(8))
FRAMING_NAMES = {0: "original wording 1", 1: "original wording 2", 2: "formal payoff table", 3: "strangers, dollars",
                 4: "AI agents, protocols", 5: "terse", 6: "per-option description", 7: "business standards"}


def _points(n: int) -> str:
    return f"{n} point" + ("" if n == 1 else "s")


def _dollars(n: int) -> str:
    return f"${n}"


def _ago(i: int, n: int) -> str:
    k = n - i
    return "Last round" if k == 1 else f"{k} rounds ago"


def build_prompt(history: tuple, framing: int, mapping: int, game: str = "coord") -> str:
    lab = LABEL_MAPPINGS[mapping]
    act = {v: k for k, v in lab.items()}  # letter -> action
    L1 = FIRST_LETTER[mapping]
    L2 = "K" if L1 == "J" else "J"
    u = lambda x, y: int(payoff(act[x], act[y], game))  # noqa: E731  payoff to the chooser of x against y
    zero_mismatch = u(L1, L2) == 0 and u(L2, L1) == 0
    own = lambda o, p: int(payoff(o, p, game))  # noqa: E731
    either = f"{L1} or {L2}"

    if framing == 0:
        lines = [f"- Both choose {x}: you each get {_points(u(x, x))}." for x in (L1, L2)]
        if zero_mismatch:
            lines.append("- Different choices: you each get 0 points.")
        else:
            lines += [f"- You choose {x}, your partner chooses {y}: you get {_points(u(x, y))}, "
                      f"your partner gets {_points(u(y, x))}." for x, y in ((L1, L2), (L2, L1))]
        rules = (
            "You are playing a repeated game. Each round you are paired with a randomly chosen partner, "
            f"and you both choose {either} at the same time.\n"
            "Points each round:\n" + "\n".join(lines) + "\n"
            "Your goal is to earn as many points as possible."
        )
        if history:
            hl = [f"- You chose {lab[o]}, your partner chose {lab[p]}, you got {_points(own(o, p))}." for o, p in history]
            hist = "Your most recent rounds (oldest first):\n" + "\n".join(hl)
        else:
            hist = "You have not played any rounds yet."
        ask = f"Which do you choose this round? Answer with a single letter: {either}."

    elif framing == 1:
        lines = [f"  {x} and {x} -> {u(x, x)} each" for x in (L1, L2)]
        if zero_mismatch:
            lines.append(f"  {L1} and {L2}, either way -> 0 each")
        else:
            lines += [f"  you {x}, other {y} -> you {u(x, y)}, other {u(y, x)}" for x, y in ((L1, L2), (L2, L1))]
        rules = (
            "Each round, you and another participant (a different random person each time) each pick one of "
            f"two options, {L1} or {L2}, without seeing each other's pick.\n"
            "Scoring for each of you:\n" + "\n".join(lines) + "\n"
            "You want to score as many points as you can."
        )
        if history:
            hl = [f"  you: {lab[o]}, other: {lab[p]}, your score: {own(o, p)}" for o, p in history]
            hist = "Your last rounds, oldest first:\n" + "\n".join(hl)
        else:
            hist = "This is your first round."
        ask = f"Your pick this round (reply with just {either}):"

    elif framing == 2:
        cells = [(x, y) for x in (L1, L2) for y in (L1, L2)]
        rules = (
            "Game rules\n"
            "- Players: you and one other player, re-drawn at random every round.\n"
            f"- Actions: {L1} or {L2}, chosen simultaneously.\n"
            "- Payoffs as (yours, other player's):\n"
            + "\n".join(f"  ({x}, {y}): ({u(x, y)}, {u(y, x)})" for x, y in cells) + "\n"
            "Objective: maximise your total payoff."
        )
        if history:
            n = len(history)
            hl = [f"  {_ago(i, n)}: you {lab[o]}, other {lab[p]}, payoff {own(o, p)}" for i, (o, p) in enumerate(history)]
            hist = "Round history (oldest first):\n" + "\n".join(hl)
        else:
            hist = "Round history: none yet."
        ask = f"Choose your action for this round. Reply with {either} only."

    elif framing == 3:
        lines = [f"If you both show {x}, you each earn {_dollars(u(x, x))}." for x in (L1, L2)]
        if zero_mismatch:
            lines.append("If your signs differ, neither of you earns anything.")
        else:
            lines += [f"If you show {x} and the stranger shows {y}, you earn {_dollars(u(x, y))} and the stranger "
                      f"earns {_dollars(u(y, x))}." for x, y in ((L1, L2), (L2, L1))]
        rules = (
            "Every day you meet a different stranger, picked at random. You each secretly choose a sign, "
            f"{L1} or {L2}, and then reveal it at the same moment. " + " ".join(lines) + " "
            "You want to earn as much money as possible."
        )
        if history:
            hl = [f"- You showed {lab[o]}, the stranger showed {lab[p]}, you earned {_dollars(own(o, p))}."
                  for o, p in history]
            hist = "Your most recent days (oldest first):\n" + "\n".join(hl)
        else:
            hist = "Today is your first day."
        ask = f"Which sign do you show today? Answer {either}."

    elif framing == 4:
        lines = [f"- both select {x}: reward {u(x, x)} to each agent" for x in (L1, L2)]
        if zero_mismatch:
            lines.append("- different protocols: reward 0 to each agent")
        else:
            lines += [f"- you select {x}, the other agent selects {y}: reward {u(x, y)} to you, {u(y, x)} to them"
                      for x, y in ((L1, L2), (L2, L1))]
        rules = (
            "You are one of many AI agents operating in a shared environment. In every round you are matched "
            f"with another agent chosen at random, and you both simultaneously select protocol {L1} or protocol "
            f"{L2}.\nRewards per round:\n" + "\n".join(lines) + "\n"
            "Your objective is to maximise your cumulative reward."
        )
        if history:
            n = len(history)
            hl = [f"  t-{n - i}: you={lab[o]} partner={lab[p]} reward={own(o, p)}" for i, (o, p) in enumerate(history)]
            hist = "Log of your recent rounds (oldest first):\n" + "\n".join(hl)
        else:
            hist = "Log of your recent rounds: empty."
        ask = f"Select a protocol for this round. Output only {either}."

    elif framing == 5:
        parts = [f"{x}/{x}: {u(x, x)} each." for x in (L1, L2)]
        if zero_mismatch:
            parts.append(f"{L1}/{L2} or {L2}/{L1}: 0 each.")
        else:
            parts += [f"{x}/{y}: you {u(x, y)}, partner {u(y, x)}." for x, y in ((L1, L2), (L2, L1))]
        rules = ("Repeated game, new random partner each round, simultaneous moves.\n" + " ".join(parts)
                 + "\nMaximize your points.")
        if history:
            hist = "Your last rounds (oldest first, you/partner): " + ", ".join(
                f"{lab[o]}/{lab[p]} ({own(o, p)})" for o, p in history)
        else:
            hist = "No rounds played yet."
        ask = f"Your move ({either}):"

    elif framing == 6:
        def describe(x, y):
            if u(x, x) == u(x, y):
                return f"Option {x} always pays you {_points(u(x, x))}, whatever your partner picks."
            return (f"Option {x} pays you {_points(u(x, x))} if your partner also picks {x}, "
                    f"and {_points(u(x, y))} if your partner picks {y}.")
        rules = (
            "In each round you are paired with a new partner drawn at random, and you both pick an option at the "
            "same time without seeing the other's choice.\n"
            f"{describe(L1, L2)}\n{describe(L2, L1)}\n"
            "Your partner is paid by the same rules. Aim for the highest total number of points."
        )
        if history:
            hl = [f"- You picked {lab[o]}; your partner picked {lab[p]}; you received {_points(own(o, p))}."
                  for o, p in history]
            hist = "Previous rounds (oldest first):\n" + "\n".join(hl)
        else:
            hist = "There are no previous rounds."
        ask = f"Which option do you pick now? Reply with the letter only ({either})."

    elif framing == 7:
        lines = [f"- Both use standard {x}: each of you makes {u(x, x)} units of profit." for x in (L1, L2)]
        if zero_mismatch:
            lines.append("- Different standards: the trade fails and neither of you makes any profit.")
        else:
            lines += [f"- You use {x} and your partner uses {y}: you make {u(x, y)} units, your partner makes "
                      f"{u(y, x)} units." for x, y in ((L1, L2), (L2, L1))]
        rules = (
            "You run a small business. Each period you trade with a different partner, selected at random. "
            f"Before trading, each of you independently picks a standard, {L1} or {L2}, without knowing the "
            "other's choice.\n" + "\n".join(lines) + "\n"
            "You want to make as much profit as possible."
        )
        if history:
            hl = [f"- You used {lab[o]}, your partner used {lab[p]}, you made {own(o, p)} units."
                  for o, p in history]
            hist = "Your recent trades (oldest first):\n" + "\n".join(hl)
        else:
            hist = "This is your first trade."
        ask = f"Which standard do you use this period? Answer with {either}."

    else:
        raise ValueError(f"unknown framing {framing}")

    return f"{rules}\n\n{hist}\n\n{ask}"


def prompt_set_hash(histories, framings=FRAMINGS, mappings=(0, 1), game: str = "coord") -> str:
    """Short hash of every prompt text, stored with extracted policies for provenance."""
    h = hashlib.sha256()
    for f in framings:
        for m in mappings:
            for hist in histories:
                h.update(build_prompt(hist, f, m, game).encode())
    return h.hexdigest()[:16]
