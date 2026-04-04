"""
Deterministic grader for persona_product_env.

Evaluates an agent's final answer for a given task and returns a score
between 0.0 and 1.0.

Scoring rubric
--------------
    correctness  (0.6) — did the agent pick the optimal product?
    reasoning    (0.4) — does the reasoning mention the right factors?
"""

from __future__ import annotations

from typing import Dict

from persona_product_env.models import Action, TaskSpec


# Keywords expected per preference (subset for grading)
_GRADING_KEYWORDS: Dict[str, list[str]] = {
    "battery": ["battery", "charge", "endurance"],
    "performance": ["performance", "speed", "processor", "fast", "multitask"],
    "camera": ["camera", "photo", "lens", "image"],
    "display": ["display", "screen", "resolution"],
    "build_quality": ["build", "durable", "durability", "quality", "rugged"],
    "storage": ["storage", "memory", "gb"],
    "portability": ["portable", "portability", "lightweight", "slim", "compact", "travel"],
    "price": ["price", "budget", "cost", "afford", "value"],
}


def grade(action: Action, task: TaskSpec) -> Dict[str, float]:
    """Return a grading dict with 'score', 'correctness', and 'reasoning'.

    Parameters
    ----------
    action : Action
        The agent's final action (selected_product + reasoning).
    task : TaskSpec
        The task specification containing the optimal answer.

    Returns
    -------
    dict with keys:
        score        : float  (0.0 – 1.0) aggregate
        correctness  : float  (0.0 – 1.0) product correctness component
        reasoning    : float  (0.0 – 1.0) reasoning quality component
    """
    correctness = _score_correctness(action.selected_product, task)
    reasoning = _score_reasoning(action.reasoning, task)

    score = round(0.6 * correctness + 0.4 * reasoning, 4)

    return {
        "score": score,
        "correctness": round(correctness, 4),
        "reasoning": round(reasoning, 4),
    }


def _score_correctness(selected: str, task: TaskSpec) -> float:
    """1.0 if exact match, 0.3 if within budget, 0.0 otherwise."""
    if selected.strip().lower() == task.optimal_product.strip().lower():
        return 1.0

    # Partial credit: product exists and is within budget
    for p in task.products:
        if p.name.strip().lower() == selected.strip().lower():
            if p.price <= task.budget:
                return 0.3
            return 0.1
    # Product not found at all
    return 0.0


def _score_reasoning(reasoning: str, task: TaskSpec) -> float:
    """Deterministic reasoning evaluation ∈ [0, 1]."""
    if not reasoning or len(reasoning.strip()) < 5:
        return 0.0

    text = reasoning.lower()
    score = 0.0

    # 1. Length adequacy (0.15)
    if len(text) >= 150:
        score += 0.15
    elif len(text) >= 80:
        score += 0.10
    elif len(text) >= 30:
        score += 0.05

    # 2. Covers task preferences (0.50)
    covered = 0
    for pref in task.preferences:
        kws = _GRADING_KEYWORDS.get(pref, [pref])
        if any(k in text for k in kws):
            covered += 1
    if task.preferences:
        score += 0.50 * (covered / len(task.preferences))

    # 3. Budget awareness (0.15)
    if any(k in text for k in ["budget", "price", "cost", "afford", "$"]):
        score += 0.15

    # 4. Comparative reasoning — mentions more than one product (0.20)
    product_names = [p.name.lower() for p in task.products]
    mentioned = sum(1 for n in product_names if n in text)
    if mentioned >= 3:
        score += 0.20
    elif mentioned >= 2:
        score += 0.15
    elif mentioned >= 1:
        score += 0.05

    return min(round(score, 4), 1.0)
