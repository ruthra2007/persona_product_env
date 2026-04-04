"""
PersonaProductEnv — OpenEnv-compatible reinforcement-learning environment.

An AI agent interacts with this environment to evaluate its ability to
recommend the best product given a user persona, preferences, and budget.

Public API
----------
    reset(task: TaskSpec) -> Observation
    step(action: Action)  -> Tuple[Observation, Reward, bool, dict]
    state()               -> Observation
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from persona_product_env.models import (
    Action,
    Observation,
    Product,
    Reward,
    TaskSpec,
)


# ── Preference-to-attribute mapping ──────────────────────────────────────

_PREF_ATTR_MAP: Dict[str, str] = {
    "battery": "battery",
    "performance": "performance",
    "camera": "camera",
    "display": "display",
    "build_quality": "build_quality",
    "build": "build_quality",
    "storage": "storage",
    "portability": "portability",
    "price": "_price",  # handled specially
}

# Keywords that indicate good reasoning per preference
_REASONING_KEYWORDS: Dict[str, List[str]] = {
    "battery": ["battery", "battery life", "endurance", "charge", "mah"],
    "performance": ["performance", "speed", "processor", "cpu", "fast", "multitask"],
    "camera": ["camera", "photo", "lens", "megapixel", "image", "photography"],
    "display": ["display", "screen", "resolution", "oled", "amoled", "visual"],
    "build_quality": ["build", "durable", "durability", "rugged", "quality", "metal", "alumin"],
    "storage": ["storage", "memory", "gb", "tb", "space"],
    "portability": ["portable", "portability", "lightweight", "light", "slim", "compact", "travel"],
    "price": ["price", "budget", "affordable", "value", "cost", "cheap", "economical"],
}


class PersonaProductEnv:
    """RL-style environment for product recommendation evaluation."""

    def __init__(self) -> None:
        self._task: Optional[TaskSpec] = None
        self._current_step: int = 0
        self._max_steps: int = 3
        self._done: bool = False
        self._previous_selections: List[str] = []
        self._rewards: List[float] = []

    # ── public API ────────────────────────────────────────────────────

    def reset(self, task: TaskSpec) -> Observation:
        """Initialise a new episode from the given task specification."""
        self._task = task
        self._current_step = 0
        self._max_steps = task.max_steps
        self._done = False
        self._previous_selections = []
        self._rewards = []
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Execute one agent action and return (obs, reward, done, info)."""
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")

        self._current_step += 1

        reward = self._compute_reward(action)
        self._rewards.append(reward.total)
        self._previous_selections.append(action.selected_product)

        if self._current_step >= self._max_steps:
            self._done = True

        obs = self._build_observation()
        info = self._build_info(action, reward)

        return obs, reward, self._done, info

    def state(self) -> Observation:
        """Return the current observation without advancing the step."""
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._build_observation()

    # ── properties ────────────────────────────────────────────────────

    @property
    def rewards(self) -> List[float]:
        return list(self._rewards)

    @property
    def done(self) -> bool:
        return self._done

    # ── internal helpers ──────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        assert self._task is not None
        return Observation(
            persona=self._task.persona,
            budget=self._task.budget,
            preferences=self._task.preferences,
            products=self._task.products,
            current_step=self._current_step,
            max_steps=self._max_steps,
            previous_selections=list(self._previous_selections),
        )

    def _build_info(self, action: Action, reward: Reward) -> dict:
        assert self._task is not None
        return {
            "task_name": self._task.name,
            "difficulty": self._task.difficulty,
            "optimal_product": self._task.optimal_product,
            "agent_selected": action.selected_product,
            "is_correct": action.selected_product == self._task.optimal_product,
            "reward_breakdown": reward.model_dump(),
            "step": self._current_step,
            "done": self._done,
        }

    # ── reward computation ────────────────────────────────────────────

    def _compute_reward(self, action: Action) -> Reward:
        assert self._task is not None
        task = self._task

        penalties: Dict[str, float] = {}

        # --- find selected product ---------------------------------
        product = self._find_product(action.selected_product)

        if product is None:
            # Invalid product → heavy penalty, minimal reward
            penalties["invalid_product"] = 0.5
            return Reward(
                total=0.0,
                budget_fit=0.0,
                preference_match=0.0,
                reasoning_quality=0.0,
                penalties=penalties,
            )

        # --- budget fit (weight 0.3) --------------------------------
        budget_score = self._score_budget(product, task.budget)
        if product.price > task.budget:
            penalties["over_budget"] = round(
                min((product.price - task.budget) / task.budget, 0.3), 4
            )
            budget_score = max(budget_score - penalties["over_budget"], 0.0)

        # --- preference match (weight 0.4) --------------------------
        pref_score = self._score_preferences(product, task)

        # --- reasoning quality (weight 0.3) -------------------------
        reasoning_score = self._score_reasoning(action.reasoning, task)

        # --- repeat penalty -----------------------------------------
        if action.selected_product in self._previous_selections:
            penalties["repeated_action"] = 0.15

        # --- aggregate ----------------------------------------------
        raw = (0.3 * budget_score) + (0.4 * pref_score) + (0.3 * reasoning_score)
        total_penalty = sum(penalties.values())
        total = round(max(min(raw - total_penalty, 1.0), 0.0), 4)

        return Reward(
            total=total,
            budget_fit=round(budget_score, 4),
            preference_match=round(pref_score, 4),
            reasoning_quality=round(reasoning_score, 4),
            penalties=penalties,
        )

    def _find_product(self, name: str) -> Optional[Product]:
        assert self._task is not None
        for p in self._task.products:
            if p.name.strip().lower() == name.strip().lower():
                return p
        return None

    @staticmethod
    def _score_budget(product: Product, budget: float) -> float:
        """Score ∈ [0, 1]: 1.0 when price ≤ budget, degrades linearly beyond."""
        if product.price <= budget:
            # Reward saving money — closer to budget is fine, big savings also fine
            ratio = product.price / budget if budget > 0 else 1.0
            # Sweet spot: 0.5-1.0 of budget → score 0.8-1.0
            if ratio >= 0.5:
                return 0.8 + 0.2 * ((ratio - 0.5) / 0.5)
            return 0.5 + 0.3 * (ratio / 0.5)
        else:
            # Over budget — linear degradation
            overshoot = (product.price - budget) / budget
            return max(1.0 - overshoot * 2, 0.0)

    @staticmethod
    def _score_preferences(product: Product, task: TaskSpec) -> float:
        """Weighted preference match ∈ [0, 1].

        Earlier preferences carry more weight (position-decayed).
        """
        if not task.preferences:
            return 0.5

        # Build weights: first pref gets highest weight
        n = len(task.preferences)
        raw_weights = [n - i for i in range(n)]
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        # Find max score across all products for normalisation
        attr_maxes: Dict[str, float] = {}
        for pref in task.preferences:
            attr = _PREF_ATTR_MAP.get(pref)
            if attr is None or attr == "_price":
                continue
            attr_maxes[pref] = max(
                getattr(p, attr, 0.0) for p in task.products
            )

        score = 0.0
        for i, pref in enumerate(task.preferences):
            attr = _PREF_ATTR_MAP.get(pref)
            if attr is None:
                continue
            if attr == "_price":
                # Lower price is better
                prices = [p.price for p in task.products]
                min_p, max_p = min(prices), max(prices)
                if max_p == min_p:
                    s = 1.0
                else:
                    s = 1.0 - (product.price - min_p) / (max_p - min_p)
            else:
                val = getattr(product, attr, 0.0)
                mx = attr_maxes.get(pref, 10.0)
                s = val / mx if mx > 0 else 0.0
            score += weights[i] * s

        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _score_reasoning(reasoning: str, task: TaskSpec) -> float:
        """Heuristic reasoning-quality score ∈ [0, 1].

        Checks:
        1. Length adequacy (>20 chars)
        2. Mentions relevant preference keywords
        3. Mentions budget / price considerations
        4. Mentions the selected product name or attributes
        """
        if not reasoning or len(reasoning.strip()) < 5:
            return 0.0

        text = reasoning.lower()
        score = 0.0

        # Length bonus (up to 0.2)
        length = len(text)
        if length >= 200:
            score += 0.2
        elif length >= 100:
            score += 0.15
        elif length >= 50:
            score += 0.1
        elif length >= 20:
            score += 0.05

        # Preference keyword coverage (up to 0.5)
        prefs_mentioned = 0
        for pref in task.preferences:
            keywords = _REASONING_KEYWORDS.get(pref, [pref])
            if any(kw in text for kw in keywords):
                prefs_mentioned += 1
        if task.preferences:
            score += 0.5 * (prefs_mentioned / len(task.preferences))

        # Budget / price awareness (up to 0.15)
        budget_kws = ["budget", "price", "cost", "afford", "expensive", "cheap", "value", "$"]
        if any(kw in text for kw in budget_kws):
            score += 0.15

        # Product name / brand mention (up to 0.15)
        product_names = [p.name.lower() for p in task.products]
        product_brands = [p.brand.lower() for p in task.products]
        mentions = sum(1 for n in product_names + product_brands if n in text)
        score += min(0.15, 0.05 * mentions)

        return min(max(round(score, 4), 0.0), 1.0)
