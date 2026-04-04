#!/usr/bin/env python3
"""
inference.py — Run the persona_product_env agent across all tasks.

Uses the OpenAI-compatible client to call an LLM that acts as the agent.
Outputs results in the strict OpenEnv format.

Environment variables
---------------------
    API_BASE_URL      : LLM API base (default: https://router.huggingface.co/v1)
    MODEL_NAME        : Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          : HuggingFace API token
    LOCAL_IMAGE_NAME  : Optional local Docker image override
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import List, Optional

from openai import OpenAI

# ── environment imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persona_product_env.env import PersonaProductEnv
from persona_product_env.models import Action, TaskSpec
from persona_product_env.grader import grade
from persona_product_env.tasks import ALL_TASKS

# ── configuration ────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

ENV_NAME = "persona_product_env"


def _build_client() -> OpenAI:
    """Construct OpenAI-compatible client."""
    api_key = HF_TOKEN or "no-key"
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


# ── prompt construction ─────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are an expert product recommendation agent. "
        "Given a user persona, budget, preferences, and a list of products, "
        "select the BEST product and explain your reasoning.\n\n"
        "You MUST respond with valid JSON only — no markdown, no extra text.\n"
        "Schema:\n"
        '{\n'
        '  "selected_product": "<exact product name>",\n'
        '  "reasoning": "<detailed reasoning>"\n'
        '}\n'
    )


def _build_user_prompt(obs_dict: dict) -> str:
    products_text = ""
    for p in obs_dict["products"]:
        products_text += (
            f"  - {p['name']} (Brand: {p['brand']}, Price: ${p['price']}, "
            f"Battery: {p['battery']}/10, Performance: {p['performance']}/10, "
            f"Camera: {p['camera']}/10, Display: {p['display']}/10, "
            f"Build: {p['build_quality']}/10, Storage: {p['storage']}/10, "
            f"Portability: {p['portability']}/10)\n"
        )

    return (
        f"Persona: {obs_dict['persona']}\n"
        f"Budget: ${obs_dict['budget']}\n"
        f"Preferences (in priority order): {', '.join(obs_dict['preferences'])}\n\n"
        f"Available Products:\n{products_text}\n"
        f"Step {obs_dict['current_step'] + 1} of {obs_dict['max_steps']}.\n\n"
        "Select the best product. Respond with JSON only."
    )


# ── LLM call ────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, obs_dict: dict) -> Action:
    """Call the LLM and parse its response into an Action."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(obs_dict)},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    parsed = json.loads(raw)
    return Action(
        selected_product=parsed["selected_product"],
        reasoning=parsed["reasoning"],
    )


# ── episode runner ───────────────────────────────────────────────────────

def run_task(client: OpenAI, task: TaskSpec) -> bool:
    """Run a single task episode. Returns True if completed successfully."""
    env = PersonaProductEnv()
    obs = env.reset(task)
    rewards: List[float] = []
    step_num = 0
    success = True

    print(f"[START] task={task.name} env={ENV_NAME} model={MODEL_NAME}")

    while not env.done:
        step_num += 1
        error_msg: Optional[str] = None
        try:
            obs_dict = obs.model_dump()
            action = _call_llm(client, obs_dict)
            obs, reward, done, info = env.step(action)

            action_str = action.selected_product
            reward_val = reward.total
            rewards.append(reward_val)

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} error=null"
            )

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")[:200]
            rewards.append(0.0)
            # Create a fallback action to advance the environment
            try:
                fallback = Action(
                    selected_product=task.products[0].name,
                    reasoning="Fallback due to error.",
                )
                obs, reward, done, info = env.step(fallback)
            except Exception:
                done = True

            print(
                f"[STEP] step={step_num} action=ERROR "
                f"reward=0.00 done={str(done).lower()} error={error_msg}"
            )
            success = False

            if done:
                break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"rewards={rewards_str}"
    )

    return success


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    client = _build_client()

    overall_success = True
    for task in ALL_TASKS:
        try:
            task_ok = run_task(client, task)
            if not task_ok:
                overall_success = False
        except Exception as exc:
            print(f"[START] task={task.name} env={ENV_NAME} model={MODEL_NAME}")
            print(
                f"[STEP] step=1 action=FATAL_ERROR reward=0.00 "
                f"done=true error={str(exc)[:200]}"
            )
            print(f"[END] success=false steps=1 rewards=0.00")
            overall_success = False

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
