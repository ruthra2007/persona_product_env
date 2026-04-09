from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

# ── setup ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persona_product_env.env import PersonaProductEnv
from persona_product_env.models import Action, TaskSpec
from persona_product_env.tasks import ALL_TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_NAME = "persona_product_env"


def _build_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")


# ── CORE LOGIC ───────────────────────────────────────

def main():
    print("[START] Running inference...")

    client = _build_client()
    results = []

    for task in ALL_TASKS:
        env = PersonaProductEnv()
        obs = env.reset(task)

        while not env.done:
            obs_dict = obs.model_dump()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Return JSON with selected_product and reasoning"},
                    {"role": "user", "content": str(obs_dict)},
                ],
                temperature=0.0,
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            action = Action(
                selected_product=parsed["selected_product"],
                reasoning=parsed["reasoning"],
            )

            obs, reward, done, _ = env.step(action)

        results.append({"task": task.name, "success": True})

    print("[END] Completed successfully")
    print(json.dumps(results, indent=2))


# ── ENTRY POINT ──────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)
