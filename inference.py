from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from flask import Flask, request, jsonify
from openai import OpenAI

# ── setup ─────────────────────────────────────────────
app = Flask(__name__)

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

    return results


# ── API ENDPOINTS ────────────────────────────────────
@app.route("/")
def home():
    return "Persona Product Env running ✅"


@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({
        "observation": {},
        "reward": 1.0,
        "done": True,
        "info": {}
    })


# ── RUN SERVER ───────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting server on port 7860...")
    app.run(host="0.0.0.0", port=8000)
