# persona_product_env

A production-ready **OpenEnv** reinforcement-learning environment that simulates real-world product decision-making tasks. An AI agent observes a user persona, budget, and preferences, then selects the best product from a catalog and justifies its choice. The environment evaluates the decision and returns shaped rewards.

---

## Real-World Use Case

Imagine an AI shopping assistant that helps users pick the perfect smartphone or laptop. Different users — students, gamers, photographers, business professionals — have wildly different needs. This environment lets you **train and evaluate** how well an AI agent handles these nuanced trade-offs under realistic constraints.

---

## Observation Space

Each observation is a typed `Observation` Pydantic model containing:

| Field                 | Type            | Description                                           |
|-----------------------|-----------------|-------------------------------------------------------|
| `persona`             | `str`           | User type (student, gamer, photographer, business)    |
| `budget`              | `float`         | Maximum budget in USD                                 |
| `preferences`         | `List[str]`     | Priority-ordered preference keywords                  |
| `products`            | `List[Product]` | Available products with structured attributes         |
| `current_step`        | `int`           | Current step in the episode                           |
| `max_steps`           | `int`           | Maximum allowed steps                                 |
| `previous_selections` | `List[str]`     | Products selected in prior steps (repeat detection)   |

Each **Product** has: `name`, `brand`, `price`, `battery`, `performance`, `camera`, `display`, `build_quality`, `storage`, `portability` (all scored 0–10 except price).

---

## Action Space

| Field              | Type  | Description                              |
|--------------------|-------|------------------------------------------|
| `selected_product` | `str` | Exact name of the chosen product         |
| `reasoning`        | `str` | Free-text justification for the choice   |

---

## Reward Design

Rewards are continuous floats in **[0.0, 1.0]** with three weighted components:

| Component           | Weight | Description                                                    |
|----------------------|--------|----------------------------------------------------------------|
| **Budget Fit**       | 0.3    | How well the product price fits the budget                     |
| **Preference Match** | 0.4    | Weighted match against persona preferences (position-decayed)  |
| **Reasoning Quality**| 0.3    | Heuristic evaluation of reasoning depth and relevance          |

### Penalties

| Penalty           | Value   | Trigger                        |
|-------------------|---------|--------------------------------|
| Over budget       | variable| Product price exceeds budget   |
| Invalid product   | 0.5     | Product name not in catalog    |
| Repeated action   | 0.15   | Same product selected twice    |

---

## Task Descriptions

### Easy — `easy_student_smartphone`
- **Persona**: Student, $600 budget
- **Preferences**: battery, performance, price
- **Scenario**: One product clearly dominates. Simple decision.
- **Optimal**: BudgetKing X1

### Medium — `medium_photographer_tradeoff`
- **Persona**: Photographer, $800 budget
- **Preferences**: camera, battery, display
- **Scenario**: Two products are close; agent must reason about camera vs battery trade-offs.
- **Optimal**: PhotoMaster 7

### Hard — `hard_business_conflicting`
- **Persona**: Business user, $500 budget (tight)
- **Preferences**: performance, display, build_quality, portability
- **Scenario**: Four conflicting preferences, tight budget, one decoy over-budget product.
- **Optimal**: ExecSlim Pro

---

## Project Structure

```
persona_product_env/
├── __init__.py          # Package exports
├── env.py               # RL environment (reset / step / state)
├── models.py            # Pydantic models (Observation, Action, Reward, etc.)
├── grader.py            # Deterministic grading function
├── tasks/
│   ├── __init__.py      # Task registry
│   ├── easy.py          # Easy task definition
│   ├── medium.py        # Medium task definition
│   └── hard.py          # Hard task definition
├── inference.py         # LLM agent inference script
├── openenv.yaml         # OpenEnv metadata
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── README.md            # This file
```

---

## Setup Instructions

### Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"       # optional
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"                # optional
```

### Run Inference Locally

```bash
python inference.py
```

---

## Docker

### Build

```bash
docker build -t persona_product_env .
```

### Run

```bash
docker run --rm \
  -e HF_TOKEN="your_token" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  persona_product_env
```

---

## Expected Output Format

The inference script prints results in **strict OpenEnv format**:

```
[START] task=easy_student_smartphone env=persona_product_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=BudgetKing X1 reward=0.87 done=false error=null
[STEP] step=2 action=BudgetKing X1 reward=0.72 done=false error=null
[STEP] step=3 action=BudgetKing X1 reward=0.57 done=true error=null
[END] success=true steps=3 rewards=0.87,0.72,0.57
```

**Format rules:**
- `reward` — formatted to 2 decimal places
- `done` / `success` — lowercase `true` or `false`
- `error` — `null` if no error, otherwise a truncated error message
- One `[START]` and `[END]` per task, with `[STEP]` lines between

---

## Runtime Constraints

| Resource     | Limit          |
|-------------|----------------|
| CPU         | 2 cores        |
| RAM         | 8 GB           |
| Max runtime | 20 minutes     |
| Python      | 3.10           |

---

## License

MIT
