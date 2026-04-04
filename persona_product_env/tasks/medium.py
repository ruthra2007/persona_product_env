"""
Medium task — trade-offs between camera vs battery.

Scenario: A *photographer* with an $800 budget who wants the best camera
but also cares about battery life for long outdoor shoots.
Two products are close; the agent must reason about trade-offs.
"""

from persona_product_env.models import Product, TaskSpec

MEDIUM_TASK = TaskSpec(
    name="medium_photographer_tradeoff",
    difficulty="medium",
    persona="photographer",
    budget=800.0,
    preferences=["camera", "battery", "display"],
    products=[
        Product(
            name="PhotoMaster 7",
            brand="CamTech",
            price=749.0,
            battery=7.0,
            performance=7.0,
            camera=9.5,
            display=8.5,
            build_quality=7.5,
            storage=8.0,
            portability=6.5,
        ),
        Product(
            name="EnduroSnap E2",
            brand="DuraCam",
            price=699.0,
            battery=9.5,
            performance=6.5,
            camera=8.0,
            display=7.0,
            build_quality=8.0,
            storage=7.0,
            portability=7.5,
        ),
        Product(
            name="GamerFury Z9",
            brand="PlayMax",
            price=799.0,
            battery=5.0,
            performance=9.5,
            camera=5.0,
            display=9.0,
            build_quality=7.0,
            storage=8.5,
            portability=5.5,
        ),
        Product(
            name="BasicLine S1",
            brand="EconoBrand",
            price=299.0,
            battery=6.0,
            performance=5.0,
            camera=4.0,
            display=5.0,
            build_quality=5.0,
            storage=4.0,
            portability=8.0,
        ),
        Product(
            name="AllRound A3",
            brand="BalanceCo",
            price=599.0,
            battery=7.5,
            performance=7.0,
            camera=7.5,
            display=7.5,
            build_quality=7.5,
            storage=7.0,
            portability=7.0,
        ),
    ],
    optimal_product="PhotoMaster 7",
    explanation=(
        "PhotoMaster 7 has the highest camera score (9.5) which is the "
        "photographer's top preference, combined with a good display (8.5) "
        "and acceptable battery (7.0). While EnduroSnap E2 has better battery, "
        "the photographer persona prioritises camera quality above all, making "
        "PhotoMaster 7 the optimal trade-off within the $800 budget."
    ),
    max_steps=3,
)
