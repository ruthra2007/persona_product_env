"""
Easy task — clear best product, simple decision.

Scenario: A *student* with a $600 budget looking for the best value
smartphone with good battery life and decent performance.
One product clearly dominates on every axis within budget.
"""

from persona_product_env.models import Product, TaskSpec

EASY_TASK = TaskSpec(
    name="easy_student_smartphone",
    difficulty="easy",
    persona="student",
    budget=600.0,
    preferences=["battery", "performance", "price"],
    products=[
        Product(
            name="BudgetKing X1",
            brand="ValueTech",
            price=349.0,
            battery=9.0,
            performance=8.0,
            camera=6.0,
            display=7.0,
            build_quality=7.0,
            storage=7.0,
            portability=8.0,
        ),
        Product(
            name="UltraGlam Pro",
            brand="LuxPhone",
            price=1199.0,
            battery=7.0,
            performance=9.5,
            camera=9.5,
            display=9.5,
            build_quality=9.5,
            storage=9.0,
            portability=6.0,
        ),
        Product(
            name="MidRange M5",
            brand="AvgCorp",
            price=549.0,
            battery=6.5,
            performance=6.5,
            camera=6.5,
            display=6.5,
            build_quality=6.0,
            storage=6.0,
            portability=7.0,
        ),
        Product(
            name="OldTimer 3",
            brand="RetroMobile",
            price=199.0,
            battery=5.0,
            performance=4.0,
            camera=3.0,
            display=4.0,
            build_quality=5.0,
            storage=4.0,
            portability=8.5,
        ),
    ],
    optimal_product="BudgetKing X1",
    explanation=(
        "BudgetKing X1 is well within the $600 budget at $349, has the best "
        "battery score (9.0) and strong performance (8.0). It clearly dominates "
        "on the student's top preferences while being the best value option."
    ),
    max_steps=3,
)
