"""
Hard task — conflicting preferences + tight budget.

Scenario: A *business_user* with a tight $500 budget who needs high
performance for multitasking, great display for presentations, AND
strong build quality for travel durability.  No single product excels
at all three within budget, so the agent must carefully weigh trade-offs
and justify a nuanced pick.
"""

from persona_product_env.models import Product, TaskSpec

HARD_TASK = TaskSpec(
    name="hard_business_conflicting",
    difficulty="hard",
    persona="business_user",
    budget=500.0,
    preferences=["performance", "display", "build_quality", "portability"],
    products=[
        Product(
            name="ExecSlim Pro",
            brand="BizTech",
            price=489.0,
            battery=6.5,
            performance=7.5,
            camera=5.0,
            display=8.0,
            build_quality=8.5,
            storage=6.0,
            portability=8.0,
        ),
        Product(
            name="PowerHouse P1",
            brand="TurboGear",
            price=479.0,
            battery=5.0,
            performance=9.0,
            camera=5.5,
            display=7.0,
            build_quality=6.0,
            storage=8.0,
            portability=5.0,
        ),
        Product(
            name="DisplayKing D4",
            brand="ScreenMax",
            price=510.0,
            battery=6.0,
            performance=6.5,
            camera=6.0,
            display=9.5,
            build_quality=6.5,
            storage=6.0,
            portability=5.5,
        ),
        Product(
            name="ChunkArmor C2",
            brand="ToughLine",
            price=449.0,
            battery=7.5,
            performance=6.0,
            camera=4.0,
            display=6.0,
            build_quality=9.5,
            storage=5.5,
            portability=4.0,
        ),
        Product(
            name="BalanceBiz B7",
            brand="EvenCo",
            price=469.0,
            battery=7.0,
            performance=7.0,
            camera=5.5,
            display=7.5,
            build_quality=7.0,
            storage=6.5,
            portability=7.0,
        ),
        Product(
            name="LuxeUltra L1",
            brand="PremiumCraft",
            price=899.0,
            battery=8.5,
            performance=9.0,
            camera=8.0,
            display=9.0,
            build_quality=9.0,
            storage=9.0,
            portability=7.0,
        ),
    ],
    optimal_product="ExecSlim Pro",
    explanation=(
        "ExecSlim Pro ($489) is the only product within the tight $500 budget "
        "that balances all four preferences: decent performance (7.5), strong "
        "display (8.0), excellent build quality (8.5), and high portability "
        "(8.0). PowerHouse P1 wins on raw performance but sacrifices build "
        "quality and portability. DisplayKing D4 exceeds the budget. "
        "ChunkArmor C2 has poor portability and display. BalanceBiz B7 is "
        "acceptable but scores lower across nearly every preference axis."
    ),
    max_steps=3,
)
