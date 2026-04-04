"""Task definitions for persona_product_env."""

from persona_product_env.tasks.easy import EASY_TASK
from persona_product_env.tasks.medium import MEDIUM_TASK
from persona_product_env.tasks.hard import HARD_TASK

ALL_TASKS = [EASY_TASK, MEDIUM_TASK, HARD_TASK]

__all__ = ["EASY_TASK", "MEDIUM_TASK", "HARD_TASK", "ALL_TASKS"]
