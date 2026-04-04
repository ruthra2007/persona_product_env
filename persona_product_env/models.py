"""
Typed Pydantic models for the PersonaProductEnv OpenEnv environment.

Defines Observation, Action, and Reward schemas used throughout the environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Product model
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """A single product with structured attributes."""

    name: str = Field(..., description="Unique product name")
    brand: str = Field(..., description="Manufacturer / brand")
    price: float = Field(..., ge=0, description="Price in USD")
    battery: float = Field(
        ..., ge=0, le=10, description="Battery score (0-10)"
    )
    performance: float = Field(
        ..., ge=0, le=10, description="Performance score (0-10)"
    )
    camera: float = Field(
        ..., ge=0, le=10, description="Camera quality score (0-10)"
    )
    display: float = Field(
        ..., ge=0, le=10, description="Display quality score (0-10)"
    )
    build_quality: float = Field(
        ..., ge=0, le=10, description="Build quality score (0-10)"
    )
    storage: float = Field(
        ..., ge=0, le=10, description="Storage score (0-10)"
    )
    portability: float = Field(
        ..., ge=0, le=10, description="Portability score (0-10)"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """State observed by the agent at each step."""

    persona: str = Field(
        ...,
        description="User persona, e.g. 'student', 'gamer', 'photographer', 'business_user'",
    )
    budget: float = Field(..., ge=0, description="Maximum budget in USD")
    preferences: List[str] = Field(
        ...,
        description="Ordered list of preference keywords, e.g. ['camera', 'battery']",
    )
    products: List[Product] = Field(
        ..., min_length=1, description="Available products to choose from"
    )
    current_step: int = Field(
        ..., ge=0, description="Current step count in the episode"
    )
    max_steps: int = Field(
        ..., ge=1, description="Maximum allowed steps in the episode"
    )
    previous_selections: List[str] = Field(
        default_factory=list,
        description="Products selected in prior steps (for repeat penalty)",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Action taken by the agent."""

    selected_product: str = Field(
        ..., description="Name of the product the agent selects"
    )
    reasoning: str = Field(
        ..., min_length=1, description="Free-text reasoning for the selection"
    )

    @field_validator("selected_product")
    @classmethod
    def product_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("selected_product must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Structured reward returned after each step."""

    total: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate reward (0.0-1.0)"
    )
    budget_fit: float = Field(
        ..., ge=0.0, le=1.0, description="Budget fitness component"
    )
    preference_match: float = Field(
        ..., ge=0.0, le=1.0, description="Preference match component"
    )
    reasoning_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Reasoning quality component"
    )
    penalties: Dict[str, float] = Field(
        default_factory=dict,
        description="Itemised penalties applied (name -> value)",
    )


# ---------------------------------------------------------------------------
# Task specification (used by easy / medium / hard modules)
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    """A deterministic task definition."""

    name: str = Field(..., description="Task identifier")
    difficulty: str = Field(..., description="easy | medium | hard")
    persona: str
    budget: float = Field(..., ge=0)
    preferences: List[str]
    products: List[Product]
    optimal_product: str = Field(
        ..., description="Name of the expected best product"
    )
    explanation: str = Field(
        ..., description="Why the optimal product is the best choice"
    )
    max_steps: int = Field(default=3, ge=1)
