"""Multi-Persona Signal Interpretation — Pydantic models."""

from typing import Literal
from pydantic import BaseModel, Field


class PersonaOpinion(BaseModel):
    """Single persona's analysis output."""
    persona: str = Field(description="Persona identifier: force_expert | div_expert | chairman")
    persona_name: str = Field(description="Display name in Korean")
    analysis: str = Field(description="Structured 2-3 sentence analysis in Korean with bold section tags")
    conviction: int = Field(ge=1, le=5, description="Conviction level 1(very low) to 5(very high)")
    key_point: str = Field(description="One-line key insight in Korean, max 40 chars")


class SignalInterpretation(BaseModel):
    """Complete multi-persona interpretation of a buy signal."""
    force_expert: PersonaOpinion
    div_expert: PersonaOpinion
    chairman: PersonaOpinion
    final_verdict: Literal["STRONG_BUY", "BUY", "CAUTIOUS_BUY", "HOLD"]
    risk_note: str = Field(description="One-line risk caveat in Korean, max 60 chars")
    confidence_score: int = Field(
        ge=1, le=100,
        description="Overall confidence derived from weighted conviction average"
    )
