"""
Pydantic models for power origin extraction.
"""
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class OriginType(str, Enum):
    """Categories of power origins."""
    MUTATION = "mutation"
    ACCIDENT = "accident"
    TECHNOLOGY = "technology"
    MYSTICAL = "mystical"
    TRAINING = "training"
    COSMIC = "cosmic"
    BIRTH = "birth"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for extractions."""
    HIGH = "high"      # Explicitly stated in text
    MEDIUM = "medium"  # Strongly implied
    LOW = "low"        # Inferred or uncertain


class ImpactLevel(str, Enum):
    """Impact levels for power significance."""
    COSMIC = "cosmic"      # Universal/multiverse level
    GLOBAL = "global"      # Planetary level
    REGIONAL = "regional"  # City/country level
    LOCAL = "local"        # Individual/team level


class PowerOrigin(BaseModel):
    """Structured representation of how a character got their powers."""

    type: OriginType = Field(
        description="The category of power origin"
    )

    description: str = Field(
        description="Detailed explanation of how the character acquired their powers"
    )

    confidence: ConfidenceLevel = Field(
        description="Confidence level of the extraction based on text evidence"
    )

    evidence: str = Field(
        description="Direct quote or paraphrase from the source text supporting this extraction"
    )


class Significance(BaseModel):
    """Structured representation of why a character's powers matter."""

    why_matters: str = Field(
        description="Explanation of the importance and impact of the character's powers"
    )

    impact_level: ImpactLevel = Field(
        description="The scope/scale of impact these powers have"
    )

    unique_capabilities: List[str] = Field(
        default_factory=list,
        description="List of unique or distinctive abilities this character possesses"
    )

    strategic_value: Optional[str] = Field(
        default=None,
        description="The strategic importance to their team or in combat scenarios"
    )


class CharacterExtraction(BaseModel):
    """Complete extraction result for a character."""

    character_name: str = Field(
        description="Name of the character"
    )

    character_id: Optional[int] = Field(
        default=None,
        description="Character page ID from the dataset"
    )

    power_origin: PowerOrigin = Field(
        description="Extracted power origin information"
    )

    significance: Significance = Field(
        description="Extracted significance/impact information"
    )

    extraction_timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp of when extraction was performed"
    )


class ValidationResult(BaseModel):
    """Validation metrics for an extraction."""

    character_name: str

    extraction_passed: bool = Field(
        description="Whether the extraction passed validation checks"
    )

    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the extraction (0-1)"
    )

    semantic_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity between extraction and source text"
    )

    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How complete the extraction is (all fields populated)"
    )

    flags: List[str] = Field(
        default_factory=list,
        description="Any issues or warnings flagged during validation"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the validation"
    )
