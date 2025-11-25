"""
API Request/Response Models for Marvel Knowledge Graph API.

Defines Pydantic models for all API endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence levels for responses."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"
    N_A = "N/A"


class QueryType(str, Enum):
    """Types of queries."""
    POWER_ORIGIN = "POWER_ORIGIN"
    POWER_ABILITIES = "POWER_ABILITIES"
    SIGNIFICANCE = "SIGNIFICANCE"
    GENETIC = "GENETIC"
    TEAM = "TEAM"
    VALIDATION = "VALIDATION"
    COMPARISON = "COMPARISON"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Request Models
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for POST /question endpoint."""
    question: str = Field(
        ...,
        description="Natural language question about Marvel characters",
        min_length=3,
        max_length=500,
        examples=["How did Spider-Man get his powers?"]
    )
    verbose: Optional[bool] = Field(
        default=False,
        description="Enable detailed processing logs"
    )
    include_context: Optional[bool] = Field(
        default=False,
        description="Include raw graph context in response"
    )


class ValidationRequest(BaseModel):
    """Request model for POST /validate-extraction endpoint."""
    character_id: Optional[str] = Field(
        default=None,
        description="Character node ID (e.g., 'character_1678')"
    )
    character_name: Optional[str] = Field(
        default=None,
        description="Character name (alternative to character_id)"
    )
    enable_multi_pass: Optional[bool] = Field(
        default=False,
        description="Enable multi-pass consistency checking (slower, more expensive)"
    )
    verbose: Optional[bool] = Field(
        default=False,
        description="Enable detailed validation logs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "character_name": "Spider-Man (Peter Parker)",
                "enable_multi_pass": False,
                "verbose": False
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class QuestionResponse(BaseModel):
    """Response model for POST /question endpoint."""
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Natural language answer")
    query_type: Optional[str] = Field(None, description="Classified query type")
    characters: List[str] = Field(
        default_factory=list,
        description="List of character names found in question"
    )
    confidence_level: str = Field(
        default="UNKNOWN",
        description="Confidence level of the answer"
    )
    context_retrieved: bool = Field(
        default=False,
        description="Whether relevant graph context was found"
    )
    raw_context: Optional[str] = Field(
        None,
        description="Raw graph context used (only if include_context=True)"
    )
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How did Spider-Man get his powers?",
                "answer": "Spider-Man gained his powers after being bitten by a radioactive spider...",
                "query_type": "POWER_ORIGIN",
                "characters": ["Spider-Man (Peter Parker)"],
                "confidence_level": "HIGH",
                "context_retrieved": True,
                "error": None
            }
        }


class CharacterNode(BaseModel):
    """Character node representation."""
    node_id: str
    name: str
    alignment: Optional[str] = None
    sex: Optional[str] = None
    alive: Optional[str] = None
    appearances: Optional[float] = None
    first_appearance: Optional[str] = None
    year: Optional[float] = None


class PowerOriginNode(BaseModel):
    """Power origin node representation."""
    node_id: str
    origin_type: str
    description: str
    confidence: str
    evidence: str


class PowerNode(BaseModel):
    """Power node representation."""
    node_id: str
    name: str
    description: str


class SignificanceNode(BaseModel):
    """Significance node representation."""
    node_id: str
    why_matters: str
    impact_level: str
    unique_capabilities: Optional[List[str]] = None
    strategic_value: Optional[str] = None


class ValidationNode(BaseModel):
    """Validation node representation."""
    node_id: str
    is_valid: bool
    confidence_score: float
    completeness_score: float
    semantic_similarity: Optional[float] = None
    consistency_score: Optional[float] = None
    validation_notes: Optional[List[str]] = None


class CharacterGraphResponse(BaseModel):
    """Response model for GET /graph/{character} endpoint."""
    character: CharacterNode
    power_origin: Optional[PowerOriginNode] = None
    powers: List[PowerNode] = Field(default_factory=list)
    significance: Optional[SignificanceNode] = None
    validation: Optional[ValidationNode] = None
    teams: List[Dict[str, Any]] = Field(default_factory=list)
    mutations: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "character": {
                    "node_id": "character_1678",
                    "name": "Spider-Man (Peter Parker)",
                    "alignment": "Good",
                    "sex": "Male",
                    "alive": "Living",
                    "appearances": 4043.0
                },
                "power_origin": {
                    "node_id": "origin_character_1678_accident",
                    "origin_type": "accident",
                    "description": "Bitten by radioactive spider",
                    "confidence": "HIGH",
                    "evidence": "Peter Parker gained his powers..."
                },
                "powers": [
                    {
                        "node_id": "power_abc123",
                        "name": "Wall-crawling",
                        "description": "Ability to adhere to surfaces"
                    }
                ],
                "significance": {
                    "node_id": "significance_character_1678",
                    "why_matters": "Protects New York City",
                    "impact_level": "LOCAL",
                    "unique_capabilities": ["Spider-sense", "Web-shooting"]
                },
                "validation": {
                    "node_id": "validation_character_1678",
                    "is_valid": True,
                    "confidence_score": 1.0,
                    "completeness_score": 0.88
                }
            }
        }


class ExtractionReportResponse(BaseModel):
    """Response model for GET /extraction-report/{character} endpoint."""
    character_id: str
    character_name: str
    validation_passed: bool
    confidence_score: float
    completeness_score: float
    semantic_similarity: Optional[float] = None
    consistency_score: Optional[float] = None
    overall_quality: float
    quality_tier: str  # "HIGH", "MEDIUM", "LOW"
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    validation_flags: List[str] = Field(default_factory=list)
    extraction_data: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "character_id": "character_1678",
                "character_name": "Spider-Man (Peter Parker)",
                "validation_passed": True,
                "confidence_score": 1.0,
                "completeness_score": 0.88,
                "semantic_similarity": 0.67,
                "overall_quality": 0.85,
                "quality_tier": "HIGH",
                "strengths": [
                    "High confidence extraction",
                    "Complete power origin data"
                ],
                "weaknesses": [
                    "Semantic similarity below threshold"
                ],
                "recommendations": [
                    "Review extraction for accuracy"
                ],
                "validation_flags": []
            }
        }


class ValidationResponse(BaseModel):
    """Response model for POST /validate-extraction endpoint."""
    character_id: str
    character_name: str
    validation_result: Dict[str, Any]
    validation_passed: bool
    confidence_score: float
    completeness_score: float
    semantic_similarity: Optional[float] = None
    consistency_score: Optional[float] = None
    processing_time_seconds: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "character_id": "character_1678",
                "character_name": "Spider-Man (Peter Parker)",
                "validation_result": {
                    "is_valid": True,
                    "confidence_score": 1.0,
                    "completeness_score": 0.88
                },
                "validation_passed": True,
                "confidence_score": 1.0,
                "completeness_score": 0.88,
                "semantic_similarity": 0.67,
                "processing_time_seconds": 2.3,
                "message": "Validation completed successfully"
            }
        }


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""
    status: str = Field(..., description="API health status")
    graph_loaded: bool = Field(..., description="Whether knowledge graph is loaded")
    total_nodes: int = Field(..., description="Total nodes in graph")
    total_edges: int = Field(..., description="Total edges in graph")
    characters_count: int = Field(..., description="Number of character nodes")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Character not found",
                "detail": "No character found with ID 'character_9999'",
                "status_code": 404
            }
        }
