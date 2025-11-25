"""
Graph schema definition for Marvel Knowledge Graph.

Defines node types, relationship types, and the overall graph structure
using NetworkX as the backend.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    CHARACTER = "Character"
    POWER_ORIGIN = "PowerOrigin"
    POWER = "Power"
    GENE = "Gene"
    TEAM = "Team"
    SIGNIFICANCE = "Significance"
    VALIDATION = "Validation"


class RelationType(str, Enum):
    """Types of relationships between nodes."""
    HAS_ORIGIN = "HAS_ORIGIN"
    POSSESSES_POWER = "POSSESSES_POWER"
    CONFERS = "CONFERS"
    HAS_MUTATION = "HAS_MUTATION"
    ENABLES = "ENABLES"
    MEMBER_OF = "MEMBER_OF"
    HAS_SIGNIFICANCE = "HAS_SIGNIFICANCE"
    EXTRACTION_VALIDATED = "EXTRACTION_VALIDATED"


# ============================================================================
# Node Data Models
# ============================================================================

class CharacterNode(BaseModel):
    """Character node in the knowledge graph."""
    node_type: NodeType = NodeType.CHARACTER
    node_id: str  # Format: "character_{page_id}"
    name: str
    page_id: Optional[int] = None
    alignment: Optional[str] = None
    sex: Optional[str] = None
    alive: Optional[str] = None
    appearances: Optional[int] = None
    first_appearance: Optional[str] = None
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class PowerOriginNode(BaseModel):
    """Power origin node in the knowledge graph."""
    node_type: NodeType = NodeType.POWER_ORIGIN
    node_id: str  # Format: "origin_{character_id}_{origin_type}"
    origin_type: str  # mutation, accident, technology, etc.
    description: str
    confidence: str  # HIGH, MEDIUM, LOW
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class PowerNode(BaseModel):
    """Power/ability node in the knowledge graph."""
    node_type: NodeType = NodeType.POWER
    node_id: str  # Format: "power_{hash}"
    name: str
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class GeneNode(BaseModel):
    """Gene/mutation node in the knowledge graph."""
    node_type: NodeType = NodeType.GENE
    node_id: str  # Format: "gene_{name}"
    name: str
    description: Optional[str] = None
    source: Optional[str] = None  # e.g., "X-gene", "gamma radiation"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class TeamNode(BaseModel):
    """Team/affiliation node in the knowledge graph."""
    node_type: NodeType = NodeType.TEAM
    node_id: str  # Format: "team_{name}"
    name: str
    affiliation_type: Optional[str] = None  # hero, villain, neutral

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class SignificanceNode(BaseModel):
    """Significance/impact node in the knowledge graph."""
    node_type: NodeType = NodeType.SIGNIFICANCE
    node_id: str  # Format: "significance_{character_id}"
    why_matters: str
    impact_level: str  # COSMIC, GLOBAL, REGIONAL, LOCAL
    unique_capabilities: List[str] = Field(default_factory=list)
    strategic_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


class ValidationNode(BaseModel):
    """Validation result node in the knowledge graph."""
    node_type: NodeType = NodeType.VALIDATION
    node_id: str  # Format: "validation_{character_id}"
    is_valid: bool
    confidence_score: float
    completeness_score: float
    validation_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for graph storage."""
        data = self.model_dump(exclude_none=True, mode='json')
        # Ensure node_type is a string
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = data['node_type']
        return data


# ============================================================================
# Relationship Data Models
# ============================================================================

class Relationship(BaseModel):
    """Base relationship model."""
    relation_type: RelationType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    def to_tuple(self) -> tuple:
        """Convert to tuple for NetworkX edge creation."""
        return (self.source_id, self.target_id, {
            "relation_type": self.relation_type.value,
            **self.properties
        })


# ============================================================================
# Graph Schema Definition
# ============================================================================

class GraphSchema:
    """
    Defines the complete schema for the Marvel Knowledge Graph.

    Schema Overview:
    ----------------
    (Character) -[HAS_ORIGIN]-> (PowerOrigin)
    (Character) -[POSSESSES_POWER]-> (Power)
    (PowerOrigin) -[CONFERS]-> (Power)
    (Character) -[HAS_MUTATION]-> (Gene)
    (Gene) -[ENABLES]-> (Power)
    (Character) -[MEMBER_OF]-> (Team)
    (Character) -[HAS_SIGNIFICANCE]-> (Significance)
    (PowerOrigin) -[EXTRACTION_VALIDATED]-> (Validation)
    """

    # Valid node types
    NODE_TYPES = {
        NodeType.CHARACTER: CharacterNode,
        NodeType.POWER_ORIGIN: PowerOriginNode,
        NodeType.POWER: PowerNode,
        NodeType.GENE: GeneNode,
        NodeType.TEAM: TeamNode,
        NodeType.SIGNIFICANCE: SignificanceNode,
        NodeType.VALIDATION: ValidationNode,
    }

    # Valid relationships (source_type -> relation -> target_type)
    VALID_RELATIONSHIPS = {
        (NodeType.CHARACTER, RelationType.HAS_ORIGIN, NodeType.POWER_ORIGIN),
        (NodeType.CHARACTER, RelationType.POSSESSES_POWER, NodeType.POWER),
        (NodeType.POWER_ORIGIN, RelationType.CONFERS, NodeType.POWER),
        (NodeType.CHARACTER, RelationType.HAS_MUTATION, NodeType.GENE),
        (NodeType.GENE, RelationType.ENABLES, NodeType.POWER),
        (NodeType.CHARACTER, RelationType.MEMBER_OF, NodeType.TEAM),
        (NodeType.CHARACTER, RelationType.HAS_SIGNIFICANCE, NodeType.SIGNIFICANCE),
        (NodeType.POWER_ORIGIN, RelationType.EXTRACTION_VALIDATED, NodeType.VALIDATION),
    }

    @staticmethod
    def validate_relationship(
        source_type: NodeType,
        relation: RelationType,
        target_type: NodeType
    ) -> bool:
        """Validate if a relationship is allowed by the schema."""
        return (source_type, relation, target_type) in GraphSchema.VALID_RELATIONSHIPS

    @staticmethod
    def get_node_model(node_type: NodeType):
        """Get the Pydantic model for a node type."""
        return GraphSchema.NODE_TYPES.get(node_type)
