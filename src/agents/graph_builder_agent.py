"""
Knowledge Graph Builder Agent using LangGraph.

Implements a state machine for building the Marvel knowledge graph from
extraction results.
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from src.models.power_origin import CharacterExtraction
from src.models.character import Character
from src.graph.operations import GraphOperations
from src.graph.schema import NodeType, RelationType
from src.utils.metrics import validate_extraction


# ============================================================================
# State Definition
# ============================================================================

class GraphBuilderState(TypedDict):
    """State for the graph builder workflow."""
    # Input
    extraction: Optional[CharacterExtraction]
    character: Optional[Character]

    # Processing
    character_id: Optional[str]
    origin_id: Optional[str]
    significance_id: Optional[str]
    power_ids: List[str]
    validation_id: Optional[str]

    # Control
    error: Optional[str]
    completed: bool
    verbose: bool


# ============================================================================
# Graph Builder Agent
# ============================================================================

class GraphBuilderAgent:
    """
    LangGraph-based agent for building the knowledge graph.

    State Machine Flow:
    -------------------
    START
      ↓
    parse_extraction → Validate input data
      ↓
    create_character_node → Add character to graph
      ↓
    create_origin_node → Add power origin
      ↓
    create_significance_node → Add significance data
      ↓
    create_power_nodes → Add individual powers
      ↓
    create_relationships → Link all nodes
      ↓
    validate_graph → Check graph integrity
      ↓
    END
    """

    def __init__(self, graph_ops: GraphOperations, verbose: bool = False):
        """
        Initialize the graph builder agent.

        Args:
            graph_ops: GraphOperations instance for graph manipulation
            verbose: Enable verbose logging
        """
        self.graph_ops = graph_ops
        self.verbose = verbose
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(GraphBuilderState)

        # Add nodes
        workflow.add_node("parse_extraction", self._parse_extraction)
        workflow.add_node("create_character_node", self._create_character_node)
        workflow.add_node("create_origin_node", self._create_origin_node)
        workflow.add_node("create_significance_node", self._create_significance_node)
        workflow.add_node("create_power_nodes", self._create_power_nodes)
        workflow.add_node("create_relationships", self._create_relationships)
        workflow.add_node("validate_graph", self._validate_graph)

        # Define edges
        workflow.set_entry_point("parse_extraction")
        workflow.add_edge("parse_extraction", "create_character_node")
        workflow.add_edge("create_character_node", "create_origin_node")
        workflow.add_edge("create_origin_node", "create_significance_node")
        workflow.add_edge("create_significance_node", "create_power_nodes")
        workflow.add_edge("create_power_nodes", "create_relationships")
        workflow.add_edge("create_relationships", "validate_graph")
        workflow.add_edge("validate_graph", END)

        return workflow.compile()

    # ========================================================================
    # State Machine Nodes
    # ========================================================================

    def _parse_extraction(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Parse and validate extraction data.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Parsing extraction data...")

        extraction = state.get("extraction")
        character = state.get("character")

        if not extraction:
            state["error"] = "No extraction data provided"
            state["completed"] = False
            return state

        if self.verbose:
            print(f"  Character: {extraction.character_name}")
            print(f"  Origin Type: {extraction.power_origin.type}")
            print(f"  Confidence: {extraction.power_origin.confidence}")

        state["error"] = None
        return state

    def _create_character_node(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Create or update character node in the graph.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Creating character node...")

        extraction = state["extraction"]
        character = state.get("character")

        # Prepare character data
        character_data = {
            "name": extraction.character_name
        }

        if character:
            character_data.update({
                "page_id": character.page_id,
                "alignment": character.align,
                "sex": character.sex,
                "alive": character.alive,
                "appearances": int(character.appearances) if character.appearances else None,
                "first_appearance": character.first_appearance,
                "year": int(character.year) if character.year else None
            })

        # Create character node
        character_id = self.graph_ops.add_character_node(**character_data)
        state["character_id"] = character_id

        if self.verbose:
            print(f"  Created character node: {character_id}")

        return state

    def _create_origin_node(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Create power origin node.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Creating power origin node...")

        extraction = state["extraction"]
        character_id = state["character_id"]

        origin = extraction.power_origin
        origin_id = self.graph_ops.add_power_origin_node(
            character_id=character_id,
            origin_type=origin.type.value,
            description=origin.description,
            confidence=origin.confidence.value,
            evidence=origin.evidence
        )

        state["origin_id"] = origin_id

        if self.verbose:
            print(f"  Created origin node: {origin_id}")

        return state

    def _create_significance_node(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Create significance node.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Creating significance node...")

        extraction = state["extraction"]
        character_id = state["character_id"]

        significance = extraction.significance
        significance_id = self.graph_ops.add_significance_node(
            character_id=character_id,
            why_matters=significance.why_matters,
            impact_level=significance.impact_level.value,
            unique_capabilities=significance.unique_capabilities,
            strategic_value=significance.strategic_value
        )

        state["significance_id"] = significance_id

        if self.verbose:
            print(f"  Created significance node: {significance_id}")

        return state

    def _create_power_nodes(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Create power nodes from unique capabilities.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Creating power nodes...")

        extraction = state["extraction"]
        power_ids = []

        for capability in extraction.significance.unique_capabilities:
            power_id = self.graph_ops.add_power_node(
                name=capability,
                description=f"{capability} - {extraction.character_name}"
            )
            power_ids.append(power_id)

            if self.verbose:
                print(f"  Created power node: {power_id} ({capability})")

        state["power_ids"] = power_ids

        return state

    def _create_relationships(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Create all relationships between nodes.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Creating relationships...")

        character_id = state["character_id"]
        origin_id = state["origin_id"]
        significance_id = state["significance_id"]
        power_ids = state["power_ids"]

        # Character -> Origin
        self.graph_ops.add_relationship(
            character_id,
            origin_id,
            RelationType.HAS_ORIGIN
        )
        if self.verbose:
            print(f"  {character_id} -[HAS_ORIGIN]-> {origin_id}")

        # Character -> Significance
        self.graph_ops.add_relationship(
            character_id,
            significance_id,
            RelationType.HAS_SIGNIFICANCE
        )
        if self.verbose:
            print(f"  {character_id} -[HAS_SIGNIFICANCE]-> {significance_id}")

        # Character -> Powers
        for power_id in power_ids:
            self.graph_ops.add_relationship(
                character_id,
                power_id,
                RelationType.POSSESSES_POWER
            )
            if self.verbose:
                print(f"  {character_id} -[POSSESSES_POWER]-> {power_id}")

        # Origin -> Powers
        for power_id in power_ids:
            self.graph_ops.add_relationship(
                origin_id,
                power_id,
                RelationType.CONFERS
            )
            if self.verbose:
                print(f"  {origin_id} -[CONFERS]-> {power_id}")

        return state

    def _validate_graph(self, state: GraphBuilderState) -> GraphBuilderState:
        """
        Validate the graph structure and add validation node.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        if self.verbose:
            print("\n[GraphBuilder] Validating graph...")

        extraction = state["extraction"]
        character_id = state["character_id"]
        origin_id = state["origin_id"]

        # Validate extraction
        validation_result = validate_extraction(extraction)

        # Create validation node
        validation_id = self.graph_ops.add_validation_node(
            character_id=character_id,
            is_valid=validation_result.extraction_passed,
            confidence_score=validation_result.confidence_score,
            completeness_score=validation_result.completeness_score,
            validation_notes=f"Flags: {', '.join(validation_result.flags) if validation_result.flags else 'None'}"
        )

        state["validation_id"] = validation_id

        # Link validation to origin
        self.graph_ops.add_relationship(
            origin_id,
            validation_id,
            RelationType.EXTRACTION_VALIDATED
        )

        if self.verbose:
            print(f"  Created validation node: {validation_id}")
            print(f"  Valid: {validation_result.extraction_passed}")
            print(f"  Confidence: {validation_result.confidence_score:.2f}")
            print(f"  Completeness: {validation_result.completeness_score:.2f}")

        state["completed"] = True
        return state

    # ========================================================================
    # Public Interface
    # ========================================================================

    async def build_character_graph(
        self,
        extraction: CharacterExtraction,
        character: Optional[Character] = None
    ) -> Dict[str, Any]:
        """
        Build graph nodes and relationships for a character.

        Args:
            extraction: Extracted character data
            character: Optional original character data

        Returns:
            Dictionary with created node IDs and status
        """
        initial_state: GraphBuilderState = {
            "extraction": extraction,
            "character": character,
            "character_id": None,
            "origin_id": None,
            "significance_id": None,
            "power_ids": [],
            "validation_id": None,
            "error": None,
            "completed": False,
            "verbose": self.verbose
        }

        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)

        return {
            "character_id": final_state.get("character_id"),
            "origin_id": final_state.get("origin_id"),
            "significance_id": final_state.get("significance_id"),
            "power_ids": final_state.get("power_ids", []),
            "validation_id": final_state.get("validation_id"),
            "completed": final_state.get("completed", False),
            "error": final_state.get("error")
        }

    def build_character_graph_sync(
        self,
        extraction: CharacterExtraction,
        character: Optional[Character] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of build_character_graph.

        Args:
            extraction: Extracted character data
            character: Optional original character data

        Returns:
            Dictionary with created node IDs and status
        """
        initial_state: GraphBuilderState = {
            "extraction": extraction,
            "character": character,
            "character_id": None,
            "origin_id": None,
            "significance_id": None,
            "power_ids": [],
            "validation_id": None,
            "error": None,
            "completed": False,
            "verbose": self.verbose
        }

        # Run the workflow synchronously
        final_state = self.workflow.invoke(initial_state)

        return {
            "character_id": final_state.get("character_id"),
            "origin_id": final_state.get("origin_id"),
            "significance_id": final_state.get("significance_id"),
            "power_ids": final_state.get("power_ids", []),
            "validation_id": final_state.get("validation_id"),
            "completed": final_state.get("completed", False),
            "error": final_state.get("error")
        }


# ============================================================================
# Helper Function
# ============================================================================

async def build_graph_from_extractions(
    extractions: List[CharacterExtraction],
    characters: Optional[List[Character]] = None,
    graph_ops: Optional[GraphOperations] = None,
    verbose: bool = False
) -> GraphOperations:
    """
    Build a complete knowledge graph from a list of extractions.

    Args:
        extractions: List of character extractions
        characters: Optional list of original character data
        graph_ops: Optional existing GraphOperations instance
        verbose: Enable verbose logging

    Returns:
        GraphOperations instance with populated graph
    """
    if graph_ops is None:
        graph_ops = GraphOperations()

    builder = GraphBuilderAgent(graph_ops, verbose=verbose)

    # Create character lookup
    char_lookup = {}
    if characters:
        for char in characters:
            char_lookup[char.name.lower()] = char

    # Build graph for each extraction
    results = []
    for extraction in extractions:
        # Find matching character
        character = char_lookup.get(extraction.character_name.lower())

        # Build graph
        result = await builder.build_character_graph(extraction, character)
        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Completed graph for: {extraction.character_name}")
            print(f"Character ID: {result['character_id']}")
            print(f"Status: {'SUCCESS' if result['completed'] else 'FAILED'}")
            if result.get("error"):
                print(f"Error: {result['error']}")

    return graph_ops


def build_graph_from_extractions_sync(
    extractions: List[CharacterExtraction],
    characters: Optional[List[Character]] = None,
    graph_ops: Optional[GraphOperations] = None,
    verbose: bool = False
) -> GraphOperations:
    """
    Synchronous version of build_graph_from_extractions.

    Args:
        extractions: List of character extractions
        characters: Optional list of original character data
        graph_ops: Optional existing GraphOperations instance
        verbose: Enable verbose logging

    Returns:
        GraphOperations instance with populated graph
    """
    if graph_ops is None:
        graph_ops = GraphOperations()

    builder = GraphBuilderAgent(graph_ops, verbose=verbose)

    # Create character lookup
    char_lookup = {}
    if characters:
        for char in characters:
            char_lookup[char.name.lower()] = char

    # Build graph for each extraction
    results = []
    for extraction in extractions:
        # Find matching character
        character = char_lookup.get(extraction.character_name.lower())

        # Build graph
        result = builder.build_character_graph_sync(extraction, character)
        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Completed graph for: {extraction.character_name}")
            print(f"Character ID: {result['character_id']}")
            print(f"Status: {'SUCCESS' if result['completed'] else 'FAILED'}")
            if result.get("error"):
                print(f"Error: {result['error']}")

    return graph_ops
