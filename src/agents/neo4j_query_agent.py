"""
Neo4j Query Agent using LangGraph for natural language question answering.

Implements a state machine for processing user questions, retrieving graph context from Neo4j,
and generating citation-grounded responses. This mirrors query_agent.py but uses Neo4j instead
of in-memory NetworkX graphs.
"""

from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum

from langgraph.graph import StateGraph, END
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.graph.neo4j_operations import Neo4jOperations
from src.graph.neo4j_queries import Neo4jQueries
from src.prompts.query_prompts import (
    QUERY_CLASSIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    CITATION_RESPONSE_PROMPT,
    COMPARISON_PROMPT,
    NO_DATA_RESPONSE_PROMPT,
    INSUFFICIENT_DATA_PROMPT,
)


# ============================================================================
# Query Types
# ============================================================================

class QueryType(str, Enum):
    """Types of queries the agent can handle."""
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
# State Definition
# ============================================================================

class QueryAgentState(TypedDict):
    """State for the query agent workflow."""
    # Input
    question: str

    # Processing
    query_type: Optional[str]
    character_names: List[str]
    character_ids: List[str]

    # Retrieved Context
    characters_data: List[Dict[str, Any]]
    origins_data: List[Dict[str, Any]]
    powers_data: List[Dict[str, Any]]
    significance_data: List[Dict[str, Any]]
    validation_data: List[Dict[str, Any]]

    # Context Assembly
    formatted_context: Optional[str]
    validation_summary: Optional[str]

    # Response
    answer: Optional[str]
    citations: List[str]
    confidence_level: Optional[str]

    # Control
    error: Optional[str]
    verbose: bool


# ============================================================================
# Neo4j Query Agent
# ============================================================================

class Neo4jQueryAgent:
    """
    LangGraph-based agent for natural language question answering using Neo4j.

    State Machine Flow:
    -------------------
    START
      ↓
    classify_query → Determine query type
      ↓
    extract_entities → Extract character names
      ↓
    retrieve_context → Get data from Neo4j knowledge graph
      ↓
    format_context → Structure context for LLM
      ↓
    generate_response → Create natural language answer
      ↓
    END
    """

    def __init__(
        self,
        neo4j_ops: Neo4jOperations,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize the Neo4j query agent.

        Args:
            neo4j_ops: Neo4jOperations instance
            llm_model: LLM model to use
            temperature: LLM temperature (higher = more creative)
            verbose: Enable verbose logging
        """
        self.neo4j_ops = neo4j_ops
        self.queries = Neo4jQueries(neo4j_ops)
        self.llm = OpenAI(model=llm_model, temperature=temperature)
        self.verbose = verbose
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(QueryAgentState)

        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("extract_entities", self._extract_entities)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("format_context", self._format_context)
        workflow.add_node("generate_response", self._generate_response)

        # Define edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "extract_entities")
        workflow.add_edge("extract_entities", "retrieve_context")
        workflow.add_edge("retrieve_context", "format_context")
        workflow.add_edge("format_context", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    # ========================================================================
    # State Machine Nodes
    # ========================================================================

    def _classify_query(self, state: QueryAgentState) -> QueryAgentState:
        """
        Classify the query type using LLM.

        Args:
            state: Current workflow state

        Returns:
            Updated state with query_type
        """
        if self.verbose:
            print("\n[Neo4jQueryAgent] Classifying query...")

        question = state["question"]

        # Use LLM to classify
        prompt = QUERY_CLASSIFICATION_PROMPT.format(question=question)
        response = self.llm.complete(prompt)
        query_type = response.text.strip().upper()

        # Validate it's a known type
        try:
            QueryType(query_type)
            state["query_type"] = query_type
        except ValueError:
            state["query_type"] = QueryType.GENERAL.value

        if self.verbose:
            print(f"  Query Type: {state['query_type']}")

        return state

    def _extract_entities(self, state: QueryAgentState) -> QueryAgentState:
        """
        Extract character names from the question.

        Args:
            state: Current workflow state

        Returns:
            Updated state with character_names
        """
        if self.verbose:
            print("\n[Neo4jQueryAgent] Extracting character names...")

        question = state["question"]

        # Use LLM to extract entities
        prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
        response = self.llm.complete(prompt)
        entities_text = response.text.strip()

        # Parse character names
        if entities_text.upper() == "NONE":
            state["character_names"] = []
        else:
            # Split by comma and clean
            names = [name.strip() for name in entities_text.split(",")]
            state["character_names"] = names

        if self.verbose:
            print(f"  Characters: {state['character_names']}")

        return state

    def _retrieve_context(self, state: QueryAgentState) -> QueryAgentState:
        """
        Retrieve relevant context from the Neo4j knowledge graph.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved data
        """
        if self.verbose:
            print("\n[Neo4jQueryAgent] Retrieving context from Neo4j...")

        character_names = state["character_names"]
        query_type = state["query_type"]

        # Initialize data lists
        state["characters_data"] = []
        state["origins_data"] = []
        state["powers_data"] = []
        state["significance_data"] = []
        state["validation_data"] = []
        state["character_ids"] = []

        # If no characters mentioned, can't retrieve specific data
        if not character_names:
            if self.verbose:
                print("  No characters specified")
            return state

        # Retrieve data for each character
        for char_name in character_names:
            # Find character
            char_node = self.queries.find_character_by_name(char_name)

            if not char_node:
                if self.verbose:
                    print(f"  Character not found: {char_name}")
                continue

            char_id = char_node["node_id"]
            state["character_ids"].append(char_id)
            state["characters_data"].append(char_node)

            if self.verbose:
                print(f"  Found: {char_node.get('name', 'Unknown')}")

            # Retrieve based on query type
            if query_type in [QueryType.POWER_ORIGIN.value, QueryType.GENERAL.value]:
                origin = self.queries.get_character_power_origin(char_id)
                if origin:
                    state["origins_data"].append(origin)

            if query_type in [QueryType.POWER_ABILITIES.value, QueryType.GENERAL.value]:
                powers = self.queries.get_character_powers(char_id)
                state["powers_data"].extend(powers)

            if query_type in [QueryType.SIGNIFICANCE.value, QueryType.GENERAL.value]:
                significance = self.queries.get_character_significance(char_id)
                if significance:
                    state["significance_data"].append(significance)

            if query_type == QueryType.VALIDATION.value:
                validation = self.queries.get_extraction_validation(char_id)
                if validation:
                    state["validation_data"].append(validation)

        if self.verbose:
            print(f"  Retrieved: {len(state['characters_data'])} characters, "
                  f"{len(state['origins_data'])} origins, {len(state['powers_data'])} powers")

        return state

    def _format_context(self, state: QueryAgentState) -> QueryAgentState:
        """
        Format retrieved context into a structured string for the LLM.

        Args:
            state: Current workflow state

        Returns:
            Updated state with formatted_context
        """
        if self.verbose:
            print("\n[Neo4jQueryAgent] Formatting context...")

        context_parts = []

        # Format character data
        for char in state["characters_data"]:
            char_context = f"\n### Character: {char.get('name', 'Unknown')}\n"
            char_context += f"- ID: {char.get('node_id', 'N/A')}\n"
            char_context += f"- Alignment: {char.get('alignment', 'Unknown')}\n"
            char_context += f"- Appearances: {char.get('appearances', 'Unknown')}\n"
            context_parts.append(char_context)

        # Format origin data
        for origin in state["origins_data"]:
            origin_context = f"\n### Power Origin:\n"
            origin_context += f"- Type: {origin.get('origin_type', 'Unknown')}\n"
            origin_context += f"- Description: {origin.get('description', 'N/A')}\n"
            origin_context += f"- Confidence: {origin.get('confidence', 'Unknown')}\n"
            origin_context += f"- Evidence: {origin.get('evidence', 'N/A')}\n"
            context_parts.append(origin_context)

        # Format powers data
        if state["powers_data"]:
            powers_context = "\n### Powers and Abilities:\n"
            for i, power in enumerate(state["powers_data"], 1):
                powers_context += f"{i}. {power.get('name', 'Unknown')}: {power.get('description', 'N/A')}\n"
            context_parts.append(powers_context)

        # Format significance data
        for sig in state["significance_data"]:
            sig_context = f"\n### Significance:\n"
            sig_context += f"- Why It Matters: {sig.get('why_matters', 'N/A')}\n"
            sig_context += f"- Impact Level: {sig.get('impact_level', 'Unknown')}\n"
            sig_context += f"- Strategic Value: {sig.get('strategic_value', 'N/A')}\n"
            context_parts.append(sig_context)

        # Format validation data
        validation_parts = []
        for val in state["validation_data"]:
            val_text = f"Confidence: {val.get('confidence_score', 0):.2f}, "
            val_text += f"Completeness: {val.get('completeness_score', 0):.2f}, "
            val_text += f"Valid: {val.get('is_valid', False)}"
            validation_parts.append(val_text)

        state["formatted_context"] = "\n".join(context_parts) if context_parts else "No data found."
        state["validation_summary"] = " | ".join(validation_parts) if validation_parts else "No validation data available."

        if self.verbose:
            print(f"  Context length: {len(state['formatted_context'])} chars")

        return state

    def _generate_response(self, state: QueryAgentState) -> QueryAgentState:
        """
        Generate natural language response using LLM.

        Args:
            state: Current workflow state

        Returns:
            Updated state with answer
        """
        if self.verbose:
            print("\n[Neo4jQueryAgent] Generating response...")

        question = state["question"]
        context = state["formatted_context"]
        validation_info = state["validation_summary"]

        # Check if we have any data
        if not state["characters_data"]:
            # No character found
            char_name = state["character_names"][0] if state["character_names"] else "the requested character"
            prompt = NO_DATA_RESPONSE_PROMPT.format(
                question=question,
                character_name=char_name
            )
            response = self.llm.complete(prompt)
            state["answer"] = response.text.strip()
            state["confidence_level"] = "N/A"
            return state

        # Check if context is insufficient
        if context == "No data found." or len(context) < 50:
            char_name = state["characters_data"][0].get("name", "Unknown")
            prompt = INSUFFICIENT_DATA_PROMPT.format(
                question=question,
                character_name=char_name,
                available_context=context
            )
            response = self.llm.complete(prompt)
            state["answer"] = response.text.strip()
            state["confidence_level"] = "LOW"
            return state

        # Generate full response
        prompt = RESPONSE_GENERATION_PROMPT.format(
            question=question,
            context=context,
            validation_info=validation_info
        )

        response = self.llm.complete(prompt)
        state["answer"] = response.text.strip()

        # Determine confidence level from validation data
        if state["validation_data"]:
            avg_confidence = sum(v.get("confidence_score", 0) for v in state["validation_data"]) / len(state["validation_data"])
            if avg_confidence >= 0.8:
                state["confidence_level"] = "HIGH"
            elif avg_confidence >= 0.6:
                state["confidence_level"] = "MEDIUM"
            else:
                state["confidence_level"] = "LOW"
        else:
            state["confidence_level"] = "UNKNOWN"

        if self.verbose:
            print(f"  Confidence: {state['confidence_level']}")

        return state

    # ========================================================================
    # Public Methods
    # ========================================================================

    def query(self, question: str, verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        Process a natural language question and return an answer.

        Args:
            question: User's question
            verbose: Override verbose setting for this query

        Returns:
            Dictionary with answer and metadata
        """
        use_verbose = verbose if verbose is not None else self.verbose

        # Initialize state
        initial_state: QueryAgentState = {
            "question": question,
            "query_type": None,
            "character_names": [],
            "character_ids": [],
            "characters_data": [],
            "origins_data": [],
            "powers_data": [],
            "significance_data": [],
            "validation_data": [],
            "formatted_context": None,
            "validation_summary": None,
            "answer": None,
            "citations": [],
            "confidence_level": None,
            "error": None,
            "verbose": use_verbose
        }

        # Run workflow
        final_state = self.workflow.invoke(initial_state)

        # Return result
        return {
            "question": question,
            "answer": final_state.get("answer", "I couldn't generate an answer."),
            "query_type": final_state.get("query_type"),
            "characters": [c.get("name") for c in final_state.get("characters_data", [])],
            "confidence_level": final_state.get("confidence_level"),
            "context_retrieved": bool(final_state.get("characters_data")),
            "error": final_state.get("error")
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_neo4j_query_agent(
    neo4j_ops: Neo4jOperations = None,
    llm_model: str = "gpt-4o-mini",
    verbose: bool = False
) -> Neo4jQueryAgent:
    """
    Create a Neo4j query agent.

    Args:
        neo4j_ops: Neo4jOperations instance (creates new if None)
        llm_model: LLM model to use
        verbose: Enable verbose logging

    Returns:
        Neo4jQueryAgent instance
    """
    if neo4j_ops is None:
        neo4j_ops = Neo4jOperations()

    return Neo4jQueryAgent(neo4j_ops, llm_model=llm_model, verbose=verbose)
