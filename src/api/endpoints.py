"""
API Endpoints for Marvel Knowledge Graph API.

Implements all REST API endpoints for querying the knowledge graph.
"""

import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.models import (
    QuestionRequest,
    QuestionResponse,
    CharacterGraphResponse,
    ExtractionReportResponse,
    ValidationRequest,
    ValidationResponse,
    HealthResponse,
    ErrorResponse,
    CharacterNode,
    PowerOriginNode,
    PowerNode,
    SignificanceNode,
    ValidationNode,
)
from src.agents.query_agent import QueryAgent
from src.agents.validation_agent import validate_character_extraction
from src.graph.operations import GraphOperations
from src.graph.queries import GraphQueries
import ast


# ============================================================================
# Helper Functions
# ============================================================================

def parse_graphml_lists(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse string representations of lists from GraphML format.

    NetworkX GraphML saves lists as strings like "['item1', 'item2']".
    This function converts them back to actual Python lists.

    Args:
        data: Dictionary with potentially string-encoded lists

    Returns:
        Dictionary with parsed lists
    """
    if not data:
        return data

    parsed = data.copy()

    # Fields that might be string representations of lists
    list_fields = ['unique_capabilities', 'validation_notes']

    for field in list_fields:
        if field in parsed and isinstance(parsed[field], str):
            value = parsed[field]
            # Check if it looks like a list string
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Safely parse the string representation
                    parsed[field] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If parsing fails, wrap in a list
                    parsed[field] = [value]
            else:
                # Single value, wrap in list
                parsed[field] = [value] if value else []

    return parsed


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter()


# Global state (will be initialized by main.py)
graph_ops: Optional[GraphOperations] = None
graph_queries: Optional[GraphQueries] = None
query_agent: Optional[QueryAgent] = None


def initialize_endpoints(
    graph_operations: GraphOperations,
    query_agent_instance: QueryAgent
):
    """
    Initialize endpoint dependencies.

    Args:
        graph_operations: GraphOperations instance
        query_agent_instance: QueryAgent instance
    """
    global graph_ops, graph_queries, query_agent
    graph_ops = graph_operations
    graph_queries = GraphQueries(graph_operations)
    query_agent = query_agent_instance


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status of the API and knowledge graph.
    """
    if not graph_ops:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "graph_loaded": False,
                "total_nodes": 0,
                "total_edges": 0,
                "characters_count": 0,
                "message": "Knowledge graph not loaded"
            }
        )

    stats = graph_ops.get_graph_stats()

    return HealthResponse(
        status="healthy",
        graph_loaded=True,
        total_nodes=stats.get("total_nodes", 0),
        total_edges=stats.get("total_edges", 0),
        characters_count=stats.get("node_counts", {}).get("Character", 0),
        message="API is operational"
    )


@router.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a natural language question about Marvel characters.

    This endpoint uses the Query Agent to process natural language questions,
    retrieve relevant context from the knowledge graph, and generate
    citation-grounded responses.

    Args:
        request: Question request with the user's question

    Returns:
        Answer with metadata and confidence level

    Raises:
        503: If query agent is not available
        500: If processing fails
    """
    if not query_agent:
        raise HTTPException(
            status_code=503,
            detail="Query agent not initialized"
        )

    try:
        # Process question
        result = query_agent.query(
            question=request.question,
            verbose=request.verbose
        )

        # Build response
        response = QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            query_type=result.get("query_type"),
            characters=result.get("characters", []),
            confidence_level=result.get("confidence_level", "UNKNOWN"),
            context_retrieved=result.get("context_retrieved", False),
            error=result.get("error")
        )

        # Add raw context if requested
        if request.include_context and hasattr(query_agent, '_last_context'):
            response.raw_context = query_agent._last_context

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@router.get("/graph/{character_identifier}", response_model=CharacterGraphResponse)
async def get_character_graph(
    character_identifier: str,
    search_by: str = Query(
        default="name",
        description="Search by 'name' or 'id'",
        regex="^(name|id)$"
    )
):
    """
    Get complete knowledge graph view for a character.

    Retrieves all nodes and relationships for a specific character including:
    - Character information
    - Power origin
    - Powers and abilities
    - Significance
    - Validation results
    - Teams (if available)
    - Mutations (if available)

    Args:
        character_identifier: Character name or ID
        search_by: Whether to search by 'name' or 'id'

    Returns:
        Complete character graph data

    Raises:
        404: If character not found
        503: If graph not loaded
    """
    if not graph_queries:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not loaded"
        )

    try:
        # Find character
        if search_by == "name":
            char_node = graph_queries.find_character_by_name(character_identifier)
        else:
            char_node = graph_queries.get_character_by_id(character_identifier)

        if not char_node:
            raise HTTPException(
                status_code=404,
                detail=f"Character not found: {character_identifier}"
            )

        char_id = char_node["node_id"]

        # Get full profile
        profile = graph_queries.get_character_full_profile(char_id)

        # Parse GraphML list strings before creating Pydantic models
        if profile.get("significance"):
            profile["significance"] = parse_graphml_lists(profile["significance"])
        if profile.get("validation"):
            profile["validation"] = parse_graphml_lists(profile["validation"])

        # Build response
        response = CharacterGraphResponse(
            character=CharacterNode(**profile["character"]),
            power_origin=PowerOriginNode(**profile["power_origin"]) if profile.get("power_origin") else None,
            powers=[PowerNode(**p) for p in profile.get("powers", [])],
            significance=SignificanceNode(**profile["significance"]) if profile.get("significance") else None,
            validation=ValidationNode(**profile["validation"]) if profile.get("validation") else None,
            teams=profile.get("teams", []),
            mutations=profile.get("mutations", [])
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving character graph: {str(e)}"
        )


@router.get("/extraction-report/{character_identifier}", response_model=ExtractionReportResponse)
async def get_extraction_report(
    character_identifier: str,
    search_by: str = Query(
        default="name",
        description="Search by 'name' or 'id'",
        regex="^(name|id)$"
    ),
    include_extraction_data: bool = Query(
        default=False,
        description="Include full extraction data in response"
    )
):
    """
    Get validation metrics and extraction report for a character.

    Provides detailed quality assessment including:
    - Validation scores (confidence, completeness, similarity)
    - Quality tier classification
    - Strengths and weaknesses
    - Actionable recommendations
    - Validation flags

    Args:
        character_identifier: Character name or ID
        search_by: Whether to search by 'name' or 'id'
        include_extraction_data: Include raw extraction data

    Returns:
        Comprehensive extraction report

    Raises:
        404: If character not found or no validation data
        503: If graph not loaded
    """
    if not graph_queries:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not loaded"
        )

    try:
        # Find character
        if search_by == "name":
            char_node = graph_queries.find_character_by_name(character_identifier)
        else:
            char_node = graph_queries.get_character_by_id(character_identifier)

        if not char_node:
            raise HTTPException(
                status_code=404,
                detail=f"Character not found: {character_identifier}"
            )

        char_id = char_node["node_id"]
        char_name = char_node["name"]

        # Get validation data
        validation = graph_queries.get_extraction_validation(char_id)

        if not validation:
            raise HTTPException(
                status_code=404,
                detail=f"No validation data found for {char_name}"
            )

        # Parse GraphML list strings
        validation = parse_graphml_lists(validation)

        # Get full profile for report generation
        profile = graph_queries.get_character_full_profile(char_id)

        # Calculate overall quality score
        confidence = validation.get("confidence_score", 0.0)
        completeness = validation.get("completeness_score", 0.0)
        similarity = validation.get("semantic_similarity", 0.0)
        overall_quality = (confidence * 0.3 + completeness * 0.3 + similarity * 0.4)

        # Determine quality tier
        if overall_quality >= 0.8:
            quality_tier = "HIGH"
        elif overall_quality >= 0.6:
            quality_tier = "MEDIUM"
        else:
            quality_tier = "LOW"

        # Generate strengths
        strengths = []
        if confidence >= 0.8:
            strengths.append("High confidence extraction")
        if completeness >= 0.8:
            strengths.append("Complete power origin data")
        if similarity >= 0.7:
            strengths.append("Strong semantic grounding")

        # Generate weaknesses
        weaknesses = []
        if confidence < 0.6:
            weaknesses.append(f"Low confidence score: {confidence:.2f}")
        if completeness < 0.6:
            weaknesses.append(f"Incomplete data: {completeness:.2f}")
        if similarity < 0.7:
            weaknesses.append(f"Low semantic similarity: {similarity:.2f}")

        # Generate recommendations
        recommendations = []
        if not validation.get("is_valid", False):
            recommendations.append("Review extraction for accuracy against source text")
        if similarity < 0.7:
            recommendations.append("Consider adjusting semantic similarity threshold")
        if confidence < 0.8 and completeness >= 0.8:
            recommendations.append("Data is complete but confidence is low - review extraction logic")

        # Build response
        response = ExtractionReportResponse(
            character_id=char_id,
            character_name=char_name,
            validation_passed=validation.get("is_valid", False),
            confidence_score=confidence,
            completeness_score=completeness,
            semantic_similarity=similarity,
            consistency_score=validation.get("consistency_score"),
            overall_quality=overall_quality,
            quality_tier=quality_tier,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            validation_flags=validation.get("validation_notes", []),
            extraction_data=profile if include_extraction_data else None
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating extraction report: {str(e)}"
        )


@router.post("/validate-extraction", response_model=ValidationResponse)
async def validate_extraction(request: ValidationRequest):
    """
    Re-validate extraction for a specific character.

    Runs the validation agent to assess extraction quality including:
    - Semantic similarity check
    - Completeness validation
    - Optional multi-pass consistency check

    Args:
        request: Validation request with character identifier

    Returns:
        Fresh validation results

    Raises:
        400: If neither character_id nor character_name provided
        404: If character not found or no extraction data
        503: If graph not loaded
    """
    if not graph_queries or not graph_ops:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not loaded"
        )

    try:
        # Determine character
        if request.character_name:
            char_node = graph_queries.find_character_by_name(request.character_name)
        elif request.character_id:
            char_node = graph_queries.get_character_by_id(request.character_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either character_id or character_name must be provided"
            )

        if not char_node:
            identifier = request.character_name or request.character_id
            raise HTTPException(
                status_code=404,
                detail=f"Character not found: {identifier}"
            )

        char_id = char_node["node_id"]
        char_name = char_node["name"]

        # Get extraction data
        profile = graph_queries.get_character_full_profile(char_id)

        if not profile.get("power_origin"):
            raise HTTPException(
                status_code=404,
                detail=f"No extraction data found for {char_name}"
            )

        # Need to reconstruct Character and CharacterExtraction objects
        # This is a limitation - we'd need to store or reconstruct these
        # For now, return existing validation with a note

        # Get existing validation
        validation = graph_queries.get_extraction_validation(char_id)

        if not validation:
            raise HTTPException(
                status_code=404,
                detail=f"No validation data available for {char_name}. Run initial extraction first."
            )

        # For actual re-validation, we'd need:
        # 1. Original Character object
        # 2. CharacterExtraction object
        # 3. Then call validate_character()
        #
        # Since we only have graph data, return existing validation
        # with a note that this is cached data

        response = ValidationResponse(
            character_id=char_id,
            character_name=char_name,
            validation_result=validation,
            validation_passed=validation.get("is_valid", False),
            confidence_score=validation.get("confidence_score", 0.0),
            completeness_score=validation.get("completeness_score", 0.0),
            semantic_similarity=validation.get("semantic_similarity"),
            consistency_score=validation.get("consistency_score"),
            processing_time_seconds=0.0,
            message="Returning cached validation data. For fresh validation, re-run extraction pipeline."
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating extraction: {str(e)}"
        )


@router.get("/characters", response_model=dict)
async def list_characters(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of characters to return"),
    offset: int = Query(default=0, ge=0, description="Number of characters to skip"),
    alignment: Optional[str] = Query(default=None, description="Filter by alignment (e.g., 'Good', 'Bad')"),
    origin_type: Optional[str] = Query(default=None, description="Filter by power origin type")
):
    """
    List all characters in the knowledge graph.

    Args:
        limit: Maximum number of results
        offset: Number of results to skip
        alignment: Filter by character alignment
        origin_type: Filter by power origin type

    Returns:
        List of characters with basic information

    Raises:
        503: If graph not loaded
    """
    if not graph_queries:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not loaded"
        )

    try:
        # Get all characters
        all_characters = graph_queries.list_all_characters()

        # Apply filters
        filtered = all_characters

        if alignment:
            filtered = [c for c in filtered if c.get("alignment", "").lower() == alignment.lower()]

        if origin_type:
            # Filter by origin type (requires looking up origin for each)
            filtered_by_origin = []
            for char in filtered:
                origin = graph_queries.get_character_power_origin(char["node_id"])
                if origin and origin.get("origin_type", "").lower() == origin_type.lower():
                    filtered_by_origin.append(char)
            filtered = filtered_by_origin

        # Apply pagination
        total = len(filtered)
        paginated = filtered[offset:offset + limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(paginated),
            "characters": paginated
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing characters: {str(e)}"
        )


@router.get("/stats", response_model=dict)
async def get_graph_stats():
    """
    Get knowledge graph statistics.

    Returns:
        Comprehensive statistics about the knowledge graph including
        node counts, edge counts, quality metrics, etc.

    Raises:
        503: If graph not loaded
    """
    if not graph_queries:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not loaded"
        )

    try:
        # Get graph summary (includes quality metrics)
        summary = graph_queries.get_graph_summary()

        return summary

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph statistics: {str(e)}"
        )
