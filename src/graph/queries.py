"""
Graph querying functions for the Marvel Knowledge Graph.

Provides high-level query functions for retrieving information from the graph.
"""

from typing import Optional, List, Dict, Any
import networkx as nx

from src.graph.schema import NodeType, RelationType
from src.graph.operations import GraphOperations


class GraphQueries:
    """High-level query interface for the Marvel knowledge graph."""

    def __init__(self, graph_ops: GraphOperations):
        """
        Initialize graph queries.

        Args:
            graph_ops: GraphOperations instance
        """
        self.ops = graph_ops
        self.graph = graph_ops.graph

    # ========================================================================
    # Character Queries
    # ========================================================================

    def find_character_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a character node by name (case-insensitive, partial match).

        Args:
            name: Character name to search for

        Returns:
            Character node data or None
        """
        characters = self.ops.get_nodes_by_type(NodeType.CHARACTER)

        # First try exact match
        for char in characters:
            if char.get("name", "").lower() == name.lower():
                return char

        # Then try partial match (e.g., "Spider-Man" matches "Spider-Man (Peter Parker)")
        for char in characters:
            char_name = char.get("name", "").lower()
            search_name = name.lower()
            if search_name in char_name or char_name.startswith(search_name):
                return char

        return None

    def get_character_by_id(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get character node by ID.

        Args:
            character_id: Character node ID

        Returns:
            Character node data or None
        """
        return self.ops.get_node(character_id)

    def list_all_characters(self) -> List[Dict[str, Any]]:
        """
        List all character nodes in the graph.

        Returns:
            List of character node data
        """
        return self.ops.get_nodes_by_type(NodeType.CHARACTER)

    # ========================================================================
    # Power Origin Queries
    # ========================================================================

    def get_character_power_origin(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the power origin for a character.

        Args:
            character_id: Character node ID

        Returns:
            Power origin node data or None
        """
        relationships = self.ops.get_relationships(
            character_id,
            RelationType.HAS_ORIGIN
        )

        if not relationships:
            return None

        # Get the first (should be only one) power origin
        _, origin_id, _ = relationships[0]
        return self.ops.get_node(origin_id)

    def get_characters_by_origin_type(self, origin_type: str) -> List[Dict[str, Any]]:
        """
        Find all characters with a specific origin type.

        Args:
            origin_type: Origin type (mutation, accident, technology, etc.)

        Returns:
            List of character node data
        """
        characters = []
        origin_nodes = self.ops.get_nodes_by_type(NodeType.POWER_ORIGIN)

        for origin in origin_nodes:
            if origin.get("origin_type", "").lower() == origin_type.lower():
                # Find characters with this origin
                incoming = self.ops.get_incoming_relationships(
                    origin["node_id"],
                    RelationType.HAS_ORIGIN
                )

                for char_id, _, _ in incoming:
                    char_data = self.ops.get_node(char_id)
                    if char_data:
                        characters.append(char_data)

        return characters

    # ========================================================================
    # Power Queries
    # ========================================================================

    def get_character_powers(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Get all powers possessed by a character.

        Args:
            character_id: Character node ID

        Returns:
            List of power node data
        """
        relationships = self.ops.get_relationships(
            character_id,
            RelationType.POSSESSES_POWER
        )

        powers = []
        for _, power_id, _ in relationships:
            power_data = self.ops.get_node(power_id)
            if power_data:
                powers.append(power_data)

        return powers

    def get_powers_from_origin(self, origin_id: str) -> List[Dict[str, Any]]:
        """
        Get all powers conferred by a specific origin.

        Args:
            origin_id: Power origin node ID

        Returns:
            List of power node data
        """
        relationships = self.ops.get_relationships(
            origin_id,
            RelationType.CONFERS
        )

        powers = []
        for _, power_id, _ in relationships:
            power_data = self.ops.get_node(power_id)
            if power_data:
                powers.append(power_data)

        return powers

    # ========================================================================
    # Significance Queries
    # ========================================================================

    def get_character_significance(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the significance/impact data for a character.

        Args:
            character_id: Character node ID

        Returns:
            Significance node data or None
        """
        relationships = self.ops.get_relationships(
            character_id,
            RelationType.HAS_SIGNIFICANCE
        )

        if not relationships:
            return None

        _, significance_id, _ = relationships[0]
        return self.ops.get_node(significance_id)

    def get_characters_by_impact_level(self, impact_level: str) -> List[Dict[str, Any]]:
        """
        Find all characters with a specific impact level.

        Args:
            impact_level: Impact level (COSMIC, GLOBAL, REGIONAL, LOCAL)

        Returns:
            List of character node data
        """
        characters = []
        significance_nodes = self.ops.get_nodes_by_type(NodeType.SIGNIFICANCE)

        for sig in significance_nodes:
            if sig.get("impact_level", "").upper() == impact_level.upper():
                # Extract character_id from significance node_id
                character_id = sig["node_id"].replace("significance_", "")
                char_data = self.ops.get_node(character_id)
                if char_data:
                    characters.append(char_data)

        return characters

    # ========================================================================
    # Gene/Mutation Queries
    # ========================================================================

    def get_character_mutations(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Get all genetic mutations for a character.

        Args:
            character_id: Character node ID

        Returns:
            List of gene node data
        """
        relationships = self.ops.get_relationships(
            character_id,
            RelationType.HAS_MUTATION
        )

        genes = []
        for _, gene_id, _ in relationships:
            gene_data = self.ops.get_node(gene_id)
            if gene_data:
                genes.append(gene_data)

        return genes

    def get_powers_enabled_by_gene(self, gene_id: str) -> List[Dict[str, Any]]:
        """
        Get all powers enabled by a specific gene.

        Args:
            gene_id: Gene node ID

        Returns:
            List of power node data
        """
        relationships = self.ops.get_relationships(
            gene_id,
            RelationType.ENABLES
        )

        powers = []
        for _, power_id, _ in relationships:
            power_data = self.ops.get_node(power_id)
            if power_data:
                powers.append(power_data)

        return powers

    # ========================================================================
    # Team Queries
    # ========================================================================

    def get_character_teams(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Get all teams a character belongs to.

        Args:
            character_id: Character node ID

        Returns:
            List of team node data
        """
        relationships = self.ops.get_relationships(
            character_id,
            RelationType.MEMBER_OF
        )

        teams = []
        for _, team_id, _ in relationships:
            team_data = self.ops.get_node(team_id)
            if team_data:
                teams.append(team_data)

        return teams

    def get_team_members(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get all members of a team.

        Args:
            team_id: Team node ID

        Returns:
            List of character node data
        """
        relationships = self.ops.get_incoming_relationships(
            team_id,
            RelationType.MEMBER_OF
        )

        characters = []
        for char_id, _, _ in relationships:
            char_data = self.ops.get_node(char_id)
            if char_data:
                characters.append(char_data)

        return characters

    # ========================================================================
    # Validation Queries
    # ========================================================================

    def get_extraction_validation(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get validation results for a character's extraction.

        Args:
            character_id: Character node ID

        Returns:
            Validation node data or None
        """
        # Get the character's power origin
        origin = self.get_character_power_origin(character_id)
        if not origin:
            return None

        # Get validation for the origin
        relationships = self.ops.get_relationships(
            origin["node_id"],
            RelationType.EXTRACTION_VALIDATED
        )

        if not relationships:
            return None

        _, validation_id, _ = relationships[0]
        return self.ops.get_node(validation_id)

    def get_high_confidence_extractions(self) -> List[Dict[str, Any]]:
        """
        Get all power origins with high confidence.

        Returns:
            List of power origin node data
        """
        origins = self.ops.get_nodes_by_type(NodeType.POWER_ORIGIN)
        return [o for o in origins if o.get("confidence", "").upper() == "HIGH"]

    def get_low_confidence_extractions(self) -> List[Dict[str, Any]]:
        """
        Get all power origins with low confidence.

        Returns:
            List of power origin node data
        """
        origins = self.ops.get_nodes_by_type(NodeType.POWER_ORIGIN)
        return [o for o in origins if o.get("confidence", "").upper() == "LOW"]

    # ========================================================================
    # Complex/Traversal Queries
    # ========================================================================

    def get_character_full_profile(self, character_id: str) -> Dict[str, Any]:
        """
        Get complete profile for a character including all relationships.

        Args:
            character_id: Character node ID

        Returns:
            Dictionary with character, origin, powers, significance, etc.
        """
        profile = {
            "character": self.get_character_by_id(character_id),
            "power_origin": self.get_character_power_origin(character_id),
            "powers": self.get_character_powers(character_id),
            "significance": self.get_character_significance(character_id),
            "mutations": self.get_character_mutations(character_id),
            "teams": self.get_character_teams(character_id),
            "validation": self.get_extraction_validation(character_id)
        }

        return profile

    def get_origin_to_powers_chain(self, character_id: str) -> Dict[str, Any]:
        """
        Get the full chain: Character -> Origin -> Powers

        Args:
            character_id: Character node ID

        Returns:
            Dictionary with character, origin, and all powers
        """
        origin = self.get_character_power_origin(character_id)

        result = {
            "character_id": character_id,
            "character": self.get_character_by_id(character_id),
            "origin": origin,
            "powers_from_origin": []
        }

        if origin:
            result["powers_from_origin"] = self.get_powers_from_origin(origin["node_id"])

        return result

    def find_characters_with_similar_origins(
        self,
        character_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find characters with similar power origins.

        Args:
            character_id: Character node ID
            limit: Maximum number of similar characters to return

        Returns:
            List of character node data
        """
        # Get the origin type of the target character
        origin = self.get_character_power_origin(character_id)
        if not origin:
            return []

        origin_type = origin.get("origin_type")

        # Find other characters with the same origin type
        similar = self.get_characters_by_origin_type(origin_type)

        # Filter out the query character
        similar = [c for c in similar if c["node_id"] != character_id]

        return similar[:limit]

    # ========================================================================
    # Search Functions
    # ========================================================================

    def search_characters(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for characters by name (partial match, case-insensitive).

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of character node data
        """
        characters = self.ops.get_nodes_by_type(NodeType.CHARACTER)
        query_lower = query.lower()

        matches = [
            c for c in characters
            if query_lower in c.get("name", "").lower()
        ]

        return matches[:limit]

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the graph.

        Returns:
            Dictionary with graph statistics and summaries
        """
        stats = self.ops.get_graph_stats()

        summary = {
            **stats,
            "high_confidence_origins": len(self.get_high_confidence_extractions()),
            "low_confidence_origins": len(self.get_low_confidence_extractions()),
            "characters_with_origins": len([
                c for c in self.list_all_characters()
                if self.get_character_power_origin(c["node_id"]) is not None
            ])
        }

        return summary
