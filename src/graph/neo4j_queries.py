"""
Neo4j querying functions for the Marvel Knowledge Graph.

Provides high-level query functions for retrieving information from Neo4j database.
This mirrors the GraphQueries interface but uses Cypher queries instead of NetworkX.
"""

from typing import Optional, List, Dict, Any
from src.graph.neo4j_operations import Neo4jOperations


class Neo4jQueries:
    """High-level query interface for the Marvel knowledge graph in Neo4j."""

    def __init__(self, neo4j_ops: Neo4jOperations):
        """
        Initialize Neo4j queries.

        Args:
            neo4j_ops: Neo4jOperations instance
        """
        self.ops = neo4j_ops
        self.driver = neo4j_ops.driver

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
        query = """
        MATCH (c:Character)
        WHERE toLower(c.name) = toLower($name)
           OR toLower(c.name) CONTAINS toLower($name)
        RETURN c
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()

            if record:
                return dict(record["c"])
            return None

    def get_character_by_id(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get character node by ID.

        Args:
            character_id: Character node ID

        Returns:
            Character node data or None
        """
        query = """
        MATCH (c:Character {node_id: $character_id})
        RETURN c
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id)
            record = result.single()

            if record:
                return dict(record["c"])
            return None

    def list_all_characters(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all character nodes in the graph.

        Args:
            limit: Maximum number of characters to return

        Returns:
            List of character node data
        """
        query = """
        MATCH (c:Character)
        RETURN c
        ORDER BY c.name
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record["c"]) for record in result]

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
        query = """
        MATCH (c:Character {node_id: $character_id})-[:HAS_ORIGIN]->(o:PowerOrigin)
        RETURN o
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id)
            record = result.single()

            if record:
                return dict(record["o"])
            return None

    def get_characters_by_origin_type(self, origin_type: str) -> List[Dict[str, Any]]:
        """
        Find all characters with a specific origin type.

        Args:
            origin_type: Origin type (mutation, accident, technology, etc.)

        Returns:
            List of character node data
        """
        query = """
        MATCH (c:Character)-[:HAS_ORIGIN]->(o:PowerOrigin)
        WHERE toLower(o.origin_type) = toLower($origin_type)
        RETURN c
        """

        with self.driver.session() as session:
            result = session.run(query, origin_type=origin_type)
            return [dict(record["c"]) for record in result]

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
        query = """
        MATCH (c:Character {node_id: $character_id})-[:POSSESSES_POWER]->(p:Power)
        RETURN p
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id)
            return [dict(record["p"]) for record in result]

    def get_powers_from_origin(self, origin_id: str) -> List[Dict[str, Any]]:
        """
        Get all powers conferred by a specific origin.

        Args:
            origin_id: Power origin node ID

        Returns:
            List of power node data
        """
        query = """
        MATCH (o:PowerOrigin {node_id: $origin_id})-[:CONFERS]->(p:Power)
        RETURN p
        """

        with self.driver.session() as session:
            result = session.run(query, origin_id=origin_id)
            return [dict(record["p"]) for record in result]

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
        query = """
        MATCH (c:Character {node_id: $character_id})-[:HAS_SIGNIFICANCE]->(s:Significance)
        RETURN s
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id)
            record = result.single()

            if record:
                return dict(record["s"])
            return None

    def get_characters_by_impact_level(self, impact_level: str) -> List[Dict[str, Any]]:
        """
        Find all characters with a specific impact level.

        Args:
            impact_level: Impact level (COSMIC, GLOBAL, REGIONAL, LOCAL)

        Returns:
            List of character node data
        """
        query = """
        MATCH (c:Character)-[:HAS_SIGNIFICANCE]->(s:Significance)
        WHERE toUpper(s.impact_level) = toUpper($impact_level)
        RETURN c
        """

        with self.driver.session() as session:
            result = session.run(query, impact_level=impact_level)
            return [dict(record["c"]) for record in result]

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
        query = """
        MATCH (c:Character {node_id: $character_id})-[:EXTRACTION_VALIDATED]->(v:Validation)
        RETURN v
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id)
            record = result.single()

            if record:
                return dict(record["v"])
            return None

    def get_high_confidence_extractions(self) -> List[Dict[str, Any]]:
        """
        Get all power origins with high confidence.

        Returns:
            List of power origin node data
        """
        query = """
        MATCH (o:PowerOrigin)
        WHERE toUpper(o.confidence) = 'HIGH'
        RETURN o
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record["o"]) for record in result]

    def get_low_confidence_extractions(self) -> List[Dict[str, Any]]:
        """
        Get all power origins with low confidence.

        Returns:
            List of power origin node data
        """
        query = """
        MATCH (o:PowerOrigin)
        WHERE toUpper(o.confidence) = 'LOW'
        RETURN o
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record["o"]) for record in result]

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
        query = """
        MATCH (c1:Character {node_id: $character_id})-[:HAS_ORIGIN]->(o1:PowerOrigin)
        MATCH (c2:Character)-[:HAS_ORIGIN]->(o2:PowerOrigin)
        WHERE o1.origin_type = o2.origin_type AND c1.node_id <> c2.node_id
        RETURN c2
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id, limit=limit)
            return [dict(record["c2"]) for record in result]

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
        cypher_query = """
        MATCH (c:Character)
        WHERE toLower(c.name) CONTAINS toLower($query)
        RETURN c
        ORDER BY c.name
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, query=query, limit=limit)
            return [dict(record["c"]) for record in result]

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the graph.

        Returns:
            Dictionary with graph statistics and summaries
        """
        stats = self.ops.get_statistics()

        summary = {
            **stats,
            "high_confidence_origins": len(self.get_high_confidence_extractions()),
            "low_confidence_origins": len(self.get_low_confidence_extractions()),
        }

        return summary
