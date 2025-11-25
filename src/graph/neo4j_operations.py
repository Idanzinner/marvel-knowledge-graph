"""
Neo4j operations for the Marvel Knowledge Graph.

Provides functions for creating, updating, and querying nodes and relationships
in a Neo4j graph database.
"""

import os
from typing import Optional, List, Dict, Any
from neo4j import GraphDatabase, Driver
from pydantic import BaseModel

from src.graph.schema import (
    NodeType, RelationType,
    CharacterNode, PowerOriginNode, PowerNode, GeneNode,
    TeamNode, SignificanceNode, ValidationNode, Relationship
)


class Neo4jOperations:
    """Handles all operations on the Marvel knowledge graph in Neo4j."""

    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (defaults to env var CONNECTION_URI)
            username: Neo4j username (defaults to env var NEO4J_USERNAME)
            password: Neo4j password (defaults to env var NEO4J_PASSWORD)
        """
        self.uri = uri or os.getenv("CONNECTION_URI", "neo4j://127.0.0.1:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "Polazin2!")

        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Verify connectivity
            self.driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            print(f"  URI: {self.uri}")
            print(f"  Username: {self.username}")
            raise

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")

    # ========================================================================
    # Database Management
    # ========================================================================

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")

    def create_constraints(self):
        """Create uniqueness constraints and indexes for better performance."""
        constraints = [
            "CREATE CONSTRAINT character_id IF NOT EXISTS FOR (c:Character) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT origin_id IF NOT EXISTS FOR (o:PowerOrigin) REQUIRE o.node_id IS UNIQUE",
            "CREATE CONSTRAINT power_id IF NOT EXISTS FOR (p:Power) REQUIRE p.node_id IS UNIQUE",
            "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.node_id IS UNIQUE",
            "CREATE CONSTRAINT team_id IF NOT EXISTS FOR (t:Team) REQUIRE t.node_id IS UNIQUE",
            "CREATE CONSTRAINT significance_id IF NOT EXISTS FOR (s:Significance) REQUIRE s.node_id IS UNIQUE",
            "CREATE CONSTRAINT validation_id IF NOT EXISTS FOR (v:Validation) REQUIRE v.node_id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")

        print("✓ Constraints created")

    # ========================================================================
    # Node Creation
    # ========================================================================

    def add_node(self, node: BaseModel) -> str:
        """
        Add a node to the graph.

        Args:
            node: Node data model (CharacterNode, PowerOriginNode, etc.)

        Returns:
            node_id: The ID of the created/updated node
        """
        node_dict = node.to_dict()
        node_id = node_dict["node_id"]
        node_type = node_dict.pop("node_type")

        # Build Cypher query
        properties_str = ", ".join([f"{k}: ${k}" for k in node_dict.keys()])
        query = f"""
        MERGE (n:{node_type} {{node_id: $node_id}})
        SET n += {{{properties_str}}}
        RETURN n.node_id as node_id
        """

        with self.driver.session() as session:
            result = session.run(query, **node_dict)
            return result.single()["node_id"]

    def add_character_node(
        self,
        name: str,
        page_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Add a character node to the graph.

        Args:
            name: Character name
            page_id: Character page ID
            **kwargs: Additional character attributes

        Returns:
            node_id: The character node ID
        """
        import hashlib
        node_id = f"character_{page_id}" if page_id else f"character_{hashlib.md5(name.encode()).hexdigest()[:12]}"

        node = CharacterNode(
            node_id=node_id,
            name=name,
            page_id=page_id,
            **kwargs
        )

        return self.add_node(node)

    def add_power_origin_node(
        self,
        character_id: str,
        origin_type: str,
        description: str,
        confidence: str,
        evidence: Optional[str] = None
    ) -> str:
        """
        Add a power origin node to the graph.

        Args:
            character_id: ID of the character this origin belongs to
            origin_type: Type of origin (mutation, accident, etc.)
            description: Description of how powers were obtained
            confidence: Confidence level (HIGH, MEDIUM, LOW)
            evidence: Evidence quote from source text

        Returns:
            node_id: The power origin node ID
        """
        node_id = f"origin_{character_id}_{origin_type.lower()}"

        node = PowerOriginNode(
            node_id=node_id,
            origin_type=origin_type,
            description=description,
            confidence=confidence,
            evidence=evidence
        )

        return self.add_node(node)

    def add_power_node(self, power_name: str, description: Optional[str] = None) -> str:
        """
        Add a power node to the graph.

        Args:
            power_name: Name of the power
            description: Optional description of the power

        Returns:
            node_id: The power node ID
        """
        import hashlib
        node_id = f"power_{hashlib.md5(power_name.encode()).hexdigest()[:12]}"

        node = PowerNode(
            node_id=node_id,
            name=power_name,
            description=description
        )

        return self.add_node(node)

    def add_significance_node(
        self,
        character_id: str,
        why_matters: str,
        impact_level: str,
        strategic_value: Optional[str] = None
    ) -> str:
        """
        Add a significance node to the graph.

        Args:
            character_id: ID of the character this significance belongs to
            why_matters: Explanation of why the powers matter
            impact_level: Impact level (COSMIC, GLOBAL, LOCAL)
            strategic_value: Optional strategic value description

        Returns:
            node_id: The significance node ID
        """
        node_id = f"significance_{character_id}"

        node = SignificanceNode(
            node_id=node_id,
            why_matters=why_matters,
            impact_level=impact_level,
            strategic_value=strategic_value
        )

        return self.add_node(node)

    def add_validation_node(
        self,
        character_id: str,
        is_valid: bool,
        confidence_score: float,
        completeness_score: float,
        validation_notes: Optional[str] = None
    ) -> str:
        """
        Add a validation node to the graph.

        Args:
            character_id: ID of the character being validated
            is_valid: Whether validation passed
            confidence_score: Confidence score (0-1)
            completeness_score: Completeness score (0-1)
            validation_notes: Optional validation notes

        Returns:
            node_id: The validation node ID
        """
        node_id = f"validation_{character_id}"

        node = ValidationNode(
            node_id=node_id,
            is_valid=is_valid,
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            validation_notes=validation_notes
        )

        return self.add_node(node)

    # ========================================================================
    # Relationship Creation
    # ========================================================================

    def add_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: RelationType,
        **properties
    ) -> bool:
        """
        Add a relationship between two nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            **properties: Additional relationship properties

        Returns:
            bool: Success status
        """
        rel_type = relationship_type.value if isinstance(relationship_type, RelationType) else relationship_type

        # Build properties string
        props_str = ""
        if properties:
            props_str = "{" + ", ".join([f"{k}: ${k}" for k in properties.keys()]) + "}"

        query = f"""
        MATCH (a {{node_id: $from_id}})
        MATCH (b {{node_id: $to_id}})
        MERGE (a)-[r:{rel_type} {props_str}]->(b)
        RETURN r
        """

        params = {"from_id": from_node_id, "to_id": to_node_id, **properties}

        with self.driver.session() as session:
            result = session.run(query, **params)
            return result.single() is not None

    # ========================================================================
    # Query Operations
    # ========================================================================

    def get_character_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get character node by name."""
        query = """
        MATCH (c:Character {name: $name})
        RETURN c
        """

        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return dict(record["c"])
        return None

    def get_character_profile(self, character_identifier: str, search_by: str = "name") -> Optional[Dict[str, Any]]:
        """
        Get full character profile with all related nodes.

        Args:
            character_identifier: Character name or ID
            search_by: 'name' or 'node_id'

        Returns:
            Dictionary with character data and relationships
        """
        if search_by == "name":
            match_clause = "MATCH (c:Character {name: $identifier})"
        else:
            match_clause = "MATCH (c:Character {node_id: $identifier})"

        query = f"""
        {match_clause}
        OPTIONAL MATCH (c)-[:HAS_ORIGIN]->(o:PowerOrigin)
        OPTIONAL MATCH (o)-[:CONFERS]->(p:Power)
        OPTIONAL MATCH (c)-[:HAS_SIGNIFICANCE]->(s:Significance)
        OPTIONAL MATCH (c)-[:EXTRACTION_VALIDATED]->(v:Validation)
        RETURN c, o, collect(DISTINCT p) as powers, s, v
        """

        with self.driver.session() as session:
            result = session.run(query, identifier=character_identifier)
            record = result.single()

            if not record:
                return None

            profile = {
                "character": dict(record["c"]) if record["c"] else None,
                "origin": dict(record["o"]) if record["o"] else None,
                "powers": [dict(p) for p in record["powers"] if p],
                "significance": dict(record["s"]) if record["s"] else None,
                "validation": dict(record["v"]) if record["v"] else None
            }

            return profile

    def list_all_characters(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all characters in the database."""
        query = """
        MATCH (c:Character)
        RETURN c
        ORDER BY c.name
        SKIP $offset
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, limit=limit, offset=offset)
            return [dict(record["c"]) for record in result]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        """

        query_rels = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        """

        with self.driver.session() as session:
            # Node counts
            node_result = session.run(query)
            nodes_by_type = {record["node_type"]: record["count"] for record in node_result}

            # Relationship counts
            rel_result = session.run(query_rels)
            edges_by_type = {record["rel_type"]: record["count"] for record in rel_result}

            total_nodes = sum(nodes_by_type.values())
            total_edges = sum(edges_by_type.values())

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes_by_type": nodes_by_type,
                "edges_by_type": edges_by_type
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
