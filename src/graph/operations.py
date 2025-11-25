"""
Graph operations for the Marvel Knowledge Graph.

Provides functions for creating, updating, and managing nodes and relationships
in a NetworkX graph.
"""

import hashlib
from typing import Optional, List, Dict, Any
import networkx as nx
from pydantic import BaseModel

from src.graph.schema import (
    NodeType, RelationType, GraphSchema,
    CharacterNode, PowerOriginNode, PowerNode, GeneNode,
    TeamNode, SignificanceNode, ValidationNode, Relationship
)


class GraphOperations:
    """Handles all operations on the Marvel knowledge graph."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize graph operations.

        Args:
            graph: Existing NetworkX directed graph, or creates new one
        """
        self.graph = graph if graph is not None else nx.DiGraph()

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

        # Add or update node
        self.graph.add_node(node_id, **node_dict)

        return node_id

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
        node_id = f"character_{page_id}" if page_id else f"character_{self._hash_string(name)}"

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
        node_id = f"origin_{character_id}_{origin_type}"

        node = PowerOriginNode(
            node_id=node_id,
            origin_type=origin_type,
            description=description,
            confidence=confidence,
            evidence=evidence
        )

        return self.add_node(node)

    def add_power_node(
        self,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Add a power/ability node to the graph.

        Args:
            name: Power name
            description: Power description

        Returns:
            node_id: The power node ID
        """
        node_id = f"power_{self._hash_string(name)}"

        node = PowerNode(
            node_id=node_id,
            name=name,
            description=description
        )

        return self.add_node(node)

    def add_gene_node(
        self,
        name: str,
        description: Optional[str] = None,
        source: Optional[str] = None
    ) -> str:
        """
        Add a gene/mutation node to the graph.

        Args:
            name: Gene name
            description: Gene description
            source: Source of mutation

        Returns:
            node_id: The gene node ID
        """
        node_id = f"gene_{self._hash_string(name)}"

        node = GeneNode(
            node_id=node_id,
            name=name,
            description=description,
            source=source
        )

        return self.add_node(node)

    def add_team_node(
        self,
        name: str,
        affiliation_type: Optional[str] = None
    ) -> str:
        """
        Add a team/affiliation node to the graph.

        Args:
            name: Team name
            affiliation_type: Type of affiliation (hero, villain, neutral)

        Returns:
            node_id: The team node ID
        """
        node_id = f"team_{self._hash_string(name)}"

        node = TeamNode(
            node_id=node_id,
            name=name,
            affiliation_type=affiliation_type
        )

        return self.add_node(node)

    def add_significance_node(
        self,
        character_id: str,
        why_matters: str,
        impact_level: str,
        unique_capabilities: List[str],
        strategic_value: Optional[str] = None
    ) -> str:
        """
        Add a significance node to the graph.

        Args:
            character_id: ID of the character
            why_matters: Why the character's powers matter
            impact_level: Impact level (COSMIC, GLOBAL, REGIONAL, LOCAL)
            unique_capabilities: List of unique capabilities
            strategic_value: Strategic value description

        Returns:
            node_id: The significance node ID
        """
        node_id = f"significance_{character_id}"

        node = SignificanceNode(
            node_id=node_id,
            why_matters=why_matters,
            impact_level=impact_level,
            unique_capabilities=unique_capabilities,
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
        Add a validation result node to the graph.

        Args:
            character_id: ID of the character
            is_valid: Whether extraction is valid
            confidence_score: Confidence score (0-1)
            completeness_score: Completeness score (0-1)
            validation_notes: Additional validation notes

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
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        **properties
    ) -> bool:
        """
        Add a relationship between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            **properties: Additional relationship properties

        Returns:
            bool: True if relationship was added successfully
        """
        # Verify both nodes exist
        if source_id not in self.graph:
            print(f"Warning: Source node {source_id} not found in graph")
            return False

        if target_id not in self.graph:
            print(f"Warning: Target node {target_id} not found in graph")
            return False

        # Add edge with relationship data
        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type.value,
            **properties
        )

        return True

    # ========================================================================
    # Node Retrieval
    # ========================================================================

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node data dictionary or None if not found
        """
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: Type of nodes to retrieve

        Returns:
            List of node data dictionaries
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == node_type.value:
                nodes.append({"node_id": node_id, **data})

        return nodes

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.graph

    # ========================================================================
    # Relationship Retrieval
    # ========================================================================

    def get_relationships(
        self,
        source_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[tuple]:
        """
        Get all relationships from a source node.

        Args:
            source_id: Source node ID
            relation_type: Optional filter by relationship type

        Returns:
            List of (source_id, target_id, edge_data) tuples
        """
        if source_id not in self.graph:
            return []

        relationships = []
        for target_id in self.graph.successors(source_id):
            edge_data = self.graph[source_id][target_id]

            if relation_type is None or edge_data.get("relation_type") == relation_type.value:
                relationships.append((source_id, target_id, edge_data))

        return relationships

    def get_incoming_relationships(
        self,
        target_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[tuple]:
        """
        Get all relationships to a target node.

        Args:
            target_id: Target node ID
            relation_type: Optional filter by relationship type

        Returns:
            List of (source_id, target_id, edge_data) tuples
        """
        if target_id not in self.graph:
            return []

        relationships = []
        for source_id in self.graph.predecessors(target_id):
            edge_data = self.graph[source_id][target_id]

            if relation_type is None or edge_data.get("relation_type") == relation_type.value:
                relationships.append((source_id, target_id, edge_data))

        return relationships

    # ========================================================================
    # Graph Statistics
    # ========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {},
            "edges_by_type": {}
        }

        # Count nodes by type
        for node_type in NodeType:
            count = len(self.get_nodes_by_type(node_type))
            stats["nodes_by_type"][node_type.value] = count

        # Count edges by type
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("relation_type", "unknown")
            stats["edges_by_type"][rel_type] = stats["edges_by_type"].get(rel_type, 0) + 1

        return stats

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _hash_string(text: str) -> str:
        """Create a hash ID from a string."""
        return hashlib.md5(text.lower().encode()).hexdigest()[:12]

    def save_graph(self, filepath: str):
        """
        Save the graph to a file.

        Args:
            filepath: Path to save the graph (supports .graphml, .gexf, .gml)
        """
        # Create a copy with serializable attributes
        export_graph = self.graph.copy()

        # Convert enum values to strings for all nodes
        for node_id in export_graph.nodes():
            node_data = export_graph.nodes[node_id]
            for key, value in list(node_data.items()):
                if hasattr(value, 'value'):  # Check if it's an enum
                    node_data[key] = value.value
                elif isinstance(value, list):  # Handle lists
                    node_data[key] = str(value)

        if filepath.endswith('.graphml'):
            nx.write_graphml(export_graph, filepath)
        elif filepath.endswith('.gexf'):
            nx.write_gexf(export_graph, filepath)
        elif filepath.endswith('.gml'):
            nx.write_gml(export_graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath}")

    @classmethod
    def load_graph(cls, filepath: str) -> 'GraphOperations':
        """
        Load a graph from a file.

        Args:
            filepath: Path to the graph file

        Returns:
            GraphOperations instance with loaded graph
        """
        if filepath.endswith('.graphml'):
            graph = nx.read_graphml(filepath)
        elif filepath.endswith('.gexf'):
            graph = nx.read_gexf(filepath)
        elif filepath.endswith('.gml'):
            graph = nx.read_gml(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath}")

        return cls(graph=graph)
