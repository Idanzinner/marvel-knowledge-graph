"""
Test script for Phase 2: Knowledge Graph Construction

Tests:
1. Loading extraction results from Phase 1
2. Building knowledge graph with LangGraph
3. Querying the graph
4. Validating graph structure
"""

import asyncio
import json
from pathlib import Path

from src.agents.graph_builder_agent import build_graph_from_extractions_sync
from src.graph.operations import GraphOperations
from src.graph.queries import GraphQueries
from src.models.power_origin import CharacterExtraction
from src.utils.data_loader import get_sample_characters


def load_extractions_from_json(filepath: str) -> list[CharacterExtraction]:
    """Load extraction results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    extractions = []
    for item in data:
        extraction = CharacterExtraction(**item)
        extractions.append(extraction)

    return extractions


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)


def test_graph_building():
    """Test building the knowledge graph from extractions."""
    print_section("PHASE 2: Knowledge Graph Construction Test")

    # Step 1: Load Phase 1 extraction results
    print("\n📂 Loading extraction results from Phase 1...")
    extraction_file = "data/processed/sample_extractions.json"

    if not Path(extraction_file).exists():
        print(f"❌ Extraction file not found: {extraction_file}")
        print("Please run test_extraction.py first to generate extractions.")
        return

    extractions = load_extractions_from_json(extraction_file)
    print(f"✅ Loaded {len(extractions)} character extractions")

    for extraction in extractions:
        print(f"  - {extraction.character_name} ({extraction.power_origin.type})")

    # Step 2: Load original character data
    print("\n📂 Loading original character data...")
    character_names = [e.character_name for e in extractions]
    data_file = "data/marvel-wikia-data-with-descriptions.pkl"
    characters = get_sample_characters(data_file, character_names)
    print(f"✅ Loaded {len(characters)} character records")

    # Step 3: Build knowledge graph
    print_section("Building Knowledge Graph with LangGraph")
    graph_ops = build_graph_from_extractions_sync(
        extractions=extractions,
        characters=characters,
        verbose=True
    )

    # Step 4: Get graph statistics
    print_section("Graph Statistics")
    stats = graph_ops.get_graph_stats()

    print(f"\n📊 Graph Overview:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")

    print(f"\n📦 Nodes by Type:")
    for node_type, count in stats['nodes_by_type'].items():
        if count > 0:
            print(f"  {node_type}: {count}")

    print(f"\n🔗 Relationships by Type:")
    for edge_type, count in stats['edges_by_type'].items():
        print(f"  {edge_type}: {count}")

    # Step 5: Test graph queries
    print_section("Testing Graph Queries")
    queries = GraphQueries(graph_ops)

    # Test 1: List all characters
    print("\n🔍 Query 1: List all characters")
    all_characters = queries.list_all_characters()
    for char in all_characters:
        print(f"  - {char['name']} (ID: {char['node_id']})")

    # Test 2: Get character power origins
    print("\n🔍 Query 2: Power origins for each character")
    for char in all_characters:
        origin = queries.get_character_power_origin(char['node_id'])
        if origin:
            print(f"\n  {char['name']}:")
            print(f"    Type: {origin['origin_type']}")
            print(f"    Confidence: {origin['confidence']}")
            print(f"    Description: {origin['description'][:100]}...")

    # Test 3: Get character full profiles
    print("\n🔍 Query 3: Full character profiles")
    for char in all_characters:
        print(f"\n  {char['name']} Profile:")
        profile = queries.get_character_full_profile(char['node_id'])

        if profile['power_origin']:
            print(f"    Origin: {profile['power_origin']['origin_type']}")

        if profile['powers']:
            print(f"    Powers ({len(profile['powers'])}):")
            for power in profile['powers'][:5]:  # Show first 5
                print(f"      - {power['name']}")

        if profile['significance']:
            sig = profile['significance']
            print(f"    Impact Level: {sig['impact_level']}")
            print(f"    Why Matters: {sig['why_matters'][:100]}...")

        if profile['validation']:
            val = profile['validation']
            print(f"    Validation: Valid={val['is_valid']}, "
                  f"Confidence={val['confidence_score']:.2f}, "
                  f"Completeness={val['completeness_score']:.2f}")

    # Test 4: Characters by origin type
    print("\n🔍 Query 4: Characters grouped by origin type")
    origin_types = set()
    for char in all_characters:
        origin = queries.get_character_power_origin(char['node_id'])
        if origin:
            origin_types.add(origin['origin_type'])

    for origin_type in sorted(origin_types):
        chars = queries.get_characters_by_origin_type(origin_type)
        print(f"\n  {origin_type.upper()} ({len(chars)} characters):")
        for char in chars:
            print(f"    - {char['name']}")

    # Test 5: Similar characters
    print("\n🔍 Query 5: Find characters with similar origins")
    if all_characters:
        test_char = all_characters[0]
        similar = queries.find_characters_with_similar_origins(test_char['node_id'], limit=5)
        print(f"\n  Characters similar to {test_char['name']}:")
        if similar:
            for char in similar:
                print(f"    - {char['name']}")
        else:
            print("    (No similar characters found)")

    # Test 6: Graph summary
    print_section("Graph Summary")
    summary = queries.get_graph_summary()

    print(f"\n📈 Extraction Quality:")
    print(f"  High Confidence Origins: {summary['high_confidence_origins']}")
    print(f"  Low Confidence Origins: {summary['low_confidence_origins']}")
    print(f"  Characters with Origins: {summary['characters_with_origins']}")
    print(f"  Coverage: {summary['characters_with_origins'] / len(all_characters) * 100:.1f}%")

    # Step 6: Save the graph
    print_section("Saving Knowledge Graph")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_file = output_dir / "marvel_knowledge_graph.graphml"
    graph_ops.save_graph(str(graph_file))
    print(f"✅ Graph saved to: {graph_file}")

    # Save graph summary
    summary_file = output_dir / "graph_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")

    print_section("Phase 2 Test Complete!")
    print("\n✅ Successfully built and tested knowledge graph!")
    print(f"   - {stats['total_nodes']} nodes")
    print(f"   - {stats['total_edges']} relationships")
    print(f"   - {len(all_characters)} characters indexed")
    print(f"\nGraph file: {graph_file}")


def test_specific_queries():
    """Test specific query scenarios."""
    print_section("Testing Specific Query Scenarios")

    # Load the saved graph
    graph_file = "data/processed/marvel_knowledge_graph.graphml"
    if not Path(graph_file).exists():
        print("❌ Please run test_graph_building() first")
        return

    print(f"📂 Loading graph from: {graph_file}")
    graph_ops = GraphOperations.load_graph(graph_file)
    queries = GraphQueries(graph_ops)

    # Scenario 1: "How did Spider-Man get his powers?"
    print("\n🎯 Scenario 1: How did Spider-Man get his powers?")
    spiderman = queries.find_character_by_name("Spider-Man")
    if spiderman:
        origin = queries.get_character_power_origin(spiderman['node_id'])
        if origin:
            print(f"\nAnswer:")
            print(f"  {spiderman['name']} got their powers through {origin['origin_type']}.")
            print(f"  {origin['description']}")
            if origin.get('evidence'):
                print(f"\n  Evidence: \"{origin['evidence'][:200]}...\"")

    # Scenario 2: "Why do Spider-Man's powers matter?"
    print("\n🎯 Scenario 2: Why do Spider-Man's powers matter?")
    if spiderman:
        significance = queries.get_character_significance(spiderman['node_id'])
        if significance:
            print(f"\nAnswer:")
            print(f"  {significance['why_matters']}")
            print(f"  Impact Level: {significance['impact_level']}")
            print(f"  Unique Capabilities:")
            for cap in significance['unique_capabilities']:
                print(f"    - {cap}")

    # Scenario 3: "Who else got their powers from technology?"
    print("\n🎯 Scenario 3: Who else got their powers from technology?")
    tech_chars = queries.get_characters_by_origin_type("technology")
    if tech_chars:
        print(f"\nAnswer: {len(tech_chars)} character(s) with technology-based powers:")
        for char in tech_chars:
            print(f"  - {char['name']}")

    # Scenario 4: "Show me all cosmic-level threats"
    print("\n🎯 Scenario 4: Show me all cosmic-level threats")
    cosmic_chars = queries.get_characters_by_impact_level("COSMIC")
    if cosmic_chars:
        print(f"\nAnswer: {len(cosmic_chars)} cosmic-level character(s):")
        for char in cosmic_chars:
            print(f"  - {char['name']}")

    print("\n✅ Query scenarios complete!")


if __name__ == "__main__":
    # Run the main graph building test
    test_graph_building()

    print("\n" + "="*80)
    input("\nPress Enter to run specific query scenarios...")

    # Run specific query tests
    test_specific_queries()
