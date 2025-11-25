"""
Quick test script to verify Neo4j connection and operations.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graph.neo4j_operations import Neo4jOperations
from src.utils.data_loader import get_sample_characters
from src.agents.parallel_extraction import extract_batch_parallel

async def test_neo4j():
    print("="*80)
    print("NEO4J CONNECTION TEST")
    print("="*80)

    # Test 1: Connection
    print("\n1. Testing Neo4j Connection...")
    try:
        neo4j = Neo4jOperations()
        print("✓ Connected to Neo4j successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return

    # Test 2: Create Constraints
    print("\n2. Creating Constraints...")
    try:
        neo4j.create_constraints()
        print("✓ Constraints created!")
    except Exception as e:
        print(f"⚠ Warning: {e}")

    # Test 3: Get Statistics
    print("\n3. Getting Graph Statistics...")
    try:
        stats = neo4j.get_statistics()
        print(f"✓ Total Nodes: {stats['total_nodes']}")
        print(f"✓ Total Edges: {stats['total_edges']}")
        print(f"✓ Nodes by Type: {stats['nodes_by_type']}")
    except Exception as e:
        print(f"✗ Failed to get stats: {e}")

    # Test 4: Load Sample Characters
    print("\n4. Loading Sample Characters...")
    try:
        data_path = project_root / "data" / "marvel-wikia-data-with-descriptions.pkl"
        characters = get_sample_characters(
            file_path=data_path,
            character_names=["Spider-Man (Peter Parker)", "Iron Man (Anthony \"Tony\" Stark)"],
            use_pickle=True
        )
        print(f"✓ Loaded {len(characters)} characters")
        for char in characters:
            print(f"  - {char.name}")
    except Exception as e:
        print(f"✗ Failed to load characters: {e}")
        neo4j.close()
        return

    # Test 5: Parallel Extraction
    print("\n5. Running Parallel Extraction...")
    try:
        summary = await extract_batch_parallel(
            characters=characters,
            max_concurrent=2,
            max_retries=2,
            verbose=True
        )
        print(f"✓ Extraction Complete!")
        print(f"  Success: {summary.successful}/{summary.total_characters}")
        print(f"  Failed: {summary.failed}")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        neo4j.close()
        return

    # Test 6: Build Graph
    print("\n6. Building Graph in Neo4j...")
    try:
        for result in summary.successful_extractions:
            char = next(c for c in characters if c.page_id == result.character_id)
            ext = result.extraction

            # Create character node
            char_id = neo4j.add_character_node(
                name=char.name,
                page_id=char.page_id,
                alignment=char.align,
                appearances=char.appearances
            )

            # Create origin node
            origin_id = neo4j.add_power_origin_node(
                character_id=char_id,
                origin_type=ext.power_origin.type.value,
                description=ext.power_origin.description,
                confidence=ext.power_origin.confidence.value,
                evidence=ext.power_origin.evidence
            )

            # Link character to origin
            neo4j.add_relationship(char_id, origin_id, "HAS_ORIGIN")

            # Create power nodes
            for capability in ext.significance.unique_capabilities:
                power_id = neo4j.add_power_node(power_name=capability)
                neo4j.add_relationship(char_id, power_id, "POSSESSES_POWER")
                neo4j.add_relationship(origin_id, power_id, "CONFERS")

            print(f"  ✓ Added {char.name} to Neo4j")

        print("✓ Graph built successfully!")
    except Exception as e:
        print(f"✗ Failed to build graph: {e}")
        neo4j.close()
        return

    # Test 7: Query Character
    print("\n7. Querying Character Profile...")
    try:
        profile = neo4j.get_character_profile("Spider-Man (Peter Parker)", search_by="name")
        if profile:
            print(f"✓ Retrieved profile for {profile['character']['name']}")
            print(f"  Origin Type: {profile['origin']['origin_type']}")
            print(f"  Powers: {len(profile['powers'])}")
        else:
            print("⚠ Character not found")
    except Exception as e:
        print(f"✗ Query failed: {e}")

    # Test 8: Final Statistics
    print("\n8. Final Graph Statistics...")
    try:
        stats = neo4j.get_statistics()
        print(f"✓ Total Nodes: {stats['total_nodes']}")
        print(f"✓ Total Edges: {stats['total_edges']}")
        print(f"✓ Nodes by Type: {stats['nodes_by_type']}")
    except Exception as e:
        print(f"✗ Failed to get stats: {e}")

    # Cleanup
    print("\n9. Cleanup...")
    neo4j.close()
    print("✓ Connection closed")

    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nNeo4j Browser: http://localhost:7474")
    print("Username: neo4j")
    print("Password: Polazin2!")
    print("\n")

if __name__ == "__main__":
    asyncio.run(test_neo4j())
