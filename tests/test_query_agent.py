"""
Test script for Phase 4: Query Agent

Demonstrates natural language question answering using the Query Agent
with LangGraph state machine.
"""

import asyncio
import json
from pathlib import Path

from src.agents.query_agent import create_query_agent


def print_separator(title: str = ""):
    """Print a nice separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"{title:^80}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\n{'-' * 80}\n")


def print_result(result: dict):
    """Pretty print query result."""
    print(f"❓ Question: {result['question']}")
    print(f"📊 Query Type: {result['query_type']}")
    print(f"👤 Characters: {', '.join(result['characters']) if result['characters'] else 'None'}")
    print(f"🎯 Confidence: {result['confidence_level']}")
    print(f"\n💬 Answer:")
    print(f"   {result['answer']}")
    print_separator()


def main():
    """Run Phase 4 Query Agent tests."""

    print_separator("PHASE 4: Query Agent Test")

    # ========================================================================
    # Load Knowledge Graph
    # ========================================================================

    graph_path = "data/processed/marvel_knowledge_graph.graphml"

    print("📂 Loading knowledge graph...")
    try:
        agent = create_query_agent(graph_path, verbose=True)
        print("✅ Query Agent initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return

    # ========================================================================
    # Sample Queries from Project Plan
    # ========================================================================

    sample_queries = [
        # Query 1: Power Origin
        "How did Spider-Man get his powers?",

        # Query 2: Significance
        "Why do Spider-Man's powers matter?",

        # Query 3: Power Abilities
        "What powers does Captain America have?",

        # Query 4: General Information
        "Tell me about Thor",

        # Query 5: Validation
        "How confident are you about Spider-Man's power origin?",

        # Query 6: Comparison (if multiple characters are in graph)
        "How are Spider-Man and Captain America different?",
    ]

    # ========================================================================
    # Execute Queries
    # ========================================================================

    print_separator("EXECUTING SAMPLE QUERIES")

    results = []

    for i, question in enumerate(sample_queries, 1):
        print(f"\n[Query {i}/{len(sample_queries)}]")

        try:
            result = agent.query(question, verbose=False)
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print_separator()
            continue

    # ========================================================================
    # Additional Test Queries
    # ========================================================================

    print_separator("ADDITIONAL TEST QUERIES")

    additional_queries = [
        "What is Thor's origin?",
        "What makes Captain America powerful?",
        "Does Spider-Man have any unique abilities?",
    ]

    for i, question in enumerate(additional_queries, 1):
        print(f"\n[Additional Query {i}/{len(additional_queries)}]")

        try:
            result = agent.query(question, verbose=False)
            results.append(result)
            print_result(result)

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print_separator()
            continue

    # ========================================================================
    # Test Edge Cases
    # ========================================================================

    print_separator("EDGE CASE TESTING")

    edge_case_queries = [
        "Who is Deadpool?",  # Character not in our small test set
        "What is the meaning of life?",  # Non-Marvel question
        "Tell me about powers",  # No specific character
    ]

    for i, question in enumerate(edge_case_queries, 1):
        print(f"\n[Edge Case {i}/{len(edge_case_queries)}]")

        try:
            result = agent.query(question, verbose=False)
            print_result(result)

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print_separator()
            continue

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print_separator("SUMMARY STATISTICS")

    successful_queries = [r for r in results if r['context_retrieved']]
    failed_queries = [r for r in results if not r['context_retrieved']]

    print(f"📊 Query Statistics:")
    print(f"   Total Queries: {len(results)}")
    print(f"   Successful: {len(successful_queries)}")
    print(f"   No Data Found: {len(failed_queries)}")
    print()

    # Query type distribution
    query_types = {}
    for r in results:
        qt = r['query_type']
        query_types[qt] = query_types.get(qt, 0) + 1

    print("📈 Query Type Distribution:")
    for qt, count in sorted(query_types.items()):
        print(f"   {qt}: {count}")
    print()

    # Confidence distribution
    confidence_levels = {}
    for r in successful_queries:
        cl = r['confidence_level']
        confidence_levels[cl] = confidence_levels.get(cl, 0) + 1

    print("🎯 Confidence Level Distribution:")
    for cl, count in sorted(confidence_levels.items()):
        print(f"   {cl}: {count}")
    print()

    # ========================================================================
    # Save Results
    # ========================================================================

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "query_results.json"

    # Format results for JSON
    json_results = {
        "total_queries": len(results),
        "successful_queries": len(successful_queries),
        "failed_queries": len(failed_queries),
        "query_type_distribution": query_types,
        "confidence_distribution": confidence_levels,
        "results": results
    }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"💾 Results saved to: {output_file}")

    # ========================================================================
    # Completion Message
    # ========================================================================

    print_separator()
    print("✅ Phase 4 Query Agent Test Complete!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Query classification (POWER_ORIGIN, SIGNIFICANCE, etc.)")
    print("  ✓ Entity extraction (character name identification)")
    print("  ✓ Graph context retrieval")
    print("  ✓ Natural language response generation")
    print("  ✓ Validation score integration")
    print("  ✓ Error handling for missing data")
    print()
    print("Next Steps:")
    print("  → Integrate into FastAPI (Phase 5)")
    print("  → Add more sophisticated routing")
    print("  → Implement caching for performance")
    print_separator()


if __name__ == "__main__":
    main()
