"""
Test script for Phase 1: Extract power origins from sample Marvel characters.

This script tests the extraction agent on a few well-known characters.
"""
import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.utils.data_loader import get_sample_characters
from src.agents.extraction_agent import extract_batch

# Load environment variables
load_dotenv()


async def main():
    """Test extraction on sample characters."""

    print("=" * 80)
    print("PHASE 1: Power Origin Extraction Test")
    print("=" * 80)

    # Define sample characters to test
    sample_characters = [
        "Spider-Man (Peter Parker)",
        "Captain America (Steven Rogers)",
        "Wolverine (James \"Logan\" Howlett)",
        "Iron Man (Anthony \"Tony\" Stark)",
        "Thor (Thor Odinson)"
    ]

    print(f"\nLoading {len(sample_characters)} sample characters...")

    # Load character data from pickle
    data_path = Path("data/marvel-wikia-data-with-descriptions.pkl")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists.")
        return

    characters = get_sample_characters(
        file_path=data_path,
        character_names=sample_characters,
        use_pickle=True
    )

    print(f"Loaded {len(characters)} characters")
    print(f"Characters: {[c.name for c in characters]}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not found in environment!")
        print("Please set it in your .env file or environment variables.")
        return

    print("\n" + "-" * 80)
    print("Running extraction agent...")
    print("-" * 80)

    # Run extraction
    extractions = await extract_batch(
        characters=characters,
        max_retries=2,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)

    # Display results
    for extraction in extractions:
        print(f"\n{'─' * 80}")
        print(f"CHARACTER: {extraction.character_name}")
        print(f"{'─' * 80}")

        print(f"\n📜 POWER ORIGIN:")
        print(f"  Type: {extraction.power_origin.type.value}")
        print(f"  Confidence: {extraction.power_origin.confidence.value.upper()}")
        print(f"  Description: {extraction.power_origin.description}")
        print(f"  Evidence: {extraction.power_origin.evidence[:200]}...")

        print(f"\n⚡ SIGNIFICANCE:")
        print(f"  Impact Level: {extraction.significance.impact_level.value}")
        print(f"  Why It Matters: {extraction.significance.why_matters}")
        print(f"  Unique Capabilities:")
        for cap in extraction.significance.unique_capabilities:
            print(f"    - {cap}")
        if extraction.significance.strategic_value:
            print(f"  Strategic Value: {extraction.significance.strategic_value}")

    # Save results to JSON
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "sample_extractions.json"

    # Convert to JSON-serializable format
    results_json = [
        {
            "character_name": e.character_name,
            "character_id": e.character_id,
            "power_origin": {
                "type": e.power_origin.type.value,
                "description": e.power_origin.description,
                "confidence": e.power_origin.confidence.value,
                "evidence": e.power_origin.evidence
            },
            "significance": {
                "why_matters": e.significance.why_matters,
                "impact_level": e.significance.impact_level.value,
                "unique_capabilities": e.significance.unique_capabilities,
                "strategic_value": e.significance.strategic_value
            },
            "extraction_timestamp": e.extraction_timestamp
        }
        for e in extractions
    ]

    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    confidence_counts = {}
    origin_type_counts = {}

    for e in extractions:
        # Count confidence levels
        conf = e.power_origin.confidence.value
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        # Count origin types
        origin = e.power_origin.type.value
        origin_type_counts[origin] = origin_type_counts.get(origin, 0) + 1

    print(f"\nTotal Extractions: {len(extractions)}")
    print(f"\nConfidence Distribution:")
    for level, count in sorted(confidence_counts.items()):
        print(f"  {level.upper()}: {count}")

    print(f"\nOrigin Type Distribution:")
    for origin, count in sorted(origin_type_counts.items()):
        print(f"  {origin}: {count}")

    print("\n" + "=" * 80)
    print("✅ Phase 1 Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
