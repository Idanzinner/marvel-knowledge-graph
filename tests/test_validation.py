"""
Phase 3: Validation System Test

This script tests the advanced validation system including:
1. Semantic similarity validation (embedding-based)
2. Comprehensive validation reports
3. Quality assessment and recommendations
4. Batch validation processing
"""
import json
import asyncio
from pathlib import Path

from src.models.character import Character
from src.models.power_origin import CharacterExtraction
from src.agents.validation_agent import validate_batch_sync
from src.utils.validation_reports import (
    generate_batch_validation_report,
    print_validation_summary,
    generate_character_validation_report
)
from src.utils.data_loader import get_sample_characters


def load_phase1_extractions():
    """Load extraction results from Phase 1."""
    extraction_file = Path("data/processed/sample_extractions.json")

    if not extraction_file.exists():
        raise FileNotFoundError(
            f"Extraction file not found: {extraction_file}\n"
            "Please run test_extraction.py first to generate extraction data."
        )

    with open(extraction_file, 'r') as f:
        data = json.load(f)

    # Handle both list format and dict format
    if isinstance(data, list):
        extraction_data = data
    else:
        extraction_data = data.get("extractions", data)

    extractions = [
        CharacterExtraction(**item) for item in extraction_data
    ]

    print(f"✅ Loaded {len(extractions)} extractions from Phase 1")
    return extractions


def main():
    """Run Phase 3 validation tests."""
    print("=" * 80)
    print("PHASE 3: VALIDATION SYSTEM TEST")
    print("=" * 80)

    # Load Phase 1 extractions
    print("\n📂 Loading extraction results from Phase 1...")
    extractions = load_phase1_extractions()

    # Load original character data
    print("\n📂 Loading original character data...")
    character_names = [e.character_name for e in extractions]
    data_file = Path("data/marvel-wikia-data-with-descriptions.pkl")
    characters = get_sample_characters(
        file_path=data_file,
        character_names=character_names,
        use_pickle=True
    )
    print(f"✅ Loaded {len(characters)} character records")

    # Run validation with semantic similarity
    print("\n" + "=" * 80)
    print("Running Advanced Validation (with Semantic Similarity)")
    print("=" * 80)

    validations = validate_batch_sync(
        extractions=extractions,
        characters=characters,
        enable_multi_pass=False,  # Disable for now (requires re-extraction)
        min_confidence_score=0.5,
        min_completeness_score=0.5,
        min_semantic_similarity=0.7,
        verbose=True
    )

    print(f"\n✅ Validation complete for {len(validations)} characters")

    # Generate comprehensive validation report
    print("\n" + "=" * 80)
    print("Generating Comprehensive Validation Report")
    print("=" * 80)

    report = generate_batch_validation_report(
        extractions=extractions,
        validations=validations,
        characters=characters,
        output_file="data/processed/validation_report.json"
    )

    # Print summary
    print_validation_summary(report)

    # Show detailed report for first character
    print("=" * 80)
    print(f"DETAILED REPORT: {extractions[0].character_name}")
    print("=" * 80)

    char_report = report["character_reports"][0]

    print(f"\n📊 Character Information:")
    print(f"  Name: {char_report['character']['name']}")
    print(f"  Page ID: {char_report['character']['page_id']}")
    print(f"  Alignment: {char_report['character']['alignment']}")
    print(f"  Appearances: {char_report['character']['appearances']}")
    print(f"  Description Length: {char_report['character']['description_length']} chars")

    print(f"\n🔍 Extraction:")
    print(f"  Origin Type: {char_report['extraction']['origin_type']}")
    print(f"  Confidence: {char_report['extraction']['confidence_level']}")
    print(f"  Impact Level: {char_report['extraction']['impact_level']}")
    print(f"  Capabilities: {len(char_report['extraction']['unique_capabilities'])}")

    print(f"\n✅ Validation:")
    print(f"  Passed: {char_report['validation']['passed']}")
    print(f"  Confidence Score: {char_report['validation']['confidence_score']:.3f}")
    print(f"  Completeness Score: {char_report['validation']['completeness_score']:.3f}")
    print(f"  Semantic Similarity: {char_report['validation']['semantic_similarity']:.3f}")

    if char_report['validation']['flags']:
        print(f"\n⚠️  Flags:")
        for flag in char_report['validation']['flags']:
            print(f"  - {flag}")

    print(f"\n⭐ Quality Assessment:")
    print(f"  Overall Quality: {char_report['quality_assessment']['overall_quality']:.3f}")

    print(f"\n💪 Strengths:")
    for strength in char_report['quality_assessment']['strengths']:
        print(f"  + {strength}")

    if char_report['quality_assessment']['weaknesses']:
        print(f"\n⚠️  Weaknesses:")
        for weakness in char_report['quality_assessment']['weaknesses']:
            print(f"  - {weakness}")

    print(f"\n💡 Recommendations:")
    for rec in char_report['quality_assessment']['recommendations']:
        print(f"  → {rec}")

    # Show quality tiers
    print("\n" + "=" * 80)
    print("QUALITY TIERS")
    print("=" * 80)

    print(f"\n🌟 High Quality Characters ({len(report['quality_tiers']['high_quality'])}):")
    for char in report['quality_tiers']['high_quality']:
        print(f"  ✅ {char['name']} (quality: {char['quality_score']:.3f})")

    if report['quality_tiers']['medium_quality']:
        print(f"\n⭐ Medium Quality Characters ({len(report['quality_tiers']['medium_quality'])}):")
        for char in report['quality_tiers']['medium_quality']:
            print(f"  ⚠️  {char['name']} (quality: {char['quality_score']:.3f})")

    if report['quality_tiers']['low_quality']:
        print(f"\n⚠️  Low Quality Characters ({len(report['quality_tiers']['low_quality'])}):")
        for char in report['quality_tiers']['low_quality']:
            print(f"  ❌ {char['name']} (quality: {char['quality_score']:.3f})")

    # Show recommendations
    print("\n" + "=" * 80)
    print("SYSTEM-WIDE RECOMMENDATIONS")
    print("=" * 80)

    if report['recommendations']['characters_needing_review']:
        print(f"\n🔍 Characters Needing Review:")
        for name in report['recommendations']['characters_needing_review']:
            print(f"  - {name}")
    else:
        print(f"\n✅ No characters need review - all meet quality standards!")

    if report['recommendations']['extraction_improvement_areas']:
        print(f"\n📈 Improvement Areas:")
        for area in report['recommendations']['extraction_improvement_areas']:
            print(f"  → {area}")

    # Test individual character report export
    print("\n" + "=" * 80)
    print("Exporting Individual Character Reports")
    print("=" * 80)

    output_dir = Path("data/processed/character_validation_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    for extraction, validation, character in zip(extractions, validations, characters):
        char_report = generate_character_validation_report(
            extraction=extraction,
            validation=validation,
            character=character
        )

        # Save to individual file
        char_name_safe = extraction.character_name.replace(" ", "_").replace("(", "").replace(")", "")
        report_file = output_dir / f"{char_name_safe}_validation.json"

        with open(report_file, 'w') as f:
            json.dump(char_report, f, indent=2)

        print(f"  ✅ Saved: {report_file.name}")

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 3 TEST COMPLETE")
    print("=" * 80)

    print(f"\n📊 Summary Statistics:")
    print(f"  Total Characters Validated: {len(validations)}")
    print(f"  Passed Validation: {sum(1 for v in validations if v.extraction_passed)}")
    print(f"  Average Confidence: {sum(v.confidence_score for v in validations) / len(validations):.3f}")
    print(f"  Average Completeness: {sum(v.completeness_score for v in validations) / len(validations):.3f}")

    similarities = [v.semantic_similarity for v in validations if v.semantic_similarity is not None]
    if similarities:
        print(f"  Average Semantic Similarity: {sum(similarities) / len(similarities):.3f}")

    print(f"\n📁 Output Files:")
    print(f"  - data/processed/validation_report.json")
    print(f"  - data/processed/character_validation_reports/*.json")

    print(f"\n✅ Phase 3 Validation System is fully operational!")
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"  ✅ Semantic similarity validation (embedding-based)")
    print(f"  ✅ Comprehensive quality assessment")
    print(f"  ✅ Strength/weakness identification")
    print(f"  ✅ Actionable recommendations")
    print(f"  ✅ Batch validation processing")
    print(f"  ✅ Detailed per-character reports")
    print(f"  ✅ Quality tier classification")
    print(f"  ✅ System-wide improvement insights")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
