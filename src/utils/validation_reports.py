"""
Comprehensive validation report generation.
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..models.character import Character
from ..models.power_origin import CharacterExtraction, ValidationResult
from .metrics import calculate_batch_metrics


def generate_character_validation_report(
    extraction: CharacterExtraction,
    validation: ValidationResult,
    character: Character
) -> Dict[str, Any]:
    """
    Generate a detailed validation report for a single character.

    Args:
        extraction: The character extraction
        validation: Validation result
        character: Original character data

    Returns:
        Dictionary with comprehensive validation report
    """
    return {
        "character": {
            "name": extraction.character_name,
            "page_id": extraction.character_id or character.page_id,
            "alignment": character.align,
            "appearances": character.appearances,
            "description_length": len(character.description_text or "")
        },
        "extraction": {
            "timestamp": extraction.extraction_timestamp,
            "origin_type": extraction.power_origin.type.value,
            "origin_description": extraction.power_origin.description,
            "confidence_level": extraction.power_origin.confidence.value,
            "evidence": extraction.power_origin.evidence,
            "impact_level": extraction.significance.impact_level.value,
            "unique_capabilities": extraction.significance.unique_capabilities,
            "why_matters": extraction.significance.why_matters
        },
        "validation": {
            "passed": validation.extraction_passed,
            "confidence_score": validation.confidence_score,
            "completeness_score": validation.completeness_score,
            "semantic_similarity": validation.semantic_similarity,
            "flags": validation.flags,
            "notes": validation.notes
        },
        "quality_assessment": {
            "overall_quality": _calculate_overall_quality(validation),
            "strengths": _identify_strengths(extraction, validation),
            "weaknesses": _identify_weaknesses(extraction, validation),
            "recommendations": _generate_recommendations(extraction, validation)
        }
    }


def generate_batch_validation_report(
    extractions: List[CharacterExtraction],
    validations: List[ValidationResult],
    characters: List[Character],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report for a batch of characters.

    Args:
        extractions: List of character extractions
        validations: List of validation results
        characters: List of original character data
        output_file: Optional path to save report as JSON

    Returns:
        Dictionary with batch validation report
    """
    if not (len(extractions) == len(validations) == len(characters)):
        raise ValueError("All input lists must have the same length")

    # Calculate aggregate metrics
    batch_metrics = calculate_batch_metrics(extractions)

    # Validation statistics
    passed_count = sum(1 for v in validations if v.extraction_passed)
    failed_count = len(validations) - passed_count

    # Quality tiers
    high_quality = []
    medium_quality = []
    low_quality = []

    for extraction, validation, character in zip(extractions, validations, characters):
        quality = _calculate_overall_quality(validation)
        char_data = {
            "name": extraction.character_name,
            "quality_score": quality,
            "passed": validation.extraction_passed
        }

        if quality >= 0.8:
            high_quality.append(char_data)
        elif quality >= 0.6:
            medium_quality.append(char_data)
        else:
            low_quality.append(char_data)

    # Semantic similarity distribution
    similarities = [
        v.semantic_similarity for v in validations
        if v.semantic_similarity is not None
    ]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    # Flag analysis
    all_flags = {}
    for v in validations:
        for flag in v.flags:
            all_flags[flag] = all_flags.get(flag, 0) + 1

    # Per-character reports
    character_reports = [
        generate_character_validation_report(extraction, validation, character)
        for extraction, validation, character in zip(extractions, validations, characters)
    ]

    # Build comprehensive report
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_characters": len(extractions),
            "version": "1.0"
        },
        "summary": {
            "validation_results": {
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": round(passed_count / len(validations), 3)
            },
            "quality_distribution": {
                "high_quality_count": len(high_quality),
                "medium_quality_count": len(medium_quality),
                "low_quality_count": len(low_quality)
            },
            "average_scores": {
                "confidence": round(
                    sum(v.confidence_score for v in validations) / len(validations), 3
                ),
                "completeness": round(
                    sum(v.completeness_score for v in validations) / len(validations), 3
                ),
                "semantic_similarity": round(avg_similarity, 3)
            }
        },
        "extraction_metrics": batch_metrics,
        "quality_tiers": {
            "high_quality": high_quality,
            "medium_quality": medium_quality,
            "low_quality": low_quality
        },
        "common_issues": dict(
            sorted(all_flags.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "character_reports": character_reports,
        "recommendations": {
            "characters_needing_review": [
                c["name"] for c in low_quality
            ],
            "extraction_improvement_areas": _identify_improvement_areas(validations),
            "overall_assessment": _generate_overall_assessment(validations, batch_metrics)
        }
    }

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Validation report saved to: {output_file}")

    return report


def _calculate_overall_quality(validation: ValidationResult) -> float:
    """Calculate overall quality score (0-1) from validation metrics."""
    # Weighted average of all scores
    confidence_weight = 0.3
    completeness_weight = 0.3
    similarity_weight = 0.4

    similarity = validation.semantic_similarity or 0.0

    quality = (
        validation.confidence_score * confidence_weight +
        validation.completeness_score * completeness_weight +
        similarity * similarity_weight
    )

    return round(quality, 3)


def _identify_strengths(
    extraction: CharacterExtraction,
    validation: ValidationResult
) -> List[str]:
    """Identify strengths of the extraction."""
    strengths = []

    if validation.confidence_score >= 0.9:
        strengths.append("High confidence extraction")

    if validation.completeness_score >= 0.9:
        strengths.append("Very complete extraction with all fields")

    if validation.semantic_similarity and validation.semantic_similarity >= 0.8:
        strengths.append("Excellent grounding in source text")

    if len(extraction.power_origin.evidence) > 50:
        strengths.append("Detailed evidence provided")

    if len(extraction.significance.unique_capabilities) >= 3:
        strengths.append("Comprehensive list of capabilities")

    if extraction.significance.strategic_value:
        strengths.append("Includes strategic value assessment")

    if not validation.flags:
        strengths.append("No validation issues flagged")

    return strengths or ["Adequate extraction quality"]


def _identify_weaknesses(
    extraction: CharacterExtraction,
    validation: ValidationResult
) -> List[str]:
    """Identify weaknesses of the extraction."""
    weaknesses = []

    if validation.confidence_score < 0.5:
        weaknesses.append("Low confidence score")

    if validation.completeness_score < 0.5:
        weaknesses.append("Incomplete extraction")

    if validation.semantic_similarity and validation.semantic_similarity < 0.6:
        weaknesses.append("Poor grounding in source text")

    if extraction.power_origin.type.value == "unknown":
        weaknesses.append("Could not determine origin type")

    if len(extraction.significance.unique_capabilities) == 0:
        weaknesses.append("No unique capabilities identified")

    if len(validation.flags) > 2:
        weaknesses.append(f"Multiple validation issues ({len(validation.flags)} flags)")

    return weaknesses or ["No significant weaknesses"]


def _generate_recommendations(
    extraction: CharacterExtraction,
    validation: ValidationResult
) -> List[str]:
    """Generate recommendations for improving the extraction."""
    recommendations = []

    if validation.confidence_score < 0.7:
        recommendations.append("Consider re-extraction with refined prompts")

    if validation.semantic_similarity and validation.semantic_similarity < 0.7:
        recommendations.append(
            "Review extraction for accuracy against source text"
        )

    if extraction.power_origin.type.value == "unknown":
        recommendations.append(
            "May need manual review or additional source material"
        )

    if len(extraction.significance.unique_capabilities) < 2:
        recommendations.append(
            "Expand unique capabilities list with more specific abilities"
        )

    if not extraction.significance.strategic_value:
        recommendations.append(
            "Add strategic value assessment for completeness"
        )

    if "Insufficient evidence" in validation.flags:
        recommendations.append(
            "Provide longer, more specific evidence quotes"
        )

    return recommendations or ["Extraction meets quality standards"]


def _identify_improvement_areas(validations: List[ValidationResult]) -> List[str]:
    """Identify systemic areas for improvement across batch."""
    areas = []

    # Check for common low scores
    avg_confidence = sum(v.confidence_score for v in validations) / len(validations)
    if avg_confidence < 0.7:
        areas.append("Overall confidence scores are low - review extraction prompts")

    avg_completeness = sum(v.completeness_score for v in validations) / len(validations)
    if avg_completeness < 0.7:
        areas.append("Completeness scores are low - ensure all fields are populated")

    similarities = [v.semantic_similarity for v in validations if v.semantic_similarity]
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        if avg_similarity < 0.7:
            areas.append("Semantic similarity is low - extractions may not be well-grounded")

    # Check for common flags
    all_flags = {}
    for v in validations:
        for flag in v.flags:
            all_flags[flag] = all_flags.get(flag, 0) + 1

    for flag, count in all_flags.items():
        if count / len(validations) > 0.3:  # More than 30% have this flag
            areas.append(f"Common issue: {flag} ({count}/{len(validations)} characters)")

    return areas or ["Extraction quality is generally good"]


def _generate_overall_assessment(
    validations: List[ValidationResult],
    batch_metrics: Dict[str, Any]
) -> str:
    """Generate an overall assessment of the batch."""
    passed_rate = sum(1 for v in validations if v.extraction_passed) / len(validations)

    if passed_rate >= 0.9:
        quality = "Excellent"
    elif passed_rate >= 0.75:
        quality = "Good"
    elif passed_rate >= 0.6:
        quality = "Acceptable"
    else:
        quality = "Needs Improvement"

    avg_confidence = sum(v.confidence_score for v in validations) / len(validations)
    confidence_desc = "high" if avg_confidence >= 0.8 else "moderate" if avg_confidence >= 0.6 else "low"

    return (
        f"{quality} extraction quality overall. "
        f"{int(passed_rate * 100)}% of extractions passed validation with "
        f"{confidence_desc} average confidence ({avg_confidence:.2f}). "
        f"{batch_metrics.get('coverage_rate', 0) * 100:.0f}% of characters have "
        f"identified power origins (non-unknown)."
    )


def print_validation_summary(report: Dict[str, Any]):
    """
    Print a human-readable summary of the validation report.

    Args:
        report: Validation report dictionary
    """
    print("\n" + "=" * 80)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 80)

    summary = report["summary"]

    print(f"\n📊 Validation Results:")
    print(f"  ✅ Passed: {summary['validation_results']['passed']}")
    print(f"  ❌ Failed: {summary['validation_results']['failed']}")
    print(f"  📈 Pass Rate: {summary['validation_results']['pass_rate'] * 100:.1f}%")

    print(f"\n🎯 Quality Distribution:")
    print(f"  🌟 High Quality: {summary['quality_distribution']['high_quality_count']}")
    print(f"  ⭐ Medium Quality: {summary['quality_distribution']['medium_quality_count']}")
    print(f"  ⚠️  Low Quality: {summary['quality_distribution']['low_quality_count']}")

    print(f"\n📏 Average Scores:")
    print(f"  Confidence: {summary['average_scores']['confidence']:.3f}")
    print(f"  Completeness: {summary['average_scores']['completeness']:.3f}")
    print(f"  Semantic Similarity: {summary['average_scores']['semantic_similarity']:.3f}")

    if report.get("common_issues"):
        print(f"\n⚠️  Most Common Issues:")
        for issue, count in list(report["common_issues"].items())[:5]:
            print(f"  - {issue}: {count} occurrences")

    print(f"\n💡 Overall Assessment:")
    print(f"  {report['recommendations']['overall_assessment']}")

    if report["recommendations"]["characters_needing_review"]:
        print(f"\n🔍 Characters Needing Review:")
        for name in report["recommendations"]["characters_needing_review"][:5]:
            print(f"  - {name}")

    print("\n" + "=" * 80 + "\n")
