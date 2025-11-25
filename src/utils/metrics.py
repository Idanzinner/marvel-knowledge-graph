"""
Validation and quality metrics for extraction results.
"""
from typing import List, Dict, Any
from ..models.power_origin import CharacterExtraction, ValidationResult, ConfidenceLevel


def calculate_completeness_score(extraction: CharacterExtraction) -> float:
    """
    Calculate how complete an extraction is (0-1 scale).

    Checks:
    - Power origin fields populated
    - Significance fields populated
    - Evidence provided
    - Unique capabilities listed

    Args:
        extraction: CharacterExtraction to evaluate

    Returns:
        Completeness score between 0.0 and 1.0
    """
    score = 0.0
    total_checks = 8

    # Power origin checks (4 points)
    if extraction.power_origin.type.value != "unknown":
        score += 1
    if extraction.power_origin.description and len(extraction.power_origin.description) > 20:
        score += 1
    if extraction.power_origin.evidence and len(extraction.power_origin.evidence) > 20:
        score += 1
    if extraction.power_origin.confidence != ConfidenceLevel.LOW:
        score += 1

    # Significance checks (4 points)
    if extraction.significance.why_matters and len(extraction.significance.why_matters) > 20:
        score += 1
    if extraction.significance.unique_capabilities and len(extraction.significance.unique_capabilities) > 0:
        score += 1
    if len(extraction.significance.unique_capabilities) >= 2:
        score += 1
    if extraction.significance.strategic_value:
        score += 1

    return score / total_checks


def confidence_to_score(confidence: ConfidenceLevel) -> float:
    """
    Convert confidence level to numeric score.

    Args:
        confidence: ConfidenceLevel enum

    Returns:
        Numeric score (0.33 for LOW, 0.66 for MEDIUM, 1.0 for HIGH)
    """
    mapping = {
        ConfidenceLevel.LOW: 0.33,
        ConfidenceLevel.MEDIUM: 0.66,
        ConfidenceLevel.HIGH: 1.0
    }
    return mapping.get(confidence, 0.33)


def validate_extraction(
    extraction: CharacterExtraction,
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5
) -> ValidationResult:
    """
    Validate an extraction result.

    Args:
        extraction: CharacterExtraction to validate
        min_confidence_score: Minimum confidence score to pass validation
        min_completeness_score: Minimum completeness score to pass validation

    Returns:
        ValidationResult with scores and flags
    """
    # Calculate scores
    confidence_score = confidence_to_score(extraction.power_origin.confidence)
    completeness_score = calculate_completeness_score(extraction)

    # Check for issues
    flags = []

    if extraction.power_origin.type.value == "unknown":
        flags.append("Unknown power origin type")

    if confidence_score < min_confidence_score:
        flags.append(f"Low confidence score: {confidence_score:.2f}")

    if completeness_score < min_completeness_score:
        flags.append(f"Low completeness score: {completeness_score:.2f}")

    if not extraction.power_origin.evidence or len(extraction.power_origin.evidence) < 20:
        flags.append("Insufficient evidence provided")

    if len(extraction.significance.unique_capabilities) == 0:
        flags.append("No unique capabilities listed")

    # Determine if extraction passed
    extraction_passed = (
        confidence_score >= min_confidence_score
        and completeness_score >= min_completeness_score
        and extraction.power_origin.type.value != "unknown"
    )

    return ValidationResult(
        character_name=extraction.character_name,
        extraction_passed=extraction_passed,
        confidence_score=confidence_score,
        completeness_score=completeness_score,
        flags=flags,
        notes=f"Validated extraction for {extraction.character_name}"
    )


def calculate_batch_metrics(
    extractions: List[CharacterExtraction]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics for a batch of extractions.

    Args:
        extractions: List of CharacterExtraction results

    Returns:
        Dictionary with aggregate statistics
    """
    if not extractions:
        return {}

    # Confidence distribution
    confidence_dist = {
        "high": 0,
        "medium": 0,
        "low": 0
    }

    # Origin type distribution
    origin_types = {}

    # Impact level distribution
    impact_levels = {}

    # Scores
    confidence_scores = []
    completeness_scores = []

    for extraction in extractions:
        # Confidence
        conf = extraction.power_origin.confidence.value
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        # Origin type
        origin = extraction.power_origin.type.value
        origin_types[origin] = origin_types.get(origin, 0) + 1

        # Impact level
        impact = extraction.significance.impact_level.value
        impact_levels[impact] = impact_levels.get(impact, 0) + 1

        # Scores
        confidence_scores.append(confidence_to_score(extraction.power_origin.confidence))
        completeness_scores.append(calculate_completeness_score(extraction))

    # Calculate averages
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    avg_completeness = sum(completeness_scores) / len(completeness_scores)

    # Coverage (non-unknown origins)
    coverage = sum(
        1 for e in extractions
        if e.power_origin.type.value != "unknown"
    ) / len(extractions)

    return {
        "total_extractions": len(extractions),
        "confidence_distribution": confidence_dist,
        "origin_type_distribution": origin_types,
        "impact_level_distribution": impact_levels,
        "average_confidence_score": round(avg_confidence, 3),
        "average_completeness_score": round(avg_completeness, 3),
        "coverage_rate": round(coverage, 3),
        "high_confidence_count": confidence_dist["high"],
        "medium_confidence_count": confidence_dist["medium"],
        "low_confidence_count": confidence_dist["low"]
    }


def generate_validation_report(
    extractions: List[CharacterExtraction],
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5
) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report for a batch of extractions.

    Args:
        extractions: List of CharacterExtraction results
        min_confidence_score: Minimum confidence threshold
        min_completeness_score: Minimum completeness threshold

    Returns:
        Dictionary containing validation report
    """
    # Validate each extraction
    validation_results = [
        validate_extraction(e, min_confidence_score, min_completeness_score)
        for e in extractions
    ]

    # Calculate pass rate
    passed = sum(1 for v in validation_results if v.extraction_passed)
    pass_rate = passed / len(validation_results) if validation_results else 0

    # Collect all flags
    all_flags = {}
    for v in validation_results:
        for flag in v.flags:
            all_flags[flag] = all_flags.get(flag, 0) + 1

    # Get batch metrics
    batch_metrics = calculate_batch_metrics(extractions)

    return {
        "summary": {
            "total_characters": len(extractions),
            "passed_validation": passed,
            "failed_validation": len(validation_results) - passed,
            "pass_rate": round(pass_rate, 3)
        },
        "batch_metrics": batch_metrics,
        "common_flags": dict(sorted(all_flags.items(), key=lambda x: x[1], reverse=True)),
        "validation_results": [
            {
                "character_name": v.character_name,
                "passed": v.extraction_passed,
                "confidence_score": v.confidence_score,
                "completeness_score": v.completeness_score,
                "flags": v.flags
            }
            for v in validation_results
        ]
    }
