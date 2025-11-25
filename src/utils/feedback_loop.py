"""
Feedback loop for re-extraction of low-confidence or failed validations.
"""
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from ..models.character import Character
from ..models.power_origin import CharacterExtraction, ValidationResult
from ..agents.extraction_agent import extract_character
from ..agents.validation_agent import validate_character_extraction


async def re_extract_failed_validations(
    extractions: List[CharacterExtraction],
    validations: List[ValidationResult],
    characters: List[Character],
    max_attempts: int = 2,
    verbose: bool = False
) -> Tuple[List[CharacterExtraction], List[ValidationResult]]:
    """
    Re-extract characters that failed validation.

    Args:
        extractions: Original extraction results
        validations: Validation results
        characters: Original character data
        max_attempts: Maximum number of re-extraction attempts
        verbose: Enable detailed logging

    Returns:
        Tuple of (updated_extractions, updated_validations)
    """
    if len(extractions) != len(validations) != len(characters):
        raise ValueError("All input lists must have the same length")

    updated_extractions = list(extractions)
    updated_validations = list(validations)

    # Find failed validations
    failed_indices = [
        i for i, v in enumerate(validations)
        if not v.extraction_passed
    ]

    if not failed_indices:
        if verbose:
            print("✅ No failed validations to re-extract")
        return updated_extractions, updated_validations

    if verbose:
        print(f"\n🔄 Re-extracting {len(failed_indices)} failed validations...")

    for idx in failed_indices:
        character = characters[idx]
        original_extraction = extractions[idx]
        validation = validations[idx]

        if verbose:
            print(f"\n  🔄 Re-extracting: {character.name}")
            print(f"     Original issues: {', '.join(validation.flags)}")

        # Attempt re-extraction
        best_extraction = original_extraction
        best_validation = validation
        best_score = _calculate_validation_quality(validation)

        for attempt in range(max_attempts):
            if verbose:
                print(f"     Attempt {attempt + 1}/{max_attempts}...")

            try:
                # Re-run extraction
                new_extraction = await extract_character(
                    character=character,
                    max_retries=1,
                    verbose=False
                )

                # Validate new extraction
                new_validation = await validate_character_extraction(
                    extraction=new_extraction,
                    character=character,
                    enable_multi_pass=False,
                    verbose=False
                )

                # Check if improvement
                new_score = _calculate_validation_quality(new_validation)

                if verbose:
                    print(f"     Quality score: {new_score:.3f} (previous: {best_score:.3f})")

                if new_score > best_score:
                    best_extraction = new_extraction
                    best_validation = new_validation
                    best_score = new_score

                    if verbose:
                        print(f"     ✅ Improved! New quality: {best_score:.3f}")

                    # If now passing, stop trying
                    if new_validation.extraction_passed:
                        if verbose:
                            print(f"     ✅ Passed validation!")
                        break

            except Exception as e:
                if verbose:
                    print(f"     ❌ Attempt failed: {e}")

        # Update with best result
        updated_extractions[idx] = best_extraction
        updated_validations[idx] = best_validation

        if verbose:
            if best_extraction != original_extraction:
                print(f"     ✅ Using improved extraction (quality: {best_score:.3f})")
            else:
                print(f"     ⚠️  No improvement found, keeping original")

    return updated_extractions, updated_validations


def re_extract_failed_validations_sync(
    extractions: List[CharacterExtraction],
    validations: List[ValidationResult],
    characters: List[Character],
    max_attempts: int = 2,
    verbose: bool = False
) -> Tuple[List[CharacterExtraction], List[ValidationResult]]:
    """
    Synchronous wrapper for re_extract_failed_validations.

    Args:
        extractions: Original extraction results
        validations: Validation results
        characters: Original character data
        max_attempts: Maximum number of re-extraction attempts
        verbose: Enable detailed logging

    Returns:
        Tuple of (updated_extractions, updated_validations)
    """
    return asyncio.run(
        re_extract_failed_validations(
            extractions=extractions,
            validations=validations,
            characters=characters,
            max_attempts=max_attempts,
            verbose=verbose
        )
    )


async def iterative_validation_improvement(
    characters: List[Character],
    max_iterations: int = 3,
    target_pass_rate: float = 0.9,
    verbose: bool = False
) -> Tuple[List[CharacterExtraction], List[ValidationResult], Dict[str, Any]]:
    """
    Iteratively extract and re-extract until target pass rate is achieved.

    Args:
        characters: List of characters to process
        max_iterations: Maximum number of improvement iterations
        target_pass_rate: Target validation pass rate (0-1)
        verbose: Enable detailed logging

    Returns:
        Tuple of (final_extractions, final_validations, improvement_metrics)
    """
    if verbose:
        print(f"\n🔄 Starting iterative validation improvement")
        print(f"   Target pass rate: {target_pass_rate * 100:.0f}%")
        print(f"   Max iterations: {max_iterations}")

    iteration_history = []

    # Initial extraction
    if verbose:
        print(f"\n📊 Iteration 1: Initial extraction...")

    extractions = []
    for char in characters:
        extraction = await extract_character(
            character=char,
            max_retries=1,
            verbose=False
        )
        extractions.append(extraction)

    # Initial validation
    validations = []
    for extraction, character in zip(extractions, characters):
        validation = await validate_character_extraction(
            extraction=extraction,
            character=character,
            enable_multi_pass=False,
            verbose=False
        )
        validations.append(validation)

    pass_rate = sum(1 for v in validations if v.extraction_passed) / len(validations)

    iteration_history.append({
        "iteration": 1,
        "pass_rate": pass_rate,
        "passed_count": sum(1 for v in validations if v.extraction_passed),
        "failed_count": sum(1 for v in validations if not v.extraction_passed),
        "avg_quality": sum(_calculate_validation_quality(v) for v in validations) / len(validations)
    })

    if verbose:
        print(f"   Pass rate: {pass_rate * 100:.1f}%")

    # Iterative improvement
    for iteration in range(2, max_iterations + 1):
        if pass_rate >= target_pass_rate:
            if verbose:
                print(f"\n✅ Target pass rate achieved!")
            break

        if verbose:
            print(f"\n📊 Iteration {iteration}: Re-extracting failed validations...")

        extractions, validations = await re_extract_failed_validations(
            extractions=extractions,
            validations=validations,
            characters=characters,
            max_attempts=2,
            verbose=verbose
        )

        new_pass_rate = sum(1 for v in validations if v.extraction_passed) / len(validations)

        iteration_history.append({
            "iteration": iteration,
            "pass_rate": new_pass_rate,
            "passed_count": sum(1 for v in validations if v.extraction_passed),
            "failed_count": sum(1 for v in validations if not v.extraction_passed),
            "avg_quality": sum(_calculate_validation_quality(v) for v in validations) / len(validations),
            "improvement": new_pass_rate - pass_rate
        })

        if verbose:
            print(f"   Pass rate: {new_pass_rate * 100:.1f}% (Δ {(new_pass_rate - pass_rate) * 100:+.1f}%)")

        if new_pass_rate <= pass_rate:
            if verbose:
                print(f"\n⚠️  No improvement in this iteration")

        pass_rate = new_pass_rate

    # Build metrics
    metrics = {
        "initial_pass_rate": iteration_history[0]["pass_rate"],
        "final_pass_rate": iteration_history[-1]["pass_rate"],
        "improvement": iteration_history[-1]["pass_rate"] - iteration_history[0]["pass_rate"],
        "iterations_performed": len(iteration_history),
        "target_achieved": pass_rate >= target_pass_rate,
        "iteration_history": iteration_history
    }

    if verbose:
        print(f"\n📊 Final Results:")
        print(f"   Initial pass rate: {metrics['initial_pass_rate'] * 100:.1f}%")
        print(f"   Final pass rate: {metrics['final_pass_rate'] * 100:.1f}%")
        print(f"   Improvement: {metrics['improvement'] * 100:+.1f}%")
        print(f"   Target achieved: {'✅ Yes' if metrics['target_achieved'] else '❌ No'}")

    return extractions, validations, metrics


def iterative_validation_improvement_sync(
    characters: List[Character],
    max_iterations: int = 3,
    target_pass_rate: float = 0.9,
    verbose: bool = False
) -> Tuple[List[CharacterExtraction], List[ValidationResult], Dict[str, Any]]:
    """
    Synchronous wrapper for iterative_validation_improvement.

    Args:
        characters: List of characters to process
        max_iterations: Maximum number of improvement iterations
        target_pass_rate: Target validation pass rate (0-1)
        verbose: Enable detailed logging

    Returns:
        Tuple of (final_extractions, final_validations, improvement_metrics)
    """
    return asyncio.run(
        iterative_validation_improvement(
            characters=characters,
            max_iterations=max_iterations,
            target_pass_rate=target_pass_rate,
            verbose=verbose
        )
    )


def _calculate_validation_quality(validation: ValidationResult) -> float:
    """
    Calculate overall quality score from validation.

    Args:
        validation: ValidationResult

    Returns:
        Quality score (0-1)
    """
    # Weighted average
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
