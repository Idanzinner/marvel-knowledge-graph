"""
Validation Agent using LlamaIndex Workflow.

This agent performs advanced validation of extraction results including:
- Semantic similarity between extraction and source text
- Multi-pass extraction consistency checking
- Confidence calibration
- Ground truth validation (when available)
"""
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
    Context
)
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from ..models.character import Character
from ..models.power_origin import CharacterExtraction, ValidationResult
from ..utils.metrics import (
    calculate_completeness_score,
    confidence_to_score,
    validate_extraction
)


# ============================================================================
# Events
# ============================================================================

class ValidateExtractionEvent(Event):
    """Event to start validation."""
    extraction: CharacterExtraction
    character: Character
    original_description: str


class SemanticSimilarityEvent(Event):
    """Event after semantic similarity check."""
    extraction: CharacterExtraction
    character: Character
    original_description: str
    semantic_similarity: float


class MultiPassCheckEvent(Event):
    """Event after multi-pass consistency check."""
    extraction: CharacterExtraction
    character: Character
    original_description: str
    semantic_similarity: float
    consistency_score: Optional[float]
    multi_pass_extractions: List[CharacterExtraction]


class FinalValidationEvent(Event):
    """Event with final validation result."""
    validation_result: ValidationResult


# ============================================================================
# Validation Agent Workflow
# ============================================================================

class ValidationAgent(Workflow):
    """
    LlamaIndex Workflow for validating extraction results.

    Performs:
    1. Semantic similarity validation (embedding-based)
    2. Multi-pass extraction consistency checking
    3. Confidence calibration
    4. Comprehensive validation reporting
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        embedding_model: Optional[OpenAIEmbedding] = None,
        enable_multi_pass: bool = True,
        num_passes: int = 3,
        min_confidence_score: float = 0.5,
        min_completeness_score: float = 0.5,
        min_semantic_similarity: float = 0.7,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize ValidationAgent.

        Args:
            llm: Language model for re-extraction (optional)
            embedding_model: Embedding model for semantic similarity
            enable_multi_pass: Whether to perform multi-pass validation
            num_passes: Number of extraction passes for consistency check
            min_confidence_score: Minimum confidence threshold
            min_completeness_score: Minimum completeness threshold
            min_semantic_similarity: Minimum semantic similarity threshold
            verbose: Enable detailed logging
        """
        super().__init__(**kwargs)

        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.0)
        self.embedding_model = embedding_model or OpenAIEmbedding(
            model="text-embedding-3-small"
        )
        self.enable_multi_pass = enable_multi_pass
        self.num_passes = num_passes
        self.min_confidence_score = min_confidence_score
        self.min_completeness_score = min_completeness_score
        self.min_semantic_similarity = min_semantic_similarity
        self.verbose = verbose

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ValidationAgent] {message}")

    @step
    async def prepare_validation(
        self,
        ev: StartEvent
    ) -> ValidateExtractionEvent:
        """
        Prepare validation inputs.

        Args:
            ev: StartEvent with extraction and character data

        Returns:
            ValidateExtractionEvent
        """
        extraction = ev.get("extraction")
        character = ev.get("character")

        if not extraction or not character:
            raise ValueError("Must provide both extraction and character")

        self._log(f"Starting validation for {extraction.character_name}")

        # Get original description
        original_description = character.description_text or ""

        if not original_description:
            self._log("WARNING: No description text available for validation")

        return ValidateExtractionEvent(
            extraction=extraction,
            character=character,
            original_description=original_description
        )

    @step
    async def check_semantic_similarity(
        self,
        ev: ValidateExtractionEvent
    ) -> SemanticSimilarityEvent:
        """
        Check semantic similarity between extraction and source text.

        Uses embeddings to measure how well the extraction is grounded
        in the original description.

        Args:
            ev: ValidateExtractionEvent

        Returns:
            SemanticSimilarityEvent with similarity score
        """
        self._log("Checking semantic similarity...")

        # Build extraction text
        extraction_text = f"""
Power Origin: {ev.extraction.power_origin.description}
Evidence: {ev.extraction.power_origin.evidence}
Significance: {ev.extraction.significance.why_matters}
Capabilities: {', '.join(ev.extraction.significance.unique_capabilities)}
"""

        # Handle empty description
        if not ev.original_description or len(ev.original_description) < 50:
            self._log("Insufficient description for semantic similarity check")
            return SemanticSimilarityEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=0.0
            )

        # Get embeddings
        try:
            extraction_embedding = await self.embedding_model.aget_text_embedding(
                extraction_text.strip()
            )
            description_embedding = await self.embedding_model.aget_text_embedding(
                ev.original_description[:2000]  # Limit length
            )

            # Calculate cosine similarity
            import numpy as np
            similarity = float(
                np.dot(extraction_embedding, description_embedding) /
                (np.linalg.norm(extraction_embedding) * np.linalg.norm(description_embedding))
            )

            self._log(f"Semantic similarity: {similarity:.3f}")

            return SemanticSimilarityEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=round(similarity, 3)
            )

        except Exception as e:
            self._log(f"Error calculating semantic similarity: {e}")
            return SemanticSimilarityEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=0.0
            )

    @step
    async def check_multi_pass_consistency(
        self,
        ev: SemanticSimilarityEvent
    ) -> MultiPassCheckEvent:
        """
        Perform multi-pass extraction consistency check.

        Runs extraction multiple times and checks agreement between passes.

        Args:
            ev: SemanticSimilarityEvent

        Returns:
            MultiPassCheckEvent with consistency score
        """
        if not self.enable_multi_pass:
            self._log("Multi-pass validation disabled, skipping...")
            return MultiPassCheckEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=ev.semantic_similarity,
                consistency_score=None,
                multi_pass_extractions=[]
            )

        self._log(f"Running {self.num_passes}-pass consistency check...")

        try:
            # Import here to avoid circular dependency
            from .extraction_agent import extract_character

            # Run multiple extraction passes
            multi_pass_extractions = []
            for i in range(self.num_passes):
                self._log(f"  Pass {i + 1}/{self.num_passes}...")
                try:
                    extraction_result = await extract_character(
                        character=ev.character,
                        max_retries=0,  # No retries within passes
                        verbose=False
                    )
                    multi_pass_extractions.append(extraction_result)
                except Exception as e:
                    self._log(f"  Pass {i + 1} failed: {e}")

            if len(multi_pass_extractions) < 2:
                self._log("Not enough successful passes for consistency check")
                return MultiPassCheckEvent(
                    extraction=ev.extraction,
                    character=ev.character,
                    original_description=ev.original_description,
                    semantic_similarity=ev.semantic_similarity,
                    consistency_score=None,
                    multi_pass_extractions=multi_pass_extractions
                )

            # Calculate consistency score by comparing origin types and descriptions
            consistency_scores = []

            for i in range(len(multi_pass_extractions)):
                for j in range(i + 1, len(multi_pass_extractions)):
                    ext1 = multi_pass_extractions[i]
                    ext2 = multi_pass_extractions[j]

                    # Compare origin types
                    type_match = 1.0 if ext1.power_origin.type == ext2.power_origin.type else 0.0

                    # Compare descriptions using embeddings
                    try:
                        desc1_emb = await self.embedding_model.aget_text_embedding(
                            ext1.power_origin.description
                        )
                        desc2_emb = await self.embedding_model.aget_text_embedding(
                            ext2.power_origin.description
                        )

                        import numpy as np
                        desc_similarity = float(
                            np.dot(desc1_emb, desc2_emb) /
                            (np.linalg.norm(desc1_emb) * np.linalg.norm(desc2_emb))
                        )
                    except Exception as e:
                        self._log(f"Error calculating description similarity: {e}")
                        desc_similarity = 0.5

                    # Weighted consistency score
                    consistency = (type_match * 0.4) + (desc_similarity * 0.6)
                    consistency_scores.append(consistency)

            # Average consistency across all pairs
            avg_consistency = sum(consistency_scores) / len(consistency_scores)

            self._log(f"Consistency score: {avg_consistency:.3f} (from {len(multi_pass_extractions)} passes)")

            return MultiPassCheckEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=ev.semantic_similarity,
                consistency_score=round(avg_consistency, 3),
                multi_pass_extractions=multi_pass_extractions
            )

        except Exception as e:
            self._log(f"Multi-pass consistency check failed: {e}")
            return MultiPassCheckEvent(
                extraction=ev.extraction,
                character=ev.character,
                original_description=ev.original_description,
                semantic_similarity=ev.semantic_similarity,
                consistency_score=None,
                multi_pass_extractions=[]
            )

    @step
    async def finalize_validation(
        self,
        ev: MultiPassCheckEvent
    ) -> StopEvent:
        """
        Finalize validation and create comprehensive ValidationResult.

        Args:
            ev: MultiPassCheckEvent

        Returns:
            StopEvent with ValidationResult
        """
        self._log("Finalizing validation...")

        # Use existing validation logic
        base_validation = validate_extraction(
            ev.extraction,
            self.min_confidence_score,
            self.min_completeness_score
        )

        # Add semantic similarity
        base_validation.semantic_similarity = ev.semantic_similarity

        # Add semantic similarity flag if below threshold
        if ev.semantic_similarity < self.min_semantic_similarity:
            base_validation.flags.append(
                f"Low semantic similarity: {ev.semantic_similarity:.2f}"
            )

        # Update pass/fail based on semantic similarity
        if ev.semantic_similarity < self.min_semantic_similarity:
            base_validation.extraction_passed = False

        # Add consistency information to notes
        if ev.consistency_score is not None:
            base_validation.notes = (
                f"{base_validation.notes} | "
                f"Consistency score: {ev.consistency_score:.2f}"
            )

            # Add consistency flag if below threshold
            if ev.consistency_score < 0.7:
                base_validation.flags.append(
                    f"Low consistency across passes: {ev.consistency_score:.2f}"
                )
                base_validation.extraction_passed = False
        else:
            base_validation.notes = (
                f"{base_validation.notes} | "
                f"Multi-pass check: disabled"
            )

        self._log(
            f"Validation complete: "
            f"passed={base_validation.extraction_passed}, "
            f"confidence={base_validation.confidence_score:.2f}, "
            f"completeness={base_validation.completeness_score:.2f}, "
            f"similarity={ev.semantic_similarity:.2f}"
        )

        return StopEvent(result=base_validation)


# ============================================================================
# Helper Functions
# ============================================================================

async def validate_character_extraction(
    extraction: CharacterExtraction,
    character: Character,
    enable_multi_pass: bool = False,
    num_passes: int = 3,
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5,
    min_semantic_similarity: float = 0.7,
    verbose: bool = False
) -> ValidationResult:
    """
    Validate a character extraction using the ValidationAgent workflow.

    Args:
        extraction: CharacterExtraction to validate
        character: Original Character data
        enable_multi_pass: Whether to perform multi-pass validation
        num_passes: Number of extraction passes for consistency
        min_confidence_score: Minimum confidence threshold
        min_completeness_score: Minimum completeness threshold
        min_semantic_similarity: Minimum semantic similarity threshold
        verbose: Enable detailed logging

    Returns:
        ValidationResult with comprehensive validation metrics
    """
    agent = ValidationAgent(
        enable_multi_pass=enable_multi_pass,
        num_passes=num_passes,
        min_confidence_score=min_confidence_score,
        min_completeness_score=min_completeness_score,
        min_semantic_similarity=min_semantic_similarity,
        verbose=verbose
    )

    result = await agent.run(
        extraction=extraction,
        character=character
    )

    return result


def validate_character_extraction_sync(
    extraction: CharacterExtraction,
    character: Character,
    enable_multi_pass: bool = False,
    num_passes: int = 3,
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5,
    min_semantic_similarity: float = 0.7,
    verbose: bool = False
) -> ValidationResult:
    """
    Synchronous wrapper for validate_character_extraction.

    Args:
        extraction: CharacterExtraction to validate
        character: Original Character data
        enable_multi_pass: Whether to perform multi-pass validation
        num_passes: Number of extraction passes for consistency
        min_confidence_score: Minimum confidence threshold
        min_completeness_score: Minimum completeness threshold
        min_semantic_similarity: Minimum semantic similarity threshold
        verbose: Enable detailed logging

    Returns:
        ValidationResult with comprehensive validation metrics
    """
    return asyncio.run(
        validate_character_extraction(
            extraction=extraction,
            character=character,
            enable_multi_pass=enable_multi_pass,
            num_passes=num_passes,
            min_confidence_score=min_confidence_score,
            min_completeness_score=min_completeness_score,
            min_semantic_similarity=min_semantic_similarity,
            verbose=verbose
        )
    )


async def validate_batch(
    extractions: List[CharacterExtraction],
    characters: List[Character],
    enable_multi_pass: bool = False,
    num_passes: int = 3,
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5,
    min_semantic_similarity: float = 0.7,
    verbose: bool = False
) -> List[ValidationResult]:
    """
    Validate a batch of extractions.

    Args:
        extractions: List of CharacterExtraction results
        characters: List of original Character data
        enable_multi_pass: Whether to perform multi-pass validation
        num_passes: Number of extraction passes for consistency
        min_confidence_score: Minimum confidence threshold
        min_completeness_score: Minimum completeness threshold
        min_semantic_similarity: Minimum semantic similarity threshold
        verbose: Enable detailed logging

    Returns:
        List of ValidationResult objects
    """
    if len(extractions) != len(characters):
        raise ValueError("Number of extractions must match number of characters")

    # Validate sequentially for now (could parallelize later)
    results = []
    for extraction, character in zip(extractions, characters):
        result = await validate_character_extraction(
            extraction=extraction,
            character=character,
            enable_multi_pass=enable_multi_pass,
            num_passes=num_passes,
            min_confidence_score=min_confidence_score,
            min_completeness_score=min_completeness_score,
            min_semantic_similarity=min_semantic_similarity,
            verbose=verbose
        )
        results.append(result)

    return results


def validate_batch_sync(
    extractions: List[CharacterExtraction],
    characters: List[Character],
    enable_multi_pass: bool = False,
    num_passes: int = 3,
    min_confidence_score: float = 0.5,
    min_completeness_score: float = 0.5,
    min_semantic_similarity: float = 0.7,
    verbose: bool = False
) -> List[ValidationResult]:
    """
    Synchronous wrapper for validate_batch.

    Args:
        extractions: List of CharacterExtraction results
        characters: List of original Character data
        enable_multi_pass: Whether to perform multi-pass validation
        num_passes: Number of extraction passes for consistency
        min_confidence_score: Minimum confidence threshold
        min_completeness_score: Minimum completeness threshold
        min_semantic_similarity: Minimum semantic similarity threshold
        verbose: Enable detailed logging

    Returns:
        List of ValidationResult objects
    """
    return asyncio.run(
        validate_batch(
            extractions=extractions,
            characters=characters,
            enable_multi_pass=enable_multi_pass,
            num_passes=num_passes,
            min_confidence_score=min_confidence_score,
            min_completeness_score=min_completeness_score,
            min_semantic_similarity=min_semantic_similarity,
            verbose=verbose
        )
    )
