"""
Parallel extraction agent with retry logic and failure tracking.

This module provides parallel character extraction with:
- Concurrent async processing using asyncio
- Automatic retry on failures
- Comprehensive failure tracking
- Progress reporting
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from llama_index.core.llms import LLM
from tqdm.asyncio import tqdm

from ..models.character import Character
from ..models.power_origin import CharacterExtraction, OriginType, ConfidenceLevel
from .extraction_agent import extract_character


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    character_name: str
    character_id: int
    success: bool
    extraction: Optional[CharacterExtraction] = None
    error_message: Optional[str] = None
    attempts: int = 1
    duration_seconds: Optional[float] = None


@dataclass
class ExtractionSummary:
    """Summary of parallel extraction run."""
    total_characters: int
    successful: int
    failed: int
    retry_count: int
    total_duration_seconds: float
    successful_extractions: List[ExtractionResult]
    failed_extractions: List[ExtractionResult]
    timestamp: str


class ParallelExtractionAgent:
    """
    Parallel extraction agent with retry logic and failure tracking.

    Features:
    - Concurrent processing with configurable parallelism
    - Automatic retry with exponential backoff
    - Detailed failure tracking
    - Progress reporting
    - Graceful degradation (continues on failures)
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        max_concurrent: int = 5,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize parallel extraction agent.

        Args:
            llm: LLM instance to use
            max_concurrent: Maximum number of concurrent extractions
            max_retries: Maximum retry attempts per character
            retry_delay: Base delay between retries (seconds)
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose

        # Tracking
        self.results: List[ExtractionResult] = []
        self.retry_count = 0

    async def extract_single_with_retry(
        self,
        character: Character,
        semaphore: asyncio.Semaphore
    ) -> ExtractionResult:
        """
        Extract a single character with retry logic.

        Args:
            character: Character to extract
            semaphore: Semaphore for concurrency control

        Returns:
            ExtractionResult with success status and data
        """
        async with semaphore:
            start_time = datetime.now()
            attempts = 0
            last_error = None

            for attempt in range(self.max_retries):
                attempts += 1

                try:
                    if self.verbose and attempt > 0:
                        print(f"  Retry {attempt}/{self.max_retries-1} for {character.name}")

                    # Attempt extraction
                    extraction = await extract_character(
                        character=character,
                        llm=self.llm,
                        max_retries=1,  # Use external retry logic
                        verbose=False  # Suppress individual logging
                    )

                    # Check if extraction was successful
                    if extraction.power_origin.type != OriginType.UNKNOWN or extraction.power_origin.confidence != ConfidenceLevel.LOW:
                        # Success!
                        duration = (datetime.now() - start_time).total_seconds()

                        if attempt > 0:
                            self.retry_count += 1

                        return ExtractionResult(
                            character_name=character.name,
                            character_id=character.page_id,
                            success=True,
                            extraction=extraction,
                            attempts=attempts,
                            duration_seconds=duration
                        )

                    # Extraction returned UNKNOWN - might need retry
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        # Last attempt failed
                        last_error = "Extraction returned UNKNOWN type"

                except Exception as e:
                    last_error = str(e)
                    if self.verbose:
                        print(f"  Error extracting {character.name} (attempt {attempts}): {e}")

                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

            # All retries exhausted
            duration = (datetime.now() - start_time).total_seconds()

            return ExtractionResult(
                character_name=character.name,
                character_id=character.page_id,
                success=False,
                error_message=last_error,
                attempts=attempts,
                duration_seconds=duration
            )

    async def extract_batch(
        self,
        characters: List[Character],
        show_progress: bool = True
    ) -> ExtractionSummary:
        """
        Extract multiple characters in parallel.

        Args:
            characters: List of characters to extract
            show_progress: Show progress bar

        Returns:
            ExtractionSummary with results and statistics
        """
        start_time = datetime.now()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PARALLEL EXTRACTION: {len(characters)} characters")
            print(f"Max Concurrent: {self.max_concurrent} | Max Retries: {self.max_retries}")
            print(f"{'='*80}\n")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create tasks
        tasks = [
            self.extract_single_with_retry(character, semaphore)
            for character in characters
        ]

        # Run with progress bar
        if show_progress:
            results = await tqdm.gather(*tasks, desc="Extracting")
        else:
            results = await asyncio.gather(*tasks)

        # Collect results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_duration = (datetime.now() - start_time).total_seconds()

        summary = ExtractionSummary(
            total_characters=len(characters),
            successful=len(successful),
            failed=len(failed),
            retry_count=self.retry_count,
            total_duration_seconds=total_duration,
            successful_extractions=successful,
            failed_extractions=failed,
            timestamp=datetime.now().isoformat()
        )

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _print_summary(self, summary: ExtractionSummary):
        """Print extraction summary."""
        print(f"\n{'='*80}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Characters:  {summary.total_characters}")
        print(f"✓ Successful:      {summary.successful} ({summary.successful/summary.total_characters*100:.1f}%)")
        print(f"✗ Failed:          {summary.failed} ({summary.failed/summary.total_characters*100:.1f}%)")
        print(f"⟳ Retries Used:    {summary.retry_count}")
        print(f"⏱ Total Duration:  {summary.total_duration_seconds:.2f}s")
        print(f"⚡ Avg per char:    {summary.total_duration_seconds/summary.total_characters:.2f}s")

        if summary.failed > 0:
            print(f"\n{'─'*80}")
            print("FAILED EXTRACTIONS:")
            for result in summary.failed_extractions:
                print(f"  ✗ {result.character_name} (ID: {result.character_id})")
                print(f"    Attempts: {result.attempts}, Error: {result.error_message}")

        print(f"{'='*80}\n")

    def save_results(
        self,
        summary: ExtractionSummary,
        output_dir: Path,
        save_successes: bool = True,
        save_failures: bool = True
    ):
        """
        Save extraction results to files.

        Args:
            summary: Extraction summary to save
            output_dir: Output directory
            save_successes: Save successful extractions
            save_failures: Save failure log
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save successful extractions
        if save_successes and summary.successful_extractions:
            success_file = output_dir / "extractions.json"

            successes_data = [
                {
                    "character_name": r.extraction.character_name,
                    "character_id": r.extraction.character_id,
                    "power_origin": {
                        "type": r.extraction.power_origin.type.value,
                        "description": r.extraction.power_origin.description,
                        "confidence": r.extraction.power_origin.confidence.value,
                        "evidence": r.extraction.power_origin.evidence
                    },
                    "significance": {
                        "why_matters": r.extraction.significance.why_matters,
                        "impact_level": r.extraction.significance.impact_level.value,
                        "unique_capabilities": r.extraction.significance.unique_capabilities,
                        "strategic_value": r.extraction.significance.strategic_value
                    },
                    "extraction_timestamp": r.extraction.extraction_timestamp,
                    "attempts": r.attempts,
                    "duration_seconds": r.duration_seconds
                }
                for r in summary.successful_extractions
            ]

            with open(success_file, 'w') as f:
                json.dump(successes_data, f, indent=2)

            print(f"✓ Saved {len(successes_data)} successful extractions to {success_file}")

        # Save failure log
        if save_failures and summary.failed_extractions:
            failure_file = output_dir / "extraction_failures.json"

            failures_data = [
                {
                    "character_name": r.character_name,
                    "character_id": r.character_id,
                    "error_message": r.error_message,
                    "attempts": r.attempts,
                    "duration_seconds": r.duration_seconds
                }
                for r in summary.failed_extractions
            ]

            with open(failure_file, 'w') as f:
                json.dump(failures_data, f, indent=2)

            print(f"✓ Saved {len(failures_data)} failures to {failure_file}")

        # Save summary
        summary_file = output_dir / "extraction_summary.json"
        summary_data = {
            "total_characters": summary.total_characters,
            "successful": summary.successful,
            "failed": summary.failed,
            "retry_count": summary.retry_count,
            "total_duration_seconds": summary.total_duration_seconds,
            "timestamp": summary.timestamp
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"✓ Saved summary to {summary_file}")


async def extract_batch_parallel(
    characters: List[Character],
    llm: Optional[LLM] = None,
    max_concurrent: int = 5,
    max_retries: int = 3,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> ExtractionSummary:
    """
    Convenience function for parallel batch extraction.

    Args:
        characters: List of characters to extract
        llm: LLM instance
        max_concurrent: Maximum concurrent extractions
        max_retries: Maximum retry attempts
        output_dir: Output directory for results
        verbose: Enable verbose logging

    Returns:
        ExtractionSummary with results
    """
    agent = ParallelExtractionAgent(
        llm=llm,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        verbose=verbose
    )

    summary = await agent.extract_batch(characters)

    if output_dir:
        agent.save_results(summary, output_dir)

    return summary
