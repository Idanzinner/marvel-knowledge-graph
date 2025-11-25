"""
Extraction Agent using LlamaIndex Workflow.

This agent extracts power origin and significance information from character descriptions.
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
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
from pydantic import BaseModel

from ..models.power_origin import (
    CharacterExtraction,
    PowerOrigin,
    Significance,
    OriginType,
    ConfidenceLevel,
    ImpactLevel
)
from ..models.character import Character
from ..prompts.extraction_prompts import format_combined_prompt


class ExtractionInput(BaseModel):
    """Input event for extraction workflow."""
    character: Character
    retry_count: int = 0


class ExtractionEvent(Event):
    """Event carrying extraction data through workflow."""
    character: Character
    raw_response: str
    retry_count: int


class ValidationEvent(Event):
    """Event for validation step."""
    character: Character
    extraction: CharacterExtraction
    needs_retry: bool
    retry_count: int


class ExtractionAgent(Workflow):
    """
    LlamaIndex Workflow for extracting character power origins and significance.

    Workflow Steps:
    1. Prepare extraction prompt
    2. Call LLM for extraction
    3. Parse and validate response
    4. Retry if needed (low confidence or parsing errors)
    5. Return structured extraction result
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize the Extraction Agent.

        Args:
            llm: LlamaIndex LLM instance (defaults to GPT-4)
            max_retries: Maximum number of retry attempts for failed extractions
            timeout: Timeout for workflow execution
            verbose: Enable verbose logging
        """
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.0)
        self.max_retries = max_retries

    @step
    async def prepare_extraction(
        self, ctx: Context, ev: StartEvent
    ) -> ExtractionEvent:
        """
        Prepare the extraction prompt and call the LLM.

        Args:
            ctx: Workflow context
            ev: Start event with character data

        Returns:
            ExtractionEvent with raw LLM response
        """
        character = ev.get("character")
        retry_count = ev.get("retry_count", 0)

        # Check if character has description
        if not character.description_text or len(character.description_text.strip()) < 100:
            # Not enough data - return minimal extraction
            minimal_json = {
                "power_origin": {
                    "type": "unknown",
                    "description": "Insufficient description available",
                    "confidence": "low",
                    "evidence": "No detailed description found"
                },
                "significance": {
                    "why_matters": "Unknown - insufficient data",
                    "impact_level": "local",
                    "unique_capabilities": [],
                    "strategic_value": None
                }
            }
            return ExtractionEvent(
                character=character,
                raw_response=json.dumps(minimal_json),
                retry_count=retry_count
            )

        # Format the extraction prompt
        prompt = format_combined_prompt(
            character_name=character.name,
            description_text=character.description_text[:4000]  # Limit to first 4000 chars
        )

        # Call LLM
        if self._verbose:
            print(f"\n[ExtractionAgent] Extracting from {character.name} (attempt {retry_count + 1})")

        response = await self.llm.acomplete(prompt)
        raw_text = str(response)

        return ExtractionEvent(
            character=character,
            raw_response=raw_text,
            retry_count=retry_count
        )

    @step
    async def parse_and_validate(
        self, ctx: Context, ev: ExtractionEvent
    ) -> ValidationEvent:
        """
        Parse the LLM response and validate the extraction.

        Args:
            ctx: Workflow context
            ev: Extraction event with raw response

        Returns:
            ValidationEvent with parsed extraction and retry flag
        """
        character = ev.character
        raw_response = ev.raw_response
        retry_count = ev.retry_count

        # Try to parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = raw_response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)

            # Build structured extraction
            power_origin = PowerOrigin(
                type=OriginType(parsed["power_origin"]["type"]),
                description=parsed["power_origin"]["description"],
                confidence=ConfidenceLevel(parsed["power_origin"]["confidence"]),
                evidence=parsed["power_origin"]["evidence"]
            )

            significance = Significance(
                why_matters=parsed["significance"]["why_matters"],
                impact_level=ImpactLevel(parsed["significance"]["impact_level"]),
                unique_capabilities=parsed["significance"]["unique_capabilities"],
                strategic_value=parsed["significance"].get("strategic_value")
            )

            extraction = CharacterExtraction(
                character_name=character.name,
                character_id=character.page_id,
                power_origin=power_origin,
                significance=significance,
                extraction_timestamp=datetime.now().isoformat()
            )

            # Check if we should retry (low confidence and retries available)
            needs_retry = (
                power_origin.confidence == ConfidenceLevel.LOW
                and retry_count < self.max_retries
                and character.description_text
                and len(character.description_text.strip()) >= 100
            )

            if self._verbose and needs_retry:
                print(f"[ExtractionAgent] Low confidence for {character.name}, will retry")

            return ValidationEvent(
                character=character,
                extraction=extraction,
                needs_retry=needs_retry,
                retry_count=retry_count
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Parsing failed - retry if attempts remaining
            if self._verbose:
                print(f"[ExtractionAgent] Parse error for {character.name}: {e}")

            if retry_count < self.max_retries:
                # Return event to trigger retry
                return ValidationEvent(
                    character=character,
                    extraction=None,
                    needs_retry=True,
                    retry_count=retry_count
                )
            else:
                # Max retries exceeded - return minimal extraction
                extraction = CharacterExtraction(
                    character_name=character.name,
                    character_id=character.page_id,
                    power_origin=PowerOrigin(
                        type=OriginType.UNKNOWN,
                        description=f"Extraction failed after {retry_count + 1} attempts",
                        confidence=ConfidenceLevel.LOW,
                        evidence="Parsing error"
                    ),
                    significance=Significance(
                        why_matters="Unknown - extraction failed",
                        impact_level=ImpactLevel.LOCAL,
                        unique_capabilities=[]
                    ),
                    extraction_timestamp=datetime.now().isoformat()
                )
                return ValidationEvent(
                    character=character,
                    extraction=extraction,
                    needs_retry=False,
                    retry_count=retry_count
                )

    @step
    async def handle_validation(
        self, ctx: Context, ev: ValidationEvent
    ) -> StartEvent | StopEvent:
        """
        Handle validation result - either retry or finish.

        Args:
            ctx: Workflow context
            ev: Validation event

        Returns:
            StartEvent for retry or StopEvent with final result
        """
        if ev.needs_retry:
            # Trigger retry by sending new StartEvent
            return StartEvent(
                character=ev.character,
                retry_count=ev.retry_count + 1
            )
        else:
            # Extraction complete
            return StopEvent(result=ev.extraction)


async def extract_character(
    character: Character,
    llm: Optional[LLM] = None,
    max_retries: int = 2,
    verbose: bool = False
) -> CharacterExtraction:
    """
    Extract power origin and significance for a single character.

    Args:
        character: Character data
        llm: Optional LLM instance
        max_retries: Maximum retry attempts
        verbose: Enable verbose logging

    Returns:
        CharacterExtraction with structured results
    """
    workflow = ExtractionAgent(
        llm=llm,
        max_retries=max_retries,
        verbose=verbose
    )

    result = await workflow.run(
        character=character,
        retry_count=0
    )

    return result


async def extract_batch(
    characters: List[Character],
    llm: Optional[LLM] = None,
    max_retries: int = 2,
    verbose: bool = False
) -> List[CharacterExtraction]:
    """
    Extract power origins and significance for multiple characters.

    Args:
        characters: List of character data
        llm: Optional LLM instance
        max_retries: Maximum retry attempts per character
        verbose: Enable verbose logging

    Returns:
        List of CharacterExtraction results
    """
    results = []
    for character in characters:
        try:
            extraction = await extract_character(
                character=character,
                llm=llm,
                max_retries=max_retries,
                verbose=verbose
            )
            results.append(extraction)
        except Exception as e:
            if verbose:
                print(f"[ExtractionAgent] Failed to extract {character.name}: {e}")
            # Add failed extraction marker
            results.append(
                CharacterExtraction(
                    character_name=character.name,
                    character_id=character.page_id,
                    power_origin=PowerOrigin(
                        type=OriginType.UNKNOWN,
                        description=f"Extraction failed: {str(e)}",
                        confidence=ConfidenceLevel.LOW,
                        evidence="Error occurred"
                    ),
                    significance=Significance(
                        why_matters="Unknown - error occurred",
                        impact_level=ImpactLevel.LOCAL,
                        unique_capabilities=[]
                    ),
                    extraction_timestamp=datetime.now().isoformat()
                )
            )

    return results
