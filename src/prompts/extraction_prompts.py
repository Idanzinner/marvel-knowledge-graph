"""
Extraction prompts for power origin and significance extraction.
"""

POWER_ORIGIN_EXTRACTION_PROMPT = """You are an expert at analyzing Marvel character descriptions and extracting structured information about how characters acquired their powers.

Analyze the following character description and extract information about their POWER ORIGIN - how they got their powers.

CHARACTER: {character_name}

DESCRIPTION:
{description_text}

INSTRUCTIONS:
1. Identify the PRIMARY mechanism through which this character acquired their powers
2. Look for keywords and phrases such as:
   - MUTATION: "mutant", "X-gene", "born with", "genetic mutation"
   - ACCIDENT: "bitten by", "exposed to radiation", "chemical accident", "experiment gone wrong"
   - TECHNOLOGY: "suit", "armor", "device", "invention", "enhancement serum"
   - MYSTICAL: "magic", "sorcery", "enchanted", "mystical artifact", "deity"
   - COSMIC: "cosmic entity", "infinity stone", "celestial", "alien technology"
   - TRAINING: "trained", "mastered", "learned", "disciplined"
   - BIRTH: "born as", "Asgardian", "alien species", "inhuman"

3. Assign a CONFIDENCE LEVEL:
   - HIGH: The origin is explicitly stated in clear terms
   - MEDIUM: The origin is strongly implied or can be reasonably inferred
   - LOW: The origin is vague, uncertain, or requires significant inference

4. Extract EVIDENCE: Find the specific sentence(s) from the description that support your extraction

IMPORTANT:
- If multiple origins are mentioned, focus on the PRIMARY/ORIGINAL source of powers
- Be specific about the mechanism (e.g., "radioactive spider bite" not just "accident")
- If no clear origin is stated, mark as "unknown" with LOW confidence

Return your answer as a valid JSON object with this exact structure:
{{
  "type": "mutation|accident|technology|mystical|cosmic|training|birth|unknown",
  "description": "Detailed explanation of how they got their powers",
  "confidence": "high|medium|low",
  "evidence": "Direct quote or paraphrase from the description"
}}
"""


SIGNIFICANCE_EXTRACTION_PROMPT = """You are an expert at analyzing Marvel characters and understanding why their powers are significant.

Analyze the following character description and extract information about the SIGNIFICANCE and IMPACT of their powers.

CHARACTER: {character_name}

DESCRIPTION:
{description_text}

POWER ORIGIN (for context):
{power_origin_description}

INSTRUCTIONS:
1. Explain WHY their powers matter:
   - Combat effectiveness and capabilities
   - Unique abilities that set them apart
   - Their role in teams or larger conflicts
   - Threat level or protective capabilities

2. Assign an IMPACT LEVEL:
   - COSMIC: Powers that affect universes, reality, or multiverse-level threats
   - GLOBAL: Powers that can affect entire planets or civilizations
   - REGIONAL: Powers that matter at city/country level or major team operations
   - LOCAL: Powers that affect individual conflicts or small-scale operations

3. List UNIQUE CAPABILITIES:
   - What can this character do that few others can?
   - What makes their powers distinctive?
   - List 2-5 key unique abilities

4. Describe STRATEGIC VALUE (optional):
   - Why would a team want this character?
   - What role do they fill in combat or missions?

IMPORTANT:
- Base your analysis on what's stated or implied in the description
- Consider both offensive and defensive capabilities
- Think about their role in the Marvel universe context

Return your answer as a valid JSON object with this exact structure:
{{
  "why_matters": "Explanation of importance and impact of powers",
  "impact_level": "cosmic|global|regional|local",
  "unique_capabilities": ["capability1", "capability2", "capability3"],
  "strategic_value": "Optional: strategic importance to teams/missions"
}}
"""


COMBINED_EXTRACTION_PROMPT = """You are an expert at analyzing Marvel character descriptions and extracting structured information.

Analyze the following character description and extract BOTH:
1. POWER ORIGIN: How they got their powers
2. SIGNIFICANCE: Why their powers matter

CHARACTER: {character_name}

DESCRIPTION:
{description_text}

POWER ORIGIN INSTRUCTIONS:
- Identify the PRIMARY mechanism: mutation, accident, technology, mystical, cosmic, training, birth, or unknown
- Be specific about the mechanism
- Assign confidence: HIGH (explicit), MEDIUM (implied), LOW (inferred/uncertain)
- Extract evidence from the text

SIGNIFICANCE INSTRUCTIONS:
- Explain why their powers matter (combat, unique abilities, role, threat level)
- Assign impact level: cosmic, global, regional, or local
- List 2-5 unique capabilities
- Optionally describe strategic value

Return your answer as a valid JSON object with this exact structure:
{{
  "power_origin": {{
    "type": "mutation|accident|technology|mystical|cosmic|training|birth|unknown",
    "description": "Detailed explanation",
    "confidence": "high|medium|low",
    "evidence": "Direct quote from description"
  }},
  "significance": {{
    "why_matters": "Explanation of importance",
    "impact_level": "cosmic|global|regional|local",
    "unique_capabilities": ["capability1", "capability2", "capability3"],
    "strategic_value": "Optional: strategic importance"
  }}
}}
"""


def format_power_origin_prompt(character_name: str, description_text: str) -> str:
    """Format the power origin extraction prompt with character data."""
    return POWER_ORIGIN_EXTRACTION_PROMPT.format(
        character_name=character_name,
        description_text=description_text
    )


def format_significance_prompt(
    character_name: str,
    description_text: str,
    power_origin_description: str
) -> str:
    """Format the significance extraction prompt with character data."""
    return SIGNIFICANCE_EXTRACTION_PROMPT.format(
        character_name=character_name,
        description_text=description_text,
        power_origin_description=power_origin_description
    )


def format_combined_prompt(character_name: str, description_text: str) -> str:
    """Format the combined extraction prompt with character data."""
    return COMBINED_EXTRACTION_PROMPT.format(
        character_name=character_name,
        description_text=description_text
    )
