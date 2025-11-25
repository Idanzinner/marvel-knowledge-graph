"""
Prompts for validation tasks.
"""

CONSISTENCY_CHECK_PROMPT = """
You are a Marvel comics expert analyzing character power origins.

Compare these two extractions for the SAME character and determine if they are consistent.

Character: {character_name}

Extraction 1:
- Origin Type: {origin1_type}
- Description: {origin1_desc}
- Evidence: {origin1_evidence}

Extraction 2:
- Origin Type: {origin2_type}
- Description: {origin2_desc}
- Evidence: {origin2_evidence}

Are these extractions CONSISTENT with each other? They don't need to be identical,
but they should describe the same fundamental origin story without contradictions.

Return JSON:
{{
  "are_consistent": true/false,
  "consistency_score": 0.0-1.0,
  "differences": ["list", "of", "key", "differences"],
  "explanation": "brief explanation of your assessment"
}}
"""

GROUND_TRUTH_VALIDATION_PROMPT = """
You are a Marvel comics expert with encyclopedic knowledge.

Validate if this power origin extraction is FACTUALLY CORRECT based on your knowledge
of canonical Marvel comics lore.

Character: {character_name}

Extracted Origin:
- Type: {origin_type}
- Description: {origin_desc}

Is this extraction factually accurate according to Marvel canon?

Return JSON:
{{
  "is_accurate": true/false,
  "accuracy_score": 0.0-1.0,
  "corrections": "what should be corrected, if anything",
  "canonical_reference": "brief mention of relevant comics/storylines",
  "explanation": "your reasoning"
}}
"""

EVIDENCE_GROUNDING_PROMPT = """
Evaluate how well this extracted power origin is grounded in the provided description text.

Character: {character_name}

Original Description:
{description}

Extracted Information:
- Origin Type: {origin_type}
- Origin Description: {origin_desc}
- Evidence Quote: {evidence}

Questions:
1. Can the extracted origin be DIRECTLY traced to the description?
2. Is the evidence quote actually present in the description?
3. Are there unsupported claims in the extraction?

Return JSON:
{{
  "is_grounded": true/false,
  "grounding_score": 0.0-1.0,
  "evidence_found": true/false,
  "unsupported_claims": ["list", "of", "claims", "not", "in", "text"],
  "explanation": "your assessment"
}}
"""

RE_EXTRACTION_PROMPT = """
The previous extraction for this character had LOW CONFIDENCE or QUALITY issues.

Please perform a CAREFUL re-extraction with HIGHER attention to detail.

Character: {character_name}
Description: {description}

Previous extraction had these issues:
{issues}

Extract:
1. POWER ORIGIN: How did this character get their powers?
2. SIGNIFICANCE: Why do their powers matter?

Focus on being as SPECIFIC and ACCURATE as possible.
ONLY extract information that is EXPLICITLY stated or STRONGLY IMPLIED.
Cite direct evidence from the text.

Return structured JSON as before.
"""
