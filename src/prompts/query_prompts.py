"""
Query processing prompts for the Marvel Knowledge Graph Query Agent.

Provides prompts for query classification, context-aware response generation,
and citation-grounded answers.
"""

# ============================================================================
# Query Classification
# ============================================================================

QUERY_CLASSIFICATION_PROMPT = """
Analyze the following user question and classify it into one of these categories:

Categories:
1. POWER_ORIGIN: Questions about how a character got their powers
   - Examples: "How did Spider-Man get his powers?", "What gave Hulk his abilities?"

2. POWER_ABILITIES: Questions about what powers/abilities a character has
   - Examples: "What can Magneto do?", "What powers does Storm have?"

3. SIGNIFICANCE: Questions about why powers matter or their impact
   - Examples: "Why are Wolverine's powers important?", "What makes Captain Marvel powerful?"

4. GENETIC: Questions about genetic mutations, X-gene, or hereditary abilities
   - Examples: "What mutation does Rogue have?", "Is Cyclops a mutant?"

5. TEAM: Questions about team affiliations or group membership
   - Examples: "Is Spider-Man an Avenger?", "Who is in the X-Men?"

6. VALIDATION: Questions about extraction confidence or data quality
   - Examples: "How confident are you about Magneto's origin?", "Is this information accurate?"

7. COMPARISON: Questions comparing multiple characters
   - Examples: "Who is stronger, Thor or Hulk?", "How are Wolverine and Deadpool similar?"

8. GENERAL: General questions about a character or broad inquiries
   - Examples: "Tell me about Iron Man", "Who is Thanos?"

User Question: {question}

Return ONLY the category name (e.g., POWER_ORIGIN, POWER_ABILITIES, etc.)
Category:"""

# ============================================================================
# Entity Extraction
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """
Extract the character name(s) from the following question.

Question: {question}

Instructions:
- Return the full character name (e.g., "Spider-Man" not just "Spider")
- If multiple characters, return all of them separated by commas
- If no character is mentioned, return "NONE"
- Handle common variations (e.g., "Spidey" -> "Spider-Man")

Character Name(s):"""

# ============================================================================
# Response Generation
# ============================================================================

RESPONSE_GENERATION_PROMPT = """
You are an expert on Marvel characters with access to a knowledge graph.

User Question: {question}

Retrieved Information from Knowledge Graph:
{context}

Validation Confidence:
{validation_info}

Instructions:
1. Answer the question using ONLY the provided context from the knowledge graph
2. Be specific and cite relevant details (origin type, power descriptions, etc.)
3. If confidence is LOW or MEDIUM, acknowledge uncertainty
4. If information is missing, state that clearly - DO NOT make up information
5. Keep your answer concise (2-4 sentences unless more detail is needed)
6. Use natural, conversational language

Answer:"""

# ============================================================================
# Citation-Grounded Response
# ============================================================================

CITATION_RESPONSE_PROMPT = """
You are an expert on Marvel characters with access to a knowledge graph.

User Question: {question}

Retrieved Information:
{context}

Validation Details:
- Extraction Confidence: {confidence}
- Completeness Score: {completeness}
- Semantic Similarity: {similarity}
- Overall Quality: {quality}

Instructions:
1. Answer the question using the provided context
2. Include inline citations referencing the source:
   - For power origins: [Origin: {origin_type}]
   - For powers: [Power: {power_name}]
   - For significance: [Impact: {impact_level}]
3. Add a confidence statement based on validation scores:
   - HIGH confidence (≥0.8): "This information is highly reliable."
   - MEDIUM confidence (0.6-0.8): "This information is moderately reliable."
   - LOW confidence (<0.6): "This information should be verified."
4. If validation scores are low, explicitly mention: "Note: This extraction has [low/medium] confidence."
5. DO NOT fabricate information not present in the context
6. Be specific and detailed while remaining conversational

Answer:"""

# ============================================================================
# Multi-Character Comparison
# ============================================================================

COMPARISON_PROMPT = """
You are comparing multiple Marvel characters based on knowledge graph data.

User Question: {question}

Character Data:
{characters_context}

Instructions:
1. Compare the characters based on the question's focus (powers, origins, significance, etc.)
2. Use specific details from the knowledge graph
3. Structure your comparison clearly (e.g., similarities first, then differences)
4. Cite confidence levels if they differ significantly between characters
5. Be objective and fact-based
6. Keep the comparison concise but informative

Comparison:"""

# ============================================================================
# Error/Missing Data Response
# ============================================================================

NO_DATA_RESPONSE_PROMPT = """
The user asked: "{question}"

However, we could not find the character "{character_name}" in the knowledge graph.

Generate a helpful response that:
1. Acknowledges the character is not in our database
2. Suggests they might try:
   - A different spelling or variation of the name
   - A more well-known character
3. Offers to answer questions about characters we do have data for
4. Be friendly and conversational

Response:"""

INSUFFICIENT_DATA_PROMPT = """
The user asked: "{question}"

We found the character "{character_name}" but the available information is insufficient to answer this question.

Available information:
{available_context}

Generate a helpful response that:
1. Shares what information IS available
2. Acknowledges what information is missing
3. Be honest about the limitations
4. Offer to answer other questions about this character if more data exists

Response:"""

# ============================================================================
# Query Routing Helper
# ============================================================================

def get_query_type_description(query_type: str) -> str:
    """
    Get a human-readable description of a query type.

    Args:
        query_type: Query type category

    Returns:
        Description string
    """
    descriptions = {
        "POWER_ORIGIN": "power origin and acquisition",
        "POWER_ABILITIES": "powers and abilities",
        "SIGNIFICANCE": "significance and impact",
        "GENETIC": "genetic mutations and heredity",
        "TEAM": "team affiliations",
        "VALIDATION": "data validation and confidence",
        "COMPARISON": "character comparison",
        "GENERAL": "general information"
    }
    return descriptions.get(query_type, "information")
