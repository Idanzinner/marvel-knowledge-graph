# Marvel Knowledge Graph - Technical Explanation

This document provides a deep technical explanation of how the Marvel Knowledge Graph system works, including implementation details, code examples, and design rationale.

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Models](#data-models)
3. [Extraction Agent Deep Dive](#extraction-agent-deep-dive)
4. [Graph Builder Deep Dive](#graph-builder-deep-dive)
5. [Validation Agent Deep Dive](#validation-agent-deep-dive)
6. [Query Agent Deep Dive](#query-agent-deep-dive)
7. [API Implementation](#api-implementation)
8. [Key Technical Decisions](#key-technical-decisions)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Overview

The Marvel Knowledge Graph is a **hybrid AI system** that combines two complementary frameworks:

### LlamaIndex Workflows
Used for **linear, multi-step pipelines** with automatic retry logic:
- **Extraction Agent**: Character description → Structured data
- **Validation Agent**: Structured data → Quality metrics

**Why LlamaIndex?**
- Event-driven architecture perfect for ETL pipelines
- Built-in retry mechanism for unreliable LLM calls
- Type-safe with async/await support
- Easy to debug with verbose mode

### LangGraph
Used for **state machines and complex routing**:
- **Graph Builder Agent**: Structured data → Knowledge graph
- **Query Agent**: Natural language → Graph query → Answer

**Why LangGraph?**
- State machine ideal for graph construction workflows
- Conditional routing for query classification
- Persistent state management
- Great for cyclic workflows (graph traversal)

---

## Data Models

All data models use **Pydantic V2** for runtime validation and type safety.

### Character Model

```python
from pydantic import BaseModel, Field
from typing import Optional

class Character(BaseModel):
    """Marvel character from dataset."""
    page_id: int = Field(description="Unique character ID")
    name: str = Field(description="Full character name")
    urlslug: Optional[str] = Field(default=None)
    id_type: Optional[str] = Field(default=None, alias="ID")
    align: Optional[str] = Field(default=None, alias="ALIGN")
    eye: Optional[str] = Field(default=None, alias="EYE")
    hair: Optional[str] = Field(default=None, alias="HAIR")
    sex: Optional[str] = Field(default=None, alias="SEX")
    gsm: Optional[str] = Field(default=None, alias="GSM")
    alive: Optional[str] = Field(default=None, alias="ALIVE")
    appearances: Optional[float] = Field(default=None)
    first_appearance: Optional[str] = Field(default=None)
    year: Optional[float] = Field(default=None)
    description_text: Optional[str] = Field(default=None)

    class Config:
        populate_by_name = True  # Allow both field name and alias
```

**Key Implementation Detail:** Custom validation for pandas DataFrame NaN values

```python
@classmethod
def model_validate(cls, obj: Any) -> "Character":
    """Custom validation to handle pandas NaN values."""
    if hasattr(obj, "to_dict"):  # pandas Series
        row_dict = obj.to_dict()
        # Convert NaN to None for Pydantic validation
        for key, value in row_dict.items():
            if pd.isna(value):
                row_dict[key] = None
        return super().model_validate(row_dict)
    return super().model_validate(obj)
```

**Why This Matters:**
- Pandas uses `float('nan')` for missing values
- Pydantic's `Optional[str]` expects `str | None`, not `float`
- Must convert NaN → None before validation

### Extraction Models

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class OriginType(str, Enum):
    """Power origin classifications."""
    MUTATION = "mutation"
    ACCIDENT = "accident"
    TECHNOLOGY = "technology"
    MYSTICAL = "mystical"
    COSMIC = "cosmic"
    TRAINING = "training"
    BIRTH = "birth"
    UNKNOWN = "unknown"

class ConfidenceLevel(str, Enum):
    """Extraction confidence levels."""
    HIGH = "high"      # Explicitly stated in text
    MEDIUM = "medium"  # Strongly implied
    LOW = "low"        # Inferred or unclear

class PowerOrigin(BaseModel):
    """How a character got their powers."""
    type: OriginType
    description: str = Field(min_length=10, max_length=500)
    confidence: ConfidenceLevel
    evidence: str = Field(description="Quote from source text")

class Significance(BaseModel):
    """Why a character's powers matter."""
    why_matters: str = Field(min_length=10, max_length=500)
    impact_level: ImpactLevel  # cosmic, global, regional, local
    unique_capabilities: List[str] = Field(min_items=1, max_items=20)
    strategic_value: Optional[str] = Field(default=None)

class CharacterExtraction(BaseModel):
    """Complete extraction result."""
    character_name: str
    character_id: Optional[int] = None
    power_origin: PowerOrigin
    significance: Significance
    extraction_timestamp: Optional[str] = None
```

**Design Decision:** Enums for categorical values
- Type-safe at runtime
- Auto-validation (only valid values allowed)
- Self-documenting code
- Easy to extend

---

## Extraction Agent Deep Dive

The Extraction Agent is a **LlamaIndex Workflow** that extracts structured information from character descriptions.

### Workflow Definition

```python
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step
)
from llama_index.llms.openai import OpenAI
import json

class ExtractionWorkflow(Workflow):
    """Extract power origins and significance from character descriptions."""

    def __init__(self, llm: Optional[OpenAI] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0.0)

    @step
    async def prepare_extraction(
        self, ev: StartEvent
    ) -> PrepareEvent:
        """Validate input and format extraction prompt."""
        character = ev.character

        # Validate sufficient description
        if not character.description_text or len(character.description_text) < 100:
            return StopEvent(result=self._minimal_extraction(character))

        # Truncate long descriptions (cost optimization)
        description = character.description_text[:4000]

        # Format prompt with character data
        prompt = EXTRACTION_PROMPT.format(
            character_name=character.name,
            description_text=description
        )

        return PrepareEvent(
            character=character,
            prompt=prompt,
            attempt=1
        )

    @step
    async def call_llm(self, ev: PrepareEvent) -> LLMResponseEvent:
        """Call LLM to extract structured data."""
        try:
            response = await self.llm.acomplete(ev.prompt)
            return LLMResponseEvent(
                character=ev.character,
                response=response.text,
                attempt=ev.attempt
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return StopEvent(result=self._minimal_extraction(ev.character))

    @step
    async def parse_and_validate(
        self, ev: LLMResponseEvent
    ) -> Union[ValidationEvent, RetryEvent, StopEvent]:
        """Parse JSON and validate with Pydantic."""
        try:
            # Extract JSON from response (handles markdown code blocks)
            json_str = self._extract_json(ev.response)
            data = json.loads(json_str)

            # Validate with Pydantic
            extraction = CharacterExtraction(
                character_name=ev.character.name,
                character_id=ev.character.page_id,
                power_origin=PowerOrigin(**data["power_origin"]),
                significance=Significance(**data["significance"]),
                extraction_timestamp=datetime.now().isoformat()
            )

            # Calculate completeness and confidence scores
            completeness = self._calculate_completeness(extraction)
            confidence = self._map_confidence(extraction.power_origin.confidence)

            # Check if retry needed
            if confidence < 0.66 and ev.attempt < MAX_RETRIES:
                return RetryEvent(
                    character=ev.character,
                    attempt=ev.attempt + 1,
                    reason="Low confidence"
                )

            return StopEvent(result=extraction)

        except Exception as e:
            logger.error(f"Parse failed: {e}")
            if ev.attempt < MAX_RETRIES:
                return RetryEvent(...)
            return StopEvent(result=self._minimal_extraction(ev.character))

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response (handles markdown)."""
        # Remove markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        return response.strip()

    def _calculate_completeness(self, extraction: CharacterExtraction) -> float:
        """Calculate how complete the extraction is."""
        score = 0.0
        total_fields = 7

        if extraction.power_origin.description:
            score += 1
        if extraction.power_origin.evidence:
            score += 1
        if extraction.significance.why_matters:
            score += 1
        if extraction.significance.unique_capabilities:
            score += 1
        if len(extraction.significance.unique_capabilities) >= 3:
            score += 1
        if extraction.significance.strategic_value:
            score += 1
        if extraction.power_origin.confidence == ConfidenceLevel.HIGH:
            score += 1

        return score / total_fields

    def _minimal_extraction(self, character: Character) -> CharacterExtraction:
        """Fallback extraction when all retries fail."""
        return CharacterExtraction(
            character_name=character.name,
            character_id=character.page_id,
            power_origin=PowerOrigin(
                type=OriginType.UNKNOWN,
                description="Insufficient data",
                confidence=ConfidenceLevel.LOW,
                evidence="N/A"
            ),
            significance=Significance(
                why_matters="Unknown",
                impact_level=ImpactLevel.LOCAL,
                unique_capabilities=["Unknown"]
            )
        )
```

### Prompt Engineering

The extraction prompt is **carefully engineered** for structured output:

```python
EXTRACTION_PROMPT = """
Analyze the following Marvel character description and extract:

1. POWER ORIGIN: How did this character get their powers?
   - Look for keywords: "bitten by", "exposed to", "born with", "trained",
     "experiment", "serum", "radiation", "genetic", "mystical"
   - Be specific about the mechanism
   - Assign confidence based on clarity:
     * HIGH: Explicitly stated with details
     * MEDIUM: Strongly implied or partial info
     * LOW: Inferred or very unclear

2. SIGNIFICANCE: Why do their powers matter?
   - Combat effectiveness and capabilities
   - Unique abilities that others don't have
   - Strategic importance to teams or world
   - Scope of impact (local, global, cosmic)

Character: {character_name}

Description:
{description_text}

Return a valid JSON object with this EXACT structure:
{{
  "power_origin": {{
    "type": "mutation|accident|technology|mystical|cosmic|training|birth|unknown",
    "description": "Detailed origin story in 1-3 sentences",
    "confidence": "high|medium|low",
    "evidence": "Direct quote from description supporting your extraction"
  }},
  "significance": {{
    "why_matters": "Explanation of significance in 1-3 sentences",
    "impact_level": "cosmic|global|regional|local",
    "unique_capabilities": ["capability 1", "capability 2", "..."],
    "strategic_value": "Optional: Strategic importance (can be null)"
  }}
}}

IMPORTANT:
- Return ONLY valid JSON, no additional text
- Use actual quotes from the description for evidence
- Be concise but specific
- If information is missing, use your best judgment and mark confidence LOW
"""
```

**Prompt Engineering Principles:**
1. **Clear Instructions**: Tell LLM exactly what to extract
2. **Examples**: Show expected format with `{{}}` syntax
3. **Keywords**: Help LLM identify relevant info
4. **Confidence Criteria**: Define HIGH/MEDIUM/LOW clearly
5. **Evidence Requirement**: Force grounding in source text
6. **JSON Schema**: Show exact structure expected

### Usage Example

```python
# Create workflow
workflow = ExtractionWorkflow(
    llm=OpenAI(model="gpt-4o-mini", temperature=0.0)
)

# Run extraction
result = await workflow.run(
    character=character,
    max_retries=2
)

# Result is a CharacterExtraction object
print(f"Origin: {result.power_origin.type}")
print(f"Confidence: {result.power_origin.confidence}")
print(f"Significance: {result.significance.why_matters}")
```

---

## Graph Builder Deep Dive

The Graph Builder is a **LangGraph state machine** that constructs the knowledge graph from extracted data.

### State Definition

```python
from typing import TypedDict, Optional, List

class GraphBuilderState(TypedDict):
    """State for graph building workflow."""
    # Input
    extraction: Optional[CharacterExtraction]
    character: Optional[Character]

    # Processing (node IDs created)
    character_id: Optional[str]
    origin_id: Optional[str]
    significance_id: Optional[str]
    power_ids: List[str]
    validation_id: Optional[str]

    # Control
    error: Optional[str]
    completed: bool
    verbose: bool
```

**Why TypedDict?**
- Type hints for IDE autocomplete
- Runtime type checking with LangGraph
- Self-documenting state structure
- Easy to add new fields

### State Machine Implementation

```python
from langgraph.graph import StateGraph, END
from src.graph.operations import GraphOperations
from src.utils.metrics import calculate_validation_metrics

def create_graph_builder_workflow(
    graph_ops: GraphOperations,
    verbose: bool = False
) -> StateGraph:
    """Create LangGraph state machine for graph building."""

    workflow = StateGraph(GraphBuilderState)

    # Define nodes (workflow steps)
    workflow.add_node("parse_extraction", parse_extraction_node)
    workflow.add_node("create_character_node", create_character_node)
    workflow.add_node("create_origin_node", create_origin_node)
    workflow.add_node("create_significance_node", create_significance_node)
    workflow.add_node("create_power_nodes", create_power_nodes)
    workflow.add_node("create_relationships", create_relationships)
    workflow.add_node("validate_graph", validate_graph_node)

    # Define edges (workflow flow)
    workflow.set_entry_point("parse_extraction")
    workflow.add_edge("parse_extraction", "create_character_node")
    workflow.add_edge("create_character_node", "create_origin_node")
    workflow.add_edge("create_origin_node", "create_significance_node")
    workflow.add_edge("create_significance_node", "create_power_nodes")
    workflow.add_edge("create_power_nodes", "create_relationships")
    workflow.add_edge("create_relationships", "validate_graph")
    workflow.add_edge("validate_graph", END)

    return workflow.compile()

# Node implementations

def parse_extraction_node(state: GraphBuilderState) -> GraphBuilderState:
    """Validate input data."""
    if not state["extraction"] or not state["character"]:
        return {**state, "error": "Missing required data", "completed": True}

    if state["verbose"]:
        print("[GraphBuilder] Parsing extraction data...")

    return state

def create_character_node(state: GraphBuilderState) -> GraphBuilderState:
    """Create Character node in graph."""
    character = state["character"]

    if state["verbose"]:
        print(f"[GraphBuilder] Creating character node for {character.name}...")

    # Create node with deterministic ID
    character_id = graph_ops.add_character_node(
        name=character.name,
        page_id=character.page_id,
        alignment=character.align,
        sex=character.sex,
        alive=character.alive,
        appearances=character.appearances,
        first_appearance=character.first_appearance,
        year=character.year
    )

    return {**state, "character_id": character_id}

def create_origin_node(state: GraphBuilderState) -> GraphBuilderState:
    """Create PowerOrigin node in graph."""
    extraction = state["extraction"]
    character_id = state["character_id"]

    if state["verbose"]:
        print("[GraphBuilder] Creating power origin node...")

    # Create origin node
    origin_id = graph_ops.add_power_origin_node(
        character_id=character_id,
        origin_type=extraction.power_origin.type.value,
        description=extraction.power_origin.description,
        confidence=extraction.power_origin.confidence.value,
        evidence=extraction.power_origin.evidence
    )

    return {**state, "origin_id": origin_id}

def create_power_nodes(state: GraphBuilderState) -> GraphBuilderState:
    """Create Power nodes for each unique capability."""
    extraction = state["extraction"]
    power_ids = []

    if state["verbose"]:
        print(f"[GraphBuilder] Creating {len(extraction.significance.unique_capabilities)} power nodes...")

    for capability in extraction.significance.unique_capabilities:
        power_id = graph_ops.add_power_node(
            name=capability,
            description=f"Power: {capability}"
        )
        power_ids.append(power_id)

    return {**state, "power_ids": power_ids}

def create_relationships(state: GraphBuilderState) -> GraphBuilderState:
    """Create all relationships between nodes."""
    character_id = state["character_id"]
    origin_id = state["origin_id"]
    significance_id = state["significance_id"]
    power_ids = state["power_ids"]

    if state["verbose"]:
        print("[GraphBuilder] Creating relationships...")

    # Character -[HAS_ORIGIN]-> PowerOrigin
    graph_ops.add_relationship(
        character_id, origin_id, RelationType.HAS_ORIGIN
    )

    # Character -[HAS_SIGNIFICANCE]-> Significance
    graph_ops.add_relationship(
        character_id, significance_id, RelationType.HAS_SIGNIFICANCE
    )

    # Character -[POSSESSES_POWER]-> Power (for each power)
    for power_id in power_ids:
        graph_ops.add_relationship(
            character_id, power_id, RelationType.POSSESSES_POWER
        )

    # PowerOrigin -[CONFERS]-> Power (for each power)
    for power_id in power_ids:
        graph_ops.add_relationship(
            origin_id, power_id, RelationType.CONFERS
        )

    return state

def validate_graph_node(state: GraphBuilderState) -> GraphBuilderState:
    """Validate extraction and create Validation node."""
    extraction = state["extraction"]
    character_id = state["character_id"]
    origin_id = state["origin_id"]

    # Calculate validation metrics
    validation_metrics = calculate_validation_metrics(extraction)

    # Create validation node
    validation_id = graph_ops.add_validation_node(
        character_id=character_id,
        is_valid=validation_metrics["is_valid"],
        confidence_score=validation_metrics["confidence_score"],
        completeness_score=validation_metrics["completeness_score"],
        validation_notes=validation_metrics.get("notes", [])
    )

    # Link to origin
    graph_ops.add_relationship(
        origin_id, validation_id, RelationType.EXTRACTION_VALIDATED
    )

    if state["verbose"]:
        print("[GraphBuilder] Validation node created")

    return {**state, "validation_id": validation_id, "completed": True}
```

### Deterministic Node IDs

**Critical Design Decision:** Use deterministic IDs for reproducibility

```python
def generate_node_id(node_type: NodeType, **kwargs) -> str:
    """Generate deterministic node ID."""
    if node_type == NodeType.CHARACTER:
        page_id = kwargs["page_id"]
        return f"character_{page_id}"

    elif node_type == NodeType.POWER_ORIGIN:
        character_id = kwargs["character_id"]
        origin_type = kwargs["origin_type"]
        return f"origin_{character_id}_{origin_type}"

    elif node_type == NodeType.POWER:
        name = kwargs["name"]
        # Hash name for deduplication
        name_hash = hashlib.md5(name.lower().encode()).hexdigest()[:8]
        return f"power_{name_hash}"

    elif node_type == NodeType.SIGNIFICANCE:
        character_id = kwargs["character_id"]
        return f"significance_{character_id}"

    elif node_type == NodeType.VALIDATION:
        character_id = kwargs["character_id"]
        return f"validation_{character_id}"

    else:
        # Fallback: UUID
        return f"{node_type.value}_{uuid.uuid4().hex[:8]}"
```

**Benefits:**
- Idempotent: Running twice produces same graph
- Natural deduplication: Same power name = same node
- Easy to query: Can construct IDs programmatically
- Deterministic testing: Predictable for assertions

---

## Validation Agent Deep Dive

The Validation Agent uses **semantic similarity** and **consistency checking** to validate extraction quality.

### Semantic Similarity Check

```python
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np

class ValidationAgent:
    """Validate extraction quality."""

    def __init__(self):
        self.embedding_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )

    async def check_semantic_similarity(
        self,
        extraction: CharacterExtraction,
        description_text: str
    ) -> float:
        """Calculate cosine similarity between extraction and source."""

        # Create extraction summary
        extraction_text = (
            f"Origin: {extraction.power_origin.description}. "
            f"Significance: {extraction.significance.why_matters}. "
            f"Powers: {', '.join(extraction.significance.unique_capabilities)}"
        )

        # Generate embeddings
        extraction_embedding = await self.embedding_model.aget_text_embedding(
            extraction_text
        )
        description_embedding = await self.embedding_model.aget_text_embedding(
            description_text[:4000]  # Truncate for API limits
        )

        # Calculate cosine similarity
        similarity = self._cosine_similarity(
            extraction_embedding,
            description_embedding
        )

        return similarity

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return float(dot_product / (norm1 * norm2))
```

**Why Semantic Similarity?**
- Checks if extraction is **grounded** in source text
- Captures **meaning**, not just word overlap
- Works with **paraphrases** and **summaries**
- Industry standard for text similarity

**Threshold Tuning:**
- **≥ 0.7**: HIGH similarity (strict)
- **0.6-0.7**: MEDIUM similarity (acceptable)
- **< 0.6**: LOW similarity (review needed)

**Note:** Sample data shows ~0.54-0.67 similarity because:
1. Source descriptions are very long (100K+ chars)
2. Extractions are concise summaries (100-500 chars)
3. Threshold may need adjustment to 0.6 for production

### Multi-Pass Consistency Check

```python
async def check_multi_pass_consistency(
    self,
    character: Character,
    num_passes: int = 3
) -> float:
    """Run extraction multiple times and check agreement."""

    extractions = []

    # Run extraction N times
    for i in range(num_passes):
        extraction = await extract_character(character, verbose=False)
        extractions.append(extraction)

    # Check origin type agreement
    origin_types = [e.power_origin.type for e in extractions]
    type_agreement = len(set(origin_types)) == 1  # All same?

    # Check description similarity (pairwise embeddings)
    descriptions = [e.power_origin.description for e in extractions]
    similarities = []

    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            emb1 = await self.embedding_model.aget_text_embedding(descriptions[i])
            emb2 = await self.embedding_model.aget_text_embedding(descriptions[j])
            sim = self._cosine_similarity(emb1, emb2)
            similarities.append(sim)

    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Combined consistency score
    consistency = (
        (1.0 if type_agreement else 0.0) * 0.4 +  # 40% weight on type
        avg_similarity * 0.6  # 60% weight on description
    )

    return consistency
```

**Why Multi-Pass?**
- Checks extraction **reliability**
- Catches **hallucinations**
- Provides **confidence signal**

**Tradeoff:**
- **Cost:** 3x extraction cost per character
- **Time:** 3x extraction time
- **Value:** Strong confidence signal

**Recommendation:** Use selectively for critical characters or when confidence is uncertain.

---

## Query Agent Deep Dive

The Query Agent uses **LangGraph** to route natural language questions to appropriate graph queries.

### Query Classification

```python
from enum import Enum

class QueryType(str, Enum):
    """Types of questions the system can answer."""
    POWER_ORIGIN = "POWER_ORIGIN"
    SIGNIFICANCE = "SIGNIFICANCE"
    POWER_ABILITIES = "POWER_ABILITIES"
    CHARACTER_INFO = "CHARACTER_INFO"
    GENERAL = "GENERAL"

def classify_query(question: str) -> QueryType:
    """Classify question into query type."""
    question_lower = question.lower()

    # Power origin questions
    if any(kw in question_lower for kw in [
        "how did", "get powers", "got powers", "acquire", "obtain",
        "origin", "where from", "how come"
    ]):
        return QueryType.POWER_ORIGIN

    # Significance questions
    if any(kw in question_lower for kw in [
        "why", "matter", "significant", "important", "useful",
        "why does", "what makes"
    ]):
        return QueryType.SIGNIFICANCE

    # Power abilities questions
    if any(kw in question_lower for kw in [
        "what powers", "abilities", "can do", "capable of",
        "powers does", "what can"
    ]):
        return QueryType.POWER_ABILITIES

    # Character info questions
    if any(kw in question_lower for kw in [
        "who is", "tell me about", "information about",
        "describe", "profile"
    ]):
        return QueryType.CHARACTER_INFO

    return QueryType.GENERAL
```

### State Machine

```python
from langgraph.graph import StateGraph, END

def create_query_agent_workflow(
    graph_queries: GraphQueries,
    llm: OpenAI
) -> StateGraph:
    """Create query agent state machine."""

    workflow = StateGraph(QueryAgentState)

    # Add nodes
    workflow.add_node("parse_question", parse_question_node)
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("extract_character_names", extract_names_node)
    workflow.add_node("route_to_graph", route_to_graph_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("format_response", format_response_node)

    # Define flow
    workflow.set_entry_point("parse_question")
    workflow.add_edge("parse_question", "classify_query")
    workflow.add_edge("classify_query", "extract_character_names")
    workflow.add_edge("extract_character_names", "route_to_graph")
    workflow.add_edge("route_to_graph", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()
```

### Context-Aware Answer Generation

```python
def generate_answer_node(state: QueryAgentState) -> QueryAgentState:
    """Generate answer using LLM with graph context."""

    question = state["question"]
    query_type = state["query_type"]
    context = state["context"]

    # Build prompt with graph facts
    prompt = f"""
You are an expert on Marvel characters. Answer the following question using
ONLY the provided factual information from the knowledge graph.

Question: {question}

Factual Information:
{context}

Instructions:
- Be concise (2-3 sentences)
- Use specific details from the facts
- Don't make up information not in the facts
- If information is missing, say so

Answer:"""

    # Call LLM
    response = llm.complete(prompt)

    return {**state, "answer": response.text}
```

**Key Principle:** Ground answers in graph facts, don't hallucinate.

---

## API Implementation

### FastAPI Application Setup

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("Loading knowledge graph...")
    global graph_ops, graph_queries, query_agent

    graph_ops = GraphOperations.load_graph(GRAPH_PATH)
    graph_queries = GraphQueries(graph_ops)
    query_agent = QueryAgent(graph_queries, llm=OpenAI(...))

    print("API ready!")

    yield

    # Shutdown
    print("Shutting down...")

# Create app
app = FastAPI(
    title="Marvel Knowledge Graph API",
    description="Query Marvel character powers and origins",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Endpoint Implementation

```python
from fastapi import APIRouter, Query
from src.api.models import QuestionRequest, QuestionResponse

router = APIRouter()

@router.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Answer natural language question about Marvel characters."""
    try:
        # Call query agent
        result = query_agent.query(
            question=request.question,
            verbose=False
        )

        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            query_type=result.get("query_type", "UNKNOWN"),
            characters=result.get("characters", []),
            confidence_level="UNKNOWN",
            context_retrieved=bool(result.get("context"))
        )

    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )

@router.get("/graph/{character}", response_model=CharacterGraphResponse)
async def get_character_graph(
    character: str,
    search_by: str = Query("name", enum=["name", "character_id"])
):
    """Get full character graph profile."""
    try:
        # Find character
        if search_by == "name":
            char = graph_queries.find_character_by_name(character)
        else:
            char = graph_queries.get_character_by_id(character)

        if not char:
            raise HTTPException(404, f"Character not found: {character}")

        # Get full profile
        profile = graph_queries.get_character_full_profile(char["node_id"])

        # Parse GraphML lists (workaround for serialization)
        profile["significance"] = parse_graphml_lists(profile["significance"])

        # Validate with Pydantic
        return CharacterGraphResponse(
            character=CharacterNode(**profile["character"]),
            power_origin=PowerOriginNode(**profile["power_origin"]),
            powers=[PowerNode(**p) for p in profile["powers"]],
            significance=SignificanceNode(**profile["significance"]),
            validation=ValidationNode(**profile["validation"]) if profile["validation"] else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        raise HTTPException(500, detail=str(e))
```

### GraphML List Parsing Workaround

**Problem:** NetworkX GraphML serialization converts Python lists to strings:
```python
['Wall-crawling', 'Spider-sense']  # Python list
"['Wall-crawling', 'Spider-sense']"  # Stored in GraphML
```

**Solution:** Pre-process data before Pydantic validation:
```python
import ast

def parse_graphml_lists(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse string representations of lists from GraphML."""
    list_fields = ["unique_capabilities", "validation_notes"]

    for field in list_fields:
        if field in data and isinstance(data[field], str):
            if data[field].startswith("["):
                try:
                    data[field] = ast.literal_eval(data[field])
                except:
                    data[field] = []

    return data

# Apply BEFORE Pydantic validation
profile["significance"] = parse_graphml_lists(profile["significance"])
SignificanceNode(**profile["significance"])  # Now works!
```

**Lesson:** Process data transformations BEFORE Pydantic, not during.

---

## Key Technical Decisions

### 1. Pydantic V2 for All Models
**Why:** Type safety, automatic validation, JSON serialization
**Tradeoff:** Learning curve, version compatibility issues

### 2. Async/Await Throughout
**Why:** Efficient I/O operations (LLM calls, embeddings)
**Tradeoff:** Complexity, harder to debug

### 3. NetworkX Over Neo4j (for now)
**Why:** Simpler setup, faster prototyping, GraphML export
**Tradeoff:** Limited scalability, no persistence
**Future:** Migrate to Neo4j for production

### 4. Temperature=0.0 for Extraction
**Why:** Deterministic, reproducible results
**Tradeoff:** Less creative, may miss edge cases

### 5. Single-Pass Extraction (Origin + Significance)
**Why:** Faster, cheaper, significance has origin context
**Tradeoff:** Longer prompts, more complex parsing

---

## Performance Optimization

### Current Performance (3 characters)
- **Extraction:** ~2-5s per character
- **Graph Building:** ~1-2s per character
- **Validation:** ~1-2s per character (semantic only)
- **Query:** ~2-5s (with LLM call)

### Optimization Strategies

#### 1. Parallel Extraction
```python
import asyncio

# Sequential (current)
for character in characters:
    result = await extract_character(character)

# Parallel (optimized)
tasks = [extract_character(c) for c in characters]
results = await asyncio.gather(*tasks)
# Expected speedup: 4-8x (limited by API rate limits)
```

#### 2. Embedding Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str) -> List[float]:
    """Cache embeddings for repeated text."""
    return embedding_model.get_text_embedding(text)
```

#### 3. Redis Caching for Queries
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def cached_query(question: str) -> Dict:
    """Cache query results in Redis."""
    cache_key = f"query:{hashlib.md5(question.encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query and cache
    result = query_agent.query(question)
    redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL

    return result
```

#### 4. Neo4j for Large Graphs
```python
from neo4j import GraphDatabase

class Neo4jGraphOperations:
    """Graph operations using Neo4j."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_character_node(self, **kwargs):
        """Add character using Cypher query."""
        with self.driver.session() as session:
            result = session.run("""
                CREATE (c:Character {
                    node_id: $node_id,
                    name: $name,
                    page_id: $page_id,
                    alignment: $alignment
                })
                RETURN c.node_id
            """, **kwargs)
            return result.single()["c.node_id"]
```

Expected benefits:
- **Query Speed:** 2-3x faster for complex traversals
- **Scalability:** Handle 16,000+ characters efficiently
- **Persistence:** No need to load graph on startup
- **Visualization:** Built-in Neo4j Browser

---

## Troubleshooting Guide

### Common Issues

#### Issue 1: pandas NaN validation error
```
ValidationError: Input should be a valid string, got NaN
```

**Solution:**
```python
# Convert NaN to None before validation
row_dict = row.to_dict()
for key, value in row_dict.items():
    if pd.isna(value):
        row_dict[key] = None
```

#### Issue 2: GraphML list serialization
```
ValidationError: Input should be a valid list, got string
```

**Solution:**
```python
import ast

def parse_graphml_lists(data: Dict[str, Any]) -> Dict[str, Any]:
    for field in ["unique_capabilities", "validation_notes"]:
        if field in data and isinstance(data[field], str):
            if data[field].startswith("["):
                data[field] = ast.literal_eval(data[field])
    return data
```

#### Issue 3: Low semantic similarity scores
```
Validation failed: semantic similarity 0.54 < 0.7
```

**Explanation:** Expected for long source texts + concise extractions

**Solution:** Adjust threshold to 0.6 or make configurable per origin type

#### Issue 4: OpenAI API rate limits
```
RateLimitError: Rate limit exceeded
```

**Solution:**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
async def call_llm_with_retry(prompt: str) -> str:
    return await llm.acomplete(prompt)
```

---

**End of Technical Explanation**

For architecture overview, see [architecture.md](architecture.md).
For API usage, see [README_PHASE5.md](../README_PHASE5.md).
