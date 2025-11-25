# Marvel Knowledge Graph - System Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Agent Architecture](#agent-architecture)
5. [Knowledge Graph Schema](#knowledge-graph-schema)
6. [API Architecture](#api-architecture)
7. [Technology Stack](#technology-stack)
8. [Deployment Architecture](#deployment-architecture)

---

## Overview

The Marvel Knowledge Graph system is a hybrid AI architecture combining **LlamaIndex Workflows** and **LangGraph** to create an intelligent, queryable knowledge graph of Marvel characters. The system processes character descriptions, extracts structured information about power origins, validates extraction quality, and provides a natural language query interface.

### Design Philosophy
- **Multi-Agent System**: Specialized agents for extraction, validation, graph building, and querying
- **Type-Safe**: Pydantic models throughout for runtime validation
- **Async-First**: Efficient async/await patterns for I/O operations
- **Modular**: Clear separation of concerns with well-defined interfaces
- **Production-Ready**: Comprehensive error handling, testing, and documentation

---

## System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Marvel Knowledge Graph System                         │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                               │   │
│  │                                                                   │   │
│  │  ┌───────────────────┐         ┌────────────────────┐          │   │
│  │  │  Marvel Dataset   │────────>│  Character Data    │          │   │
│  │  │    (16K chars)    │         │  (~16,000 records) │          │   │
│  │  └───────────────────┘         └────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   v                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Processing Layer                            │   │
│  │                                                                   │   │
│  │  ┌──────────────────┐     ┌──────────────────┐                 │   │
│  │  │ Extraction Agent │────>│ Validation Agent │                 │   │
│  │  │ (LlamaIndex)     │     │ (LlamaIndex)     │                 │   │
│  │  └──────────────────┘     └──────────────────┘                 │   │
│  │           │                         │                            │   │
│  │           └─────────────┬───────────┘                           │   │
│  │                         v                                        │   │
│  │              ┌──────────────────────┐                           │   │
│  │              │ Graph Builder Agent  │                           │   │
│  │              │    (LangGraph)       │                           │   │
│  │              └──────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   v                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Storage Layer                               │   │
│  │                                                                   │   │
│  │              ┌──────────────────────┐                           │   │
│  │              │  Knowledge Graph     │                           │   │
│  │              │    (NetworkX)        │                           │   │
│  │              │  23 nodes, 31 edges  │                           │   │
│  │              └──────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   v                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       Query Layer                                │   │
│  │                                                                   │   │
│  │              ┌──────────────────────┐                           │   │
│  │              │   Query Agent        │                           │   │
│  │              │   (LangGraph)        │                           │   │
│  │              └──────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   v                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        API Layer                                 │   │
│  │                                                                   │   │
│  │              ┌──────────────────────┐                           │   │
│  │              │    REST API          │                           │   │
│  │              │    (FastAPI)         │                           │   │
│  │              │  8 endpoints         │                           │   │
│  │              └──────────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   v                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Client Layer                                │   │
│  │                                                                   │   │
│  │    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │    │   Web   │  │   CLI   │  │  cURL   │  │  Other  │         │   │
│  │    │   UI    │  │  Tools  │  │ Scripts │  │ Clients │         │   │
│  │    └─────────┘  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: EXTRACTION                                              │
└─────────────────────────────────────────────────────────────────┘

Marvel Dataset (PKL)
      │
      ├─> Character Record
      │   (name, description, metadata)
      │
      v
┌──────────────────────┐
│  Extraction Agent    │  LlamaIndex Workflow
│  (LlamaIndex)        │
└──────────────────────┘
      │
      ├─> Prepare Extraction (validate, format)
      ├─> Call LLM (GPT-4o-mini, temp=0.0)
      ├─> Parse JSON Response
      ├─> Validate with Pydantic
      └─> Retry if LOW confidence
      │
      v
CharacterExtraction
  ├─> PowerOrigin (type, description, confidence, evidence)
  └─> Significance (why_matters, impact_level, capabilities)

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: VALIDATION                                              │
└─────────────────────────────────────────────────────────────────┘

CharacterExtraction + Character
      │
      v
┌──────────────────────┐
│  Validation Agent    │  LlamaIndex Workflow
│  (LlamaIndex)        │
└──────────────────────┘
      │
      ├─> Prepare Validation
      ├─> Semantic Similarity (OpenAI embeddings)
      ├─> Multi-Pass Consistency (optional)
      └─> Finalize Validation
      │
      v
ValidationResult
  ├─> is_valid, confidence_score, completeness_score
  ├─> semantic_similarity_score, consistency_score
  └─> validation_notes, quality_tier

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: GRAPH BUILDING                                          │
└─────────────────────────────────────────────────────────────────┘

CharacterExtraction + Character + ValidationResult
      │
      v
┌──────────────────────┐
│ Graph Builder Agent  │  LangGraph State Machine
│   (LangGraph)        │
└──────────────────────┘
      │
      ├─> Parse Extraction
      ├─> Create Character Node
      ├─> Create PowerOrigin Node
      ├─> Create Significance Node
      ├─> Create Power Nodes (from capabilities)
      ├─> Create Relationships (HAS_ORIGIN, POSSESSES_POWER, etc.)
      └─> Create Validation Node
      │
      v
Knowledge Graph (NetworkX)
  ├─> 7 Node Types (Character, PowerOrigin, Power, etc.)
  └─> 8 Relationship Types (HAS_ORIGIN, POSSESSES_POWER, etc.)

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: QUERYING                                                │
└─────────────────────────────────────────────────────────────────┘

Natural Language Question
      │
      v
┌──────────────────────┐
│   Query Agent        │  LangGraph State Machine
│   (LangGraph)        │
└──────────────────────┘
      │
      ├─> Parse Question
      ├─> Classify Query Type (POWER_ORIGIN, SIGNIFICANCE, etc.)
      ├─> Route to Graph Queries
      ├─> Retrieve Context (character, origin, powers, significance)
      ├─> Construct LLM Prompt with Context
      ├─> Generate Answer (GPT-4o-mini)
      └─> Format Response
      │
      v
Natural Language Answer
  ├─> answer, query_type, characters
  └─> confidence_level, context_retrieved

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: API                                                     │
└─────────────────────────────────────────────────────────────────┘

HTTP Request (/question POST)
      │
      v
┌──────────────────────┐
│    FastAPI Router    │
└──────────────────────┘
      │
      ├─> Validate Request (Pydantic)
      ├─> Call Query Agent
      └─> Format Response (Pydantic)
      │
      v
HTTP Response (JSON)
```

---

## Agent Architecture

### Agent 1: Extraction Agent (LlamaIndex Workflow)

**Purpose:** Extract structured information about power origins and significance from character descriptions

**Workflow Steps:**
```
START
  ↓
prepare_extraction
  - Validate character has description (≥100 chars)
  - Format prompt with character data
  - Truncate description to 4000 chars
  ↓
call_llm
  - Call GPT-4o-mini (temperature=0.0)
  - Request structured JSON output
  - Handle API errors
  ↓
parse_and_validate
  - Extract JSON from response
  - Validate against Pydantic models
  - Calculate completeness score
  ↓
check_confidence
  - If confidence < MEDIUM: retry
  - Max retries: 2
  - Else: proceed
  ↓
END
  - Return CharacterExtraction
```

**Key Features:**
- Async/await for efficiency
- Automatic retry for low confidence
- Graceful degradation on failure
- Verbose logging mode

**Input:**
```python
Character(
    page_id=1678,
    name="Spider-Man (Peter Parker)",
    description_text="..."
)
```

**Output:**
```python
CharacterExtraction(
    character_name="Spider-Man (Peter Parker)",
    power_origin=PowerOrigin(
        type=OriginType.ACCIDENT,
        description="Bitten by radioactive spider...",
        confidence=ConfidenceLevel.HIGH,
        evidence="Peter Parker gained his powers..."
    ),
    significance=Significance(
        why_matters="Protects NYC...",
        impact_level=ImpactLevel.LOCAL,
        unique_capabilities=[...]
    )
)
```

---

### Agent 2: Graph Builder Agent (LangGraph)

**Purpose:** Construct knowledge graph from extracted data

**State Machine:**
```
START
  ↓
parse_extraction
  - Validate input data
  - Check for required fields
  ↓
create_character_node
  - Create Character node with metadata
  - Generate deterministic ID: character_{page_id}
  ↓
create_origin_node
  - Create PowerOrigin node
  - Generate ID: origin_{character_id}_{origin_type}
  ↓
create_significance_node
  - Create Significance node
  - Generate ID: significance_{character_id}
  ↓
create_power_nodes
  - For each unique_capability:
  -   Create Power node (deduplicated by name hash)
  -   Generate ID: power_{hash(name)}
  ↓
create_relationships
  - Character -[HAS_ORIGIN]-> PowerOrigin
  - Character -[HAS_SIGNIFICANCE]-> Significance
  - Character -[POSSESSES_POWER]-> Power (×N)
  - PowerOrigin -[CONFERS]-> Power (×N)
  ↓
validate_graph
  - Check graph integrity
  - Create Validation node
  - Generate ID: validation_{character_id}
  ↓
END
  - Return node IDs created
```

**State Definition:**
```python
class GraphBuilderState(TypedDict):
    # Input
    extraction: Optional[CharacterExtraction]
    character: Optional[Character]

    # Processing
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

---

### Agent 3: Validation Agent (LlamaIndex Workflow)

**Purpose:** Validate extraction quality using semantic similarity and consistency checks

**Workflow Steps:**
```
START
  ↓
prepare_validation
  - Validate inputs
  - Get character description
  ↓
check_semantic_similarity
  - Generate extraction embedding (OpenAI)
  - Generate description embedding
  - Calculate cosine similarity
  - Compare to threshold (0.7)
  ↓
check_multi_pass_consistency (optional)
  - Run extraction N times (default: 3)
  - Compare origin types
  - Compare descriptions (embedding similarity)
  - Calculate consistency score
  ↓
finalize_validation
  - Combine all metrics
  - Generate validation flags
  - Determine pass/fail
  - Calculate overall quality score
  - Assign quality tier (HIGH/MEDIUM/LOW)
  ↓
END
  - Return ValidationResult
```

**Validation Metrics:**
- **Confidence Score**: 0.33 (LOW), 0.66 (MEDIUM), 1.0 (HIGH)
- **Completeness Score**: 0-1 based on field population
- **Semantic Similarity**: 0-1 cosine similarity
- **Consistency Score**: 0-1 agreement across passes
- **Overall Quality**: Weighted average (30% conf, 30% comp, 40% sim)

---

### Agent 4: Query Agent (LangGraph)

**Purpose:** Answer natural language questions using graph context

**State Machine:**
```
START
  ↓
parse_question
  - Tokenize question
  - Extract character names
  ↓
classify_query_type
  - POWER_ORIGIN: "how did X get powers?"
  - SIGNIFICANCE: "why do X powers matter?"
  - POWER_ABILITIES: "what powers does X have?"
  - GENERAL: other questions
  ↓
route_to_graph_queries
  - If POWER_ORIGIN: get_character_power_origin()
  - If SIGNIFICANCE: get_character_significance()
  - If POWER_ABILITIES: get_character_powers()
  - If GENERAL: get_character_full_profile()
  ↓
retrieve_context
  - Execute graph queries
  - Collect character, origin, powers, significance
  ↓
construct_prompt
  - Build prompt with graph context
  - Include retrieved facts
  - Add query-specific instructions
  ↓
generate_answer
  - Call GPT-4o-mini (temperature=0.3)
  - Generate natural language response
  ↓
format_response
  - Structure response with metadata
  - Include query_type, characters
  - Add confidence_level if available
  ↓
END
  - Return QuestionResponse
```

**Query Routing:**
```python
def classify_query_type(question: str) -> QueryType:
    question_lower = question.lower()

    if any(kw in question_lower for kw in ["how", "get", "acquire", "obtain"]):
        if "power" in question_lower:
            return QueryType.POWER_ORIGIN

    if any(kw in question_lower for kw in ["why", "matter", "significant", "important"]):
        return QueryType.SIGNIFICANCE

    if any(kw in question_lower for kw in ["what power", "abilities", "can do"]):
        return QueryType.POWER_ABILITIES

    return QueryType.GENERAL
```

---

## Knowledge Graph Schema

### Node Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        Node Types                                │
└─────────────────────────────────────────────────────────────────┘

CHARACTER
├─ node_id: character_{page_id}
├─ name: string
├─ page_id: int
├─ alignment: string ("Good", "Bad", "Neutral")
├─ sex: string
├─ alive: string
├─ appearances: float
├─ first_appearance: string
└─ year: float

POWER_ORIGIN
├─ node_id: origin_{character_id}_{origin_type}
├─ origin_type: enum (accident, technology, birth, mutation, etc.)
├─ description: string
├─ confidence: enum (high, medium, low)
└─ evidence: string (quote from source)

POWER
├─ node_id: power_{hash(name)}
├─ name: string
└─ description: string

SIGNIFICANCE
├─ node_id: significance_{character_id}
├─ why_matters: string
├─ impact_level: enum (cosmic, global, regional, local)
├─ unique_capabilities: list[string]
└─ strategic_value: string

VALIDATION
├─ node_id: validation_{character_id}
├─ is_valid: bool
├─ confidence_score: float (0-1)
├─ completeness_score: float (0-1)
├─ semantic_similarity_score: float (0-1, optional)
└─ validation_notes: list[string]

GENE (future)
├─ node_id: gene_{hash(name)}
├─ name: string
├─ description: string
└─ source: string

TEAM (future)
├─ node_id: team_{hash(name)}
├─ name: string
└─ affiliation_type: string
```

### Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                     Relationship Types                           │
└─────────────────────────────────────────────────────────────────┘

Character -[HAS_ORIGIN]-> PowerOrigin
  - Describes how the character got their powers

Character -[HAS_SIGNIFICANCE]-> Significance
  - Describes why the character's powers matter

Character -[POSSESSES_POWER]-> Power
  - Character has this specific power/ability

PowerOrigin -[CONFERS]-> Power
  - This origin grants these specific powers

Character -[HAS_MUTATION]-> Gene (future)
  - Character has this genetic mutation

Gene -[ENABLES]-> Power (future)
  - Gene enables these powers

Character -[MEMBER_OF]-> Team (future)
  - Character is member of this team

PowerOrigin -[EXTRACTION_VALIDATED]-> Validation
  - Links extraction to its validation results
```

### Graph Visualization

```
Spider-Man (Peter Parker) [Character]
    │
    ├──[HAS_ORIGIN]──> Accident Origin [PowerOrigin]
    │                      │
    │                      ├──[CONFERS]──> Wall-crawling [Power]
    │                      ├──[CONFERS]──> Super strength [Power]
    │                      ├──[CONFERS]──> Spider-sense [Power]
    │                      └──[CONFERS]──> Web-shooting [Power]
    │
    ├──[POSSESSES_POWER]──> Wall-crawling [Power]
    ├──[POSSESSES_POWER]──> Super strength [Power]
    ├──[POSSESSES_POWER]──> Spider-sense [Power]
    ├──[POSSESSES_POWER]──> Web-shooting [Power]
    │
    ├──[HAS_SIGNIFICANCE]──> NYC Protector [Significance]
    │
    └──[EXTRACTION_VALIDATED]──> Validation Result [Validation]
```

---

## API Architecture

### FastAPI Application Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
└─────────────────────────────────────────────────────────────────┘

src/api/main.py
├─ FastAPI app instance
├─ Lifespan management
│  ├─ Startup: Load graph, initialize query agent
│  └─ Shutdown: Cleanup resources
├─ CORS middleware
├─ Request logging middleware
└─ Include routers

src/api/endpoints.py
├─ Global state (graph_ops, graph_queries, query_agent)
├─ Route handlers (8 endpoints)
│  ├─ GET  /health
│  ├─ GET  /
│  ├─ POST /question
│  ├─ GET  /graph/{character}
│  ├─ GET  /extraction-report/{character}
│  ├─ POST /validate-extraction
│  ├─ GET  /characters
│  └─ GET  /stats
└─ Error handling (HTTPException)

src/api/models.py
├─ Request models (Pydantic)
│  ├─ QuestionRequest
│  └─ ValidationRequest
├─ Response models (Pydantic)
│  ├─ QuestionResponse
│  ├─ CharacterGraphResponse
│  ├─ ExtractionReportResponse
│  ├─ ValidationResponse
│  ├─ HealthResponse
│  └─ ErrorResponse
└─ Node models (Pydantic)
   ├─ CharacterNode
   ├─ PowerOriginNode
   ├─ PowerNode
   ├─ SignificanceNode
   └─ ValidationNode
```

### Request/Response Flow

```
Client Request (HTTP)
  │
  v
┌──────────────────────┐
│  ASGI Server         │  Uvicorn
│  (Async)             │
└──────────────────────┘
  │
  v
┌──────────────────────┐
│  FastAPI Router      │
└──────────────────────┘
  │
  ├─> Request Validation (Pydantic)
  │   - Parse JSON body
  │   - Validate types
  │   - Return 400 if invalid
  │
  v
┌──────────────────────┐
│  Endpoint Handler    │
└──────────────────────┘
  │
  ├─> Business Logic
  │   - Query graph (graph_ops, graph_queries)
  │   - Call Query Agent (query_agent)
  │   - Process data
  │
  ├─> Error Handling
  │   - 404: Resource not found
  │   - 500: Internal error
  │   - Log errors
  │
  v
┌──────────────────────┐
│  Response Formation  │
└──────────────────────┘
  │
  ├─> Response Validation (Pydantic)
  │   - Validate response model
  │   - Serialize to JSON
  │
  v
HTTP Response (JSON)
```

---

## Technology Stack

### Core Framework Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Technology Stack                              │
└─────────────────────────────────────────────────────────────────┘

AI/ML Frameworks
├─ LlamaIndex (0.11.20+)
│  ├─ llama-index-core: Core functionality
│  ├─ llama-index-llms-openai: OpenAI LLM integration
│  └─ llama-index-embeddings-openai: Embedding generation
├─ LangGraph (0.2.0+)
│  └─ State machine framework
└─ LangChain (0.3.0+)
   └─ Base framework for LangGraph

Web Framework
├─ FastAPI (0.115.6)
│  ├─ Modern async web framework
│  ├─ Automatic OpenAPI docs
│  └─ Pydantic integration
└─ Uvicorn (0.34.0)
   └─ ASGI server

Data Processing
├─ Pandas (2.2.3)
│  └─ DataFrame operations
├─ Numpy (2.0.2)
│  └─ Numerical operations
└─ Pydantic (2.10.3)
   └─ Data validation

Graph Database
└─ NetworkX (3.4.2)
   ├─ In-memory graph operations
   └─ GraphML export

External APIs
└─ OpenAI API
   ├─ GPT-4o-mini: Text generation
   └─ text-embedding-3-small: Embeddings

Supporting Libraries
├─ Python-dotenv (1.0.1): Environment variables
├─ Python-multipart (0.0.20): Form data
└─ Requests (2.31.0): HTTP client
```

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────────────────────────────┐
│                   Local Development Setup                        │
└─────────────────────────────────────────────────────────────────┘

Developer Machine
├─ Python 3.12 Virtual Environment
├─ Source Code (/Users/hadaszinner/sandbox/marvel_knowledge_grpah/)
├─ Data Files (data/marvel-wikia-data-with-descriptions.pkl)
├─ Environment Variables (../.env)
│  └─ OPENAI_API_KEY
├─ NetworkX Graph (in-memory)
└─ FastAPI Server (localhost:8000)
   ├─ Uvicorn ASGI server
   └─ Single worker (development mode)
```

### Production Deployment (Proposed)

```
┌─────────────────────────────────────────────────────────────────┐
│                 Production Deployment Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│         Load Balancer               │
│     (e.g., AWS ALB, nginx)          │
└─────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────┐
│    Application Servers (4 workers)   │
│                                      │
│  ┌────────────┐  ┌────────────┐    │
│  │  Worker 1  │  │  Worker 2  │    │
│  │  FastAPI   │  │  FastAPI   │    │
│  └────────────┘  └────────────┘    │
│  ┌────────────┐  ┌────────────┐    │
│  │  Worker 3  │  │  Worker 4  │    │
│  │  FastAPI   │  │  FastAPI   │    │
│  └────────────┘  └────────────┘    │
│                                      │
│  Gunicorn + Uvicorn Workers          │
└─────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────┐
│         Graph Database               │
│     (Neo4j or NetworkX)              │
│  - Persistent storage                │
│  - Optimized for graph queries       │
└─────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────┐
│         Caching Layer                │
│          (Redis)                     │
│  - Query result caching              │
│  - Rate limiting                     │
│  - Session management                │
└─────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────┐
│      External Services               │
│                                      │
│  ┌──────────────────────────────┐  │
│  │     OpenAI API                │  │
│  │  - GPT-4o-mini (generation)   │  │
│  │  - Embeddings (validation)    │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Environment variables
ENV GRAPH_PATH=/app/data/processed/marvel_knowledge_graph.graphml
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run application
CMD ["python", "-m", "src.api.main"]
```

```yaml
# docker-compose.yml (proposed)
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GRAPH_PATH=/app/data/marvel_knowledge_graph.graphml
      - LLM_MODEL=gpt-4o-mini
      - LLM_TEMPERATURE=0.3
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/Polazin2!
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  neo4j_data:
```

---

## Scalability Considerations

### Current System (3 characters, 23 nodes, 31 edges)
- **Memory Usage:** ~200 MB
- **Response Time:** 100-200ms (graph queries), 2-5s (LLM queries)
- **Throughput:** 10-20 concurrent requests
- **Cost:** ~$0.001-0.003 per question

### Scaling to 16,000 Characters
- **Memory Usage:** ~1-2 GB (NetworkX in-memory)
- **Graph Load Time:** 10-20 seconds (startup)
- **Node Count:** ~100,000+ nodes
- **Edge Count:** ~150,000+ edges
- **Mitigation:** Migrate to Neo4j for better performance

### Optimization Strategies
1. **Caching:** Redis for frequent queries (~50% hit rate expected)
2. **Parallel Processing:** Batch extraction with asyncio.gather (~4-8x speedup)
3. **Database:** Neo4j for large graphs (~2-3x faster queries)
4. **CDN:** Static assets (docs, examples)
5. **Rate Limiting:** Prevent abuse, manage costs
6. **Monitoring:** Prometheus + Grafana for observability

---

## Security Considerations

### Current State (Development)
- No authentication
- No rate limiting
- Open CORS (*)
- API keys in environment variables

### Production Recommendations
1. **Authentication:** OAuth2 + JWT tokens
2. **Rate Limiting:** Per-user/IP limits (100 req/hour)
3. **CORS:** Whitelist specific domains
4. **Input Validation:** Already implemented (Pydantic)
5. **API Key Rotation:** Regular rotation schedule
6. **HTTPS:** TLS/SSL certificates
7. **Secrets Management:** AWS Secrets Manager, Azure Key Vault
8. **Logging:** Centralized logging (ELK stack)
9. **Monitoring:** Anomaly detection, alerting

---

## Monitoring & Observability

### Metrics to Track
1. **Application Metrics**
   - Request rate (requests/second)
   - Response time (p50, p95, p99)
   - Error rate (4xx, 5xx)
   - Active users/sessions

2. **Business Metrics**
   - Questions asked (by type)
   - Characters queried (popularity)
   - Extraction quality (confidence distribution)
   - Validation pass rate

3. **Infrastructure Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network bandwidth

4. **External API Metrics**
   - OpenAI API latency
   - OpenAI API cost
   - Token usage
   - Error rates

### Logging Strategy
```python
import logging

# Structured logging
logger.info(
    "Question answered",
    extra={
        "question": question,
        "query_type": query_type,
        "characters": characters,
        "response_time_ms": response_time,
        "llm_tokens": token_count
    }
)
```

---

## Disaster Recovery & Backup

### Data Persistence
1. **Graph Backups:** Daily GraphML exports to S3/GCS
2. **Extraction Results:** Versioned JSON exports
3. **Validation Reports:** Archived per run
4. **Database Snapshots:** Neo4j backup schedule

### Recovery Procedures
1. **Graph Corruption:** Restore from latest GraphML backup
2. **API Failure:** Automatic restart (Docker/K8s)
3. **Data Loss:** Re-run extraction pipeline from source data
4. **External API Outage:** Queue requests, retry with exponential backoff

---

**End of Architecture Document**

For implementation details, see [technical_explanation.md](technical_explanation.md).
For API usage, see [README_PHASE5.md](../README_PHASE5.md).
