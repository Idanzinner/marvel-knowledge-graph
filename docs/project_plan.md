# Marvel Knowledge Graph + LLM Project Plan

## Project Overview
Build a hybrid AI system combining LlamaIndex Workflows and LangGraph to create a knowledge graph of Marvel characters, their genetic mutations, powers, and affiliations. The system will extract power origin information from character descriptions and validate extraction success.

## Key Objectives
1. Extract "how they got their powers" from character descriptions
2. Extract "why it matters" (significance/impact of their powers)
3. Validate LLM extraction success
4. Build multi-agent system using LlamaIndex Workflows + LangGraph
5. Create API endpoints for querying

## Architecture Components

### 1. Data Processing Pipeline
**Input**: Marvel character dataset with descriptions

**Extraction Tasks**:
- **Power Origin Extraction**: How did the character acquire their powers?
  - Genetic mutation (X-gene)
  - Radioactive accident (spider bite, gamma radiation)
  - Technology/Enhancement (Iron Man suit, Super Soldier serum)
  - Mystical/Cosmic (Sorcerer Supreme, Infinity Stones)
  - Training/Natural ability
  
- **Significance Extraction**: Why do their powers matter?
  - Combat capabilities
  - Strategic importance
  - Threat level
  - Unique abilities
  - Team role

**Validation Mechanism**:
- Confidence scoring for extractions
- Cross-validation with multiple LLM passes
- Ground truth comparison (for known characters)
- Extraction completeness check

### 2. Knowledge Graph Schema

#### Nodes
- **Character**: name, alignment, physical traits, sex, alive status, appearances
- **PowerOrigin**: origin_type, description, confidence_score
- **Power**: name, description, power_level
- **Gene/Mutation**: name, description, source
- **Team**: name, affiliation
- **SignificanceContext**: why_matters, impact_level, strategic_value

#### Relationships
- (Character) -[HAS_ORIGIN]-> (PowerOrigin)
- (Character) -[POSSESSES_POWER]-> (Power)
- (PowerOrigin) -[CONFERS]-> (Power)
- (Character) -[HAS_MUTATION]-> (Gene)
- (Gene) -[ENABLES]-> (Power)
- (Character) -[MEMBER_OF]-> (Team)
- (Character) -[HAS_SIGNIFICANCE]-> (SignificanceContext)
- (Power) -[EXTRACTION_VALIDATED]-> (ValidationResult)

### 3. Multi-Agent System Architecture

#### Agent 1: Extraction Agent (LlamaIndex Workflow)
**Responsibilities**:
- Parse character descriptions
- Extract power origins using structured prompts
- Extract significance/impact information
- Output structured data with confidence scores

**Tools**:
- LlamaIndex text splitter
- LLM-based extraction with structured output
- Retry mechanism for low-confidence extractions

#### Agent 2: Knowledge Graph Builder (LangGraph)
**Responsibilities**:
- Take extracted data and build graph structure
- Create nodes and relationships
- Handle data validation and deduplication
- Update existing nodes with new information

**State Machine**:
```
START -> Parse Extraction -> Create/Update Nodes -> Create Relationships -> Validate Graph -> END
```

#### Agent 3: Validation Agent (LlamaIndex Workflow)
**Responsibilities**:
- Verify extraction quality
- Cross-reference extracted origins with description text
- Calculate confidence/accuracy metrics
- Flag uncertain extractions for review

**Validation Methods**:
- Semantic similarity between extraction and source text
- Fact consistency checking
- Multiple-pass extraction comparison
- Ground truth validation (where available)

#### Agent 4: Query Agent (LangGraph)
**Responsibilities**:
- Process user questions
- Query knowledge graph for relevant facts
- Construct LLM prompts with graph context
- Generate natural language responses

**Query Flow**:
```
User Question -> Parse Intent -> Graph Query -> Retrieve Context -> LLM Generation -> Format Response
```

### 4. Technology Stack

**Core Frameworks**:
- **LlamaIndex**: Workflows for extraction and validation agents
- **LangGraph**: State machine for knowledge graph operations and query routing
- **Neo4j** or **NetworkX**: Graph database
- **OpenAI API**: LLM for extraction and generation

**Supporting Libraries**:
- FastAPI: REST API endpoints
- Pydantic: Data validation and structured outputs
- Redis: Caching (optional enhancement)
- Docker: Containerization

## Implementation Phases

### Phase 1: Data Preparation & Extraction (Days 1-2)
1. Load and explore the Marvel dataset
2. Design extraction prompts for:
   - Power origin identification
   - Significance/impact extraction
3. Implement Extraction Agent using LlamaIndex Workflow
4. Test extraction on sample characters
5. Build validation metrics

**Deliverable**: Extracted structured data with confidence scores

### Phase 2: Knowledge Graph Construction (Days 3-4)
1. Design complete graph schema
2. Implement Knowledge Graph Builder using LangGraph
3. Create state machine for graph operations
4. Build graph from extracted data
5. Implement graph querying functions

**Deliverable**: Populated knowledge graph with all relationships

### Phase 3: Validation System (Day 5)
1. Implement Validation Agent
2. Create validation metrics:
   - Extraction recall (did we find power origins?)
   - Extraction precision (are extractions accurate?)
   - Confidence calibration (are confidence scores reliable?)
3. Generate validation report for each character
4. Implement feedback loop for low-confidence extractions

**Deliverable**: Validation report showing extraction success rates

### Phase 4: Query & Response System (Days 6-7)
1. Implement Query Agent using LangGraph
2. Build query routing logic
3. Create context-aware prompt construction
4. Implement response generation
5. Add citation/grounding to graph facts

**Deliverable**: Working query system with graph-grounded responses

### Phase 5: API & Integration (Day 8)
1. Build FastAPI endpoints:
   - `POST /question`: Natural language queries
   - `GET /graph/{character}`: Character graph view
   - `GET /extraction-report/{character}`: Validation metrics
   - `POST /validate-extraction`: Re-validate specific extraction
2. Add request/response models
3. Implement error handling
4. Add optional caching layer

**Deliverable**: Complete REST API

### Phase 6: Testing & Documentation (Days 9-10)
1. Create comprehensive test suite
2. Document API endpoints (OpenAPI/Swagger)
3. Write README with setup instructions
4. Create sample queries and expected responses
5. Generate architecture diagram
6. Write technical explanation document

**Deliverable**: Complete documentation and test suite

## Sample Queries to Implement

### Query 1: Power Origin
**Question**: "How did Spider-Man get his powers?"

**Expected Process**:
1. Query graph for Spider-Man node
2. Retrieve HAS_ORIGIN relationship
3. Get PowerOrigin node (radioactive spider bite)
4. Construct response with graph facts + LLM narration

### Query 2: Significance Check
**Question**: "Why do Spider-Man's powers matter?"

**Expected Process**:
1. Query for Spider-Man's significance context
2. Retrieve related powers and their impacts
3. Generate response explaining strategic importance

### Query 3: Genetic Mechanism
**Question**: "What genetic mutation gives Wolverine his healing factor?"

**Expected Process**:
1. Query Wolverine -> HAS_MUTATION -> Gene
2. Follow Gene -> ENABLES -> Power (healing)
3. Provide detailed explanation with gene information

### Query 4: Extraction Validation
**Question**: "How confident are you about Magneto's power origin?"

**Expected Process**:
1. Retrieve extraction validation results
2. Show confidence scores
3. Display source text evidence
4. Indicate if validation passed

## Validation Metrics to Track

### Extraction Success Metrics
1. **Coverage**: % of characters with extracted power origins
2. **Confidence Distribution**: Histogram of confidence scores
3. **Validation Pass Rate**: % of extractions validated as accurate
4. **Missing Data Rate**: % of characters with insufficient description

### Quality Metrics
1. **Semantic Similarity**: Cosine similarity between extraction and source text
2. **Fact Consistency**: Do multiple extraction passes agree?
3. **Completeness**: Are all extraction fields populated?
4. **Groundedness**: Can extraction be directly traced to description text?

### System Metrics
1. **Query Response Time**: Average time for question answering
2. **Graph Query Performance**: Time to traverse relationships
3. **LLM Token Usage**: Monitor API costs
4. **Cache Hit Rate**: If caching is implemented

## Project Structure

```
marvel-knowledge-graph/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── data/
│   ├── raw/
│   │   └── marvel_characters.csv
│   ├── processed/
│   │   ├── extracted_origins.json
│   │   └── validation_results.json
│   └── graph/
│       └── neo4j_data/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── extraction_agent.py      # LlamaIndex Workflow
│   │   ├── graph_builder_agent.py    # LangGraph
│   │   ├── validation_agent.py       # LlamaIndex Workflow
│   │   └── query_agent.py            # LangGraph
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── operations.py
│   │   └── queries.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── character.py
│   │   ├── power_origin.py
│   │   └── validation.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── extraction_prompts.py
│   │   ├── validation_prompts.py
│   │   └── query_prompts.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py
│   │   └── metrics.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── endpoints.py
│       └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_graph_builder.py
│   ├── test_validation.py
│   └── test_api.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_extraction_testing.ipynb
│   └── 03_validation_analysis.ipynb
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── graph_schema.png
│   └── technical_explanation.md
└── examples/
    ├── sample_queries.json
    └── expected_responses.json
```

## Key Design Decisions

### Why LlamaIndex Workflows?
- Perfect for multi-step extraction pipelines
- Built-in retry and error handling
- Easy to compose complex agent behaviors
- Excellent for structured output generation

### Why LangGraph?
- State machine perfect for graph operations
- Conditional routing for complex query logic
- Persistent state management
- Great for cyclic workflows (graph traversal)

### Extraction Prompt Strategy
```python
extraction_prompt = """
Analyze the following character description and extract:

1. POWER ORIGIN: How did this character get their powers?
   - Look for: accidents, mutations, experiments, training, birth, technology
   - Be specific about the mechanism
   - Assign confidence: HIGH (explicitly stated), MEDIUM (strongly implied), LOW (inferred)

2. SIGNIFICANCE: Why do their powers matter?
   - Combat effectiveness
   - Unique capabilities
   - Strategic importance to their team
   - Threat level

Character: {character_name}
Description: {description_text}

Return a structured JSON with:
{{
  "power_origin": {{
    "type": "mutation|accident|technology|mystical|training|unknown",
    "description": "detailed origin story",
    "confidence": "high|medium|low",
    "evidence": "quote from description"
  }},
  "significance": {{
    "why_matters": "explanation",
    "impact_level": "cosmic|global|regional|local",
    "unique_capabilities": ["list", "of", "unique", "abilities"]
  }}
}}
"""
```

### Validation Strategy
1. **Extraction-Source Alignment**: Use embedding similarity to check if extraction is grounded in source text
2. **Multi-Pass Consistency**: Run extraction multiple times, check for agreement
3. **Confidence Calibration**: Track prediction confidence vs actual accuracy
4. **Human-in-the-Loop**: Flag low-confidence extractions for review

## Success Criteria

### Must Have
- [x] Extract power origins for 90%+ of characters with sufficient description
- [x] Build complete knowledge graph with all relationships
- [x] Validation system showing extraction success metrics
- [x] Working API with `/question` and `/graph/{character}` endpoints
- [x] Sample queries demonstrating graph-grounded responses

### Should Have
- [x] Confidence scores for all extractions
- [x] Validation report per character
- [x] Multi-agent coordination between LlamaIndex and LangGraph
- [x] Docker containerization
- [x] Comprehensive documentation

### Nice to Have
- [ ] Caching layer for repeated queries
- [ ] Web UI for visualization
- [ ] Graph visualization endpoint
- [ ] Batch processing for large datasets
- [ ] A/B testing different extraction prompts

## Timeline
**Total Duration**: 10 days

- **Days 1-2**: Data preparation & extraction system
- **Days 3-4**: Knowledge graph construction
- **Day 5**: Validation system
- **Days 6-7**: Query & response system
- **Day 8**: API development
- **Days 9-10**: Testing & documentation

## Next Steps
1. Set up development environment
2. Install dependencies (LlamaIndex, LangGraph, Neo4j/NetworkX)
3. Begin Phase 1: Data exploration and extraction prompt design
4. Create initial extraction agent prototype
5. Test on 5-10 sample characters

---

**Note**: This plan prioritizes the unique aspects of your requirements—extracting power origins, validating extraction success, and using both LlamaIndex Workflows and LangGraph in a complementary way.
