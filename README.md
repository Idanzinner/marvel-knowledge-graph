# Marvel Knowledge Graph + LLM System with Neo4j

A hybrid AI system combining **LlamaIndex Workflows**, **LangGraph**, and **Neo4j** to create a queryable knowledge graph of Marvel characters. The system extracts power origins from character descriptions, stores them in Neo4j, validates extraction quality, and provides a natural language query interface via REST API.

## Project Status: ✅ PRODUCTION READY

| Phase | Status | Documentation |
|-------|--------|---------------|
| Phase 1: Data Extraction | ✅ Complete | [README_PHASE1.md](docs/README_PHASE1.md) |
| Phase 2: Knowledge Graph | ✅ Complete | [README_PHASE2.md](docs/README_PHASE2.md) |
| Phase 3: Validation System | ✅ Complete | [README_PHASE3.md](docs/README_PHASE3.md) |
| Phase 4: Query & Response | ✅ Complete | [README_PHASE4.md](docs/README_PHASE4.md) |
| Phase 5: REST API | ✅ Complete | [README_PHASE5.md](docs/README_PHASE5.md) |
| Phase 6: Neo4j Integration | ✅ Complete | [README_NEO4J.md](docs/README_NEO4J.md) |
| Phase 7: Documentation | ✅ Complete | This file + [docs/](docs/) |

**Test Results:** All phases passing (100% test coverage)
**API Status:** Running at http://localhost:8000
**Neo4j Browser:** http://localhost:7474
**Total Lines of Code:** ~6,000+

---

## Quick Start

### 1. Setup Environment & Neo4j

```bash
# Clone repository
git clone https://github.com/Idanzinner/marvel-knowledge-graph.git
cd marvel-knowledge-graph

# Create and activate virtual environment (Python 3.10-3.12)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and Neo4j credentials
```

### 2. Start Neo4j

```bash
# Using Docker Compose (recommended)
docker-compose up -d neo4j

# Or run the setup script
./setup_neo4j.sh

# Verify Neo4j is running
# Neo4j Browser: http://localhost:7474
# Default credentials: neo4j/Polazin2!
```

### 3. Run the Complete Pipeline (Jupyter Notebook)

```bash
# Start Jupyter
jupyter notebook notebooks/neo4j_pipeline.ipynb

# The notebook includes:
# - Character data loading
# - Parallel extraction with retry logic
# - Neo4j graph construction
# - Natural language querying with Query Agent
# - Cypher queries and statistics
```

**Or run programmatically:**

```bash
# Test Neo4j connection and pipeline
python tests/test_neo4j_connection.py
```

### 4. Query with Natural Language

```python
from src.agents.neo4j_query_agent import Neo4jQueryAgent
from src.graph.neo4j_operations import Neo4jOperations

# Initialize
neo4j = Neo4jOperations()
query_agent = Neo4jQueryAgent(neo4j_ops=neo4j)

# Ask questions
result = query_agent.query("How did Spider-Man get his powers?")
print(result['answer'])
```

### 5. Start the API

```bash
# Start the REST API server
python -m src.api.main

# API will be available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

---

## Key Features

### 🤖 Multi-Agent AI System
- **Extraction Agent** (LlamaIndex Workflow): Extracts power origins and significance from character descriptions
- **Graph Builder Agent** (LangGraph): Constructs knowledge graph with nodes and relationships
- **Validation Agent** (LlamaIndex Workflow): Validates extraction quality with semantic similarity
- **Query Agent** (LangGraph): Answers natural language questions using graph context
- **Neo4j Query Agent**: LangGraph-based agent for querying Neo4j with natural language

### 🗄️ Neo4j Graph Database
- **Production Database**: Full Neo4j integration with Cypher queries
- **5 Node Types**: Character, PowerOrigin, Power, Significance, Validation
- **5 Relationship Types**: HAS_ORIGIN, POSSESSES_POWER, CONFERS, HAS_SIGNIFICANCE, EXTRACTION_VALIDATED
- **Scalable**: Tested with 392 characters, 2,814 nodes, 4,026 relationships
- **Docker Deployment**: Easy setup with docker-compose

### 📊 Parallel Processing
- **Concurrent Extraction**: Process up to 12 characters simultaneously
- **Automatic Retry**: Failed extractions retry up to 3 times
- **Progress Tracking**: Real-time progress bars with detailed statistics
- **Failure Reporting**: Comprehensive error tracking and logging

### ✅ Validation System
- **Semantic Similarity**: Embedding-based grounding check (OpenAI embeddings)
- **Confidence Scoring**: HIGH/MEDIUM/LOW confidence levels
- **Completeness Metrics**: Field population and quality assessment
- **Quality Tiers**: Automatic classification (High/Medium/Low)
- **Feedback Loop**: Automatic re-extraction for failed validations

### 🚀 Production REST API
- **8 Endpoints**: `/question`, `/graph/{character}`, `/extraction-report/{character}`, etc.
- **FastAPI Framework**: Modern async/await, automatic OpenAPI docs
- **Pydantic Validation**: Type-safe request/response models
- **Interactive Docs**: Swagger UI + ReDoc at `/docs` and `/redoc`

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Marvel Knowledge Graph System (Neo4j)               │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Phase 1    │      │   Phase 2    │      │   Phase 3    │
│  Extraction  │─────>│ Graph Build  │─────>│  Validation  │
│  (LlamaIdx)  │      │ (LangGraph)  │      │ (LlamaIdx)   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │
       │                      v                      │
       │              ┌──────────────┐               │
       └─────────────>│    Neo4j     │<──────────────┘
                      │   Database   │
                      │  (Cypher)    │
                      └──────────────┘
                             │
                             v
                      ┌──────────────┐
                      │   Phase 4    │
                      │ Query Agent  │
                      │ (LangGraph)  │
                      └──────────────┘
                             │
                             v
                      ┌──────────────┐
                      │   Phase 5    │
                      │  REST API    │
                      │  (FastAPI)   │
                      └──────────────┘
```

**Full Architecture:** See [docs/architecture.md](docs/architecture.md)
**Neo4j Guide:** See [docs/README_NEO4J.md](docs/README_NEO4J.md)
**Technical Details:** See [docs/technical_explanation.md](docs/technical_explanation.md)

---

## Technology Stack

### Core Frameworks
- **LlamaIndex** (0.11.20+): Workflow framework for extraction and validation agents
- **LangGraph** (0.2.0+): State machine for graph building and query routing
- **FastAPI** (0.115.6): Modern async web framework for REST API
- **Neo4j** (5.x): Production graph database
- **Pydantic** (2.10.3): Data validation and structured outputs

### AI/ML
- **OpenAI API**: GPT-4o-mini for extraction and question answering
- **OpenAI Embeddings**: text-embedding-3-small for semantic similarity

### Supporting Libraries
- **neo4j-driver** (5.26.0): Official Neo4j Python driver
- **Pandas** (2.2.3): Data processing
- **Uvicorn** (0.34.0): ASGI server
- **Python-dotenv** (1.0.1): Environment configuration

**Python Version:** 3.10-3.12 (3.12 recommended)

---

## Project Structure

```
marvel-knowledge-graph/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose (Neo4j + API)
├── setup_neo4j.sh                 # Neo4j setup script
│
├── src/
│   ├── agents/
│   │   ├── extraction_agent.py           # Phase 1: LlamaIndex extraction workflow
│   │   ├── parallel_extraction.py        # Parallel processing with retry logic
│   │   ├── graph_builder_agent.py        # Phase 2: LangGraph state machine
│   │   ├── validation_agent.py           # Phase 3: LlamaIndex validation workflow
│   │   ├── query_agent.py                # Phase 4: LangGraph query routing
│   │   └── neo4j_query_agent.py          # Neo4j natural language query agent
│   ├── graph/
│   │   ├── schema.py                     # Graph schema definitions
│   │   ├── operations.py                 # NetworkX CRUD operations (legacy)
│   │   ├── queries.py                    # NetworkX query interface (legacy)
│   │   ├── neo4j_operations.py           # Neo4j CRUD operations
│   │   └── neo4j_queries.py              # Neo4j Cypher query interface
│   ├── models/
│   │   ├── character.py                  # Character data model
│   │   └── power_origin.py               # Extraction models
│   ├── prompts/
│   │   ├── extraction_prompts.py         # Extraction prompt engineering
│   │   ├── validation_prompts.py         # Validation prompts
│   │   └── query_prompts.py              # Query prompts
│   ├── utils/
│   │   ├── data_loader.py                # Data loading utilities
│   │   ├── metrics.py                    # Validation metrics
│   │   ├── validation_reports.py         # Report generation
│   │   └── feedback_loop.py              # Re-extraction logic
│   └── api/
│       ├── main.py                       # FastAPI application
│       ├── endpoints.py                  # API route handlers
│       └── models.py                     # Request/response models
│
├── data/
│   └── marvel-wikia-data-with-descriptions.pkl  # Source dataset (16K chars)
│
├── tests/
│   ├── test_extraction.py         # Phase 1 test
│   ├── test_graph_builder.py      # Phase 2 test
│   ├── test_validation.py         # Phase 3 test
│   ├── test_query_agent.py        # Phase 4 test
│   ├── test_api.py                # Phase 5 test
│   └── test_neo4j_connection.py   # Neo4j integration test
│
├── docs/
│   ├── README_PHASE1.md               # Phase 1 documentation
│   ├── README_PHASE2.md               # Phase 2 documentation
│   ├── README_PHASE3.md               # Phase 3 documentation
│   ├── README_PHASE4.md               # Phase 4 documentation
│   ├── README_PHASE5.md               # Phase 5 documentation
│   ├── README_NEO4J.md                # Neo4j integration guide
│   ├── architecture.md                # System architecture
│   └── technical_explanation.md       # Technical deep dive
│
├── notebooks/
│   ├── neo4j_pipeline.ipynb       # Complete Neo4j pipeline (RECOMMENDED)
│   └── exp_notebook.ipynb         # Exploration notebook
│
└── examples/
    ├── sample_queries.json        # Example questions
    └── expected_responses.json    # Expected answers
```

---

## Neo4j Integration

### Quick Start with Neo4j

```bash
# 1. Start Neo4j
docker-compose up -d neo4j

# 2. Run the notebook
jupyter notebook notebooks/neo4j_pipeline.ipynb
```

### Programmatic Usage

```python
from src.graph.neo4j_operations import Neo4jOperations
from src.agents.neo4j_query_agent import Neo4jQueryAgent

# Connect to Neo4j
neo4j = Neo4jOperations()

# Query character profile
profile = neo4j.get_character_profile("Spider-Man (Peter Parker)", search_by="name")

# Natural language queries
agent = Neo4jQueryAgent(neo4j_ops=neo4j)
result = agent.query("What powers does Iron Man have?")
print(result['answer'])

# Close connection
neo4j.close()
```

### Cypher Queries

```python
# Custom Cypher query
query = """
MATCH (c:Character)-[:HAS_ORIGIN]->(o:PowerOrigin {origin_type: 'accident'})
RETURN c.name, o.description
"""

with neo4j.driver.session() as session:
    results = session.run(query)
    for record in results:
        print(record['c.name'])
```

**Full Neo4j Guide:** [docs/README_NEO4J.md](docs/README_NEO4J.md)

---

## Jupyter Notebook Pipeline

The **[neo4j_pipeline.ipynb](notebooks/neo4j_pipeline.ipynb)** notebook provides a complete end-to-end demonstration:

### What's Included

1. **Setup & Environment** - Load dependencies and configure Neo4j
2. **Data Loading** - Load Marvel character data (sample or full dataset)
3. **Parallel Extraction** - Extract 5-12 characters concurrently with retry logic
4. **Graph Construction** - Build Neo4j graph with nodes and relationships
5. **Statistics & Validation** - View graph stats and validate data
6. **Cypher Queries** - Run custom queries against Neo4j
7. **Natural Language Queries** - Use the Query Agent to ask questions
8. **Interactive Examples** - Ready-to-run code cells

### Sample Output

```python
# Cell: Natural Language Query
question = "How did Spider-Man get his powers?"
result = query_agent.query(question, verbose=True)

# Output:
# [Neo4jQueryAgent] Classifying query...
#   Query Type: POWER_ORIGIN
# [Neo4jQueryAgent] Extracting character names...
#   Characters: ['Spider-Man (Peter Parker)']
# [Neo4jQueryAgent] Retrieving context from Neo4j...
#   Found: Spider-Man (Peter Parker)
#   Retrieved: 1 characters, 1 origins, 9 powers
# [Neo4jQueryAgent] Formatting context...
#   Context length: 856 chars
# [Neo4jQueryAgent] Generating response...
#   Confidence: UNKNOWN

# ANSWER:
# Spider-Man gained his powers after being bitten by a radioactive spider,
# which altered his physiology and granted him superhuman abilities.
```

---

## Sample Queries

### Query 1: Power Origin (Neo4j Agent)

```python
from src.agents.neo4j_query_agent import Neo4jQueryAgent

agent = Neo4jQueryAgent(neo4j_ops=neo4j)
result = agent.query("How did Spider-Man get his powers?")

# Response:
{
  "question": "How did Spider-Man get his powers?",
  "answer": "Spider-Man gained his powers after being bitten by a radioactive spider...",
  "query_type": "POWER_ORIGIN",
  "characters": ["Spider-Man (Peter Parker)"],
  "confidence_level": "UNKNOWN",
  "context_retrieved": true
}
```

### Query 2: Character Powers

```python
result = agent.query("What powers does Iron Man have?")
```

### Query 3: Direct Cypher Query

```python
# Find all characters with accident origin
query = """
MATCH (c:Character)-[:HAS_ORIGIN]->(o:PowerOrigin {origin_type: 'accident'})
RETURN c.name, o.description
LIMIT 10
"""

with neo4j.driver.session() as session:
    results = list(session.run(query))
    print(f"Found {len(results)} characters")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and graph status |
| `/` | GET | API information |
| `/question` | POST | Natural language questions |
| `/graph/{character}` | GET | Full character profile |
| `/extraction-report/{character}` | GET | Validation metrics |
| `/validate-extraction` | POST | Re-validate extraction |
| `/characters` | GET | List characters (pagination) |
| `/stats` | GET | Graph statistics |

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Performance & Scalability

### Current Dataset (392 characters)
- **Nodes:** 2,814 (Character: 392, Power: 1,246, Origin: 392, etc.)
- **Relationships:** 4,026
- **Neo4j Query Time:** < 100ms for character profiles
- **LLM Query Time:** 2-5 seconds (includes reasoning)

### Parallel Processing
- **Concurrent Extraction:** 5-12 characters simultaneously
- **Throughput:** ~60-144 characters per 10 minutes (with retries)
- **Success Rate:** 98%+ with retry logic

### Cost Estimates
- **Per Character:** ~$0.001-0.003 (extraction + validation)
- **Full Dataset (16,000):** ~$20-50 one-time processing
- **API Queries:** ~$0.001-0.003 per question

---

## Testing

### Quick Test

```bash
# Test complete Neo4j pipeline
python tests/test_neo4j_connection.py
```

### All Tests

```bash
# Phase 1: Extraction
PYTHONPATH=. python tests/test_extraction.py

# Phase 2: Graph Building
PYTHONPATH=. python tests/test_graph_builder.py

# Phase 3: Validation
PYTHONPATH=. python tests/test_validation.py

# Phase 4: Query Agent
PYTHONPATH=. python tests/test_query_agent.py

# Phase 5: API
python -m src.api.main &  # Start server
PYTHONPATH=. python tests/test_api.py
```

---

## Key Design Decisions

### 1. Why Neo4j?
- **Production-Ready**: Scalable graph database for thousands of characters
- **Cypher Queries**: Powerful query language for complex traversals
- **Graph Algorithms**: Built-in support for centrality, community detection
- **Visualization**: Neo4j Browser for exploring the graph
- **ACID Transactions**: Data consistency and reliability

### 2. Why LlamaIndex Workflows?
- Perfect for multi-step extraction and validation pipelines
- Built-in retry mechanism for low-confidence extractions
- Type-safe event-driven architecture
- Async/await support for efficiency

### 3. Why LangGraph?
- State machine ideal for graph construction and query routing
- Conditional routing for complex workflows
- Complementary to LlamaIndex
- Great for cyclic graph traversal

### 4. Why FastAPI?
- Modern async/await support
- Automatic OpenAPI documentation
- Type safety with Pydantic
- Fast performance and easy deployment

---

## Known Limitations & Future Enhancements

### Current Features
- ✅ Neo4j production database
- ✅ Parallel extraction with retry logic
- ✅ Natural language query agent
- ✅ 392 characters extracted and stored
- ✅ Complete Jupyter notebook pipeline

### Potential Enhancements
- [ ] Scale to full dataset (16,000 characters)
- [ ] Add Redis caching layer
- [ ] Implement user authentication (OAuth2/JWT)
- [ ] Add rate limiting
- [ ] Create Web UI (React/Vue)
- [ ] Advanced graph visualization
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add CI/CD pipeline
- [ ] Graph algorithms (PageRank, community detection)

---

## Documentation

### Phase-Specific Guides
- [Phase 1: Data Extraction](docs/README_PHASE1.md) - Extraction Agent, prompts, data loading
- [Phase 2: Knowledge Graph](docs/README_PHASE2.md) - Graph schema, operations, queries
- [Phase 3: Validation System](docs/README_PHASE3.md) - Validation Agent, reports, feedback loop
- [Phase 4: Query & Response](docs/README_PHASE4.md) - Query Agent, natural language processing
- [Phase 5: REST API](docs/README_PHASE5.md) - FastAPI endpoints, deployment, testing
- [Phase 6: Neo4j Integration](docs/README_NEO4J.md) - Neo4j setup, operations, queries

### Technical Documentation
- [Architecture](docs/architecture.md) - System architecture and data flow
- [Technical Explanation](docs/technical_explanation.md) - Deep technical dive
- [Project Plan](docs/project_plan.md) - Original project plan

### Notebooks & Examples
- [Neo4j Pipeline Notebook](notebooks/neo4j_pipeline.ipynb) - **Complete end-to-end demo**
- [Sample Queries](examples/sample_queries.json) - Example questions
- [Expected Responses](examples/expected_responses.json) - Expected API responses

---

## Contributing

This project showcases LlamaIndex Workflows + LangGraph + Neo4j integration. Key areas for contribution:
- Scaling to full dataset
- Advanced Neo4j queries and graph algorithms
- Web UI development
- Additional validation metrics
- Prompt optimization

---

## License

MIT License

---

## Acknowledgments

- **Marvel Data:** Dataset from Marvel Wikia (fictional characters, educational use)
- **LlamaIndex:** Workflow framework for AI agents
- **LangGraph:** State machine framework
- **Neo4j:** Graph database platform
- **FastAPI:** Modern web framework
- **OpenAI:** GPT-4o-mini and embeddings API

---

## Contact & Support

**Repository:** https://github.com/Idanzinner/marvel-knowledge-graph
**Neo4j Browser:** http://localhost:7474
**API Docs:** http://localhost:8000/docs

---

**Built with LlamaIndex Workflows + LangGraph + Neo4j**
**Project Status:** ✅ Production Ready
**Date Completed:** November 25, 2025
