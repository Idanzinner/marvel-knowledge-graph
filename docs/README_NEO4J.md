# Neo4j Integration Guide

This guide explains how to use the Neo4j graph database backend for the Marvel Knowledge Graph project.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Parallel Extraction](#parallel-extraction)
- [Usage](#usage)
- [Notebook Tutorial](#notebook-tutorial)
- [API Integration](#api-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Marvel Knowledge Graph now supports **Neo4j** as a production-ready graph database backend, replacing or complementing the in-memory NetworkX solution.

### Key Benefits

- **Scalability**: Handle 16,000+ characters efficiently
- **Persistence**: Data survives application restarts
- **Performance**: Optimized graph queries with indexes
- **Visualization**: Built-in Neo4j Browser for graph exploration
- **Cypher**: Powerful query language for complex graph operations

---

## Features

### 1. Neo4j Operations Module

Location: `src/graph/neo4j_operations.py`

Provides full CRUD operations for Neo4j:
- **Node Creation**: Characters, PowerOrigins, Powers, Significance, Validation
- **Relationship Management**: All relationship types (HAS_ORIGIN, POSSESSES_POWER, etc.)
- **Querying**: Get character profiles, list characters, statistics
- **Constraints**: Automatic uniqueness constraints for data integrity

### 2. Parallel Extraction Agent

Location: `src/agents/parallel_extraction.py`

Features:
- **Concurrent Processing**: Extract multiple characters simultaneously
- **Automatic Retry**: Retry failed extractions with exponential backoff
- **Failure Tracking**: Comprehensive logging of successes and failures
- **Progress Reporting**: Real-time progress bars
- **Graceful Degradation**: Continue processing even if some characters fail

### 3. Comprehensive Pipeline Notebook

Location: `notebooks/neo4j_pipeline.ipynb`

Step-by-step guide covering:
- Neo4j connection setup
- Data loading strategies
- Parallel extraction execution
- Graph building in Neo4j
- Querying and analysis
- Custom Cypher queries

---

## Setup

### Prerequisites

1. **Docker Desktop** installed and running
2. **Python 3.10+** with virtual environment
3. **OpenAI API Key** in `.env` file

### Step 1: Install Dependencies

```bash
# Activate virtual environment
cd /Users/hadaszinner/sandbox/marvel_knowledge_grpah
source ../.venv/bin/activate

# Install Neo4j driver
pip install -r requirements.txt
```

### Step 2: Configure Environment

Ensure your `.env` file (in parent directory) contains:

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-your-key-here

# Neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=Polazin2!
CONNECTION_URI=neo4j://127.0.0.1:7687
```

### Step 3: Start Neo4j

```bash
# Start Neo4j using Docker Compose
docker-compose up -d neo4j

# Check if Neo4j is running
docker ps | grep marvel_neo4j

# View logs
docker logs marvel_neo4j
```

### Step 4: Verify Connection

```bash
# Using curl
curl -u neo4j:Polazin2! http://localhost:7474

# Or open Neo4j Browser
open http://localhost:7474
```

**Login Credentials:**
- Username: `neo4j`
- Password: `Polazin2!`

---

## Parallel Extraction

### Basic Usage

```python
from src.utils.data_loader import get_sample_characters
from src.agents.parallel_extraction import extract_batch_parallel
from pathlib import Path

# Load characters
characters = get_sample_characters(
    file_path="data/marvel-wikia-data-with-descriptions.pkl",
    character_names=["Spider-Man (Peter Parker)", "Iron Man (Anthony \"Tony\" Stark)"],
    use_pickle=True
)

# Run parallel extraction
summary = await extract_batch_parallel(
    characters=characters,
    max_concurrent=5,      # Process 5 characters at once
    max_retries=3,         # Retry failed extractions up to 3 times
    output_dir=Path("data/processed"),
    verbose=True
)

# Check results
print(f"Success: {summary.successful}/{summary.total_characters}")
print(f"Failed: {summary.failed}")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent` | 5 | Number of concurrent extractions |
| `max_retries` | 3 | Retry attempts per character |
| `retry_delay` | 2.0 | Base delay between retries (seconds) |
| `verbose` | True | Enable progress logging |
| `output_dir` | None | Save results to directory |

### Output Files

When `output_dir` is specified, the following files are created:

```
data/processed/
├── extractions.json              # Successful extractions
├── extraction_failures.json      # Failed extraction details
└── extraction_summary.json       # Overall summary statistics
```

### Failure Tracking

Failed extractions include:
- Character name and ID
- Error message
- Number of attempts made
- Duration of all attempts

Example failure log:

```json
{
  "character_name": "Unknown Character",
  "character_id": 12345,
  "error_message": "Extraction returned UNKNOWN type",
  "attempts": 3,
  "duration_seconds": 15.3
}
```

---

## Usage

### 1. Python Script Example

```python
import asyncio
from src.graph.neo4j_operations import Neo4jOperations
from src.utils.data_loader import get_sample_characters
from src.agents.parallel_extraction import extract_batch_parallel
from pathlib import Path

async def main():
    # Connect to Neo4j
    neo4j = Neo4jOperations()
    neo4j.create_constraints()

    # Load characters
    characters = get_sample_characters(
        file_path=Path("data/marvel-wikia-data-with-descriptions.pkl"),
        character_names=["Spider-Man (Peter Parker)"],
        use_pickle=True
    )

    # Extract in parallel
    summary = await extract_batch_parallel(
        characters=characters,
        max_concurrent=5,
        max_retries=3,
        verbose=True
    )

    # Build graph for successful extractions
    for result in summary.successful_extractions:
        char = next(c for c in characters if c.page_id == result.character_id)
        ext = result.extraction

        # Create character node
        char_id = neo4j.add_character_node(
            name=char.name,
            page_id=char.page_id,
            alignment=char.align
        )

        # Create origin node
        origin_id = neo4j.add_power_origin_node(
            character_id=char_id,
            origin_type=ext.power_origin.type.value,
            description=ext.power_origin.description,
            confidence=ext.power_origin.confidence.value
        )

        # Link them
        neo4j.add_relationship(char_id, origin_id, "HAS_ORIGIN")

        print(f"✓ Added {char.name} to Neo4j")

    # Query results
    profile = neo4j.get_character_profile("Spider-Man (Peter Parker)")
    print(f"\\nProfile: {profile}")

    # Cleanup
    neo4j.close()

# Run
asyncio.run(main())
```

### 2. Query Examples

#### Get Character Profile

```python
neo4j = Neo4jOperations()
profile = neo4j.get_character_profile("Spider-Man (Peter Parker)", search_by="name")

if profile:
    print(f"Character: {profile['character']['name']}")
    print(f"Origin Type: {profile['origin']['origin_type']}")
    print(f"Powers: {len(profile['powers'])}")
```

#### List All Characters

```python
characters = neo4j.list_all_characters(limit=10)
for char in characters:
    print(f"- {char['name']}")
```

#### Get Statistics

```python
stats = neo4j.get_statistics()
print(f"Total Nodes: {stats['total_nodes']}")
print(f"Total Edges: {stats['total_edges']}")
print(f"Nodes by Type: {stats['nodes_by_type']}")
```

### 3. Custom Cypher Queries

```python
with neo4j.driver.session() as session:
    # Find characters with specific origin type
    query = """
    MATCH (c:Character)-[:HAS_ORIGIN]->(o:PowerOrigin {origin_type: 'mutation'})
    RETURN c.name as name, o.description as origin
    """
    result = session.run(query)

    for record in result:
        print(f"{record['name']}: {record['origin']}")
```

---

## Notebook Tutorial

The comprehensive pipeline notebook ([notebooks/neo4j_pipeline.ipynb](../notebooks/neo4j_pipeline.ipynb)) provides a step-by-step guide:

### Steps Covered

1. **Setup and Imports** - Environment configuration
2. **Connect to Neo4j** - Establish database connection
3. **Load Character Data** - Various loading strategies
4. **Run Parallel Extraction** - Extract with retry logic
5. **Review Results** - Examine successes and failures
6. **Clear Database** - Optional fresh start
7. **Build Knowledge Graph** - Create nodes and relationships
8. **Verify Statistics** - Check what was created
9. **Query Characters** - Retrieve profiles
10. **List All Characters** - Overview of database
11. **Custom Queries** - Advanced Cypher examples
12. **Cleanup** - Close connections

### Running the Notebook

```bash
# Start Jupyter
cd notebooks
jupyter notebook neo4j_pipeline.ipynb

# Or use JupyterLab
jupyter lab neo4j_pipeline.ipynb
```

---

## API Integration

### Updating API to Use Neo4j

To integrate Neo4j with the FastAPI backend:

1. **Update main.py** to initialize Neo4j instead of NetworkX:

```python
from src.graph.neo4j_operations import Neo4jOperations

# Replace
graph_ops = GraphOperations()

# With
neo4j_ops = Neo4jOperations()
```

2. **Update endpoints** to use Neo4j operations:

```python
@app.get("/graph/{character}")
async def get_character_graph(character: str, search_by: str = "name"):
    profile = neo4j_ops.get_character_profile(character, search_by)
    if not profile:
        raise HTTPException(status_code=404, detail="Character not found")
    return profile
```

3. **Update startup/shutdown**:

```python
@app.on_event("startup")
async def startup_event():
    neo4j_ops.create_constraints()

@app.on_event("shutdown")
async def shutdown_event():
    neo4j_ops.close()
```

---

## Troubleshooting

### Neo4j Won't Start

```bash
# Check Docker
docker ps -a | grep marvel_neo4j

# View logs
docker logs marvel_neo4j

# Restart Neo4j
docker-compose restart neo4j

# Full reset
docker-compose down -v
docker-compose up -d neo4j
```

### Connection Errors

```python
# Test connection
from src.graph.neo4j_operations import Neo4jOperations

try:
    neo4j = Neo4jOperations()
    stats = neo4j.get_statistics()
    print("✓ Connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

**Common Issues:**
- Neo4j not running: `docker-compose up -d neo4j`
- Wrong credentials: Check `.env` file
- Port conflict: Ensure ports 7474 and 7687 are free

### Extraction Failures

If many characters fail extraction:

1. **Check OpenAI API Key**:
   ```python
   import os
   print(os.getenv("OPENAI_API_KEY"))
   ```

2. **Reduce Concurrency**:
   ```python
   # Lower max_concurrent if rate limited
   summary = await extract_batch_parallel(
       characters=characters,
       max_concurrent=2,  # Reduced from 5
       max_retries=3
   )
   ```

3. **Review Failure Log**:
   ```python
   # Check extraction_failures.json
   import json
   with open("data/processed/extraction_failures.json") as f:
       failures = json.load(f)
       for fail in failures:
           print(f"{fail['character_name']}: {fail['error_message']}")
   ```

### Memory Issues

For large-scale extraction (1000+ characters):

```python
# Process in batches
batch_size = 100
all_characters = load_all_characters(...)

for i in range(0, len(all_characters), batch_size):
    batch = all_characters[i:i+batch_size]
    summary = await extract_batch_parallel(batch, ...)
    print(f"Batch {i//batch_size + 1} complete")
```

---

## Performance Tips

### 1. Batch Size

- **Small batches (10-50)**: Best for testing
- **Medium batches (100-500)**: Good balance
- **Large batches (1000+)**: Use batch processing

### 2. Concurrency

- **Low (1-2)**: Avoid rate limits
- **Medium (5-10)**: Good throughput
- **High (20+)**: Maximum speed (may hit rate limits)

### 3. Neo4j Optimization

```cypher
-- Create indexes for faster queries
CREATE INDEX char_name IF NOT EXISTS FOR (c:Character) ON (c.name);
CREATE INDEX origin_type IF NOT EXISTS FOR (o:PowerOrigin) ON (o.origin_type);
```

### 4. Caching

Consider adding Redis caching for frequently accessed queries:

```python
# In requirements.txt
# redis==5.2.1

# Cache character profiles
import redis
cache = redis.Redis(host='localhost', port=6379)
```

---

## Next Steps

1. **Scale Up**: Process more characters (100, 1000, 16K)
2. **Validation**: Add validation agent integration
3. **API Migration**: Update FastAPI to use Neo4j
4. **Visualization**: Create graph visualization tools
5. **Analytics**: Build analytics dashboard
6. **Export**: Add GraphML export for Neo4j Desktop

---

## Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/)
- [Docker Compose Guide](https://docs.docker.com/compose/)

---

## Summary

The Neo4j integration provides:
- ✅ Production-ready graph database
- ✅ Parallel extraction with retry logic
- ✅ Comprehensive failure tracking
- ✅ Easy-to-follow notebook tutorial
- ✅ Scalable to thousands of characters

Ready to build your Marvel Knowledge Graph! 🦸‍♂️
