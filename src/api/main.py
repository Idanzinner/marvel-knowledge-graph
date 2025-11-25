"""
Marvel Knowledge Graph API - Main Application

FastAPI application for querying the Marvel Knowledge Graph.
Provides REST API endpoints for natural language questions, graph exploration,
and extraction validation.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.api.endpoints import router, initialize_endpoints
from src.graph.operations import GraphOperations
from src.agents.query_agent import QueryAgent


# ============================================================================
# Configuration
# ============================================================================

class APIConfig:
    """API Configuration."""
    # Graph settings
    GRAPH_PATH = os.getenv(
        "GRAPH_PATH",
        "data/processed/marvel_knowledge_graph.graphml"
    )

    # LLM settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # API settings
    API_TITLE = "Marvel Knowledge Graph API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = """
## Marvel Knowledge Graph API

A hybrid AI system combining LlamaIndex Workflows and LangGraph to create a
knowledge graph of Marvel characters, their genetic mutations, powers, and affiliations.

### Features

* **Natural Language Queries**: Ask questions about Marvel characters in plain English
* **Knowledge Graph Exploration**: View complete character profiles with all relationships
* **Extraction Validation**: Get quality metrics and validation reports
* **Citation-Grounded Responses**: All answers are grounded in graph facts

### Sample Questions

* "How did Spider-Man get his powers?"
* "Why do Spider-Man's powers matter?"
* "What are Thor's abilities?"
* "How confident are you about Magneto's power origin?"

### Technologies

* **LlamaIndex Workflows**: For extraction and validation agents
* **LangGraph**: For knowledge graph operations and query routing
* **NetworkX**: For graph database
* **OpenAI GPT-4**: For LLM-powered extraction and responses
* **FastAPI**: For REST API framework
"""

    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]


config = APIConfig()


# ============================================================================
# Application State
# ============================================================================

app_state = {
    "graph_ops": None,
    "query_agent": None,
}


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Loads the knowledge graph on startup and cleans up on shutdown.
    """
    # Startup
    print("\n" + "=" * 80)
    print("🚀 Marvel Knowledge Graph API - Starting Up")
    print("=" * 80)

    try:
        # Load knowledge graph
        graph_path = Path(config.GRAPH_PATH)

        if not graph_path.exists():
            print(f"\n⚠️  WARNING: Graph file not found at {graph_path}")
            print("   API will start but endpoints will return 503 errors.")
            print("   Please run data extraction and graph building first.\n")
        else:
            print(f"\n📂 Loading knowledge graph from: {graph_path}")
            graph_ops = GraphOperations.load_graph(str(graph_path))
            stats = graph_ops.get_graph_stats()

            print(f"✅ Graph loaded successfully!")
            print(f"   - Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"   - Total Edges: {stats.get('total_edges', 0)}")
            print(f"   - Characters: {stats.get('node_counts', {}).get('Character', 0)}")

            # Initialize query agent
            print(f"\n🤖 Initializing Query Agent...")
            print(f"   - Model: {config.LLM_MODEL}")
            print(f"   - Temperature: {config.LLM_TEMPERATURE}")

            query_agent = QueryAgent(
                graph_ops=graph_ops,
                llm_model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                verbose=False
            )

            print(f"✅ Query Agent initialized!")

            # Initialize endpoints
            initialize_endpoints(graph_ops, query_agent)

            # Store in app state
            app_state["graph_ops"] = graph_ops
            app_state["query_agent"] = query_agent

        print("\n" + "=" * 80)
        print("✨ API Ready!")
        print("=" * 80)
        print(f"\n📚 Documentation: http://localhost:8000/docs")
        print(f"🔍 ReDoc: http://localhost:8000/redoc")
        print(f"❤️  Health Check: http://localhost:8000/health\n")

    except Exception as e:
        print(f"\n❌ ERROR during startup: {e}")
        print("   API will start but may not function correctly.\n")

    # Yield control to application
    yield

    # Shutdown
    print("\n" + "=" * 80)
    print("👋 Marvel Knowledge Graph API - Shutting Down")
    print("=" * 80 + "\n")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================================
# Middleware
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    print(f"📥 {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"📤 {request.method} {request.url.path} - Status: {response.status_code}")
    return response


# ============================================================================
# Routes
# ============================================================================

# Include API router
app.include_router(router, prefix="", tags=["Marvel Knowledge Graph"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        Welcome message and links to documentation
    """
    return {
        "message": "Welcome to the Marvel Knowledge Graph API",
        "version": config.API_VERSION,
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "question": "POST /question - Ask natural language questions",
            "graph": "GET /graph/{character} - Get character graph view",
            "report": "GET /extraction-report/{character} - Get validation report",
            "validate": "POST /validate-extraction - Re-validate extraction",
            "characters": "GET /characters - List all characters",
            "stats": "GET /stats - Get graph statistics"
        },
        "sample_questions": [
            "How did Spider-Man get his powers?",
            "What are Thor's abilities?",
            "Why do Captain America's powers matter?",
            "How confident are you about Magneto's power origin?"
        ]
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"The requested resource was not found: {request.url.path}",
            "status_code": 404
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    print(f"❌ Internal Server Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please check server logs.",
            "status_code": 500
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("\n" + "=" * 80)
    print("🚀 Starting Marvel Knowledge Graph API Server")
    print("=" * 80)
    print(f"\nHost: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Graph: {config.GRAPH_PATH}\n")

    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
