#!/bin/bash
# Setup and run Neo4j for Marvel Knowledge Graph

set -e

echo "=========================================="
echo "Marvel Knowledge Graph - Neo4j Setup"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=marvelgraph123

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
EOF
    echo "✓ Created .env file"
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo ""
fi

# Start Neo4j
echo "Starting Neo4j..."
docker-compose up -d

echo ""
echo "Waiting for Neo4j to start..."
sleep 10

# Check if Neo4j is ready
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec marvel_neo4j cypher-shell -u neo4j -p marvelgraph123 "RETURN 1" > /dev/null 2>&1; then
        echo "✓ Neo4j is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "  Waiting... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "✗ Neo4j failed to start"
    exit 1
fi

echo ""
echo "=========================================="
echo "Neo4j is running!"
echo "=========================================="
echo ""
echo "Neo4j Browser: http://localhost:7474"
echo "Bolt URL: bolt://localhost:7687"
echo "Username: neo4j"
echo "Password: marvelgraph123"
echo ""
echo "Next steps:"
echo "1. Set your OPENAI_API_KEY in .env file"
echo "2. Run: source .env"
echo "3. Run: python3 extract_graph_data.py --test"
echo "4. Run: python3 upload_to_neo4j.py --clear"
echo ""
