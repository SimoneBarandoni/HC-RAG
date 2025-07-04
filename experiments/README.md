# HC-RAG: Hybrid Knowledge Graph + RAG System

This project demonstrates a complete **Hybrid RAG (Retrieval-Augmented Generation)** system that combines:
- **Neo4j Knowledge Graph** for structured data relationships  
- **Vector Embeddings** for semantic similarity search
- **Interactive Subgraph Visualization** using Cytoscape.js
- **Dynamic Entity Recognition** with intelligent query parsing

## Features

✅ **Knowledge Graph Construction**: Build Neo4j graphs from CSV data and PDF documents  
✅ **Dynamic Embedding Generation**: Automatically create embeddings for diverse data types  
✅ **Semantic Search**: Find similar content using vector similarity  
✅ **Interactive Visualization**: Explore knowledge graph subgraphs in your browser  
✅ **Query Understanding**: Parse natural language queries into structured entities  
✅ **Relationship Discovery**: Find connections between similar items  

## 🎨 Interactive Subgraph Visualization

The system creates beautiful, interactive visualizations of knowledge graph subgraphs based on your search results:

- **Node Size**: Represents similarity score to your query
- **Node Color**: High similarity (red) → Medium (teal) → Low (blue)  
- **Relationships**: Shows actual connections between similar items
- **Click Interactions**: Click nodes/edges for detailed information
- **Drag & Zoom**: Fully interactive graph exploration

![Subgraph Visualization Features](docs/visualization-preview.png)

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j (Docker)
docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest
```

### 2. Build Knowledge Graph + Embeddings

```bash
# Complete setup (creates graph + embeddings)
python main.py
```

### 3. Try Interactive Visualization

```bash
# Interactive demo with multiple queries
python demo_visualization.py

# Or test specific visualization features
python main.py --viz-test
```

## Usage Examples

### Basic RAG Search

```python
from main import EmbeddingRAGSystem

# Initialize system
rag_system = EmbeddingRAGSystem('data/knowledge_graph_embeddings.pkl')

# Search for similar content
results = rag_system.process_query("mountain bike components", top_k=5)

# View results
for result in results["results"]:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['content']}")
```

### Interactive Subgraph Visualization

```python
# Create interactive visualization
query_results, viz_file = rag_system.visualize_query_results(
    "road bike frames", 
    top_k=8,
    similarity_threshold=0.25
)

# Opens in browser automatically - explore the subgraph!
print(f"Visualization saved: {viz_file}")
```

### Category-Filtered Search

```python
# Search only in specific content types
filtered_results = rag_system.search_by_category(
    "bike accessories", 
    category_filter="database_table"
)
```

## Available Commands

| Command | Description |
|---------|-------------|
| `python main.py` | Full setup: build graph + generate embeddings |
| `python main.py --rag-only` | Test RAG search functionality only |
| `python main.py --viz-test` | Test visualization features |
| `python demo_visualization.py` | Interactive visualization demo |

## Data Structure

The system works with:

- **CSV Files**: Product catalogs, categories, descriptions
- **PDF Documents**: Product manuals and specifications  
- **JSON Tables**: Extracted structured data from PDFs
- **Images**: Product photos and diagrams

All data types are automatically processed and embedded for semantic search.

## Visualization Technology Stack

- **Backend**: Python + Neo4j + SentenceTransformers
- **Frontend**: Cytoscape.js + HTML5 + CSS3
- **Graph Database**: Neo4j Community Edition
- **Embeddings**: Sentence-BERT models (all-MiniLM-L6-v2)

## Environment Variables

Create a `.env` file (copy from `example.env`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Jupyter Notebook

For interactive development:

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies + Jupyter
pip install -r requirements.txt
python -m ipykernel install --user --name=hc-rag

# Start Jupyter
jupyter notebook
```

Then use the `hc-rag` kernel in your notebooks.

## Architecture

```
Query → Entity Parsing → Vector Search → Subgraph Extraction → Visualization
  ↓           ↓              ↓               ↓                    ↓
Text     Structured     Embeddings      Neo4j Query        Cytoscape.js
Input     Entities      Similarity      Relationships       Interactive
                        Ranking           Discovery            Graph
```

## Advanced Features

### Dynamic Column Detection
The system automatically detects column types in CSV data:
- **Identifiers**: Primary keys, IDs
- **Text Content**: Names, descriptions  
- **Categories**: Classification fields
- **Measurements**: Prices, dimensions
- **Attributes**: Colors, sizes, features

### Intelligent Query Parsing
Natural language queries are parsed to extract:
- **Product Information**: Names, features, categories
- **Document References**: Manual types, specifications
- **Relationship Context**: Compatibility, similarity

### Multi-Modal Embeddings
Creates embeddings for:
- **Database Rows**: "Product: HL Road Frame. Category: Road Frames. Price: $1431.50"
- **JSON Tables**: "Technical specs. Document: Mountain Bike Manual. Tire Pressure: 30-50 PSI"
- **Document Content**: Full-text PDF content with metadata

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.