# Property Graph Builder with LlamaIndex

A **general-purpose** LlamaIndex Property Graph system that can automatically discover entities and relationships from **any type of data** without hardcoding schemas or column names.

## 🚀 Key Features

- **Multi-Format Data Ingestion**: Automatically handles CSV, JSON, text, PDF, and other file types
- **Schema-Free**: No need to predefine entities or relationships - they're discovered automatically
- **Multiple Extraction Modes**: Choose between simple, dynamic, or implicit extraction
- **Flexible Backends**: Works with simple in-memory storage or Neo4j database
- **Domain Agnostic**: Works with any data domain (e-commerce, medical, financial, etc.)
- **Interactive Querying**: Built-in query interface with multiple retrieval modes

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Neo4j settings if using Neo4j backend
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

**Note**: The system now uses local HuggingFace embeddings by default, so you don't need OpenAI API access for embeddings!

3. **Configure your settings**:
Edit `config.py` to customize:
- Data path and file types
- Extraction mode
- Backend choice (simple vs Neo4j)
- LLM and embedding models

## 📊 Quick Start

### Step 1: Configure Your Data
Edit `config.py`:
```python
# Point to your data directory or file
DATA_PATH = "path/to/your/data"

# Optionally filter by file type
FILE_EXTENSIONS = ['.csv', '.json', '.txt']  # or None for all

# Choose extraction mode
EXTRACTION_MODE = "auto"  # auto, simple, dynamic, or implicit
```

### Step 2: Build Your Graph
```bash
python graph_builder.py
```

This will:
- Automatically detect and process all supported files
- Convert structured data (CSV/JSON) to natural language
- Use LLM to extract entities and relationships
- Save the graph for querying

### Step 3: Query Your Graph
```bash
python query_interface.py
```

This starts an interactive session where you can:
- See graph statistics and structure
- Get suggested queries based on your data
- Ask questions in natural language
- Use different query modes (vector, keyword, context)

## 🎯 Extraction Modes

### `"auto"` (Recommended)
- Uses both simple and dynamic extraction
- Best for most use cases
- Automatically discovers schema

### `"simple"`
- Basic LLM-based extraction
- Fast and reliable
- Good for well-structured data

### `"dynamic"`
- Discovers schema automatically
- Adapts to data structure
- Best for complex, varied data

### `"implicit"`
- Uses embeddings for relationships
- Good for unstructured text
- Discovers implicit connections

## 💡 Example Use Cases

### E-commerce Data
```python
# config.py
DATA_PATH = "ecommerce_data"
FILE_EXTENSIONS = ['.csv']
EXTRACTION_MODE = "dynamic"
```

**Automatic Discovery**: Products, Categories, Orders, Customers
**Queries**: "What are the most popular products?", "Show me customer purchase patterns"

### Research Papers
```python
# config.py
DATA_PATH = "research_papers"
FILE_EXTENSIONS = ['.pdf', '.txt']
EXTRACTION_MODE = "auto"
```

**Automatic Discovery**: Authors, Topics, Citations, Institutions
**Queries**: "What are the main research themes?", "Which authors collaborate most?"

### Mixed Data Sources
```python
# config.py
DATA_PATH = "mixed_data"
FILE_EXTENSIONS = None  # All supported files
EXTRACTION_MODE = "auto"
```

**Automatic Discovery**: Any entities and relationships in your data
**Queries**: "Summarize the key information", "What patterns exist in this data?"

## 🔧 Advanced Configuration

### Custom Processing
The system automatically handles:
- **CSV**: Converts rows to natural language descriptions
- **JSON**: Processes nested objects and arrays
- **Text/PDF**: Chunks and processes content
- **Mixed encodings**: Tries multiple encodings automatically

### Query Modes
- **Vector**: Semantic similarity search
- **Keyword**: Exact term matching
- **Context**: Relationship-aware queries
- **Default**: General-purpose queries

### Backends
- **Simple**: In-memory graph (default)
- **Neo4j**: Full graph database with Cypher support

## 📈 Scaling for Large Data

For large datasets:

1. **Use Neo4j backend**:
```python
USE_NEO4J = True
```

2. **Process in chunks**:
```python
CHUNK_SIZE = 512
NUM_WORKERS = 4
```

3. **Filter file types**:
```python
FILE_EXTENSIONS = ['.csv']  # Focus on specific types
```

## 🔍 Example Queries

The system works with **any domain** and **any data structure**. Here are examples:

### General Queries
- "What are the main entities in this data?"
- "What relationships exist between different entities?"
- "Summarize the key patterns in this dataset"

### Specific Queries (auto-discovered)
- "Tell me about [entity type] entities"
- "Show me [relationship type] relationships"
- "What are the properties of [discovered entity]?"

### Complex Analysis
- "Analyze the patterns in this data"
- "Explain the relationships between different components"
- "What insights can you derive from this information?"

## 🤝 Contributing

This is a **general-purpose system** - it should work with any data type and domain. If you find data it doesn't handle well:

1. Check the console output for processing errors
2. Try different extraction modes
3. Adjust chunk sizes for your content type
4. Add custom processing for new file types

## 📚 Key Files

- `graph_builder.py`: Main graph building logic
- `query_interface.py`: Interactive query system
- `config.py`: All configuration options
- `requirements.txt`: Dependencies
- `data/`: Your data directory (customize path in config)

## 🎯 Next Steps

1. **Try it with your data**: Point `DATA_PATH` to your files
2. **Experiment with modes**: Try different `EXTRACTION_MODE` values
3. **Scale up**: Use Neo4j for larger datasets
4. **Customize queries**: Add domain-specific query templates
5. **Integrate**: Use the graph in your applications

---

**The beauty of this system**: You don't need to know your data structure ahead of time. Just point it at your data, and it will automatically discover the entities, relationships, and patterns that matter for your domain! 🎉 