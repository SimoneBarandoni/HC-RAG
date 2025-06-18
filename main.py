#!/usr/bin/env python3
"""
Main Setup Script for Knowledge Graph + Embeddings System

This script orchestrates the complete setup of:
1. Neo4j Knowledge Graph creation
2. Embedding generation for all data
3. Integration between graph nodes and embeddings

"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_graph import (
    KnowledgeGraphBuilder, 
    NEO4J_URI, 
    NEO4J_USER, 
    NEO4J_PASSWORD,
    test_neo4j_connection,
    load_csv_data,
    analyze_ingested_documents
)
from embedding_generator import DynamicEmbeddingGenerator
from query_parser import QueryParser, test_query_parser


class EmbeddingRAGSystem:
    """
    Complete RAG system that integrates query parsing with embedding-based search
    """
    
    def __init__(self, embeddings_path="data/knowledge_graph_embeddings.pkl", 
                 embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG system with embeddings and query parsing
        
        Args:
            embeddings_path: Path to saved embeddings file
            embedding_model_name: Name of SentenceTransformer model to use
        """
        print("Initializing Embedding RAG System...")
        
        # Initialize embedding generator and load pre-computed embeddings
        self.embedding_generator = DynamicEmbeddingGenerator(embedding_model_name)
        
        print(f"Loading embeddings from {embeddings_path}")
        self.embeddings_data = self.embedding_generator.load_embeddings(embeddings_path)
        
        # Extract components for faster access
        # Convert embeddings list back to numpy array for operations like .shape and cosine_similarity
        self.embeddings_matrix = np.array(self.embeddings_data["embeddings"])
        self.metadata_list = self.embeddings_data["metadata"] 
        self.texts_list = self.embeddings_data["texts"]
        
        print(f"Loaded {len(self.embeddings_matrix)} embeddings")
        print(f"Embedding dimensions: {self.embeddings_matrix.shape[1]}")
        
        # Initialize query parser
        self.query_parser = QueryParser()
        
        print("RAG System ready!")
    
    def parse_query(self, query):
        """Parse user query to extract structured entities"""
        try:
            return self.query_parser.parse_query(query)
        except Exception as e:
            print(f"Error parsing query: {e}")
            return {"search_text": query}  # Fallback to raw query
    
    def process_query(self, query, top_k=5, similarity_threshold=0.3):
        """
        Complete query processing pipeline:
        1. Parse query to extract entities
        2. Generate embedding for query text  
        3. Search for similar content
        4. Return structured results
        """
        print(f"Processing query: '{query}'")
        
        # Step 1: Parse the query (extract entities)
        print("Step 1: Parsing query...")
        parsed_entities = self.parse_query(query)
        
        # Step 2: Generate embedding for the search text
        search_text = parsed_entities.get("search_text", query)
        print("Step 2: Generating query embedding...")
        query_embedding = self.embedding_generator.model.encode([search_text])
        
        # Step 3: Find similar content using embeddings
        print("Step 3: Searching for similar content...")
        search_results = self.find_similar_content(
            query_embedding[0], 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )
        
        print(f"Found {len(search_results)} relevant results")
        
        # Step 4: Create summary
        avg_similarity = np.mean([r['similarity_score'] for r in search_results]) if search_results else 0
        summary = f"Found {len(search_results)} results with average similarity: {avg_similarity:.3f}"
        
        return {
            "parsed_query": parsed_entities,
            "search_text": search_text,
            "results": search_results,
            "summary": summary,
            "query_embedding": query_embedding[0]
        }
    
    def find_similar_content(self, query_embedding, top_k=5, similarity_threshold=0.3):
        """
        Find content similar to query embedding
        
        Args:
            query_embedding: Numpy array of query embedding
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        """
        # Calculate cosine similarity between query and all stored embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get indices of most similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by similarity threshold and create results
        results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                result = {
                    'content': self.texts_list[idx],
                    'metadata': self.metadata_list[idx],
                    'similarity_score': float(similarities[idx])
                }
                results.append(result)
        
        return results
    
    def search_by_category(self, query, category_filter=None, top_k=5):
        """
        Search with category filtering
        
        Args:
            query: Search query text
            category_filter: Filter by content type (e.g., 'database_table', 'json_table')
            top_k: Number of results to return
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([query])
        
        # Filter indices by category if specified
        if category_filter:
            valid_indices = [
                i for i, meta in enumerate(self.metadata_list) 
                if meta.get("type") == category_filter
            ]
            print(f"🔍 Filtering to {category_filter} only ({len(valid_indices)} items)")
        else:
            valid_indices = list(range(len(self.embeddings_matrix)))
        
        if not valid_indices:
            return {"results": [], "summary": "No items match the filter criteria"}
        
        # Calculate similarities for filtered items
        filtered_embeddings = self.embeddings_matrix[valid_indices]
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            original_idx = valid_indices[idx]
            result = {
                'rank': rank + 1,
                'similarity_score': float(similarities[idx]),
                'content': self.texts_list[original_idx],
                'metadata': self.metadata_list[original_idx]
            }
            results.append(result)
        
        return {
            "results": results,
            "summary": f"Found {len(results)} results in {category_filter or 'all categories'}"
        }
    
    def get_content_statistics(self):
        """Get statistics about the loaded content"""
        stats = {
            "total_entries": len(self.embeddings_matrix),
            "embedding_dimensions": self.embeddings_matrix.shape[1],
            "content_types": {},
            "database_tables": {}
        }
        
        for meta in self.metadata_list:
            # Count content types
            content_type = meta.get("type", "unknown")
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            # Count database tables
            if content_type == "database_table":
                table_name = meta.get("table_name", "unknown")
                stats["database_tables"][table_name] = stats["database_tables"].get(table_name, 0) + 1
        
        return stats


def setup_environment():
    print("KNOWLEDGE GRAPH + EMBEDDINGS SETUP")
    print("Testing Neo4j connection...")
    connected, message = test_neo4j_connection()

    if not connected:
        print(f"Error: {message}")
        print("To start Neo4j with Docker:")
        print("docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest")
        return False
    
    print(f"{message}")

    print("\nLoading CSV data...")
    try:
        csv_data = load_csv_data()
        print(f"Loaded {len(csv_data)} CSV files")
        
        print("\nAnalyzing document structure...")
        document_structure = analyze_ingested_documents()
        print(f"Found {len(document_structure)} document groups")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Step 3: Setup Knowledge Graph
    print("\nSetting up Knowledge Graph...")
    try:
        kg_builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        print("   Cleaning database...")
        kg_builder.clear_database()
        
        print("   Creating indexes...")
        kg_builder.create_indexes()
        
        print("   Creating product nodes...")
        kg_builder.create_product_nodes(
            csv_data["products"], 
            csv_data["categories"], 
            csv_data["models"]
        )
        
        print("   Creating document nodes...")
        kg_builder.create_document_nodes(document_structure)
        
        print("   Creating relationships...")
        kg_builder.create_product_relationships()
        kg_builder.create_product_document_relationships()
        
        print("Knowledge Graph created successfully")
        
    except Exception as e:
        print(f"Error creating knowledge graph: {e}")
        return False
    
    # Step 4: Generate Embeddings
    print("\nGenerating embeddings...")
    try:
        embeddings_path = kg_builder.generate_and_store_embeddings("data")
        print(f"Embeddings saved to: {embeddings_path}")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        kg_builder.close()
        return False
    
    # Step 5: Get final statistics
    print("\nFinal Statistics:")
    try:
        graph_stats = kg_builder.get_graph_statistics()
        embedding_stats = kg_builder.get_embedding_statistics()
        
        print(f"  Graph Nodes: {graph_stats['nodes']}")
        print(f"  Graph Relationships: {graph_stats['relationships']}")
        print(f"  Total Embeddings: {embedding_stats['total_embeddings']}")
        print(f"  Embedding Dimension: {embedding_stats['embedding_dimension']}")
        print(f"  Content Types: {embedding_stats['content_types']}")
        
    except Exception as e:
        print(f"Warning: Could not get statistics: {e}")
    
    finally:
        kg_builder.close()
    
    # Step 6: Verify setup
    print("\nSETUP COMPLETED SUCCESSFULLY!")
    print("-" * 30)
    print("Neo4j Browser: http://localhost:7474")
    print("Embeddings file: data/knowledge_graph_embeddings.pkl")
    print()
    print("Available capabilities:")
    print("  Graph queries and traversals via Neo4j")
    print("  Semantic similarity search via embeddings")
    print("  Hybrid search combining both approaches")
    
    return True


def verify_data_files():
    """
    Verify that all required data files exist
    """
    data_path = Path("data")
    required_files = [
        "Product.csv",
        "ProductCategory.csv", 
        "ProductDescription.csv",
        "ProductModel.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    # Check for IngestedDocuments directory
    if not (data_path / "IngestedDocuments").exists():
        print("Warning: IngestedDocuments directory not found")
        print("   JSON document processing will be skipped")
    
    return True


def test_query_parsing():    
    try:
        # Initialize the query parser
        parser = QueryParser()
        
        # Test with a single query
        test_query = "Find products similar to the black mountain bike size 58"
        print(f"Test Query: '{test_query}'")
        
        # Parse the query
        result = parser.parse_query(test_query)
        
        print("\nExtracted Entities:")
        print("-" * 20)
        
        # Display product information
        if result["product"]["name"]:
            print(f"Product Name: {result['product']['name']}")
        if result["product"]["features"]:
            print(f"Features: {', '.join(result['product']['features'])}")
        if result["product"]["category"]:
            print(f"Category: {result['product']['category']}")
            
        # Display document information
        if result["document"]["type"] or result["document"]["name"]:
            print(f"Document Type: {result['document']['type']}")
            print(f"Document Name: {result['document']['name']}")
            
        # Display relationship information
        if result["relationship"]["type"]:
            print(f"Relationship Type: {result['relationship']['type']}")
            print(f"Relationship Direction: {result['relationship']['direction']}")
        
        print("\nQuery parsing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing query parser: {e}")
        return False


def test_embedding_rag_system():
    """
    Test the complete RAG system with query embedding and similarity search
    """
    print("\nTesting Complete RAG System (Query Embedding + Graph Search)")
    print("=" * 70)
    
    try:
        # Check if embeddings file exists
        embeddings_path = "data/knowledge_graph_embeddings.pkl"
        if not Path(embeddings_path).exists():
            print(f"Embeddings file not found: {embeddings_path}")
            print("   Please run the setup first to generate embeddings")
            return False
        
        # Initialize the RAG system
        rag_system = EmbeddingRAGSystem(embeddings_path)
        
        # Show content statistics
        print("\nCONTENT STATISTICS:")
        stats = rag_system.get_content_statistics()
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
        print("  Content types:")
        for content_type, count in stats["content_types"].items():
            print(f"    - {content_type}: {count}")
        
        # Test queries with different complexity levels
        test_queries = [
            "Find information about mountain bikes",
            "Show me black bicycle components", 
            "What products are available in red color?",
            "Find bike frames with size 58"
        ]
        
        print(f"\nTESTING {len(test_queries)} DIFFERENT QUERIES")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTEST {i}: '{query}'")
            print("-" * 40)
            
            # Process the query through the complete pipeline
            results = rag_system.process_query(query, top_k=3, similarity_threshold=0.2)
            
            # Show parsed query information
            parsed = results["parsed_query"]
            if any(str(v) != "{}" and str(v) != "[]" and v for v in parsed.values()):
                print("PARSED ENTITIES:")
                for entity_type, entity_data in parsed.items():
                    if entity_data and str(entity_data) not in ["{}", "[]", "None"]:
                        print(f"  {entity_type}: {entity_data}")
            
            print(f"Search text: '{results['search_text']}'")
            
            # Show similarity search results
            search_results = results["results"]
            print(f"\nSIMILARITY SEARCH RESULTS ({len(search_results)} found):")
            
            if search_results:
                for j, result in enumerate(search_results, 1):
                    score = result["similarity_score"]
                    content = result["content"]
                    metadata = result["metadata"]
                    
                    # Truncate content for display
                    display_content = content[:120] + "..." if len(content) > 120 else content
                    
                    print(f"  {j}. Similarity: {score:.3f}")
                    print(f"     Type: {metadata.get('type', 'unknown')}")
                    if metadata.get('table_name'):
                        print(f"     Table: {metadata['table_name']}")
                    print(f"     Content: {display_content}")
                    print()
            else:
                print("  No results found above similarity threshold")
            
            print(f"{results['summary']}")
            
            # Show query embedding info
            embedding_norm = np.linalg.norm(results['query_embedding'])
            print(f"Query embedding norm: {embedding_norm:.3f}")
        
        # Test category filtering
        print("\nTESTING CATEGORY-FILTERED SEARCH")
        print("-" * 40)
        
        category_results = rag_system.search_by_category(
            "bike components", 
            category_filter="database_table", 
            top_k=3
        )
        
        print("Results for 'bike components' filtered to database tables:")
        for result in category_results["results"]:
            score = result['similarity_score']
            table_name = result['metadata'].get('table_name', 'unknown')
            print(f"  - {table_name}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for the setup script
    """
    print("Knowledge Graph + Embeddings Environment Setup")
    print("=" * 60)
    
    # Verify data files exist
    print("Verifying data files...")
    if not verify_data_files():
        print("Setup failed: Missing required data files")
        sys.exit(1)
    
    print("All required data files found")
    
    # Run the complete setup
    success = setup_environment()
    
    if success:
        # Test query parsing functionality
        print("\n" + "="*60)
        print("TESTING SYSTEM COMPONENTS")
        print("="*60)
        
        # Test basic query parsing
        query_parsing_success = test_query_parsing()
        
        # Test complete RAG system with embeddings
        rag_system_success = test_embedding_rag_system()
        
        if query_parsing_success and rag_system_success:
            print("\nALL TESTS PASSED!")
            sys.exit(0)
        else:
            print("\nSome tests failed, but basic setup is complete.")
            sys.exit(0)
    else:
        print("\nEnvironment setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


def test_rag_only():
    """
    Test only the RAG system functionality (assumes embeddings already exist)
    """
    print("RAG SYSTEM TEST ONLY")
    print("=" * 40)
    print("This will test the complete query embedding and search pipeline")
    print("(Assumes embeddings have already been generated)\n")
    
    success = test_embedding_rag_system()
    
    if success:
        print("\nRAG SYSTEM TEST COMPLETED SUCCESSFULLY!")
    else:
        print("\nRAG SYSTEM TEST FAILED")


if __name__ == "__main__":
    # Check command line arguments for different modes
    if len(sys.argv) > 1 and sys.argv[1] == "--test-rag":
        test_rag_only()
    else:
        main() 