#!/usr/bin/env python3
"""
Interactive Subgraph Visualization Demo

This script demonstrates how to create interactive visualizations of 
the knowledge graph subgraphs based on RAG search results.

Usage:
    python demo_visualization.py

Features:
- Interactive Cytoscape.js visualization
- Neo4j subgraph extraction
- Similarity-based node styling
- Click interactions and info panels
"""

import sys
from pathlib import Path

# Add current directory to path so we can import from main.py
sys.path.append(str(Path(__file__).parent))

from main import EmbeddingRAGSystem, test_neo4j_connection, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def demo_interactive_visualization():
    """
    Interactive demo of the subgraph visualization features
    """
    print("🎨 INTERACTIVE SUBGRAPH VISUALIZATION DEMO")
    print("=" * 55)
    print("This demo will show you how to visualize knowledge graph subgraphs")
    print("based on similarity search results from your RAG system.\n")
    
    # Check prerequisites
    print("📋 Checking prerequisites...")
    
    # Check Neo4j connection
    connected, message = test_neo4j_connection()
    if not connected:
        print(f"❌ Neo4j connection failed: {message}")
        print("   Please start Neo4j with:")
        print("   docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest")
        return False
    print(f"✅ Neo4j: {message}")
    
    # Check embeddings file
    embeddings_path = "data/knowledge_graph_embeddings.pkl"
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        print("   Please run: python main.py")
        return False
    print(f"✅ Embeddings found: {embeddings_path}")
    
    print("\n🚀 Initializing RAG system...")
    try:
        rag_system = EmbeddingRAGSystem(embeddings_path)
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False
    
    # Interactive demo
    print("\n" + "="*55)
    print("🎯 INTERACTIVE VISUALIZATION DEMO")
    print("="*55)
    
    # Predefined interesting queries
    demo_queries = [
        {
            "query": "mountain bike components",
            "description": "Find mountain bike parts and see how they're related",
            "top_k": 8,
            "threshold": 0.25
        },
        {
            "query": "road bike frames",
            "description": "Explore road bike frame products and their relationships",
            "top_k": 6,
            "threshold": 0.3
        },
        {
            "query": "HL Road Frame Black",
            "description": "Find a specific product and its similar items",
            "top_k": 5,
            "threshold": 0.2
        },
        {
            "query": "bicycle handlebars",
            "description": "Discover handlebar products and their connections",
            "top_k": 7,
            "threshold": 0.25
        }
    ]
    
    while True:
        print("\nChoose a demo query:")
        for i, demo in enumerate(demo_queries, 1):
            print(f"  {i}. \"{demo['query']}\" - {demo['description']}")
        print(f"  {len(demo_queries)+1}. Enter your own query")
        print(f"  0. Exit")
        
        try:
            choice = input(f"\nEnter your choice (0-{len(demo_queries)+1}): ").strip()
            
            if choice == "0":
                break
            elif choice == str(len(demo_queries)+1):
                # Custom query
                custom_query = input("Enter your search query: ").strip()
                if not custom_query:
                    continue
                
                print(f"\n🔍 Processing custom query: '{custom_query}'")
                query_results, viz_file = rag_system.visualize_query_results(
                    custom_query,
                    top_k=8,
                    similarity_threshold=0.2,
                    title_suffix="Custom Query"
                )
                
            elif choice.isdigit() and 1 <= int(choice) <= len(demo_queries):
                # Predefined query
                demo = demo_queries[int(choice)-1]
                
                print(f"\n🔍 Processing query: '{demo['query']}'")
                print(f"   {demo['description']}")
                
                query_results, viz_file = rag_system.visualize_query_results(
                    demo['query'],
                    top_k=demo['top_k'],
                    similarity_threshold=demo['threshold'],
                    title_suffix=f"Demo {choice}"
                )
                
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Show results summary
            results = query_results["results"]
            if results:
                print(f"\n📊 Found {len(results)} similar items:")
                for j, result in enumerate(results[:5], 1):  # Show top 5
                    score = result['similarity_score']
                    metadata = result['metadata']
                    table_name = metadata.get('table_name', 'unknown')
                    entity_id = metadata.get('entity_id', 'unknown')
                    print(f"   {j}. {table_name}[{entity_id}]: {score:.3f}")
                
                if len(results) > 5:
                    print(f"   ... and {len(results)-5} more items")
            else:
                print("   No results found above similarity threshold")
            
            if viz_file:
                print(f"\n🎨 Interactive visualization opened in browser!")
                print(f"   File saved as: {viz_file}")
                print("\n💡 In the visualization you can:")
                print("   • Click and drag nodes to rearrange them")
                print("   • Zoom in/out with mouse wheel")
                print("   • Click nodes to see detailed information")
                print("   • Click edges to see relationship types")
                print("   • Node size = similarity score")
                print("   • Node color = similarity level (red=high, blue=low)")
                
                input("\n⏸️  Press Enter to continue with next query...")
            else:
                print("   No visualization created (no graph relationships found)")
        
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error during demo: {e}")
            continue
    
    # Cleanup
    rag_system.close()
    
    print("\n🎉 Demo completed!")
    print("\nWhat you learned:")
    print("✅ How to extract subgraphs from RAG search results")
    print("✅ How to create interactive Cytoscape.js visualizations")
    print("✅ How similarity scores affect node styling")
    print("✅ How to explore relationships between similar items")
    
    print(f"\n🔧 Integration in your code:")
    print("```python")
    print("from main import EmbeddingRAGSystem")
    print("")
    print("rag_system = EmbeddingRAGSystem('data/knowledge_graph_embeddings.pkl')")
    print("query_results, viz_file = rag_system.visualize_query_results('your query')")
    print("```")
    
    return True


if __name__ == "__main__":
    try:
        success = demo_interactive_visualization()
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed. Check prerequisites.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 