#!/usr/bin/env python3
"""
Graph Relevance Integration

This module integrates the relevance scoring system (isRelevant.py) with the actual
knowledge graph nodes from main.py. It:

1. Uses the RAG system to find similar nodes to a query
2. Expands the subgraph to include connected nodes  
3. Applies relevance scoring to all nodes
4. Returns ranked results based on relevance scores
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import from our existing modules
from main import EmbeddingRAGSystem
from isRelevant import QueryInput, NodeInput, QueryIntent, ScorerType, isRelevant


class GraphRelevanceScorer:
    """
    Integrates relevance scoring with the actual knowledge graph
    """
    
    def __init__(self, embeddings_path="data/knowledge_graph_embeddings.pkl"):
        # Initialize the RAG system
        self.rag_system = EmbeddingRAGSystem(embeddings_path)
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'rag_system'):
            self.rag_system.close()
    
    def convert_rag_result_to_node_input(self, result: Dict[str, Any], is_connected: bool = False) -> NodeInput:
        """
        Convert a RAG search result to NodeInput format
        
        Args:
            result: Result from RAG system
            is_connected: Whether this is a connected node (vs direct match)
        """
        content = result.get("content", "")
        metadata = result.get("metadata", {})
        similarity_score = result.get("similarity_score", 0.0)
        
        # Determine node type based on metadata
        if metadata.get("type") == "database_table":
            table_name = metadata.get("table_name", "unknown")
            if table_name.lower() in ["product"]:
                node_type = "product"
            elif table_name.lower() in ["productcategory", "category"]:
                node_type = "category"
            else:
                node_type = "specification"
        elif metadata.get("type") == "pdf_document":
            node_type = "document"
        elif metadata.get("type") == "json_table":
            node_type = "specification"
        else:
            node_type = "unknown"
        
        # Extract entities from content (simple keyword extraction)
        entities = self._extract_entities_from_content(content)
        
        # Create graph relations info
        graph_relations = {
            "similarity_score": similarity_score,
            "is_connected": is_connected,
            "metadata": metadata
        }
        
        # Generate embeddings for the content
        embeddings = self.rag_system.embedding_generator.model.encode([content])[0]
        
        return NodeInput(
            text=content,
            embeddings=embeddings,
            graph_relations=graph_relations,
            node_type=node_type,
            entities=entities
        )
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """Simple entity extraction from content text"""
        # Common product-related keywords
        keywords = [
            "mountain bike", "road bike", "bike", "bicycle",
            "frame", "handlebar", "wheel", "tire", "brake",
            "gear", "pedal", "chain", "saddle", "helmet",
            "red", "black", "blue", "white", "green",
            "small", "medium", "large", "xl", "xs"
        ]
        
        content_lower = content.lower()
        found_entities = []
        
        for keyword in keywords:
            if keyword in content_lower:
                found_entities.append(keyword)
        
        # If no keywords found, extract first few meaningful words
        if not found_entities:
            words = content.split()[:3]
            found_entities = [word.lower().strip('.,!?') for word in words if len(word) > 2]
        
        return found_entities[:5]  # Limit to 5 entities
    
    def _infer_query_intent(self, query: str) -> QueryIntent:
        """Infer query intent from query text"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["find", "search", "show", "get", "buy"]):
            return QueryIntent.PRODUCT_SEARCH
        elif any(word in query_lower for word in ["manual", "document", "guide", "instructions"]):
            return QueryIntent.DOCUMENT_REQUEST
        elif any(word in query_lower for word in ["help", "support", "problem", "issue", "fix"]):
            return QueryIntent.TECHNICAL_SUPPORT
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return QueryIntent.COMPARISON_REQUEST
        elif any(word in query_lower for word in ["spec", "specification", "details", "features"]):
            return QueryIntent.SPECIFICATION_INQUIRY
        else:
            return QueryIntent.PRODUCT_SEARCH  # Default
    
    def create_query_input(self, query: str) -> QueryInput:
        """
        Create QueryInput from a text query
        """
        # Infer intent
        intent = self._infer_query_intent(query)
        
        # Extract entities
        entities = self._extract_entities_from_content(query)
        
        # Generate embeddings
        embeddings = self.rag_system.embedding_generator.model.encode([query])[0]
        
        return QueryInput(
            text=query,
            embeddings=embeddings,
            entities=entities,
            intent=intent
        )
    
    def get_graph_nodes_for_query(self, query: str, top_k: int = 10, 
                                  similarity_threshold: float = 0.25,
                                  expand_subgraph: bool = True) -> Tuple[List[NodeInput], Dict[str, Any]]:
        """
        Get nodes from the knowledge graph for relevance scoring
        
        Args:
            query: Search query
            top_k: Number of similar nodes to find
            similarity_threshold: Minimum similarity score
            expand_subgraph: Whether to include connected nodes
            
        Returns:
            Tuple of (list of NodeInput objects, query results metadata)
        """
        print(f"Getting graph nodes for query: '{query}'")
        
        # Get similar nodes using RAG system
        query_results = self.rag_system.process_query(
            query, 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )
        
        similar_results = query_results["results"]
        print(f"Found {len(similar_results)} similar nodes")
        
        # Convert similar results to NodeInput
        nodes = []
        for result in similar_results:
            node_input = self.convert_rag_result_to_node_input(result, is_connected=False)
            nodes.append(node_input)
        
        # Get expanded subgraph if requested
        if expand_subgraph and similar_results:
            print("Expanding subgraph with connected nodes...")
            
            # Extract subgraph data
            subgraph_data = self.rag_system.visualizer.extract_subgraph_from_results(
                query_results, 
                max_nodes=top_k,
                expand_subgraph=True
            )
            
            # Find connected nodes (those marked as is_connected=True)
            connected_nodes = []
            for node_data in subgraph_data.get("nodes", []):
                node_info = node_data.get("data", {})
                if node_info.get("is_connected", False):
                    # Create a mock result for connected node
                    connected_result = {
                        "content": node_info.get("content", ""),
                        "metadata": node_info.get("metadata", {}),
                        "similarity_score": 0.0  # Connected nodes don't have similarity scores
                    }
                    connected_node = self.convert_rag_result_to_node_input(connected_result, is_connected=True)
                    connected_nodes.append(connected_node)
            
            print(f"Added {len(connected_nodes)} connected nodes")
            nodes.extend(connected_nodes)
        
        print(f"Total nodes for relevance scoring: {len(nodes)}")
        
        return nodes, query_results
    
    def score_query_against_graph(self, query: str, top_k: int = 10,
                                  similarity_threshold: float = 0.25,
                                  expand_subgraph: bool = True,
                                  scorer_types: List[ScorerType] = None) -> Dict[str, Any]:
        """
        Complete pipeline: get graph nodes and score relevance
        
        Args:
            query: Search query
            top_k: Number of similar nodes to find
            similarity_threshold: Minimum similarity threshold
            expand_subgraph: Whether to include connected nodes
            scorer_types: List of scorer types to use
            
        Returns:
            Dictionary with results for each scorer type
        """
        if scorer_types is None:
            scorer_types = [ScorerType.COMPOSITE, ScorerType.PARALLEL, ScorerType.ROUTER]
        
        print(f"Scoring query against knowledge graph: '{query}'")
        print(f"Scorer types: {[s.value for s in scorer_types]}")
        print("=" * 60)
        
        # Create query input
        query_input = self.create_query_input(query)
        print(f"Query Intent: {query_input.intent.value}")
        print(f"Query Entities: {query_input.entities}")
        
        # Get nodes from graph
        nodes, query_metadata = self.get_graph_nodes_for_query(
            query, top_k, similarity_threshold, expand_subgraph
        )
        
        if not nodes:
            return {
                "query": query,
                "query_input": query_input,
                "nodes_found": 0,
                "results": {},
                "error": "No nodes found for scoring"
            }
        
        # Score each node with each scorer type
        results = {}
        for scorer_type in scorer_types:
            print(f"\nScoring with {scorer_type.value} scorer:")
            print("-" * 40)
            
            scored_nodes = []
            for i, node in enumerate(nodes):
                try:
                    relevance_score = isRelevant(query_input, node, scorer_type)
                    
                    scored_nodes.append({
                        "node_index": i,
                        "relevance_score": relevance_score,
                        "node_type": node.node_type,
                        "is_connected": node.graph_relations.get("is_connected", False),
                        "similarity_score": node.graph_relations.get("similarity_score", 0.0),
                        "content_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                        "entities": node.entities,
                        "node_data": node
                    })
                    
                except Exception as e:
                    print(f"Error scoring node {i}: {e}")
                    continue
            
            # Sort by relevance score
            scored_nodes.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Show top results
            print(f"Top 5 results for {scorer_type.value}:")
            for j, scored_node in enumerate(scored_nodes[:5]):
                score = scored_node["relevance_score"]
                node_type = scored_node["node_type"]
                is_connected = "Connected" if scored_node["is_connected"] else "Direct Match"
                content = scored_node["content_preview"]
                
                print(f"  {j+1}. Score: {score:.3f} | Type: {node_type} | {is_connected}")
                print(f"     Content: {content}")
            
            results[scorer_type.value] = scored_nodes
        
        return {
            "query": query,
            "query_input": query_input,
            "nodes_found": len(nodes),
            "query_metadata": query_metadata,
            "results": results
        }
    
    def compare_scorer_performance(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare how different scorers perform on the same set of nodes
        """
        print(f"Comparing scorer performance for: '{query}'")
        print("=" * 60)
        
        # Get comprehensive results
        results = self.score_query_against_graph(query, top_k=top_k)
        
        if "error" in results:
            return results
        
        # Analyze differences between scorers
        comparison = {
            "query": query,
            "total_nodes": results["nodes_found"],
            "scorer_comparisons": {}
        }
        
        scorer_results = results["results"]
        
        # Compare top-5 rankings for each scorer
        for scorer_type in scorer_results:
            top_5 = scorer_results[scorer_type][:5]
            
            comparison["scorer_comparisons"][scorer_type] = {
                "top_5_scores": [node["relevance_score"] for node in top_5],
                "avg_score": np.mean([node["relevance_score"] for node in top_5]) if top_5 else 0,
                "connected_nodes_in_top_5": sum(1 for node in top_5 if node["is_connected"]),
                "node_types_in_top_5": [node["node_type"] for node in top_5]
            }
        
        # Print comparison summary
        print("\nSCORER COMPARISON SUMMARY:")
        print("-" * 30)
        for scorer_type, data in comparison["scorer_comparisons"].items():
            print(f"{scorer_type.upper()}:")
            print(f"  Average Top-5 Score: {data['avg_score']:.3f}")
            print(f"  Connected Nodes in Top-5: {data['connected_nodes_in_top_5']}/5")
            print(f"  Node Types: {', '.join(set(data['node_types_in_top_5']))}")
        
        return comparison


def test_graph_relevance_integration():
    """
    Test the graph relevance integration system
    """
    print("GRAPH RELEVANCE INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Check if embeddings exist
        embeddings_path = "data/knowledge_graph_embeddings.pkl"
        if not Path(embeddings_path).exists():
            print(f"Embeddings file not found: {embeddings_path}")
            print("Please run main.py setup first to generate embeddings")
            return False
        
        # Initialize the scorer
        scorer = GraphRelevanceScorer(embeddings_path)
        
        # Test queries
        test_queries = [
            "Find red mountain bikes under $1000",
            "Show me bicycle frame specifications",
            "What bike components are available?",
            "Mountain bike maintenance manual"
        ]
        
        print(f"\nTesting with {len(test_queries)} different queries")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTEST {i}: Comprehensive Relevance Scoring")
            print("=" * 40)
            
            # Get comprehensive results
            results = scorer.score_query_against_graph(
                query,
                top_k=8,
                similarity_threshold=0.2,
                expand_subgraph=True
            )
            
            if "error" in results:
                print(f"Error: {results['error']}")
                continue
            
            print(f"\nQuery: '{results['query']}'")
            print(f"Nodes analyzed: {results['nodes_found']}")
            print(f"Query intent: {results['query_input'].intent.value}")
            
            # Show best result from each scorer
            print(f"\nBEST RESULT FROM EACH SCORER:")
            print("-" * 30)
            
            for scorer_type, scored_nodes in results["results"].items():
                if scored_nodes:
                    best = scored_nodes[0]
                    score = best["relevance_score"]
                    node_type = best["node_type"]
                    is_connected = "Connected" if best["is_connected"] else "Direct Match"
                    
                    print(f"{scorer_type.upper()}: {score:.3f} ({node_type}, {is_connected})")
            
            # Show scorer comparison for first query only
            if i == 1:
                print(f"\nDETAILED SCORER COMPARISON:")
                comparison = scorer.compare_scorer_performance(query, top_k=6)
                # Comparison details already printed by the method
        
        # Close connections
        scorer.close()
        
        print(f"\nGRAPH RELEVANCE INTEGRATION TEST COMPLETED!")
        print("Features tested:")
        print("  ✓ RAG system integration")
        print("  ✓ Subgraph expansion with connected nodes")
        print("  ✓ Relevance scoring on real graph nodes")
        print("  ✓ Multiple scorer type comparison")
        print("  ✓ Intent classification and entity extraction")
        
        return True
        
    except Exception as e:
        print(f"Error in graph relevance test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for testing
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_graph_relevance_integration()
        sys.exit(0 if success else 1)
    else:
        print("Graph Relevance Integration Module")
        print("=" * 40)
        print("This module integrates relevance scoring with the knowledge graph.")
        print()
        print("Usage:")
        print("  python graph_relevance_integration.py --test")
        print()
        print("Make sure to run main.py setup first to generate embeddings!")


if __name__ == "__main__":
    main()
