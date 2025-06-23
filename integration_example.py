"""
Integration Example: Using isRelevant with Knowledge Graph

This example shows how to integrate the isRelevant function with your existing
knowledge graph and RAG system for node ranking and selection.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from relevance_scorer import (
    isRelevant, QueryInput, NodeInput, ScorerType, QueryIntent
)
from sentence_transformers import SentenceTransformer


class KnowledgeGraphRelevanceIntegration:
    """Integration class for using relevance scoring with knowledge graph."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data()
    
    def load_data(self):
        """Load your existing data."""
        try:
            self.products = pd.read_csv('data/Product.csv', delimiter=';')
            self.categories = pd.read_csv('data/ProductCategory.csv', delimiter=';')
            print(f"Loaded {len(self.products)} products and {len(self.categories)} categories")
        except FileNotFoundError:
            print("Data files not found, using sample data")
            self.products = pd.DataFrame({
                'ProductID': [1, 2, 3, 4, 5],
                'Name': ['Mountain Bike Pro', 'Road Bike Elite', 'Touring Bike Classic', 
                        'HL Road Frame - Red', 'Sport-100 Helmet'],
                'Color': ['Red', 'Blue', 'Black', 'Red', 'Red'],
                'ListPrice': [899.99, 1299.99, 799.99, 1059.31, 34.99]
            })
            self.categories = pd.DataFrame({
                'ProductCategoryID': [1, 2, 3, 4, 5],
                'Name': ['Bikes', 'Components', 'Clothing', 'Mountain Bikes', 'Road Bikes']
            })
    
    def create_query_input(self, user_query: str) -> QueryInput:
        """Convert user query to QueryInput format."""
        # Simple intent classification
        intent = self._classify_intent(user_query)
        
        # Simple entity extraction
        entities = self._extract_entities(user_query)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(user_query)
        
        return QueryInput(
            text=user_query,
            embeddings=np.array(embeddings),
            parsed_entities=entities,
            intent=intent
        )
    
    def create_node_input(self, data_row: pd.Series, node_type: str) -> NodeInput:
        """Convert database row to NodeInput format."""
        # Create text representation
        if node_type == 'product':
            text = f"Product: {data_row.get('Name', 'Unknown')}"
            if 'Color' in data_row and pd.notna(data_row['Color']):
                text += f", Color: {data_row['Color']}"
            if 'ListPrice' in data_row and pd.notna(data_row['ListPrice']):
                text += f", Price: ${data_row['ListPrice']}"
        else:
            text = f"Category: {data_row.get('Name', 'Unknown')}"
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(text)
        
        # Mock graph relations (in real implementation, query from Neo4j)
        graph_relations = self._get_mock_graph_relations(data_row, node_type)
        
        return NodeInput(
            text_or_db=data_row.to_dict(),
            embeddings=np.array(embeddings),
            graph_relations=graph_relations,
            node_type=node_type
        )
    
    def rank_nodes_by_relevance(self, user_query: str, scorer_type: ScorerType = ScorerType.COMPOSITE) -> List[Tuple[Dict, float, str]]:
        """
        Main function: Rank all nodes by relevance to user query.
        
        Returns:
            List of (node_data, relevance_score, node_type) tuples, sorted by relevance
        """
        query_input = self.create_query_input(user_query)
        
        scored_nodes = []
        
        # Score product nodes
        print(f"Scoring {len(self.products)} product nodes...")
        for _, product in self.products.iterrows():
            node_input = self.create_node_input(product, 'product')
            score = isRelevant(query_input, node_input, scorer_type)
            scored_nodes.append((product.to_dict(), score, 'product'))
        
        # Score category nodes
        print(f"Scoring {len(self.categories)} category nodes...")
        for _, category in self.categories.iterrows():
            node_input = self.create_node_input(category, 'category')
            score = isRelevant(query_input, node_input, scorer_type)
            scored_nodes.append((category.to_dict(), score, 'category'))
        
        # Sort by relevance score (descending)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return scored_nodes
    
    def select_top_nodes_for_context(self, scored_nodes: List[Tuple[Dict, float, str]], 
                                   max_context_length: int = 2000) -> List[Tuple[Dict, float, str]]:
        """
        Select top nodes for RAG context considering length constraints.
        
        This is a simplified version - in practice you'd estimate token counts.
        """
        selected_nodes = []
        current_length = 0
        
        for node_data, score, node_type in scored_nodes:
            # Estimate content length (simplified)
            estimated_length = len(str(node_data)) + 50  # Add buffer for formatting
            
            if current_length + estimated_length <= max_context_length:
                selected_nodes.append((node_data, score, node_type))
                current_length += estimated_length
            else:
                break
        
        return selected_nodes
    
    def demonstrate_rag_integration(self, user_query: str):
        """Demonstrate complete RAG integration workflow."""
        print("=" * 60)
        print(f"RAG INTEGRATION DEMO")
        print(f"Query: '{user_query}'")
        print("=" * 60)
        
        # Step 1: Rank all nodes by relevance
        print("\n1. RANKING NODES BY RELEVANCE")
        print("-" * 30)
        
        scored_nodes = self.rank_nodes_by_relevance(user_query, ScorerType.COMPOSITE)
        
        print(f"Top 10 most relevant nodes:")
        for i, (node_data, score, node_type) in enumerate(scored_nodes[:10]):
            name = node_data.get('Name', 'Unknown')
            print(f"{i+1:2d}. {name:30} ({node_type:8}) - Score: {score:.3f}")
        
        # Step 2: Select nodes for context
        print(f"\n2. SELECTING NODES FOR CONTEXT")
        print("-" * 30)
        
        selected_nodes = self.select_top_nodes_for_context(scored_nodes, max_context_length=1000)
        
        print(f"Selected {len(selected_nodes)} nodes for context:")
        for node_data, score, node_type in selected_nodes:
            name = node_data.get('Name', 'Unknown')
            print(f"  - {name} ({node_type}) - Score: {score:.3f}")
        
        # Step 3: Compare scorer types
        print(f"\n3. COMPARING SCORER TYPES")
        print("-" * 30)
        
        # Test with top product
        if self.products is not None and len(self.products) > 0:
            query_input = self.create_query_input(user_query)
            node_input = self.create_node_input(self.products.iloc[0], 'product')
            
            composite_score = isRelevant(query_input, node_input, ScorerType.COMPOSITE)
            parallel_score = isRelevant(query_input, node_input, ScorerType.PARALLEL)
            router_score = isRelevant(query_input, node_input, ScorerType.ROUTER)
            
            print(f"Scores for '{self.products.iloc[0]['Name']}':")
            print(f"  Composite: {composite_score:.3f}")
            print(f"  Parallel:  {parallel_score:.3f}")
            print(f"  Router:    {router_score:.3f}")
        
        return selected_nodes
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Simple intent classification."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['find', 'search', 'show', 'get']):
            if 'manual' in query_lower or 'documentation' in query_lower:
                return QueryIntent.DOCUMENT_REQUEST
            elif 'compare' in query_lower:
                return QueryIntent.COMPARISON_REQUEST
            else:
                return QueryIntent.PRODUCT_SEARCH
        elif 'help' in query_lower or 'support' in query_lower:
            return QueryIntent.TECHNICAL_SUPPORT
        elif 'spec' in query_lower or 'specification' in query_lower:
            return QueryIntent.SPECIFICATION_INQUIRY
        else:
            return QueryIntent.PRODUCT_SEARCH
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Simple entity extraction."""
        entities = {}
        query_lower = query.lower()
        
        # Product entities
        product_entities = {}
        
        # Colors
        colors = ['red', 'blue', 'black', 'white', 'green', 'yellow']
        found_colors = [color for color in colors if color in query_lower]
        if found_colors:
            product_entities['features'] = found_colors
        
        # Categories
        if 'bike' in query_lower:
            if 'mountain' in query_lower:
                product_entities['category'] = 'mountain bikes'
            elif 'road' in query_lower:
                product_entities['category'] = 'road bikes'
            else:
                product_entities['category'] = 'bikes'
        
        if product_entities:
            entities['product'] = product_entities
        
        return entities
    
    def _get_mock_graph_relations(self, data_row: pd.Series, node_type: str) -> Dict[str, Any]:
        """Generate mock graph relations (replace with real Neo4j queries)."""
        if node_type == 'product':
            return {
                'incoming': ['category_1', 'brand_1'],
                'outgoing': ['review_1', 'review_2', 'spec_1'],
                'relationship_weights': {'category': 0.8, 'brand': 0.6}
            }
        else:  # category
            return {
                'incoming': ['parent_category'] if 'ParentProductCategoryID' in data_row else [],
                'outgoing': ['product_1', 'product_2', 'product_3'],
                'relationship_weights': {'products': 0.9}
            }


def main():
    """Main demonstration function."""
    # Initialize the integration
    integration = KnowledgeGraphRelevanceIntegration()
    
    # Test queries
    test_queries = [
        "Find red mountain bikes under $1000",
        "Show me all bike categories", 
        "I need blue road bikes",
        "Find bike components"
    ]
    
    for query in test_queries:
        selected_nodes = integration.demonstrate_rag_integration(query)
        print(f"\nSelected {len(selected_nodes)} nodes for RAG context\n")
        print("=" * 60)


if __name__ == "__main__":
    main() 