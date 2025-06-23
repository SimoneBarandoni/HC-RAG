"""
Test script for the isRelevant function implementation.

This script demonstrates how to use the relevance scorer with real data
from the knowledge graph and provides various testing scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from relevance_scorer import (
    isRelevant, QueryInput, NodeInput, ScorerType, QueryIntent,
    create_sample_query, create_sample_node
)
from sentence_transformers import SentenceTransformer
import json
import os


class RelevanceTestSuite:
    """Test suite for the relevance scoring system."""
    
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load real data
        self.product_data = self._load_product_data()
        self.category_data = self._load_category_data()
        
    def _load_product_data(self) -> pd.DataFrame:
        """Load product data from CSV if available."""
        try:
            return pd.read_csv('data/Product.csv', delimiter=';')
        except FileNotFoundError:
            print("Product.csv not found, using sample data")
            return pd.DataFrame({
                'ProductID': [1, 2, 3],
                'Name': ['Mountain Bike Pro', 'Road Bike Elite', 'Touring Bike Classic'],
                'Color': ['Red', 'Blue', 'Black'],
                'ListPrice': [899.99, 1299.99, 799.99]
            })
    
    def _load_category_data(self) -> pd.DataFrame:
        """Load category data from CSV if available."""
        try:
            return pd.read_csv('data/ProductCategory.csv', delimiter=';')
        except FileNotFoundError:
            print("ProductCategory.csv not found, using sample data")
            return pd.DataFrame({
                'ProductCategoryID': [1, 2, 3],
                'Name': ['Bikes', 'Components', 'Clothing'],
                'ParentProductCategoryID': [None, None, None]
            })
    
    def create_query_from_text(self, query_text: str, intent: QueryIntent, 
                              entities: Dict[str, Any] = None) -> QueryInput:
        """Create a QueryInput from text with real embeddings."""
        if entities is None:
            entities = self._parse_entities_from_text(query_text)
        
        embeddings = self.embedding_model.encode(query_text)
        
        return QueryInput(
            text=query_text,
            embeddings=np.array(embeddings),
            parsed_entities=entities,
            intent=intent
        )
    
    def create_node_from_product(self, product_row: pd.Series) -> NodeInput:
        """Create a NodeInput from a product database row."""
        # Format product data as text for embedding
        product_text = f"Product: {product_row.get('Name', 'Unknown')}"
        if 'Color' in product_row and pd.notna(product_row['Color']):
            product_text += f", Color: {product_row['Color']}"
        if 'ListPrice' in product_row and pd.notna(product_row['ListPrice']):
            product_text += f", Price: ${product_row['ListPrice']}"
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(product_text)
        
        # Create database entry dict
        db_entry = product_row.to_dict()
        
        # Mock graph relations
        graph_relations = {
            'incoming': ['category_1', 'brand_1'],
            'outgoing': ['review_1', 'review_2', 'spec_1'],
            'relationship_weights': {'category': 0.8, 'brand': 0.6, 'reviews': 0.4}
        }
        
        return NodeInput(
            text_or_db=db_entry,
            embeddings=np.array(embeddings),
            graph_relations=graph_relations,
            node_type='product'
        )
    
    def create_node_from_category(self, category_row: pd.Series) -> NodeInput:
        """Create a NodeInput from a category database row."""
        category_text = f"Category: {category_row.get('Name', 'Unknown')}"
        
        embeddings = self.embedding_model.encode(category_text)
        db_entry = category_row.to_dict()
        
        # Mock graph relations for categories (typically more connected)
        graph_relations = {
            'incoming': ['parent_category'],
            'outgoing': ['product_1', 'product_2', 'product_3', 'subcategory_1'],
            'relationship_weights': {'products': 0.9, 'subcategories': 0.7}
        }
        
        return NodeInput(
            text_or_db=db_entry,
            embeddings=np.array(embeddings),
            graph_relations=graph_relations,
            node_type='category'
        )
    
    def _parse_entities_from_text(self, query_text: str) -> Dict[str, Any]:
        """Simple entity parsing from query text."""
        entities = {}
        query_lower = query_text.lower()
        
        # Product entity parsing
        product_entities = {}
        
        # Extract colors
        colors = ['red', 'blue', 'black', 'white', 'green', 'yellow']
        found_colors = [color for color in colors if color in query_lower]
        if found_colors:
            product_entities['features'] = found_colors
        
        # Extract categories
        if 'bike' in query_lower or 'bicycle' in query_lower:
            if 'mountain' in query_lower:
                product_entities['category'] = 'mountain bikes'
            elif 'road' in query_lower:
                product_entities['category'] = 'road bikes'
            elif 'touring' in query_lower:
                product_entities['category'] = 'touring bikes'
            else:
                product_entities['category'] = 'bikes'
        
        # Extract price information
        if 'under' in query_lower and '$' in query_text:
            # Simple price extraction
            import re
            price_match = re.search(r'\$(\d+)', query_text)
            if price_match:
                product_entities['price_range'] = f"under ${price_match.group(1)}"
        
        if product_entities:
            entities['product'] = product_entities
        
        # Document entity parsing
        if 'manual' in query_lower or 'documentation' in query_lower:
            entities['document'] = {'type': 'manual'}
        
        return entities
    
    def run_comprehensive_test(self):
        """Run a comprehensive test of the relevance scoring system."""
        print("=" * 60)
        print("RELEVANCE SCORER COMPREHENSIVE TEST")
        print("=" * 60)
        
        # Test 1: Product search with exact match
        print("\n1. PRODUCT SEARCH - Exact Match Test")
        print("-" * 40)
        
        query1 = self.create_query_from_text(
            "Find red mountain bikes under $1000",
            QueryIntent.PRODUCT_SEARCH
        )
        
        # Test with first few products
        for idx, (_, product) in enumerate(self.product_data.head(3).iterrows()):
            node = self.create_node_from_product(product)
            
            composite_score = isRelevant(query1, node, ScorerType.COMPOSITE)
            parallel_score = isRelevant(query1, node, ScorerType.PARALLEL)
            router_score = isRelevant(query1, node, ScorerType.ROUTER)
            
            print(f"Product: {product.get('Name', 'Unknown')}")
            print(f"  Composite: {composite_score:.3f}")
            print(f"  Parallel:  {parallel_score:.3f}")
            print(f"  Router:    {router_score:.3f}")
            print()
        
        # Test 2: Category search
        print("\n2. CATEGORY SEARCH Test")
        print("-" * 40)
        
        query2 = self.create_query_from_text(
            "Show me all bike categories",
            QueryIntent.PRODUCT_SEARCH,
            {'product': {'category': 'bikes'}}
        )
        
        for idx, (_, category) in enumerate(self.category_data.head(3).iterrows()):
            node = self.create_node_from_category(category)
            
            composite_score = isRelevant(query2, node, ScorerType.COMPOSITE)
            parallel_score = isRelevant(query2, node, ScorerType.PARALLEL)
            router_score = isRelevant(query2, node, ScorerType.ROUTER)
            
            print(f"Category: {category.get('Name', 'Unknown')}")
            print(f"  Composite: {composite_score:.3f}")
            print(f"  Parallel:  {parallel_score:.3f}")
            print(f"  Router:    {router_score:.3f}")
            print()
        
        # Test 3: Document request
        print("\n3. DOCUMENT REQUEST Test")
        print("-" * 40)
        
        query3 = self.create_query_from_text(
            "I need the user manual for mountain bikes",
            QueryIntent.DOCUMENT_REQUEST,
            {'document': {'type': 'manual'}, 'product': {'category': 'mountain bikes'}}
        )
        
        # Create a mock document node
        doc_node = NodeInput(
            text_or_db="Mountain Bike User Manual - Complete guide for assembly, maintenance, and safety procedures for mountain bicycles.",
            embeddings=self.embedding_model.encode("Mountain Bike User Manual - Complete guide for assembly, maintenance, and safety procedures for mountain bicycles."),
            graph_relations={
                'incoming': ['product_mountain_bike'],
                'outgoing': ['safety_section', 'maintenance_section'],
                'relationship_weights': {'products': 0.9, 'sections': 0.8}
            },
            node_type='document'
        )
        
        composite_score = isRelevant(query3, doc_node, ScorerType.COMPOSITE)
        parallel_score = isRelevant(query3, doc_node, ScorerType.PARALLEL)
        router_score = isRelevant(query3, doc_node, ScorerType.ROUTER)
        
        print(f"Document: Mountain Bike Manual")
        print(f"  Composite: {composite_score:.3f}")
        print(f"  Parallel:  {parallel_score:.3f}")
        print(f"  Router:    {router_score:.3f}")
        print()
        
        # Test 4: Comparison of scorer types
        print("\n4. SCORER TYPE COMPARISON")
        print("-" * 40)
        
        test_queries = [
            ("Find blue road bikes", QueryIntent.PRODUCT_SEARCH),
            ("Show me bike documentation", QueryIntent.DOCUMENT_REQUEST),
            ("Compare mountain bike features", QueryIntent.COMPARISON_REQUEST)
        ]
        
        for query_text, intent in test_queries:
            print(f"\nQuery: '{query_text}'")
            query = self.create_query_from_text(query_text, intent)
            
            # Test with first product
            if not self.product_data.empty:
                node = self.create_node_from_product(self.product_data.iloc[0])
                
                scores = {
                    'Composite': isRelevant(query, node, ScorerType.COMPOSITE),
                    'Parallel': isRelevant(query, node, ScorerType.PARALLEL),
                    'Router': isRelevant(query, node, ScorerType.ROUTER)
                }
                
                for scorer_type, score in scores.items():
                    print(f"  {scorer_type:10}: {score:.3f}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "=" * 60)
        print("EDGE CASES AND ERROR HANDLING")
        print("=" * 60)
        
        # Test 1: Empty query
        print("\n1. Empty Query Test")
        empty_query = QueryInput(
            text="",
            embeddings=np.zeros(384),
            parsed_entities={},
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        if not self.product_data.empty:
            node = self.create_node_from_product(self.product_data.iloc[0])
            score = isRelevant(empty_query, node, ScorerType.COMPOSITE)
            print(f"Empty query score: {score:.3f}")
        
        # Test 2: Malformed embeddings
        print("\n2. Malformed Embeddings Test")
        try:
            bad_query = QueryInput(
                text="test query",
                embeddings=np.array([]),  # Empty embedding
                parsed_entities={},
                intent=QueryIntent.PRODUCT_SEARCH
            )
            
            if not self.product_data.empty:
                node = self.create_node_from_product(self.product_data.iloc[0])
                score = isRelevant(bad_query, node, ScorerType.COMPOSITE)
                print(f"Malformed embedding score: {score:.3f}")
        except Exception as e:
            print(f"Handled error: {e}")
        
        # Test 3: Missing graph relations
        print("\n3. Missing Graph Relations Test")
        minimal_node = NodeInput(
            text_or_db="Minimal test node",
            embeddings=self.embedding_model.encode("Minimal test node"),
            graph_relations={},  # Empty relations
            node_type='unknown'
        )
        
        query = self.create_query_from_text("test", QueryIntent.PRODUCT_SEARCH)
        score = isRelevant(query, minimal_node, ScorerType.COMPOSITE)
        print(f"Minimal node score: {score:.3f}")
    
    def benchmark_performance(self, num_queries: int = 10, num_nodes: int = 20):
        """Benchmark the performance of different scorer types."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        import time
        
        # Create test queries and nodes
        queries = []
        nodes = []
        
        # Generate test queries
        test_query_texts = [
            "Find red mountain bikes",
            "Show me bike accessories",
            "I need documentation",
            "Compare road bikes",
            "Blue touring bicycles"
        ]
        
        for i in range(num_queries):
            query_text = test_query_texts[i % len(test_query_texts)]
            query = self.create_query_from_text(query_text, QueryIntent.PRODUCT_SEARCH)
            queries.append(query)
        
        # Generate test nodes
        for i in range(min(num_nodes, len(self.product_data))):
            node = self.create_node_from_product(self.product_data.iloc[i])
            nodes.append(node)
        
        # Benchmark each scorer type
        scorer_types = [ScorerType.COMPOSITE, ScorerType.PARALLEL, ScorerType.ROUTER]
        
        for scorer_type in scorer_types:
            start_time = time.time()
            total_scores = 0
            
            for query in queries:
                for node in nodes:
                    score = isRelevant(query, node, scorer_type)
                    total_scores += score
            
            end_time = time.time()
            avg_score = total_scores / (len(queries) * len(nodes))
            
            print(f"{scorer_type.value.capitalize():10}: "
                  f"{end_time - start_time:.3f}s total, "
                  f"{avg_score:.3f} avg score, "
                  f"{len(queries) * len(nodes)} calculations")


def main():
    """Main function to run all tests."""
    print("Initializing Relevance Scorer Test Suite...")
    test_suite = RelevanceTestSuite()
    
    # Run basic functionality test
    print("\nTesting basic functionality with sample data...")
    query = create_sample_query()
    node = create_sample_node()
    
    print(f"Sample Test Results:")
    print(f"  Composite Score: {isRelevant(query, node, ScorerType.COMPOSITE):.3f}")
    print(f"  Parallel Score:  {isRelevant(query, node, ScorerType.PARALLEL):.3f}")
    print(f"  Router Score:    {isRelevant(query, node, ScorerType.ROUTER):.3f}")
    
    # Run comprehensive tests
    test_suite.run_comprehensive_test()
    
    # Test edge cases
    test_suite.test_edge_cases()
    
    # Performance benchmark
    test_suite.benchmark_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main() 