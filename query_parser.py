"""
Query Parser Module for Knowledge Graph System

This module provides functionality to parse user queries and extract relevant entities
for product knowledge graph searches.

Author: Knowledge Graph System
Date: 2024
"""

import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductEntity(BaseModel):
    """Represents a product entity extracted from user queries"""
    name: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class DocumentEntity(BaseModel):
    """Represents a document entity extracted from user queries"""
    type: Optional[str] = None
    name: Optional[str] = None


class RelationshipEntity(BaseModel):
    """Represents a relationship entity extracted from user queries"""
    type: Optional[str] = None
    direction: Optional[str] = None


class QueryEntities(BaseModel):
    """Container for all extracted entities from a user query"""
    product: ProductEntity = Field(default_factory=ProductEntity)
    document: DocumentEntity = Field(default_factory=DocumentEntity)
    relationship: RelationshipEntity = Field(default_factory=RelationshipEntity)


class QueryParser:
    
    def __init__(self, base_url="http://localhost:11434/v1", model="gemma3:1b"):
        """
        Initialize the query parser.
        
        Args:
            base_url: The base URL for the API (defaults to local Ollama)
            model: The model to use for parsing (defaults to gemma3:1b)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=model,  # Using model name as API key for Ollama
        )
        self.model = model

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query and extract relevant entities.
        
        Args:
            query: The user's natural language query
            
        Returns:
            A dictionary containing the extracted entities with the following structure:
            {
                "product": {"name": str, "features": List[str], "category": str},
                "document": {"type": str, "name": str},
                "relationship": {"type": str, "direction": str}
            }
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are a query parser for a product knowledge graph system.
                        Your task is to extract relevant entities from user queries about products, documents, and relationships.
                        
                        Extract the following entities:
                        
                        1. Product:
                           - name: the main product name or identifier
                           - features: list of features, attributes, or specifications (color, size, material, etc.)
                           - category: the product category or type
                        
                        2. Document:
                           - type: the type of document requested (manual, specification, guide, etc.)
                           - name: the specific document name or identifier
                        
                        3. Relationship:
                           - type: the type of relationship requested (similar, compatible, related, etc.)
                           - direction: the direction of the relationship (incoming, outgoing, bidirectional)
                        
                        Guidelines:
                        - If an entity is not present in the query, use null as the value
                        - Use English for all extracted values
                        - Be specific and accurate in entity extraction
                        - Consider synonyms and variations in product names
                        - Extract all relevant features mentioned in the query
                        """,
                    },
                    {"role": "user", "content": query},
                ],
                response_format=QueryEntities,
            )

            # Get the response content and parse it
            result_str = completion.choices[0].message.content
            result_dict = json.loads(result_str)
            
            # Convert to our structured format
            return QueryEntities(**result_dict).model_dump()

        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}")
            # Return empty structure in case of error
            return {
                "product": {"name": None, "features": [], "category": None},
                "document": {"type": None, "name": None},
                "relationship": {"type": None, "direction": None},
            }

    def parse_queries_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple queries in batch.
        
        Args:
            queries: List of user queries to parse
            
        Returns:
            List of parsed query dictionaries
        """
        results = []
        for query in queries:
            result = self.parse_query(query)
            results.append(result)
        return results


def test_query_parser():
    """
    Test function to demonstrate query parser functionality
    """
    print("🔍 Testing Query Parser")
    print("=" * 40)
    
    # Initialize parser
    parser = QueryParser()
    
    # Test queries
    test_queries = [
        "Find products similar to the black mountain bike size 58",
        "Show me the mountain bike manual",
        "What red road bikes are available in size large?",
        "I need the specification document for touring frames",
        "Find products compatible with mountain bike handlebars",
        "What accessories go with helmets?",
    ]
    
    print("Testing individual queries:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            result = parser.parse_query(query)
            print("   Extracted entities:")
            
            # Pretty print the results
            if result["product"]["name"]:
                print(f"   Product: {result['product']['name']}")
                if result["product"]["features"]:
                    print(f"   Features: {', '.join(result['product']['features'])}")
                if result["product"]["category"]:
                    print(f"   Category: {result['product']['category']}")
            
            if result["document"]["type"] or result["document"]["name"]:
                print(f"   Document: {result['document']['type']} - {result['document']['name']}")
            
            if result["relationship"]["type"]:
                print(f"   Relationship: {result['relationship']['type']} ({result['relationship']['direction']})")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\nQuery parser test completed!")


if __name__ == "__main__":
    test_query_parser() 