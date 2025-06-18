import json
import requests
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the structured output format
class ProductEntity(BaseModel):
    name: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class DocumentEntity(BaseModel):
    type: Optional[str] = None
    name: Optional[str] = None


class RelationshipEntity(BaseModel):
    type: Optional[str] = None
    direction: Optional[str] = None


class QueryEntities(BaseModel):
    product: ProductEntity = Field(default_factory=ProductEntity)
    document: DocumentEntity = Field(default_factory=DocumentEntity)
    relationship: RelationshipEntity = Field(default_factory=RelationshipEntity)


class QueryParser:
    def __init__(self):
        """
        Initialize the query parser using OpenAI.
        """
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="gemma3:1b",  # or your actual API key if using OpenAI
        )

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query and extract relevant entities.

        Args:
            query: The user's query

        Returns:
            A dictionary containing the extracted entities
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gemma3:1b",  # or your preferred model
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are a query parser for a product knowledge graph.
                        Your task is to extract relevant entities from user queries.
                        
                        Extract the following entities:
                        1. Product:
                           - name: the main product name
                           - features: list of features (color, size, etc.)
                           - category: the product category
                        
                        2. Document:
                           - type: the type of document requested
                           - name: the specific document name
                        
                        3. Relationship:
                           - type: the type of relationship requested
                           - direction: the direction of the relationship
                        
                        If an entity is not present in the query, use null as value.
                        Use English for all extracted values.
                        """,
                    },
                    {"role": "user", "content": query},
                ],
                response_format=QueryEntities,
            )

            # Get the response content and parse it
            result_str = completion.choices[0].message.content
            result_dict = json.loads(result_str)
            
            # Convert to our structured format using model_dump() instead of dict()
            return QueryEntities(**result_dict).model_dump()

        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            # Return empty dictionary in case of error
            return {
                "product": {"name": None, "features": [], "category": None},
                "document": {"type": None, "name": None},
                "relationship": {"type": None, "direction": None},
            }




# Usage example
if __name__ == "__main__":
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # Initialize RAG system
    rag = RAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Test with some queries
        test_queries = [
            "Find products similar to the black mountain bike size 58",
            #"Show me the mountain bike manual",
            #"What products are compatible with the mountain bike handlebar?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = rag.process_query(query)
            print("Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    finally:
        rag.close()
