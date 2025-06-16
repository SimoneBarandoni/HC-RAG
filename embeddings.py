from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j import GraphDatabase
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the embedding generator.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        
    def generate_embeddings(self):
        """Generate and add embeddings to all nodes in the graph"""
        with self.driver.session() as session:
            # 1. Add embeddings to Product nodes
            self._add_product_embeddings(session)
            
            # 2. Add embeddings to Document nodes
            self._add_document_embeddings(session)
            
            # 3. Add embeddings to Annotation nodes
            self._add_annotation_embeddings(session)
            
        logger.info("Added embeddings to all nodes")
        
    def _add_product_embeddings(self, session):
        """Add embeddings to Product nodes"""
        products = session.run("""
            MATCH (p:Product)
            RETURN p.product_id as id, 
                   p.name as name,
                   p.category_name as category
        """)
        
        for product in products:
            # Create text representation
            text = f"{product['name']} {product['category']}"
            # Generate embedding
            embedding = self.model.encode(text)
            # Add to node
            session.run("""
                MATCH (p:Product {product_id: $id})
                SET p.embedding = $embedding
            """, id=product['id'], embedding=embedding.tolist())
            
        logger.info("Added embeddings to Product nodes")
        
    def _add_document_embeddings(self, session):
        """Add embeddings to Document nodes"""
        documents = session.run("""
            MATCH (d:Document)
            RETURN d.filename as filename,
                   d.document_name as name
        """)
        
        for doc in documents:
            # Create text representation
            text = f"{doc['name']} {doc['filename']}"
            # Generate embedding
            embedding = self.model.encode(text)
            # Add to node
            session.run("""
                MATCH (d:Document {filename: $filename})
                SET d.embedding = $embedding
            """, filename=doc['filename'], embedding=embedding.tolist())
            
        logger.info("Added embeddings to Document nodes")
        
    def _add_annotation_embeddings(self, session):
        """Add embeddings to Annotation nodes"""
        annotations = session.run("""
            MATCH (a:Annotation)
            RETURN a.filename as filename,
                   a.content as content
        """)
        
        for ann in annotations:
            # Create text representation
            text = f"{ann['filename']} {ann['content'] if ann['content'] else ''}"
            # Generate embedding
            embedding = self.model.encode(text)
            # Add to node
            session.run("""
                MATCH (a:Annotation {filename: $filename})
                SET a.embedding = $embedding
            """, filename=ann['filename'], embedding=embedding.tolist())
            
        logger.info("Added embeddings to Annotation nodes")

# Usage example
if __name__ == "__main__":
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # Initialize and run
    generator = EmbeddingGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        generator.generate_embeddings()
    finally:
        generator.close() 