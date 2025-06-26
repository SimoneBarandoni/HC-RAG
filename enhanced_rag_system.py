#!/usr/bin/env python3
"""
Enhanced Knowledge Graph RAG System

This builds upon your existing EmbeddingRAGSystem to create a complete RAG pipeline
that integrates:
1. Your advanced relevance scoring (isRelevant.py)
2. Knowledge graph structure (Neo4j)
3. LLM generation with context optimization
4. Query parsing and entity extraction
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import your existing components
from main import EmbeddingRAGSystem
from isRelevant import (
    QueryInput, NodeInput, QueryIntent, ScorerType, 
    isRelevant, priority_matrix
)
from graph_relevance_integration import GraphRelevanceScorer
from openai import OpenAI


@dataclass
class RAGResponse:
    """Structured response from RAG system"""
    query: str
    answer: str
    context_used: List[Dict[str, Any]]
    relevance_scores: List[float]
    metadata: Dict[str, Any]
    processing_time: float


class EnhancedKnowledgeGraphRAG:
    """
    Enhanced RAG system that combines your existing components with generation
    """
    
    def __init__(self, 
                 embeddings_path: str = "data/knowledge_graph_embeddings.pkl",
                 llm_base_url: str = "http://localhost:11434/v1",
                 llm_model: str = "gemma3:1b",
                 max_context_tokens: int = 2000):
        """
        Initialize the enhanced RAG system
        
        Args:
            embeddings_path: Path to embeddings file
            llm_base_url: Base URL for LLM API
            llm_model: LLM model name
            max_context_tokens: Maximum tokens for context window
        """
        print("🚀 Initializing Enhanced Knowledge Graph RAG System...")
        
        # Initialize core components
        self.embedding_rag = EmbeddingRAGSystem(embeddings_path)
        self.relevance_scorer = GraphRelevanceScorer(embeddings_path)
        
        # Initialize LLM client for generation
        self.llm_client = OpenAI(
            base_url=llm_base_url,
            api_key=llm_model
        )
        self.llm_model = llm_model
        self.max_context_tokens = max_context_tokens
        
        print("✅ Enhanced RAG System ready!")
    
    def close(self):
        """Close all connections"""
        if hasattr(self, 'embedding_rag'):
            self.embedding_rag.close()
        if hasattr(self, 'relevance_scorer'):
            self.relevance_scorer.close()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _infer_query_intent(self, query: str) -> QueryIntent:
        """Infer query intent from text"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['manual', 'documentation', 'guide', 'instruction']):
            return QueryIntent.DOCUMENT_REQUEST
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return QueryIntent.COMPARISON_REQUEST
        elif any(word in query_lower for word in ['spec', 'specification', 'technical', 'details']):
            return QueryIntent.SPECIFICATION_INQUIRY
        elif any(word in query_lower for word in ['help', 'support', 'troubleshoot', 'fix', 'problem']):
            return QueryIntent.TECHNICAL_SUPPORT
        else:
            return QueryIntent.PRODUCT_SEARCH
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction (can be enhanced with NER)"""
        # This is a simplified version - you could use spaCy or other NER tools
        keywords = []
        important_words = [word.lower().strip('.,!?') for word in text.split() 
                          if len(word) > 3 and word.lower() not in ['find', 'show', 'what', 'where', 'when', 'how']]
        return important_words[:5]  # Limit to 5 entities
    
    def retrieve_and_rank(self, query: str, 
                         top_k: int = 20, 
                         similarity_threshold: float = 0.25,
                         scorer_type: ScorerType = ScorerType.COMPOSITE) -> Tuple[List[Dict], QueryInput]:
        """
        Retrieve relevant nodes and rank them using advanced relevance scoring
        """
        print(f"🔍 Retrieving and ranking nodes for: '{query}'")
        
        # Step 1: Get initial candidates using embedding similarity
        rag_results = self.embedding_rag.process_query(
            query, 
            top_k=top_k * 2,  # Get more candidates for ranking
            similarity_threshold=similarity_threshold * 0.7  # Lower threshold for initial retrieval
        )
        
        if not rag_results["results"]:
            return [], None
        
        # Step 2: Create QueryInput for relevance scoring
        query_input = QueryInput(
            text=query,
            embeddings=rag_results["query_embedding"],
            entities=self._extract_entities_simple(query),
            intent=self._infer_query_intent(query)
        )
        
        # Step 3: Convert RAG results to NodeInput and score relevance
        scored_nodes = []
        
        for result in rag_results["results"]:
            # Create NodeInput
            node_input = self._create_node_input_from_result(result)
            
            # Calculate relevance score
            relevance_score = isRelevant(query_input, node_input, scorer_type)
            
            # Combine with similarity score (weighted average)
            combined_score = (relevance_score * 0.7 + result["similarity_score"] * 0.3)
            
            scored_nodes.append({
                "content": result["content"],
                "metadata": result["metadata"],
                "similarity_score": result["similarity_score"],
                "relevance_score": relevance_score,
                "combined_score": combined_score,
                "node_input": node_input
            })
        
        # Step 4: Sort by combined score and return top_k
        scored_nodes.sort(key=lambda x: x["combined_score"], reverse=True)
        
        print(f"📊 Ranked {len(scored_nodes)} nodes, top score: {scored_nodes[0]['combined_score']:.3f}")
        
        return scored_nodes[:top_k], query_input
    
    def _create_node_input_from_result(self, result: Dict) -> NodeInput:
        """Convert RAG result to NodeInput for relevance scoring"""
        content = result["content"]
        metadata = result["metadata"]
        
        # Determine node type
        if metadata.get("type") == "database_table":
            table_name = metadata.get("table_name", "").lower()
            if "product" in table_name:
                node_type = "product"
            elif "category" in table_name:
                node_type = "category"
            else:
                node_type = "specification"
        elif metadata.get("type") == "pdf_document":
            node_type = "document"
        else:
            node_type = "specification"
        
        # Extract entities from content
        entities = self._extract_entities_simple(content)
        
        # Create embeddings (you might want to use cached embeddings)
        embeddings = self.embedding_rag.embedding_generator.model.encode([content])[0]
        
        return NodeInput(
            text=content,
            embeddings=embeddings,
            graph_relations={"metadata": metadata},
            node_type=node_type,
            entities=entities
        )
    
    def optimize_context(self, scored_nodes: List[Dict], query_input: QueryInput) -> List[Dict]:
        """
        Optimize context selection based on token limits and relevance
        """
        print(f"🎯 Optimizing context from {len(scored_nodes)} candidates...")
        
        selected_nodes = []
        total_tokens = 0
        
        # Reserve tokens for system prompt, query, and generation
        available_tokens = self.max_context_tokens - 500
        
        for node in scored_nodes:
            content = node["content"]
            estimated_tokens = self._estimate_tokens(content)
            
            if total_tokens + estimated_tokens <= available_tokens:
                selected_nodes.append(node)
                total_tokens += estimated_tokens
            else:
                break
        
        print(f"✅ Selected {len(selected_nodes)} nodes using ~{total_tokens} tokens")
        
        return selected_nodes
    
    def generate_response(self, query: str, context_nodes: List[Dict], query_input: QueryInput) -> str:
        """
        Generate response using LLM with optimized context
        """
        print("🤖 Generating response...")
        
        # Build context from selected nodes
        context_parts = []
        for i, node in enumerate(context_nodes, 1):
            content = node["content"]
            metadata = node["metadata"]
            score = node["combined_score"]
            
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{content}\n")
        
        context_text = "\n".join(context_parts)
        
        # Create system prompt based on query intent
        system_prompts = {
            QueryIntent.PRODUCT_SEARCH: "You are a helpful product search assistant. Use the provided product information to answer the user's query about finding products.",
            QueryIntent.DOCUMENT_REQUEST: "You are a documentation assistant. Use the provided documents and manuals to help the user find the information they need.",
            QueryIntent.TECHNICAL_SUPPORT: "You are a technical support specialist. Use the provided technical information to help troubleshoot and solve the user's problem.",
            QueryIntent.COMPARISON_REQUEST: "You are a product comparison specialist. Use the provided information to help compare different products or options.",
            QueryIntent.SPECIFICATION_INQUIRY: "You are a technical specifications expert. Use the provided technical details to answer the user's specification questions."
        }
        
        system_prompt = system_prompts.get(query_input.intent, system_prompts[QueryIntent.PRODUCT_SEARCH])
        
        # Create the full prompt
        full_prompt = f"""Based on the following information, please answer the user's question.

CONTEXT INFORMATION:
{context_text}

USER QUESTION: {query}

Please provide a helpful, accurate response based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I found {len(context_nodes)} relevant pieces of information, but encountered an error generating the response. The most relevant information relates to: {context_nodes[0]['content'][:200]}..." if context_nodes else "No relevant information found."
    
    def query(self, 
              query: str,
              top_k: int = 10,
              similarity_threshold: float = 0.25,
              scorer_type: ScorerType = ScorerType.COMPOSITE,
              include_debug: bool = False) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve → rank → optimize → generate
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🔥 PROCESSING RAG QUERY")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Retrieve and rank
            scored_nodes, query_input = self.retrieve_and_rank(
                query, top_k, similarity_threshold, scorer_type
            )
            
            if not scored_nodes:
                return RAGResponse(
                    query=query,
                    answer="I couldn't find any relevant information for your query.",
                    context_used=[],
                    relevance_scores=[],
                    metadata={"error": "No relevant nodes found"},
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Optimize context
            optimized_context = self.optimize_context(scored_nodes, query_input)
            
            # Step 3: Generate response
            answer = self.generate_response(query, optimized_context, query_input)
            
            # Prepare metadata
            metadata = {
                "query_intent": query_input.intent.value,
                "nodes_retrieved": len(scored_nodes),
                "nodes_used": len(optimized_context),
                "scorer_type": scorer_type.value,
                "top_relevance_score": scored_nodes[0]["combined_score"] if scored_nodes else 0
            }
            
            if include_debug:
                metadata["debug"] = {
                    "all_scores": [(node["combined_score"], node["content"][:100]) for node in scored_nodes[:5]],
                    "query_entities": query_input.entities
                }
            
            processing_time = time.time() - start_time
            print(f"✅ Query processed in {processing_time:.2f} seconds")
            
            return RAGResponse(
                query=query,
                answer=answer,
                context_used=[{
                    "content": node["content"],
                    "metadata": node["metadata"],
                    "relevance_score": node["combined_score"]
                } for node in optimized_context],
                relevance_scores=[node["combined_score"] for node in optimized_context],
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return RAGResponse(
                query=query,
                answer=f"An error occurred processing your query: {str(e)}",
                context_used=[],
                relevance_scores=[],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )


def demo_enhanced_rag():
    """Demonstrate the enhanced RAG system"""
    print("🎬 ENHANCED KNOWLEDGE GRAPH RAG DEMO")  
    print("="*60)
    
    # Initialize system
    rag = EnhancedKnowledgeGraphRAG()
    
    # Test queries
    test_queries = [
        "Find red mountain bikes under $1000",
        "What are the technical specifications for mountain bike brakes?",
        "Show me the maintenance guide for bicycles",
        "Compare different mountain bike frame materials",
        "I need help with bike assembly"
    ]
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 TEST {i}: {query}")
            print("-"*50)
            
            # Process query
            response = rag.query(query, include_debug=True)
            
            # Display results
            print(f"📝 Answer: {response.answer}")
            print(f"⏱️ Processing time: {response.processing_time:.2f}s")
            print(f"📊 Context used: {response.metadata['nodes_used']} nodes")
            print(f"🎯 Intent: {response.metadata['query_intent']}")
            print(f"⭐ Top relevance: {response.metadata['top_relevance_score']:.3f}")
            
            if response.context_used:
                print(f"🔍 Top context source: {response.context_used[0]['content'][:150]}...")
    
    finally:
        rag.close()


if __name__ == "__main__":
    demo_enhanced_rag() 