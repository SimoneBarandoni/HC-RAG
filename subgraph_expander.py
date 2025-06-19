#!/usr/bin/env python3
"""
Subgraph Expander

This module provides functionality to expand subgraphs by adding connected nodes
through specific relationships. It augments the initial matched nodes with their
related context from the knowledge graph.

Key Features:
- Expands subgraphs with ANNOTATION and DESCRIBED_BY relationships
- Preserves original similarity scores for matched nodes
- Adds context nodes with appropriate styling
- Maintains relationship information for visualization
"""

from typing import Dict, List, Any, Set, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubgraphExpander:
    """
    Expands subgraphs by adding connected nodes through specific relationships
    """
    
    def __init__(self, neo4j_driver):
        """
        Initialize with Neo4j driver
        
        Args:
            neo4j_driver: Neo4j driver instance for database queries
        """
        self.driver = neo4j_driver
        
        # Define which relationships to follow for expansion
        self.expansion_relationships = [
            "ANNOTATION",      # Annotations connected to documents
            "DESCRIBED_BY"     # Products described by documents
        ]
    
    def expand_subgraph(self, subgraph_data: Dict[str, List], 
                       expansion_depth: int = 1,
                       max_connected_nodes: int = 20) -> Dict[str, List]:
        """
        Expand subgraph by adding connected nodes through specific relationships
        
        Args:
            subgraph_data: Original subgraph with nodes and edges
            expansion_depth: How many relationship hops to follow (1 or 2)
            max_connected_nodes: Maximum number of connected nodes to add
            
        Returns:
            Expanded subgraph with additional nodes and relationships
        """
        if not self.driver:
            logger.warning("No Neo4j driver available for expansion")
            return subgraph_data
        
        original_nodes = subgraph_data.get("nodes", [])
        original_edges = subgraph_data.get("edges", [])
        
        if not original_nodes:
            logger.info("No nodes to expand")
            return subgraph_data
        
        logger.info(f"Expanding subgraph from {len(original_nodes)} matched nodes...")
        
        # Extract Neo4j identifiable nodes from original subgraph
        expandable_nodes = self._extract_expandable_nodes(original_nodes)
        
        if not expandable_nodes:
            logger.info("No expandable nodes found (no Neo4j connections)")
            return subgraph_data
        
        # Find connected nodes through expansion relationships
        connected_data = self._find_connected_nodes(
            expandable_nodes, 
            expansion_depth, 
            max_connected_nodes
        )
        
        # Merge original and connected data
        expanded_subgraph = self._merge_subgraph_data(
            original_nodes,
            original_edges,
            connected_data["nodes"],
            connected_data["edges"]
        )
        
        logger.info(f"Expanded subgraph: {len(expanded_subgraph['nodes'])} nodes, "
                   f"{len(expanded_subgraph['edges'])} edges")
        
        return expanded_subgraph
    
    def _extract_expandable_nodes(self, original_nodes: List[Dict]) -> List[Dict]:
        """
        Extract nodes that can be expanded (have Neo4j connections)
        
        Args:
            original_nodes: Original matched nodes from RAG search
            
        Returns:
            List of nodes with Neo4j identifiers for expansion
        """
        expandable_nodes = []
        
        for node in original_nodes:
            node_data = node.get("data", {})
            
            # Check if node has Neo4j connection info
            if node_data.get("has_neo4j") and node_data.get("neo4j_data"):
                neo4j_data = node_data["neo4j_data"]
                node_type = node_data.get("type", "Unknown")
                
                # Extract Neo4j identifier based on node type
                neo4j_id = None
                if node_type == "Product" and "id" in neo4j_data:
                    neo4j_id = neo4j_data["id"]
                elif node_type in ["Document", "Annotation"] and "id" in neo4j_data:
                    neo4j_id = neo4j_data["id"]
                elif node_type == "Category" and "id" in neo4j_data:
                    neo4j_id = neo4j_data["id"]
                
                if neo4j_id is not None:
                    expandable_nodes.append({
                        "result_id": node_data["id"],
                        "neo4j_type": node_type,
                        "neo4j_id": neo4j_id,
                        "original_node": node
                    })
        
        logger.info(f"Found {len(expandable_nodes)} expandable nodes")
        return expandable_nodes
    
    def _find_connected_nodes(self, expandable_nodes: List[Dict], 
                            expansion_depth: int,
                            max_connected_nodes: int) -> Dict[str, List]:
        """
        Find nodes connected through expansion relationships
        
        Args:
            expandable_nodes: Nodes that can be expanded
            expansion_depth: Relationship depth to follow
            max_connected_nodes: Maximum connected nodes to add
            
        Returns:
            Dictionary with connected nodes and edges
        """
        connected_nodes = []
        connected_edges = []
        seen_node_ids = set()
        
        with self.driver.session() as session:
            for expandable in expandable_nodes:
                neo4j_type = expandable["neo4j_type"]
                neo4j_id = expandable["neo4j_id"]
                result_id = expandable["result_id"]
                
                # Build query based on node type and relationships
                connected_data = self._query_connected_nodes(
                    session, neo4j_type, neo4j_id, expansion_depth
                )
                
                # Process connected nodes
                for conn_node, relationship_type, direction in connected_data:
                    # Create unique ID for connected node
                    conn_node_id = self._create_node_id(conn_node)
                    
                    if conn_node_id not in seen_node_ids and len(connected_nodes) < max_connected_nodes:
                        # Add connected node
                        connected_nodes.append(self._create_connected_node(conn_node, conn_node_id))
                        seen_node_ids.add(conn_node_id)
                    
                    # Add relationship edge
                    if conn_node_id in seen_node_ids or len(connected_nodes) < max_connected_nodes:
                        edge = self._create_relationship_edge(
                            result_id, conn_node_id, relationship_type, direction
                        )
                        if edge and edge not in connected_edges:
                            connected_edges.append(edge)
        
        logger.info(f"Found {len(connected_nodes)} connected nodes, {len(connected_edges)} relationships")
        
        return {
            "nodes": connected_nodes,
            "edges": connected_edges
        }
    
    def _query_connected_nodes(self, session, node_type: str, node_id: Any, 
                             expansion_depth: int) -> List[Tuple]:
        """
        Query Neo4j for connected nodes through expansion relationships
        
        Args:
            session: Neo4j session
            node_type: Type of the source node
            node_id: ID of the source node
            expansion_depth: How many hops to follow
            
        Returns:
            List of (connected_node, relationship_type, direction) tuples
        """
        connected_data = []
        
        try:
            # Build query based on node type
            if node_type == "Product":
                query = """
                MATCH (p:Product {product_id: $node_id})-[r:DESCRIBED_BY]-(connected)
                RETURN connected, type(r) as rel_type, 
                       CASE WHEN startNode(r) = p THEN 'outgoing' ELSE 'incoming' END as direction
                UNION
                MATCH (p:Product {product_id: $node_id})-[r]-(d:Document)-[r2:ANNOTATION]-(a:Annotation)
                RETURN a as connected, 'ANNOTATION' as rel_type, 'indirect' as direction
                """
            elif node_type == "Document":
                query = """
                MATCH (d:Document {filename: $node_id})-[r:DESCRIBED_BY]-(connected)
                RETURN connected, type(r) as rel_type,
                       CASE WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END as direction
                UNION
                MATCH (d:Document {filename: $node_id})-[r:ANNOTATION]-(connected)
                RETURN connected, type(r) as rel_type,
                       CASE WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END as direction
                """
            elif node_type == "Annotation":
                query = """
                MATCH (a:Annotation {filename: $node_id})-[r:ANNOTATION]-(connected)
                RETURN connected, type(r) as rel_type,
                       CASE WHEN startNode(r) = a THEN 'outgoing' ELSE 'incoming' END as direction
                """
            else:
                # Generic query for unknown node types
                query = """
                MATCH (n)-[r]-(connected)
                WHERE (n.product_id = $node_id OR n.filename = $node_id OR id(n) = $node_id)
                AND type(r) IN ['ANNOTATION', 'DESCRIBED_BY']
                RETURN connected, type(r) as rel_type,
                       CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END as direction
                """
            
            result = session.run(query, node_id=node_id)
            
            for record in result:
                connected_node = dict(record["connected"])
                rel_type = record["rel_type"]
                direction = record["direction"]
                connected_data.append((connected_node, rel_type, direction))
                
        except Exception as e:
            logger.warning(f"Error querying connected nodes for {node_type}[{node_id}]: {e}")
        
        return connected_data
    
    def _create_node_id(self, node_data: Dict) -> str:
        """
        Create unique ID for a connected node
        
        Args:
            node_data: Neo4j node properties
            
        Returns:
            Unique node identifier
        """
        # Try different ID fields based on node properties
        if "product_id" in node_data:
            return f"connected_product_{node_data['product_id']}"
        elif "filename" in node_data:
            return f"connected_file_{hash(node_data['filename']) % 10000}"
        elif "ProductCategoryID" in node_data:
            return f"connected_category_{node_data['ProductCategoryID']}"
        else:
            # Fallback to hash of all properties
            return f"connected_node_{hash(str(sorted(node_data.items()))) % 10000}"
    
    def _create_connected_node(self, node_data: Dict, node_id: str) -> Dict:
        """
        Create visualization node for connected node
        
        Args:
            node_data: Neo4j node properties
            node_id: Unique node identifier
            
        Returns:
            Node data for visualization
        """
        # Determine node label and type
        if "name" in node_data:
            label = str(node_data["name"])
            node_type = "Product" if "product_id" in node_data else "Unknown"
        elif "filename" in node_data:
            label = str(node_data["filename"])
            if "document_name" in node_data:
                node_type = "Document"
            elif "annotation_type" in node_data:
                node_type = "Annotation"
            else:
                node_type = "File"
        elif "Name" in node_data:  # Category nodes
            label = str(node_data["Name"])
            node_type = "Category"
        else:
            label = f"Node {node_id}"
            node_type = "Unknown"
        
        return {
            "data": {
                "id": node_id,
                "label": label[:50],  # Truncate long labels
                "type": node_type,
                "similarity_score": 0.0,  # Connected nodes have no similarity score
                "is_connected": True,  # Mark as connected node
                "neo4j_data": node_data,
                "has_neo4j": True
            }
        }
    
    def _create_relationship_edge(self, source_id: str, target_id: str, 
                                relationship_type: str, direction: str) -> Dict:
        """
        Create visualization edge for relationship
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            direction: Direction of relationship
            
        Returns:
            Edge data for visualization
        """
        # Determine edge direction
        if direction == "incoming":
            # Swap source and target for incoming relationships
            actual_source, actual_target = target_id, source_id
        else:
            actual_source, actual_target = source_id, target_id
        
        edge_id = f"{actual_source}-{actual_target}-{relationship_type}"
        
        return {
            "data": {
                "id": edge_id,
                "source": actual_source,
                "target": actual_target,
                "relationship": relationship_type,
                "is_expansion": True  # Mark as expansion relationship
            }
        }
    
    def _merge_subgraph_data(self, original_nodes: List[Dict], original_edges: List[Dict],
                           connected_nodes: List[Dict], connected_edges: List[Dict]) -> Dict[str, List]:
        """
        Merge original and connected subgraph data
        
        Args:
            original_nodes: Original matched nodes
            original_edges: Original relationships
            connected_nodes: Connected nodes from expansion
            connected_edges: Connected relationships from expansion
            
        Returns:
            Merged subgraph data
        """
        # Combine nodes (original + connected)
        all_nodes = original_nodes.copy()
        
        # Add connected nodes, avoiding duplicates
        existing_ids = {node["data"]["id"] for node in original_nodes}
        for conn_node in connected_nodes:
            if conn_node["data"]["id"] not in existing_ids:
                all_nodes.append(conn_node)
        
        # Combine edges (original + connected)
        all_edges = original_edges.copy()
        
        # Add connected edges, avoiding duplicates
        existing_edge_ids = {edge["data"]["id"] for edge in original_edges}
        for conn_edge in connected_edges:
            if conn_edge["data"]["id"] not in existing_edge_ids:
                all_edges.append(conn_edge)
        
        return {
            "nodes": all_nodes,
            "edges": all_edges
        }


def create_subgraph_expander(neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> SubgraphExpander:
    """
    Factory function to create SubgraphExpander with Neo4j connection
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        Configured SubgraphExpander instance
    """
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        return SubgraphExpander(driver)
    except ImportError:
        logger.error("Neo4j driver not available. Install with: pip install neo4j")
        return SubgraphExpander(None)
    except Exception as e:
        logger.error(f"Failed to create Neo4j connection: {e}")
        return SubgraphExpander(None) 