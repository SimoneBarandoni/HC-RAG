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
import json
import webbrowser
import tempfile
import os

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
from subgraph_expander import create_subgraph_expander


class SubgraphVisualizer:
    """
    Interactive subgraph visualization for RAG search results
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        """Initialize with Neo4j connection"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user  
        self.neo4j_password = neo4j_password
        
        # Try to import Neo4j driver
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        except ImportError:
            print("Warning: Neo4j driver not available. Install with: pip install neo4j")
            self.driver = None
        
        # Initialize subgraph expander
        self.expander = create_subgraph_expander(neo4j_uri, neo4j_user, neo4j_password)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
        if hasattr(self.expander, 'driver') and self.expander.driver:
            self.expander.driver.close()
    
    def extract_subgraph_from_results(self, rag_results, max_nodes=20, expand_subgraph=True):
        """
        Extract subgraph containing the similar nodes found by RAG system
        
        Args:
            rag_results: Results from RAG system process_query
            max_nodes: Maximum number of nodes to include
            expand_subgraph: Whether to expand with connected nodes via ANNOTATION/DESCRIBED_BY
            
        Returns:
            dict: Subgraph data with nodes and relationships
        """
        if not self.driver:
            print("Error: No Neo4j connection available")
            return {"nodes": [], "edges": []}
        
        search_results = rag_results.get("results", [])
        if not search_results:
            print("No search results to visualize")
            return {"nodes": [], "edges": []}
        
        print(f"Extracting subgraph from {len(search_results)} similar items...")
        
        # Extract all entities from search results - any type
        entity_mappings = []
        node_data = {}
        
        for i, result in enumerate(search_results[:max_nodes]):
            metadata = result.get("metadata", {})
            similarity_score = result.get("similarity_score", 0)
            content = result.get("content", "")
            
            # Create a unique identifier for this search result
            result_id = f"result_{i}"
            
            # Store data for this result
            node_data[result_id] = {
                "similarity_score": similarity_score,
                "content": content,
                "metadata": metadata,
                "result_index": i
            }
            
            # Map to Neo4j entities if possible
            if metadata.get("type") == "database_table":
                table_name = metadata.get("table_name", "")
                entity_id = metadata.get("entity_id")
                if entity_id and table_name:
                    entity_mappings.append({
                        "result_id": result_id,
                        "neo4j_table": table_name,
                        "neo4j_id": entity_id,
                        "neo4j_id_field": self._get_id_field_for_table(table_name)
                    })
            elif metadata.get("type") == "pdf_document":
                # Map PDF chunks to their parent Document node
                document_name = metadata.get("document_name", "")
                if document_name:
                    # PDF filename should match the document_name
                    pdf_filename = f"{document_name}.pdf"
                    entity_mappings.append({
                        "result_id": result_id,
                        "neo4j_table": "Document",
                        "neo4j_id": pdf_filename,
                        "neo4j_id_field": "filename"
                    })
            elif metadata.get("type") == "json_table":
                # Map JSON tables to their parent Document node via filename
                parent_document = metadata.get("parent_document", "")
                if parent_document:
                    pdf_filename = f"{parent_document}.pdf"
                    entity_mappings.append({
                        "result_id": result_id,
                        "neo4j_table": "Document", 
                        "neo4j_id": pdf_filename,
                        "neo4j_id_field": "filename"
                    })
        
        # Query Neo4j for basic subgraph (matched nodes + their direct relationships)
        subgraph_data = self._query_neo4j_subgraph_generic(entity_mappings, node_data)
        
        print(f"Base subgraph: {len(subgraph_data['nodes'])} nodes, {len(subgraph_data['edges'])} edges")
        
        # Expand subgraph with connected nodes via ANNOTATION and DESCRIBED_BY relationships
        if expand_subgraph and self.expander and len(subgraph_data['nodes']) > 0:
            print("Expanding subgraph with connected nodes...")
            subgraph_data = self.expander.expand_subgraph(
                subgraph_data, 
                expansion_depth=1,
                max_connected_nodes=15
            )
        
        print(f"Final subgraph: {len(subgraph_data['nodes'])} nodes, {len(subgraph_data['edges'])} edges")
        
        return subgraph_data
    
    def _get_id_field_for_table(self, table_name):
        """Get the ID field name for different table types"""
        id_field_mapping = {
            "Product": "product_id",
            "ProductCategory": "ProductCategoryID", 
            "ProductModel": "ProductModelID",
            "Document": "filename",
            "Annotation": "filename"
        }
        return id_field_mapping.get(table_name, "id")
    
    def _query_neo4j_subgraph_generic(self, entity_mappings, node_data):
        """Query Neo4j to get subgraph with relationships - works with any entity type"""
        nodes = []
        edges = []
        neo4j_node_ids = []
        
        with self.driver.session() as session:
            # Create nodes for all search results, whether they map to Neo4j or not
            for result_id, data in node_data.items():
                similarity_score = data.get("similarity_score", 0)
                content = data.get("content", "")
                metadata = data.get("metadata", {})
                
                # Try to find corresponding Neo4j node
                neo4j_data = None
                for mapping in entity_mappings:
                    if mapping["result_id"] == result_id:
                        # Query Neo4j for this entity
                        table_name = mapping["neo4j_table"]
                        entity_id = mapping["neo4j_id"]
                        id_field = mapping["neo4j_id_field"]
                        
                        # Dynamic query based on table type
                        if table_name == "Product":
                            node_query = f"""
                            MATCH (n:Product {{{id_field}: $entity_id}})
                            RETURN n.{id_field} as id, n.name as name, n.category_name as category,
                                   n.list_price as price, n.color as color, n.size as size, 
                                   labels(n) as labels
                            """
                        elif table_name == "ProductCategory":
                            node_query = f"""
                            MATCH (n:Category {{{id_field}: $entity_id}})
                            RETURN n.{id_field} as id, n.Name as name, n.ParentProductCategoryID as parent,
                                   labels(n) as labels
                            """
                        elif table_name in ["Document", "Annotation"]:
                            node_query = f"""
                            MATCH (n:{table_name} {{{id_field}: $entity_id}})
                            RETURN n.{id_field} as id, n.filename as name, n.document_name as document,
                                   n.annotation_type as type, labels(n) as labels
                            """
                        else:
                            # Generic query for unknown table types
                            node_query = f"""
                            MATCH (n) WHERE n.{id_field} = $entity_id
                            RETURN n.{id_field} as id, 
                                   COALESCE(n.name, n.Name, n.filename, n.title, toString(n.{id_field})) as name,
                                   labels(n) as labels, properties(n) as props
                            """
                        
                        try:
                            result = session.run(node_query, entity_id=entity_id)
                            record = result.single()
                            if record:
                                neo4j_data = dict(record)
                                neo4j_node_ids.append(f"{table_name}_{entity_id}")
                        except Exception as e:
                            print(f"Warning: Could not query {table_name} with {id_field}={entity_id}: {e}")
                
                # Create node (with or without Neo4j data)
                if neo4j_data:
                    # Node exists in Neo4j
                    node_label = neo4j_data.get("name", content[:50])
                    node_type = neo4j_data.get("labels", ["Unknown"])[0] if neo4j_data.get("labels") else "Unknown"
                    
                    nodes.append({
                        "data": {
                            "id": result_id,
                            "label": str(node_label),
                            "similarity_score": similarity_score,
                            "type": node_type,
                            "content": content[:200],
                            "neo4j_data": neo4j_data,
                            "has_neo4j": True
                        }
                    })
                else:
                    # Node doesn't exist in Neo4j or couldn't be found
                    # Extract meaningful label from content or metadata
                    if metadata.get("type") == "database_table":
                        node_label = f"{metadata.get('table_name', 'Unknown')}[{metadata.get('entity_id', '?')}]"
                        node_type = metadata.get("table_name", "Unknown")
                    elif metadata.get("type") == "json_table":
                        parent_doc = metadata.get("parent_document", "")
                        node_label = f"Table: {parent_doc}" if parent_doc else "JSON Table"
                        node_type = "JSON Table"
                    elif metadata.get("type") == "pdf_document":
                        doc_name = metadata.get("document_name", "")
                        chunk_idx = metadata.get("chunk_index", 0)
                        node_label = f"PDF: {doc_name} (chunk {chunk_idx + 1})" if doc_name else "PDF Chunk"
                        node_type = "PDF Chunk"
                    else:
                        # Extract first meaningful words from content
                        words = content.split()[:5]
                        node_label = " ".join(words) if words else "Unknown Content"
                        node_type = metadata.get("type", "Unknown")
                    
                    nodes.append({
                        "data": {
                            "id": result_id,
                            "label": node_label,
                            "similarity_score": similarity_score,
                            "type": node_type,
                            "content": content[:200],
                            "metadata": metadata,
                            "has_neo4j": False
                        }
                    })
            
            # Get relationships between Neo4j nodes if we have any
            if len(neo4j_node_ids) > 1:
                # Build a generic relationship query
                relationship_query = """
                MATCH (n1)-[r]-(n2)
                WHERE (
                    (n1:Product OR n1:Category OR n1:Document OR n1:Annotation) AND
                    (n2:Product OR n2:Category OR n2:Document OR n2:Annotation)
                )
                RETURN DISTINCT 
                    COALESCE(n1.product_id, n1.ProductCategoryID, n1.filename, id(n1)) as source_id,
                    COALESCE(n2.product_id, n2.ProductCategoryID, n2.filename, id(n2)) as target_id,
                    labels(n1)[0] as source_type,
                    labels(n2)[0] as target_type,
                    type(r) as relationship_type
                LIMIT 100
                """
                
                try:
                    result = session.run(relationship_query)
                    
                    # Map Neo4j IDs back to our result IDs
                    id_to_result = {}
                    for mapping in entity_mappings:
                        neo4j_key = f"{mapping['neo4j_table']}_{mapping['neo4j_id']}"
                        id_to_result[str(mapping['neo4j_id'])] = mapping['result_id']
                        id_to_result[mapping['neo4j_id']] = mapping['result_id']
                    
                    for record in result:
                        source_id = str(record["source_id"])
                        target_id = str(record["target_id"])
                        
                        source_result = id_to_result.get(source_id)
                        target_result = id_to_result.get(target_id)
                        
                        if source_result and target_result and source_result != target_result:
                            edges.append({
                                "data": {
                                    "id": f"{source_result}-{target_result}",
                                    "source": source_result,
                                    "target": target_result,
                                    "relationship": record["relationship_type"]
                                }
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not query relationships: {e}")
        
        return {"nodes": nodes, "edges": edges}
    
    def create_cytoscape_visualization(self, subgraph_data, title="RAG Search Subgraph"):
        """
        Create interactive Cytoscape.js visualization
        
        Args:
            subgraph_data: Subgraph data from extract_subgraph_from_results
            title: Title for the visualization
        """
        if not subgraph_data["nodes"]:
            print("No nodes to visualize")
            return None
        
        # Create HTML content with Cytoscape.js
        html_content = self._generate_cytoscape_html(subgraph_data, title)
        
        # Save to temporary file and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_file = f.name
        
        print(f"Opening visualization in browser: {temp_file}")
        webbrowser.open(f"file://{temp_file}")
        
        return temp_file
    
    def _generate_cytoscape_html(self, subgraph_data, title):
        """Generate HTML content with Cytoscape.js visualization"""
        
        nodes_json = json.dumps(subgraph_data["nodes"], indent=2)
        edges_json = json.dumps(subgraph_data["edges"], indent=2)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/cytoscape@3.23.0/dist/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            color: #333;
            margin: 0;
        }}
        
        .header p {{
            color: #666;
            margin: 5px 0;
        }}
        
        .container {{
            display: flex;
            gap: 20px;
            height: 80vh;
        }}
        
        #cy {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            flex: 1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .info-panel {{
            width: 300px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
        }}
        
        .info-panel h3 {{
            margin-top: 0;
            color: #333;
        }}
        
        .node-info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        
        .node-info h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        
        .node-info p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        
        .similarity-score {{
            background-color: #e3f2fd;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
            color: #1976d2;
        }}
        
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        
                         .legend h4 {{
                     margin: 0 0 10px 0;
                 }}
                 
                 .legend-item {{
                     display: flex;
                     align-items: center;
                     margin: 5px 0;
                     font-size: 14px;
                 }}
                 
                 .legend-color {{
                     width: 20px;
                     height: 20px;
                     border-radius: 3px;
                     margin-right: 10px;
                 }}
                 
                 .expansion-info {{
                     background-color: #fff3cd;
                     padding: 10px;
                     border-radius: 5px;
                     margin-bottom: 15px;
                     border-left: 4px solid #ffc107;
                 }}
        
        .stats {{
            background-color: #e8f5e8;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Interactive visualization of knowledge graph nodes similar to your search query</p>
    </div>
    
    <div class="container">
        <div id="cy"></div>
        <div class="info-panel">
            <div class="stats">
                <h3>Graph Statistics</h3>
                <p><strong>Nodes:</strong> <span id="node-count">{len(subgraph_data["nodes"])}</span></p>
                <p><strong>Relationships:</strong> <span id="edge-count">{len(subgraph_data["edges"])}</span></p>
            </div>
            
            <div id="selection-info">
                <h3>Node Information</h3>
                <p>Click on a node to see details</p>
            </div>
            
                         <div class="expansion-info">
                 <h4>🔗 Expanded Subgraph</h4>
                 <p>Shows matched nodes + connected nodes via ANNOTATION and DESCRIBED_BY relationships</p>
             </div>
             
             <div class="legend">
                 <h4>Node Types</h4>
                 <div class="legend-item">
                     <div class="legend-color" style="background-color: #ff6b6b;"></div>
                     <span>High Similarity (>0.7)</span>
                 </div>
                 <div class="legend-item">
                     <div class="legend-color" style="background-color: #4ecdc4;"></div>
                     <span>Medium Similarity (0.4-0.7)</span>
                 </div>
                 <div class="legend-item">
                     <div class="legend-color" style="background-color: #45b7d1;"></div>
                     <span>Low Similarity (<0.4)</span>
                 </div>
                 <div class="legend-item">
                     <div class="legend-color" style="background-color: #95a5a6; border: 2px dashed #7f8c8d;"></div>
                     <span>Connected Nodes</span>
                 </div>
             </div>
        </div>
    </div>
    
    <script>
        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            
            elements: [
                ...{nodes_json},
                ...{edges_json}
            ],
            
            style: [
                                 {{
                     selector: 'node',
                     style: {{
                         'background-color': function(node) {{
                             const isConnected = node.data('is_connected');
                             if (isConnected) return '#95a5a6';  // Gray for connected nodes
                             
                             const score = node.data('similarity_score');
                             if (score > 0.7) return '#ff6b6b';
                             if (score > 0.4) return '#4ecdc4';
                             return '#45b7d1';
                         }},
                         'label': 'data(label)',
                         'text-valign': 'center',
                         'text-halign': 'center',
                         'text-wrap': 'wrap',
                         'text-max-width': '120px',
                         'width': function(node) {{
                             const isConnected = node.data('is_connected');
                             if (isConnected) return 50;  // Fixed size for connected nodes
                             
                             const score = node.data('similarity_score');
                             return Math.max(40, score * 80);
                         }},
                         'height': function(node) {{
                             const isConnected = node.data('is_connected');
                             if (isConnected) return 50;  // Fixed size for connected nodes
                             
                             const score = node.data('similarity_score');
                             return Math.max(40, score * 80);
                         }},
                         'font-size': '12px',
                         'font-weight': 'bold',
                         'color': 'white',
                         'text-outline-width': 2,
                         'text-outline-color': 'rgba(0,0,0,0.5)',
                         'border-width': function(node) {{
                             const isConnected = node.data('is_connected');
                             return isConnected ? 3 : 2;
                         }},
                         'border-color': function(node) {{
                             const isConnected = node.data('is_connected');
                             return isConnected ? '#7f8c8d' : '#333';
                         }},
                         'border-style': function(node) {{
                             const isConnected = node.data('is_connected');
                             return isConnected ? 'dashed' : 'solid';
                         }}
                     }}
                 }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 3,
                        'line-color': '#666',
                        'target-arrow-color': '#666',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(relationship)',
                        'font-size': '10px',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#ffeb3b'
                    }}
                }}
            ],
            
            layout: {{
                name: 'cose',
                idealEdgeLength: 100,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: 400000,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1
            }}
        }});
        
                 // Handle node selection
         cy.on('tap', 'node', function(evt) {{
             const node = evt.target;
             const data = node.data();
             
                           // Build dynamic info based on available data
              const isConnected = data.is_connected;
              const nodeTypeLabel = isConnected ? "Connected Node" : "Matched Node";
              
              let infoHTML = `
                  <h3>Node Information</h3>
                  <div class="node-info">
                      <h4>${{data.label}}</h4>
                      <p><strong>Type:</strong> ${{data.type}} (${{nodeTypeLabel}})</p>
              `;
              
              if (!isConnected) {{
                  infoHTML += `
                      <div class="similarity-score">
                          Similarity: ${{(data.similarity_score * 100).toFixed(1)}}%
                      </div>
                  `;
              }} else {{
                  infoHTML += `
                      <div style="background-color: #e9ecef; padding: 5px 10px; border-radius: 3px; color: #495057;">
                          Connected via ANNOTATION/DESCRIBED_BY
                      </div>
                  `;
              }}
             
             // Add Neo4j data if available
             if (data.has_neo4j && data.neo4j_data) {{
                 const neo4j = data.neo4j_data;
                 infoHTML += '<p><strong>Neo4j Properties:</strong></p><ul>';
                 
                 // Show relevant properties
                 if (neo4j.category) infoHTML += `<li>Category: ${{neo4j.category}}</li>`;
                 if (neo4j.price) infoHTML += `<li>Price: $$${{neo4j.price}}</li>`;
                 if (neo4j.color) infoHTML += `<li>Color: ${{neo4j.color}}</li>`;
                 if (neo4j.size) infoHTML += `<li>Size: ${{neo4j.size}}</li>`;
                 if (neo4j.parent) infoHTML += `<li>Parent ID: ${{neo4j.parent}}</li>`;
                 if (neo4j.document) infoHTML += `<li>Document: ${{neo4j.document}}</li>`;
                 
                 infoHTML += '</ul>';
             }} else {{
                 // Show metadata for non-Neo4j nodes
                 if (data.metadata) {{
                     infoHTML += '<p><strong>Metadata:</strong></p><ul>';
                     if (data.metadata.table_name) infoHTML += `<li>Table: ${{data.metadata.table_name}}</li>`;
                     if (data.metadata.entity_id) infoHTML += `<li>Entity ID: ${{data.metadata.entity_id}}</li>`;
                     if (data.metadata.filename) infoHTML += `<li>File: ${{data.metadata.filename}}</li>`;
                     infoHTML += '</ul>';
                 }}
             }}
             
             // Show content preview
             if (data.content && data.content.length > 0) {{
                 infoHTML += `<p><strong>Content:</strong></p><p style="font-size: 12px; color: #666; max-height: 100px; overflow-y: auto;">${{data.content}}...</p>`;
             }}
             
             infoHTML += '</div>';
             
             document.getElementById('selection-info').innerHTML = infoHTML;
         }});
        
        // Handle edge selection
        cy.on('tap', 'edge', function(evt) {{
            const edge = evt.target;
            const data = edge.data();
            
            const infoHTML = `
                <h3>Relationship Information</h3>
                <div class="node-info">
                    <h4>${{data.relationship}}</h4>
                    <p><strong>From:</strong> ${{data.source}}</p>
                    <p><strong>To:</strong> ${{data.target}}</p>
                </div>
            `;
            
            document.getElementById('selection-info').innerHTML = infoHTML;
        }});
        
        // Handle background tap
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                document.getElementById('selection-info').innerHTML = `
                    <h3>Node Information</h3>
                    <p>Click on a node to see details</p>
                `;
            }}
        }});
        
        console.log('Graph loaded with', cy.nodes().length, 'nodes and', cy.edges().length, 'edges');
    </script>
</body>
</html>
        """
        
        return html_template


class EmbeddingRAGSystem:
    """
    Complete RAG system using embeddings for semantic search
    """
    
    def __init__(self, embeddings_path="data/knowledge_graph_embeddings.pkl", 
                 embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG system
        
        Args:
            embeddings_path: Path to stored embeddings pickle file
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
        
        # Initialize subgraph visualizer
        self.visualizer = SubgraphVisualizer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        print("RAG System ready!")
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'visualizer'):
            self.visualizer.close()
    
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
    
    def visualize_query_results(self, query, top_k=10, similarity_threshold=0.3, 
                               title_suffix="", open_browser=True, expand_subgraph=True):
        """
        Create interactive visualization of query results subgraph
        
        Args:
            query: Search query
            top_k: Number of similar items to find
            similarity_threshold: Minimum similarity score
            title_suffix: Additional text for visualization title
            open_browser: Whether to automatically open browser
            expand_subgraph: Whether to expand with connected nodes via ANNOTATION/DESCRIBED_BY
            
        Returns:
            tuple: (query_results, visualization_file_path)
        """
        print(f"Creating visualization for query: '{query}'")
        
        # Process the query to get similar items
        query_results = self.process_query(query, top_k=top_k, 
                                         similarity_threshold=similarity_threshold)
        
        if not query_results["results"]:
            print("No results found for visualization")
            return query_results, None
        
        # Extract subgraph from results
        subgraph_data = self.visualizer.extract_subgraph_from_results(query_results, 
                                                                     max_nodes=top_k,
                                                                     expand_subgraph=expand_subgraph)
        
        if not subgraph_data["nodes"]:
            print("No nodes found for visualization")
            return query_results, None
        
        # Create visualization
        title = f"Search Results: {query}"
        if title_suffix:
            title += f" - {title_suffix}"
        
        if open_browser:
            viz_file = self.visualizer.create_cytoscape_visualization(subgraph_data, title)
        else:
            # Just save without opening browser
            html_content = self.visualizer._generate_cytoscape_html(subgraph_data, title)
            viz_file = f"subgraph_viz_{hash(query) % 10000}.html"
            with open(viz_file, 'w') as f:
                f.write(html_content)
            print(f"Visualization saved to: {viz_file}")
        
        return query_results, viz_file


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


def test_rag_only():
    """
    Test only the RAG system functionality (assumes embeddings already exist)
    """
    print("RAG SYSTEM TEST ONLY")
    print("=" * 40)
    print("This will test the complete query embedding and search pipeline")
    print("Make sure Neo4j is running and embeddings exist!")
    
    success = test_embedding_rag_system()
    
    if success:
        print("\nRAG SYSTEM TEST COMPLETED SUCCESSFULLY!")
    else:
        print("\nRAG SYSTEM TEST FAILED")


def test_subgraph_visualization():
    """
    Test the subgraph visualization functionality
    """
    print("SUBGRAPH VISUALIZATION TEST")
    print("=" * 50)
    
    try:
        # Check if embeddings exist
        embeddings_path = "data/knowledge_graph_embeddings.pkl"
        if not Path(embeddings_path).exists():
            print(f"Embeddings file not found: {embeddings_path}")
            print("   Please run the main setup first to generate embeddings")
            return False
        
        print("Initializing RAG system with visualization...")
        rag_system = EmbeddingRAGSystem(embeddings_path)
        
        # Test queries for visualization
        test_queries = [
            "mountain bike components",
            "road bike frames", 
            "bicycle handlebars",
            "bike accessories",
            "cycling clothing"
        ]
        
        print(f"\nTesting visualization with {len(test_queries)} different queries")
        print("-" * 40)
        
        for i, query in enumerate(test_queries[:2], 1):  # Test only first 2 to avoid too many browser windows
            print(f"\nVISUALIZATION TEST {i}: '{query}'")
            print("-" * 30)
            
            # Create visualization
            query_results, viz_file = rag_system.visualize_query_results(
                query, 
                top_k=8, 
                similarity_threshold=0.25,
                title_suffix=f"Test {i}",
                open_browser=(i == 1)  # Only open browser for first test
            )
            
            # Show results summary
            results = query_results["results"]
            print(f"Found {len(results)} similar items:")
            
            for j, result in enumerate(results[:3], 1):  # Show top 3
                score = result['similarity_score']
                metadata = result['metadata']
                table_name = metadata.get('table_name', 'unknown')
                entity_id = metadata.get('entity_id', 'unknown')
                print(f"  {j}. {table_name}[{entity_id}]: {score:.3f}")
            
            if viz_file:
                print(f"Visualization saved: {viz_file}")
            else:
                print("No visualization created (no graph data found)")
        
        # Test with a more specific product query
        print(f"\nSPECIFIC PRODUCT VISUALIZATION TEST")
        print("-" * 40)
        
        specific_query = "HL Road Frame Black 58"
        query_results, viz_file = rag_system.visualize_query_results(
            specific_query,
            top_k=6,
            similarity_threshold=0.2,
            title_suffix="Product Similarity",
            open_browser=False  # Don't open browser for this test
        )
        
        print(f"Specific query: '{specific_query}'")
        print(f"Results found: {len(query_results['results'])}")
        
        if viz_file:
            print(f"Product similarity visualization saved: {viz_file}")
        
        # Close connections
        rag_system.close()
        
        print(f"\nVISUALIZATION TEST COMPLETED!")
        print("Features tested:")
        print("  ✓ Subgraph extraction from RAG results")
        print("  ✓ Neo4j relationship queries")
        print("  ✓ Interactive Cytoscape.js visualization")
        print("  ✓ Similarity-based node styling")
        print("  ✓ Click interactions and info panels")
        
        return True
        
    except Exception as e:
        print(f"Error in visualization test: {e}")
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--rag-only":
            test_rag_only()
        elif sys.argv[1] == "--viz-test":
            test_subgraph_visualization()
        else:
            print("Usage:")
            print("  python main.py              # Full setup")
            print("  python main.py --rag-only   # Test RAG system only")
            print("  python main.py --viz-test   # Test visualization only")
    else:
        main() 