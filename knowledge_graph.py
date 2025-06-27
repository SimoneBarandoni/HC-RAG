import pandas as pd
import json
from pathlib import Path
from neo4j import GraphDatabase
# Import embedding generator from separate module
from embedding_generator import DynamicEmbeddingGenerator
# Add other required imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Initialize embedding generator
        self.embedding_generator = None

    def close(self):
        self.driver.close()

    def initialize_embeddings(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_generator = DynamicEmbeddingGenerator(model_name)
        return self.embedding_generator

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_indexes(self):
        with self.driver.session() as session:
            session.run(
                "CREATE INDEX product_id IF NOT EXISTS FOR (p:Product) ON (p.product_id)"
            )
            session.run(
                "CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)"
            )
            session.run(
                "CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename)"
            )
            session.run(
                "CREATE INDEX annotation_filename IF NOT EXISTS FOR (a:Annotation) ON (a.filename)"
            )
            # Add embedding-related indexes with proper syntax
            try:
                session.run(
                    "CREATE INDEX embedding_id_index IF NOT EXISTS FOR (n:Product) ON (n.embedding_id)"
                )
                session.run(
                    "CREATE INDEX embedding_id_doc_index IF NOT EXISTS FOR (n:Document) ON (n.embedding_id)"
                )
                session.run(
                    "CREATE INDEX embedding_id_ann_index IF NOT EXISTS FOR (n:Annotation) ON (n.embedding_id)"
                )
            except Exception as e:
                print(f"Warning: Could not create embedding indexes: {e}")
                # Try alternative syntax for older Neo4j versions
                try:
                    session.run("CREATE INDEX ON :Product(embedding_id)")
                    session.run("CREATE INDEX ON :Document(embedding_id)")
                    session.run("CREATE INDEX ON :Annotation(embedding_id)")
                except Exception as e2:
                    print(f"Warning: Alternative index creation also failed: {e2}")

    def create_product_nodes(self, products_df, categories_df, models_df):
        with self.driver.session() as session:
            for _, product in products_df.iterrows():
                try:
                    category_name = "Unknown"
                    if pd.notna(product.get("ProductCategoryID")):
                        category = categories_df[
                            categories_df["ProductCategoryID"]
                            == product["ProductCategoryID"]
                        ]
                        if not category.empty:
                            category_name = str(category["Name"].iloc[0])

                    model_name = "Unknown"
                    if pd.notna(product.get("ProductModelID")):
                        model = models_df[
                            models_df["ProductModelID"] == product["ProductModelID"]
                        ]
                        if not model.empty:
                            model_name = str(model["Name"].iloc[0])

                    # Create embedding ID for linking
                    embedding_id = f"Product_{product['ProductID']}"

                    query = """
                    CREATE (p:Product {
                        product_id: $product_id,
                        name: $name,
                        product_number: $product_number,
                        color: $color,
                        standard_cost: $standard_cost,
                        list_price: $list_price,
                        size: $size,
                        weight: $weight,
                        category_id: $category_id,
                        category_name: $category_name,
                        model_id: $model_id,
                        model_name: $model_name,
                        sell_start_date: $sell_start_date,
                        embedding_id: $embedding_id
                    })
                    """

                    params = {
                        "product_id": int(product["ProductID"]),
                        "name": str(product["Name"]),
                        "product_number": str(product["ProductNumber"]),
                        "color": str(
                            product.get("Color", "")
                            if pd.notna(product.get("Color"))
                            else ""
                        ),
                        "standard_cost": float(
                            str(product.get("StandardCost", "0")).replace(",", ".")
                        )
                        if pd.notna(product.get("StandardCost"))
                        else 0.0,
                        "list_price": float(
                            str(product.get("ListPrice", "0")).replace(",", ".")
                        )
                        if pd.notna(product.get("ListPrice"))
                        else 0.0,
                        "size": str(
                            product.get("Size", "")
                            if pd.notna(product.get("Size"))
                            else ""
                        ),
                        "weight": str(
                            product.get("Weight", "")
                            if pd.notna(product.get("Weight"))
                            else ""
                        ),
                        "category_id": int(product["ProductCategoryID"])
                        if pd.notna(product.get("ProductCategoryID"))
                        else None,
                        "category_name": category_name,
                        "model_id": int(product["ProductModelID"])
                        if pd.notna(product.get("ProductModelID"))
                        else None,
                        "model_name": model_name,
                        "sell_start_date": str(
                            product.get("SellStartDate", "")
                            if pd.notna(product.get("SellStartDate"))
                            else ""
                        ),
                        "embedding_id": embedding_id,
                    }

                    session.run(query, **params)

                except Exception as e:
                    continue

    def create_document_nodes(self, document_structure):
        with self.driver.session() as session:
            for doc_name, doc_data in document_structure.items():
                if doc_data["pdf"]:
                    pdf_file = doc_data["pdf"]
                    
                    # Create embedding ID for document
                    embedding_id = f"Document_{pdf_file.stem}"

                    query = """
                    CREATE (d:Document {
                        filename: $filename,
                        document_name: $document_name,
                        file_path: $file_path,
                        file_type: 'PDF',
                        file_size: $file_size,
                        embedding_id: $embedding_id
                    })
                    """

                    session.run(
                        query,
                        filename=pdf_file.name,
                        document_name=doc_name,
                        file_path=str(pdf_file),
                        file_size=pdf_file.stat().st_size if pdf_file.exists() else 0,
                        embedding_id=embedding_id,
                    )

                    for annotation_file in doc_data["annotations"]:
                        ann_type = (
                            "Image" if annotation_file.suffix == ".jpg" else "Table"
                        )

                        content = None
                        if annotation_file.suffix == ".json":
                            try:
                                with open(annotation_file, "r") as f:
                                    content = json.load(f)
                            except Exception as e:
                                print(f"Error reading {annotation_file}: {e}")

                        # Create embedding ID for annotation
                        annotation_embedding_id = f"Annotation_{annotation_file.stem}"

                        ann_query = """
                        CREATE (a:Annotation {
                            filename: $filename,
                            annotation_type: $annotation_type,
                            file_path: $file_path,
                            content: $content,
                            file_size: $file_size,
                            embedding_id: $embedding_id
                        })
                        """

                        session.run(
                            ann_query,
                            filename=annotation_file.name,
                            annotation_type=ann_type,
                            file_path=str(annotation_file),
                            content=json.dumps(content) if content else None,
                            file_size=annotation_file.stat().st_size
                            if annotation_file.exists()
                            else 0,
                            embedding_id=annotation_embedding_id,
                        )

                        rel_query = """
                        MATCH (d:Document {filename: $doc_filename})
                        MATCH (a:Annotation {filename: $ann_filename})
                        CREATE (a)-[:ANNOTATION]->(d)
                        """

                        session.run(
                            rel_query,
                            doc_filename=pdf_file.name,
                            ann_filename=annotation_file.name,
                        )

    def generate_and_store_embeddings(self, data_directory="data"):
        """Generate embeddings for all data and store them"""
        if not self.embedding_generator:
            self.initialize_embeddings()
        
        print("🔥 Generating embeddings for all data...")
        
        # Process all data to generate embeddings
        self.embedding_generator.process_all_data(data_directory)
        
        # Save embeddings to file
        embeddings_path = Path(data_directory) / "knowledge_graph_embeddings.pkl"
        self.embedding_generator.save_embeddings(str(embeddings_path))
        
        # Update Neo4j nodes with embedding information
        self.update_nodes_with_embedding_info()
        
        return embeddings_path

    def update_nodes_with_embedding_info(self):
        """Update Neo4j nodes with embedding indices and metadata"""
        print("🔗 Linking Neo4j nodes with embeddings...")
        
        if not self.embedding_generator:
            print("⚠️ No embedding generator available")
            return
        
        embeddings_metadata = self.embedding_generator.embeddings_data['metadata']
        
        with self.driver.session() as session:
            for idx, metadata in enumerate(embeddings_metadata):
                try:
                    if metadata['type'] == 'database_table':
                        table_name = metadata['table_name']
                        entity_id = metadata.get('entity_id')
                        
                        if table_name == 'Product' and entity_id:
                            # Update Product nodes
                            query = """
                            MATCH (p:Product {product_id: $entity_id})
                            SET p.embedding_index = $embedding_index,
                                p.embedding_text = $embedding_text
                            """
                            session.run(
                                query,
                                entity_id=int(entity_id),
                                embedding_index=idx,
                                embedding_text=self.embedding_generator.embeddings_data['texts'][idx][:200]
                            )
                        
                        elif table_name == 'ProductCategory' and entity_id:
                            # Create or update Category nodes
                            query = """
                            MERGE (c:Category {category_id: $entity_id})
                            SET c.embedding_index = $embedding_index,
                                c.embedding_text = $embedding_text,
                                c.embedding_id = $embedding_id
                            """
                            session.run(
                                query,
                                entity_id=int(entity_id),
                                embedding_index=idx,
                                embedding_text=self.embedding_generator.embeddings_data['texts'][idx][:200],
                                embedding_id=metadata['id']
                            )
                    
                    elif metadata['type'] == 'json_table':
                        # Update Annotation nodes for JSON files
                        filename = metadata['filename']
                        query = """
                        MATCH (a:Annotation)
                        WHERE a.filename CONTAINS $filename
                        SET a.embedding_index = $embedding_index,
                            a.embedding_text = $embedding_text
                        """
                        session.run(
                            query,
                            filename=filename.split(' Table ')[0] if ' Table ' in filename else filename,
                            embedding_index=idx,
                            embedding_text=self.embedding_generator.embeddings_data['texts'][idx][:200]
                        )
                
                except Exception as e:
                    print(f"  Error updating node with embedding info: {e}")
                    continue

    def get_embedding_statistics(self):
        """Get statistics about embeddings in the graph"""
        if not self.embedding_generator:
            return "No embeddings generated yet"
        
        stats = {
            "total_embeddings": len(self.embedding_generator.embeddings_data['embeddings']),
            "embedding_dimension": len(self.embedding_generator.embeddings_data['embeddings'][0]) if self.embedding_generator.embeddings_data['embeddings'] else 0,
            "content_types": {}
        }
        
        # Count by content type
        for metadata in self.embedding_generator.embeddings_data['metadata']:
            content_type = metadata['type']
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
        
        return stats

    def create_product_relationships(self):
        with self.driver.session() as session:
            category_query = """
            MATCH (p1:Product), (p2:Product)
            WHERE p1.category_id = p2.category_id 
            AND p1.product_id <> p2.product_id
            AND p1.category_id IS NOT NULL
            CREATE (p1)-[:SAME_CATEGORY]->(p2)
            """
            result1 = session.run(category_query)

            model_query = """
            MATCH (p1:Product), (p2:Product)
            WHERE p1.model_id = p2.model_id 
            AND p1.product_id <> p2.product_id
            AND p1.model_id IS NOT NULL
            CREATE (p1)-[:SAME_MODEL]->(p2)
            """
            result2 = session.run(model_query)

            price_query = """
            MATCH (p1:Product), (p2:Product)
            WHERE p1.product_id <> p2.product_id
            AND p1.list_price > 0 AND p2.list_price > 0
            AND abs(p1.list_price - p2.list_price) / p1.list_price <= 0.20
            CREATE (p1)-[:SIMILAR_PRICE]->(p2)
            """
            result3 = session.run(price_query)

            manual_relations = [
                {
                    "filter1": "p1.name CONTAINS 'Road Frame'",
                    "filter2": "p2.name CONTAINS 'Road Frame'",
                    "relation": "COMPATIBLE_PRODUCT",
                },
                {
                    "filter1": "p1.name CONTAINS 'Mountain'",
                    "filter2": "p2.name CONTAINS 'Mountain'",
                    "relation": "COMPATIBLE_PRODUCT",
                },
                {
                    "filter1": "p1.name CONTAINS 'Helmet'",
                    "filter2": "p2.name CONTAINS 'Jersey'",
                    "relation": "COMPLEMENTARY_PRODUCT",
                },
                {
                    "filter1": "p1.name CONTAINS 'Frame'",
                    "filter2": "p2.name CONTAINS 'Handlebars'",
                    "relation": "COMPLEMENTARY_PRODUCT",
                },
            ]

            for relation in manual_relations:
                manual_query = f"""
                MATCH (p1:Product), (p2:Product)
                WHERE {relation["filter1"]}
                AND {relation["filter2"]}
                AND p1.product_id <> p2.product_id
                CREATE (p1)-[:{relation["relation"]}]->(p2)
                """
                session.run(manual_query)

    def create_product_document_relationships(self):
        with self.driver.session() as session:
            connect_query = """
            MATCH (p:Product), (d:Document)
            WHERE d.document_name CONTAINS p.name 
            OR p.name CONTAINS d.document_name
            OR (d.document_name CONTAINS 'Mountain' AND p.name CONTAINS 'Mountain')
            OR (d.document_name CONTAINS 'Handlebars' AND p.name CONTAINS 'Handlebars')
            OR (d.document_name CONTAINS 'Jersey' AND p.name CONTAINS 'Jersey')
            CREATE (p)-[:DESCRIBED_BY]->(d)
            """

            session.run(connect_query)

    def get_graph_statistics(self):
        with self.driver.session() as session:
            stats = {}

            node_counts = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY count DESC
            """)

            stats["nodes"] = {
                record["node_type"]: record["count"] for record in node_counts
            }

            rel_counts = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """)

            stats["relationships"] = {
                record["relationship_type"]: record["count"] for record in rel_counts
            }

            return stats

    def query_similar_products(self, product_id, limit=5):
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {product_id: $product_id})-[r]-(similar:Product)
            RETURN similar.name as product_name, 
                   similar.product_id as product_id,
                   type(r) as relationship_type,
                   similar.list_price as price
            ORDER BY similar.list_price
            LIMIT $limit
            """

            result = session.run(query, product_id=product_id, limit=limit)
            return [dict(record) for record in result]

    def hybrid_search_example(self, search_term, limit=5):
        """
        Example of hybrid search combining Neo4j graph queries with embedding similarity
        """
        if not self.embedding_generator:
            return "No embeddings available"
        
        print(f"Hybrid search for: '{search_term}'")
        
        # Generate embedding for search term
        search_embedding = self.embedding_generator.model.encode([search_term])
        embeddings_matrix = np.array(self.embedding_generator.embeddings_data['embeddings'])
        
        # Find most similar embeddings
        similarities = cosine_similarity(search_embedding, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:limit*2]  # Get more for filtering
        
        # Step 2: Use graph to get related products
        similar_items = []
        with self.driver.session() as session:
            for idx in top_indices:
                metadata = self.embedding_generator.embeddings_data['metadata'][idx]
                
                if metadata['type'] == 'database_table' and metadata['table_name'] == 'Product':
                    entity_id = metadata.get('entity_id')
                    if entity_id:
                        # Get product info from graph
                        query = """
                        MATCH (p:Product {product_id: $product_id})
                        OPTIONAL MATCH (p)-[:SAME_CATEGORY]-(related:Product)
                        RETURN p.name as name, p.list_price as price, p.category_name as category,
                               p.embedding_index as embedding_index,
                               collect(DISTINCT related.name)[0..3] as related_products
                        """
                        result = session.run(query, product_id=int(entity_id))
                        record = result.single()
                        
                        if record:
                            similar_items.append({
                                'name': record['name'],
                                'price': record['price'],
                                'category': record['category'],
                                'similarity_score': similarities[idx],
                                'related_products': record['related_products'],
                                'embedding_text': self.embedding_generator.embeddings_data['texts'][idx][:100] + "..."
                            })
                
                if len(similar_items) >= limit:
                    break
        
        return similar_items

def load_csv_data():
    data_path = Path("data")
    products_df = pd.read_csv(data_path / "Product.csv", sep=";")
    categories_df = pd.read_csv(data_path / "ProductCategory.csv", sep=";")
    descriptions_df = pd.read_csv(data_path / "ProductDescription.csv", sep=";")
    models_df = pd.read_csv(data_path / "ProductModel.csv", sep=";")

    return {
        "products": products_df,
        "categories": categories_df,
        "descriptions": descriptions_df,
        "models": models_df,
    }

def analyze_ingested_documents():
    docs_path = Path("data/IngestedDocuments")

    files = list(docs_path.glob("*"))
    documents = {}

    for file in files:
        name = file.name

        if name.endswith(".pdf"):
            base_name = name.replace(".pdf", "")
            if base_name not in documents:
                documents[base_name] = {"pdf": None, "annotations": []}
            documents[base_name]["pdf"] = file

        elif name.endswith(".jpg"):
            base_name = name.replace(".jpg", "")
            if " Fig " in base_name:
                base_name = base_name.split(" Fig ")[0]
            if base_name not in documents:
                documents[base_name] = {"pdf": None, "annotations": []}
            documents[base_name]["annotations"].append(file)

        elif name.endswith(".json"):
            base_name = name.replace(".json", "")
            if " Table " in base_name:
                base_name = base_name.split(" Table ")[0]
            if base_name not in documents:
                documents[base_name] = {"pdf": None, "annotations": []}
            documents[base_name]["annotations"].append(file)

    for doc_name, doc_data in documents.items():
        print(f"Documento: {doc_name}")
        print(f"  PDF: {'Sì' if doc_data['pdf'] else 'Mancante'}")
        print(f"  Annotazioni: {len(doc_data['annotations'])}")
        for ann in doc_data["annotations"]:
            print(f"    - {ann.name}")

    return documents

# Utility functions that can be imported
def test_neo4j_connection():
    try:
        test_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with test_driver.session() as session:
            result = session.run("RETURN 'Neo4j is connected!' as message")
            message = result.single()["message"]
            test_driver.close()
            return True, message
    except Exception as e:
        return False, str(e)


# Main execution (only runs when script is executed directly)
if __name__ == "__main__":
    # load csv data
    csv_data = load_csv_data()
    document_structure = analyze_ingested_documents()

    # docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest

    connected, message = test_neo4j_connection()

    if connected:
        build_graph = True
    else:
        print(f"Error: {message}")
        print("docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest")
        build_graph = False

    if build_graph:
        try:
            kg_builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

            print("🧹 Cleaning database...")
            kg_builder.clear_database()

            print("📊 Creating indexes...")
            kg_builder.create_indexes()

            print("🏗️ Creating product nodes...")
            kg_builder.create_product_nodes(
                csv_data["products"], csv_data["categories"], csv_data["models"]
            )

            print("📄 Creating document nodes...")
            kg_builder.create_document_nodes(document_structure)

            print("🔗 Creating product relationships...")
            kg_builder.create_product_relationships()

            print("📋 Connecting products with documents...")
            kg_builder.create_product_document_relationships()

            print("🧠 Generating embeddings for all nodes...")
            embeddings_path = kg_builder.generate_and_store_embeddings("data")
            
            print("📊 Graph and embedding statistics:")
            graph_stats = kg_builder.get_graph_statistics()
            embedding_stats = kg_builder.get_embedding_statistics()
            
            print(f"   Graph Nodes: {graph_stats['nodes']}")
            print(f"   Graph Relationships: {graph_stats['relationships']}")
            print(f"   Total Embeddings: {embedding_stats['total_embeddings']}")
            print(f"   Embedding Dimension: {embedding_stats['embedding_dimension']}")
            print(f"   Content Types: {embedding_stats['content_types']}")
            print(f"   Embeddings saved to: {embeddings_path}")

            print("=== KNOWLEDGE GRAPH + EMBEDDINGS COMPLETED! ===")
            print("Neo4j Browser: http://localhost:7474")
            print("Embeddings file: data/knowledge_graph_embeddings.pkl")
            print()
            print("Available capabilities:")
            print("  • Graph queries and traversals via Neo4j")
            print("  • Semantic similarity search via embeddings")
            print("  • Hybrid search combining both approaches")
            print("  • Each node has embedding_id and embedding_index properties")
            
            # Demo hybrid search
            print("\n HYBRID SEARCH DEMO:")
            print("-" * 40)
            
            demo_searches = ["mountain bike", "black frame", "road bicycle"]
            for search_term in demo_searches:
                try:
                    results = kg_builder.hybrid_search_example(search_term, limit=3)
                    print(f"\n🔍 Results for '{search_term}':")
                    for i, item in enumerate(results, 1):
                        print(f"  {i}. {item['name']} (${item['price']:.2f})")
                        print(f"     Category: {item['category']}")
                        print(f"     Similarity: {item['similarity_score']:.3f}")
                        if item['related_products']:
                            print(f"     Related: {', '.join(item['related_products'][:2])}")
                        print()
                except Exception as e:
                    print(f"  Error in demo search: {e}")
            
            kg_builder.close()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Neo4j is not connected.")
        print("To start Neo4j with Docker:")
        print(
            "docker run -p 7474:7474 -p 7687:7687 -d --env NEO4J_AUTH=neo4j/password neo4j:latest"
        )

    print("\n" + "="*60)
    print("🎉 SCRIPT COMPLETED")
    print("="*60)
    print("If successful, you now have:")
    print("📊 Neo4j knowledge graph with all your data")
    print("🧠 Embeddings for semantic search")
    print("🔗 Hybrid search capabilities")
    print("📁 Saved embeddings file: data/knowledge_graph_embeddings.pkl")
    print()
    print("Next steps:")
    print("1. Explore the graph at http://localhost:7474")
    print("2. Use the embeddings for semantic search")
    print("3. Try the hybrid search in your applications")
    print("4. Check the notebook for RAG system examples")







