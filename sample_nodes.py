from typing import List
import numpy as np
from isRelevant import NodeInput

def create_sample_nodes() -> List[NodeInput]:
    """Create a list of sample nodes with different characteristics for testing relevance scoring"""
    nodes = []
    
    # Node 1: Highly relevant red mountain bike product
    nodes.append(NodeInput(
        text="Premium Red Mountain Bike - Trail Blazer X1 with advanced suspension and lightweight frame, perfect for off-road adventures under $900",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "001", "category": "mountain_bikes"},
        node_type="product",
        entities=["red mountain bike", "trail", "suspension", "lightweight"]
    ))
    
    # Node 2: Partially relevant - mountain bike but different color
    nodes.append(NodeInput(
        text="Blue Mountain Bike - Rugged terrain specialist with 21-speed gear system, priced at $750",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "002", "category": "mountain_bikes"},
        node_type="product",
        entities=["blue mountain bike", "terrain", "gear system"]
    ))
    
    # Node 3: Document about mountain bike maintenance
    nodes.append(NodeInput(
        text="Mountain Bike Maintenance Guide - Complete handbook for maintaining your mountain bike including brake adjustments, tire care, and gear tuning",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_001", "category": "maintenance"},
        node_type="document",
        entities=["mountain bike", "maintenance", "brake", "tire", "gear"]
    ))
    
    # Node 4: Specification document
    nodes.append(NodeInput(
        text="Technical Specifications for Mountain Bike Components - Detailed specs for handlebars, frames, wheels, and suspension systems",
        embeddings=np.random.rand(384),
        graph_relations={"type": "specification", "id": "spec_001", "category": "components"},
        node_type="specification",
        entities=["mountain bike", "handlebars", "frames", "wheels", "suspension"]
    ))
    
    # Node 5: Category node
    nodes.append(NodeInput(
        text="Mountain Bikes Category - Browse our complete selection of mountain bikes for all skill levels and terrains",
        embeddings=np.random.rand(384),
        graph_relations={"type": "category", "id": "cat_001", "parent": "bikes"},
        node_type="category",
        entities=["mountain bikes", "selection", "terrain"]
    ))
    
    # Node 6: Irrelevant node - road bike
    nodes.append(NodeInput(
        text="Professional Road Racing Bike - Lightweight carbon fiber frame designed for speed on paved roads, $2500",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "003", "category": "road_bikes"},
        node_type="product",
        entities=["road bike", "carbon fiber", "racing", "paved roads"]
    ))
    
    # Node 7: Completely unrelated node
    nodes.append(NodeInput(
        text="Camping Tent Setup Instructions - How to properly set up your 4-person camping tent for outdoor adventures",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_002", "category": "camping"},
        node_type="document",
        entities=["camping tent", "setup", "outdoor", "adventures"]
    ))
    
    # Node 8: Another red mountain bike - budget option
    nodes.append(NodeInput(
        text="Budget Red Mountain Bike - Affordable entry-level mountain bike with basic components, perfect for beginners at $450",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "004", "category": "mountain_bikes"},
        node_type="product",
        entities=["red mountain bike", "budget", "entry-level", "beginners"]
    ))
    
    # Node 9: Green mountain bike - color variant
    nodes.append(NodeInput(
        text="Forest Green Mountain Bike - Eco-friendly paint with 27-speed transmission and disc brakes, priced at $850",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "005", "category": "mountain_bikes"},
        node_type="product",
        entities=["green mountain bike", "eco-friendly", "transmission", "disc brakes"]
    ))
    
    # Node 10: Mountain bike accessories
    nodes.append(NodeInput(
        text="Mountain Bike Accessories Kit - Complete set including helmet, gloves, water bottle, and repair tools",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "acc_001", "category": "accessories"},
        node_type="product",
        entities=["mountain bike", "accessories", "helmet", "gloves", "repair tools"]
    ))
    
    # Node 11: Electric mountain bike
    nodes.append(NodeInput(
        text="Electric Red Mountain Bike - E-bike with 50-mile range, pedal assist, and rugged design for trails, $1800",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "006", "category": "electric_bikes"},
        node_type="product",
        entities=["electric red mountain bike", "e-bike", "pedal assist", "trails"]
    ))
    
    # Node 12: Mountain bike review document
    nodes.append(NodeInput(
        text="Mountain Bike Reviews 2024 - Comprehensive review of top mountain bikes including performance tests and user ratings",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_003", "category": "reviews"},
        node_type="document",
        entities=["mountain bike", "reviews", "performance", "ratings"]
    ))
    
    # Node 13: Bike shop category
    nodes.append(NodeInput(
        text="Local Bike Shop Services - Expert bicycle repair, maintenance, and custom builds for all bike types",
        embeddings=np.random.rand(384),
        graph_relations={"type": "category", "id": "cat_002", "parent": "services"},
        node_type="category",
        entities=["bike shop", "repair", "maintenance", "custom builds"]
    ))
    
    # Node 14: BMX bike - different category
    nodes.append(NodeInput(
        text="BMX Freestyle Bike - Designed for tricks and stunts with reinforced frame and special pegs, $650",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "007", "category": "bmx_bikes"},
        node_type="product",
        entities=["bmx bike", "freestyle", "tricks", "stunts", "reinforced frame"]
    ))
    
    # Node 15: Mountain biking trail guide
    nodes.append(NodeInput(
        text="Mountain Biking Trail Guide - Best trails for mountain biking with difficulty ratings and scenic descriptions",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_004", "category": "guides"},
        node_type="document",
        entities=["mountain biking", "trails", "difficulty", "scenic"]
    ))
    
    # Node 16: Vintage mountain bike
    nodes.append(NodeInput(
        text="Vintage 1990s Mountain Bike - Classic steel frame mountain bike, restored condition, collector's item at $1200",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "008", "category": "vintage_bikes"},
        node_type="product",
        entities=["vintage mountain bike", "1990s", "steel frame", "restored", "collector"]
    ))
    
    # Node 17: Bike parts specification
    nodes.append(NodeInput(
        text="Mountain Bike Brake System Specifications - Technical details for hydraulic and mechanical brake systems",
        embeddings=np.random.rand(384),
        graph_relations={"type": "specification", "id": "spec_002", "category": "brake_systems"},
        node_type="specification",
        entities=["mountain bike", "brake system", "hydraulic", "mechanical"]
    ))
    
    # Node 18: Completely unrelated - kitchen appliances
    nodes.append(NodeInput(
        text="Stainless Steel Kitchen Blender - High-performance blender with multiple speed settings for smoothies and soups",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "009", "category": "kitchen_appliances"},
        node_type="product",
        entities=["kitchen blender", "stainless steel", "smoothies", "soups"]
    ))
    
    return nodes 