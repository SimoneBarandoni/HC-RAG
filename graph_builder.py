"""
Property Graph Builder
===============================

A general-purpose LlamaIndex property graph builder that can ingest various data formats
and automatically discover entities, relationships, and schema without hardcoding.

This approach works with:
- CSV files (any structure)
- JSON files
- Text documents
- PDFs
- Mixed data sources
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from llama_index.core import PropertyGraphIndex, Document, SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    DynamicLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
import requests
from typing import Any
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import time

load_dotenv()


class CustomLLMClient(CustomLLM):
    """Custom LLM client for your deployed model endpoint."""
    
    context_window: int = 32768
    num_output: int = 2048
    model_name: str = "custom"
    api_url: str = ""
    api_key: str = ""
    
    def __init__(self, api_url: str, model_name: str, api_key: str = "", **kwargs):
        super().__init__(
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Send completion request to your custom endpoint."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # OpenAI-compatible API format
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=1200
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return CompletionResponse(text=text)
            
        except Exception as e:
            print(f"Error calling custom LLM: {e}")
            return CompletionResponse(text=f"Error: {str(e)}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion from your custom endpoint."""
        # For now, just use the complete method and yield the full response
        # You can implement proper streaming later if your endpoint supports it
        response = self.complete(prompt, **kwargs)
        yield response


class GraphBuilder:
    """Property graph builder for any data type."""

    def __init__(self, use_neo4j: bool = False, extraction_mode: str = "auto"):
        """Initialize the graph builder.

        Args:
            use_neo4j: Whether to use Neo4j backend
            extraction_mode: Type of extraction ('auto', 'dynamic', 'simple', 'implicit')
        """
        self.use_neo4j = use_neo4j
        self.extraction_mode = extraction_mode
        self.documents = []

        # Import configuration for settings
        import config

        # Configure LlamaIndex settings - use custom LLM
        Settings.llm = CustomLLMClient(
            api_url=config.LLM_URL,
            model_name=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY
        )

        # Configure embedding model based on type
        if config.EMBEDDING_TYPE == "huggingface":
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL
            )
        else:
            Settings.embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL)

        # Initialize graph store
        if use_neo4j:
            self.graph_store = Neo4jPropertyGraphStore(
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                url=config.NEO4J_URL,
            )
        else:
            self.graph_store = SimplePropertyGraphStore()

    def ingest_directory(
        self, directory_path: str, file_extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """Automatically ingest all supported files from a directory.

        Args:
            directory_path: Path to directory containing data files
            file_extensions: List of file extensions to include (None = all supported)

        Returns:
            List of processed documents
        """
        documents = []

        # Use SimpleDirectoryReader for automatic file type detection
        reader = SimpleDirectoryReader(
            input_dir=directory_path, required_exts=file_extensions, recursive=True
        )

        try:
            # Load documents automatically
            base_documents = reader.load_data()
            print(f"Loaded {len(base_documents)} documents from directory")

            # Process different file types
            for doc in base_documents:
                processed_docs = self._process_document_by_type(doc)
                documents.extend(processed_docs)

        except Exception as e:
            print(f"Error loading directory {directory_path}: {e}")
            # Fallback: manually process known file types
            documents = self._manual_directory_processing(
                directory_path, file_extensions
            )

        return documents

    def _process_document_by_type(self, document: Document) -> List[Document]:
        """Process a document based on its type/content."""
        processed_docs = []

        # Get file path from metadata
        file_path = document.metadata.get("file_path", "")
        file_name = document.metadata.get("file_name", "")

        print(f"Processing: {file_name}")

        # Detect file type and process accordingly
        if file_path.endswith(".csv"):
            processed_docs = self._process_csv_content(document)
        elif file_path.endswith(".json"):
            processed_docs = self._process_json_content(document)
        elif file_path.endswith((".txt", ".md")):
            processed_docs = self._process_text_content(document)
        else:
            # For other types (PDF, etc.), use as-is but chunk appropriately
            processed_docs = self._chunk_document(document)

        return processed_docs

    def _process_csv_content(self, document: Document) -> List[Document]:
        """Process CSV content by converting rows to natural language."""
        try:
            file_path = document.metadata.get("file_path")

            # Try different separators and encodings
            separators = [",", ";", "\t", "|"]
            encodings = ["utf-8", "latin-1", "cp1252"]

            df = None
            for sep in separators:
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if (
                            len(df.columns) > 1
                        ):  # Valid CSV should have multiple columns
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break

            if df is None or len(df.columns) <= 1:
                print(f"Could not parse CSV: {file_path}")
                return [document]

            print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            documents = []
            file_name = document.metadata.get("file_name", "unknown")

            # Convert each row to a natural language description
            for idx, row in df.iterrows():
                # Create natural language description of the row
                content_parts = [f"Record from {file_name}:"]

                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        content_parts.append(f"{col}: {value}")

                if len(content_parts) > 1:  # Only create document if we have content
                    content = ". ".join(content_parts)

                    # Create metadata with essential info only
                    metadata = {
                        "source": file_name,
                        "source_type": "csv",
                        "row_index": idx,
                        "columns": list(
                            row.keys()
                        ),  # Just store column names, not all values
                    }

                    documents.append(
                        Document(
                            text=content,
                            metadata=metadata,
                            doc_id=f"{file_name}_row_{idx}",
                        )
                    )

            return documents

        except Exception as e:
            print(f"Error processing CSV: {e}")
            return [document]

    def _process_json_content(self, document: Document) -> List[Document]:
        """Process JSON content by converting objects to natural language."""
        try:
            file_path = document.metadata.get("file_path")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            file_name = document.metadata.get("file_name", "unknown")

            def process_json_object(obj, path="root", idx=0):
                """Recursively process JSON objects."""
                if isinstance(obj, dict):
                    content_parts = [f"JSON object from {file_name} at {path}:"]

                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            content_parts.append(f"{key}: [nested structure]")
                        else:
                            content_parts.append(f"{key}: {value}")

                    content = ". ".join(content_parts)
                    metadata = {
                        "source": file_name,
                        "source_type": "json",
                        "json_path": path,
                        "object_index": idx,
                        "keys": [
                            k
                            for k in obj.keys()
                            if not isinstance(obj[k], (dict, list))
                        ],  # Just store keys, not all values
                    }

                    return [
                        Document(
                            text=content,
                            metadata=metadata,
                            doc_id=f"{file_name}_{path}_{idx}",
                        )
                    ]

                elif isinstance(obj, list):
                    docs = []
                    for i, item in enumerate(obj):
                        docs.extend(process_json_object(item, f"{path}[{i}]", i))
                    return docs

                return []

            documents = process_json_object(data)
            return documents

        except Exception as e:
            print(f"Error processing JSON: {e}")
            return [document]

    def _process_text_content(self, document: Document) -> List[Document]:
        """Process text content by chunking appropriately."""
        return self._chunk_document(document)

    def _chunk_document(self, document: Document) -> List[Document]:
        """Chunk a document into smaller pieces for better processing."""
        try:
            import config

            # Use sentence splitter for better chunking
            splitter = SentenceSplitter(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
            )

            nodes = splitter.get_nodes_from_documents([document])

            # Convert nodes back to documents
            chunked_docs = []
            for i, node in enumerate(nodes):
                doc_metadata = {**document.metadata, "chunk_index": i}
                chunked_docs.append(
                    Document(
                        text=node.text,
                        metadata=doc_metadata,
                        doc_id=f"{document.doc_id}_chunk_{i}"
                        if document.doc_id
                        else f"doc_chunk_{i}",
                    )
                )

            return chunked_docs

        except Exception as e:
            print(f"Error chunking document: {e}")
            return [document]

    def _manual_directory_processing(
        self, directory_path: str, file_extensions: Optional[List[str]]
    ) -> List[Document]:
        """Manually process directory when SimpleDirectoryReader fails."""
        documents = []
        path_obj = Path(directory_path)

        for file_path in path_obj.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()

                if file_extensions and ext not in file_extensions:
                    continue

                try:
                    # Create a basic document for manual processing
                    doc = Document(
                        text="",  # Will be populated by type-specific processing
                        metadata={
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "file_type": ext,
                        },
                    )

                    processed_docs = self._process_document_by_type(doc)
                    documents.extend(processed_docs)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return documents

    def create_extractors(self) -> List:
        """Create appropriate extractors based on extraction mode."""
        import config

        extractors = []

        if self.extraction_mode == "auto" or self.extraction_mode == "simple":
            # Simple LLM extraction - works with any domain
            extractors.append(
                SimpleLLMPathExtractor(
                    llm=Settings.llm,
                    max_paths_per_chunk=config.MAX_PATHS_PER_CHUNK,
                    num_workers=config.NUM_WORKERS,
                )
            )

        if self.extraction_mode == "auto" or self.extraction_mode == "dynamic":
            # Dynamic extraction that discovers schema automatically
            extractors.append(
                DynamicLLMPathExtractor(
                    llm=Settings.llm, num_workers=config.NUM_WORKERS
                )
            )

        if self.extraction_mode == "implicit":
            # Implicit extraction using embeddings
            extractors.append(ImplicitPathExtractor())

        # Default to simple if no extractors
        if not extractors:
            extractors.append(SimpleLLMPathExtractor(llm=Settings.llm))

        return extractors

    def build_graph(
        self, data_path: str, file_extensions: Optional[List[str]] = None
    ) -> PropertyGraphIndex:
        """Build property graph from any data source.

        Args:
            data_path: Path to data directory or file
            file_extensions: File extensions to include (e.g., ['.csv', '.json', '.txt'])

        Returns:
            PropertyGraphIndex
        """
        print(f"🔄 Starting graph building from: {data_path}")
        print(f"📊 Extraction mode: {self.extraction_mode}")

        # Ingest documents
        if os.path.isdir(data_path):
            documents = self.ingest_directory(data_path, file_extensions)
        else:
            # Single file
            reader = SimpleDirectoryReader(input_files=[data_path])
            base_docs = reader.load_data()
            documents = []
            for doc in base_docs:
                documents.extend(self._process_document_by_type(doc))

        print(f"📄 Processed {len(documents)} documents")

        if not documents:
            raise ValueError("No documents were successfully processed")

        # Create extractors
        extractors = self.create_extractors()
        print(
            f"🔧 Using {len(extractors)} extractors: {[type(e).__name__ for e in extractors]}"
        )

        # Build the property graph
        print("🏗️  Building property graph...")

        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self.graph_store,
            kg_extractors=extractors,
            show_progress=True,
        )

        print("✅ Property graph built successfully!")
        return index

    def save_graph(self, index: PropertyGraphIndex, output_path: str = "graph"):
        """Save the graph."""
        if not self.use_neo4j:
            import pickle

            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "graph_store.pkl"), "wb") as f:
                pickle.dump(self.graph_store, f)
            print(f"💾 Graph saved to {output_path}")
        else:
            print("💾 Graph data persisted in Neo4j database")


def main():
    """Demo the graph builder."""
    start_time = time.time()
    # Import configuration
    import config

    # Validate configuration
    if not config.validate_config():
        print("❌ Please fix configuration errors before proceeding.")
        return None, None

    print("🚀 Property Graph Builder")
    print("=" * 50)

    # Print configuration
    config.print_config()
    print()

    # Build graph
    builder = GraphBuilder(
        use_neo4j=config.USE_NEO4J, extraction_mode=config.EXTRACTION_MODE
    )

    try:
        index = builder.build_graph(
            data_path=config.DATA_PATH, file_extensions=config.get_file_extensions()
        )

        builder.save_graph(index, config.OUTPUT_PATH)
        index.property_graph_store.save_networkx_graph(name="./kg.html")
        end_time = time.time()
        print(f"🕒 Time taken: {end_time - start_time:.2f} seconds")
        print("\n🎉 Success! Your property graph is ready.")

        retriever = index.as_retriever(
            include_text=True,  # include source text, default True
        )

        nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

        for node in nodes:
            print(node.text)

        query_engine = index.as_query_engine(
            include_text=True,
        )

        response = query_engine.query("What happened at Interleaf and Viaweb?")

        print(str(response))

        return index, builder

    except Exception as e:
        print(f"❌ Error building graph: {e}")
        return None, None


if __name__ == "__main__":
    index, builder = main()
