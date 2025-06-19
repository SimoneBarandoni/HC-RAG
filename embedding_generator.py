
import pandas as pd
import json
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import PyPDF2
import pdfplumber


class DynamicEmbeddingGenerator:
    """
    A flexible embedding generator that adapts to any table structure
    and generates embeddings for diverse data types.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings_data = {
            'embeddings': [],
            'metadata': [],
            'texts': []
        }
    
    def analyze_data_patterns(self, df):
        """
        Analyze data patterns to categorize fields by semantic importance
        """
        field_analysis = {}
        
        for col in df.columns:
            # Get sample of non-null values
            sample_values = df[col].dropna().head(10).astype(str).tolist()
            if not sample_values:
                continue
            
            # Calculate characteristics
            avg_length = np.mean([len(str(val)) for val in sample_values])
            unique_ratio = len(df[col].dropna().unique()) / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
            
            # Determine semantic importance
            if unique_ratio > 0.9 and avg_length > 20:
                importance = "high"  # Likely descriptions, names
            elif unique_ratio > 0.8:
                importance = "medium"  # Likely identifiers, codes
            elif avg_length > 10:
                importance = "medium"  # Longer text fields
            else:
                importance = "low"  # Short codes, numbers, flags
            
            field_analysis[col] = {
                'importance': importance,
                'avg_length': avg_length,
                'unique_ratio': unique_ratio,
                'sample_values': sample_values[:3]
            }
        
        return field_analysis
    
    def create_smart_text_representation(self, row, df, table_name=None):
        """
        Create intelligent text representation based on data analysis
        """
        field_analysis = self.analyze_data_patterns(df)
        
        # Categorize fields by importance
        high_importance = []
        medium_importance = []
        low_importance = []
        
        for col, analysis in field_analysis.items():
            if pd.notna(row.get(col)) and str(row[col]).strip():
                value = str(row[col]).strip()
                field_info = f"{col}: {value}"
                
                if analysis['importance'] == 'high':
                    high_importance.append(field_info)
                elif analysis['importance'] == 'medium':
                    medium_importance.append(field_info)
                else:
                    low_importance.append(field_info)
        
        # Build text with prioritized information
        text_parts = []
        
        if table_name:
            text_parts.append(f"Table: {table_name}")
        
        # Add high importance fields first (descriptions, names)
        if high_importance:
            text_parts.extend(high_importance)
        
        # Add medium importance fields (IDs, codes, categories)
        if medium_importance:
            text_parts.extend(medium_importance[:3])  # Limit to avoid too long text
        
        # Add some low importance fields for context
        if low_importance:
            text_parts.extend(low_importance[:2])  # Just a few for context
        
        return ". ".join(text_parts)
    
    def process_csv_table(self, csv_path, related_data=None):
        """Process any CSV table dynamically"""
        print(f"Processing CSV: {csv_path}")
        
        # Load the CSV
        df = pd.read_csv(csv_path, sep=';')
        table_name = Path(csv_path).stem
        
        print(f"  Found {len(df)} rows")
        
        # Generate embeddings for each row
        for idx, row in df.iterrows():
            try:
                # Create smart text representation
                text = self.create_smart_text_representation(row, df, table_name)
                
                if text.strip():  # Only process non-empty texts
                    # Generate embedding
                    embedding = self.model.encode([text])[0]
                    
                    # Store everything
                    self.embeddings_data['embeddings'].append(embedding.tolist())
                    self.embeddings_data['texts'].append(text)
                    
                    # Create metadata
                    metadata = {
                        'id': f"{table_name}_{idx}",
                        'type': 'database_table',
                        'table_name': table_name,
                        'row_index': idx,
                        'source_file': str(csv_path)
                    }
                    
                    # Add entity ID if available (for linking with Neo4j)
                    id_columns = [col for col in df.columns if 'id' in col.lower() or 'ID' in col]
                    if id_columns:
                        entity_id = row.get(id_columns[0])
                        if pd.notna(entity_id):
                            metadata['entity_id'] = int(entity_id) if str(entity_id).isdigit() else str(entity_id)
                    
                    self.embeddings_data['metadata'].append(metadata)
                    
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                continue
    
    def flatten_json_to_text(self, json_obj, prefix=""):
        """Recursively flatten JSON to readable text"""
        text_parts = []
        
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    text_parts.extend(self.flatten_json_to_text(value, current_prefix))
                else:
                    text_parts.append(f"{current_prefix}: {value}")
        
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                current_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                if isinstance(item, (dict, list)):
                    text_parts.extend(self.flatten_json_to_text(item, current_prefix))
                else:
                    text_parts.append(f"{current_prefix}: {item}")
        
        else:
            text_parts.append(f"{prefix}: {json_obj}" if prefix else str(json_obj))
        
        return text_parts
    
    def process_json_table(self, json_path, parent_document=None):
        """Process JSON table files dynamically"""
        print(f"Processing JSON: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Flatten JSON to text
            text_parts = self.flatten_json_to_text(json_data)
            
            # Create comprehensive text
            filename = Path(json_path).stem
            document_context = parent_document or filename
            
            # Create semantic text representation
            full_text = f"Document: {document_context}. Contains structured information. "
            full_text += ". ".join(text_parts[:20])  # Limit to avoid too long text
            
            # Generate embedding
            embedding = self.model.encode([full_text])[0]
            
            # Store data
            self.embeddings_data['embeddings'].append(embedding.tolist())
            self.embeddings_data['texts'].append(full_text)
            
            # Create metadata
            metadata = {
                'id': f"json_{filename}",
                'type': 'json_table',
                'filename': filename,
                'parent_document': parent_document,
                'source_file': str(json_path),
                'json_keys': list(json_data.keys()) if isinstance(json_data, dict) else []
            }
            
            self.embeddings_data['metadata'].append(metadata)
            
        except Exception as e:
            print(f"  Error processing JSON {json_path}: {e}")
    
    def extract_text_from_pdf_pypdf2(self, pdf_path):
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"Page {page_num + 1}: {page_text.strip()}")
                    except Exception as e:
                        print(f"    Error extracting page {page_num + 1}: {e}")
                        continue
            
            return "\n".join(text_content)
        except Exception as e:
            print(f"    PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (preferred method)"""
        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Clean up the text
                            cleaned_text = " ".join(page_text.split())
                            text_content.append(f"Page {page_num + 1}: {cleaned_text}")
                    except Exception as e:
                        print(f"    Error extracting page {page_num + 1}: {e}")
                        continue
            
            return "\n".join(text_content)
        except Exception as e:
            print(f"    pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using the best available method"""
        print(f"  Extracting text from PDF: {pdf_path}")
        
        # Try pdfplumber first (usually better for text extraction)
        text_content = self.extract_text_from_pdf_pdfplumber(pdf_path)
        
        # If pdfplumber fails or returns empty, try PyPDF2
        if not text_content.strip():
            print("    pdfplumber failed, trying PyPDF2...")
            text_content = self.extract_text_from_pdf_pypdf2(pdf_path)
        
        # If still no content, return a placeholder
        if not text_content.strip():
            print("    Warning: No text could be extracted from PDF")
            return f"PDF Document: {Path(pdf_path).stem}. Text extraction failed - may be image-based PDF or corrupted."
        
        return text_content
    
    def chunk_text(self, text, max_chunk_size=1000, overlap=100):
        """Split text into overlapping chunks for better embeddings"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + max_chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdf_document(self, pdf_path, document_name=None):
        """Process a PDF document and create embeddings for its content"""
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            full_text = self.extract_text_from_pdf(pdf_path)
            
            if not full_text.strip():
                print(f"  Warning: No text extracted from {pdf_path}")
                return
            
            # Get document name
            doc_name = document_name or Path(pdf_path).stem
            
            # Create document context
            document_context = f"PDF Document: {doc_name}. "
            
            # Chunk the text for better embeddings
            text_chunks = self.chunk_text(full_text, max_chunk_size=800, overlap=100)
            
            print(f"  Created {len(text_chunks)} text chunks")
            
            # Process each chunk
            for chunk_idx, chunk in enumerate(text_chunks):
                try:
                    # Create full text with context
                    full_chunk_text = document_context + chunk
                    
                    # Generate embedding
                    embedding = self.model.encode([full_chunk_text])[0]
                    
                    # Store data
                    self.embeddings_data['embeddings'].append(embedding.tolist())
                    self.embeddings_data['texts'].append(full_chunk_text)
                    
                    # Create metadata
                    metadata = {
                        'id': f"pdf_{doc_name}_chunk_{chunk_idx}",
                        'type': 'pdf_document',
                        'document_name': doc_name,
                        'source_file': str(pdf_path),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(text_chunks),
                        'text_length': len(chunk),
                        'file_size': Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0
                    }
                    
                    self.embeddings_data['metadata'].append(metadata)
                    
                except Exception as e:
                    print(f"    Error processing chunk {chunk_idx}: {e}")
                    continue
            
            print(f"  Successfully processed PDF with {len(text_chunks)} chunks")
            
        except Exception as e:
            print(f"  Error processing PDF {pdf_path}: {e}")
    
    def process_all_data(self, data_directory):
        """Process all data files in the directory"""
        data_path = Path(data_directory)
        
        print("🔄 Starting comprehensive data processing...")
        
        print("📊 Processing CSV files...")
        csv_files = list(data_path.glob("*.csv"))
        for csv_file in csv_files:
            self.process_csv_table(csv_file)
        
        print("📋 Processing JSON files...")
        json_dir = data_path / "IngestedDocuments"
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            for json_file in json_files:
                # Try to determine parent document from filename
                parent_doc = None
                filename = json_file.stem
                if " Table " in filename:
                    parent_doc = filename.split(" Table ")[0]
                
                self.process_json_table(json_file, parent_doc)
        
        print("📄 Processing PDF files...")
        if json_dir.exists():
            pdf_files = list(json_dir.glob("*.pdf"))
            for pdf_file in pdf_files:
                # Extract document name (remove .pdf extension)
                document_name = pdf_file.stem
                self.process_pdf_document(pdf_file, document_name)
        
        print(f"✅ Processed {len(self.embeddings_data['texts'])} items total")
        
        # Print processing summary
        self.print_processing_summary()
    
    def print_processing_summary(self):
        """Print a summary of what was processed"""
        if not self.embeddings_data['metadata']:
            print("No data was processed.")
            return
        
        # Count by type
        type_counts = {}
        for metadata in self.embeddings_data['metadata']:
            content_type = metadata.get('type', 'unknown')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        print("\n📈 Processing Summary:")
        print("-" * 40)
        for content_type, count in type_counts.items():
            print(f"  {content_type}: {count} items")
        print(f"  Total: {len(self.embeddings_data['texts'])} embeddings generated")
        print("-" * 40)
    
    def save_embeddings(self, output_path="knowledge_graph_embeddings.pkl"):
        """Save embeddings with metadata"""
        print(f"Saving embeddings to {output_path}")
        
        # Add generation metadata
        self.embeddings_data["generation_info"] = {
            "model_name": getattr(self.model, 'model_name', 'all-MiniLM-L6-v2'),
            "total_entries": len(self.embeddings_data["texts"]),
            "embedding_dimension": len(self.embeddings_data["embeddings"][0]) if self.embeddings_data["embeddings"] else 0,
            "generation_timestamp": pd.Timestamp.now().isoformat(),
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(self.embeddings_data, f)
        
        print(f"Saved {len(self.embeddings_data['texts'])} embeddings")
    
    def load_embeddings(self, input_path="knowledge_graph_embeddings.pkl"):
        """Load previously saved embeddings"""
        print(f"Loading embeddings from {input_path}")
        
        with open(input_path, "rb") as f:
            self.embeddings_data = pickle.load(f)
        
        print(f"Loaded {len(self.embeddings_data['texts'])} embeddings")
        return self.embeddings_data
    
    def get_statistics(self):
        """Get statistics about the generated embeddings"""
        if not self.embeddings_data['embeddings']:
            return "No embeddings generated yet"
        
        stats = {
            "total_embeddings": len(self.embeddings_data['embeddings']),
            "embedding_dimension": len(self.embeddings_data['embeddings'][0]),
            "content_types": {}
        }
        
        # Count by content type
        for metadata in self.embeddings_data['metadata']:
            content_type = metadata['type']
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
        
        return stats


# Utility functions for standalone usage
def generate_embeddings_for_directory(data_directory, output_path=None, model_name='all-MiniLM-L6-v2'):
    """
    Standalone function to generate embeddings for a data directory
    
    Args:
        data_directory: Path to directory containing CSV and JSON files
        output_path: Where to save embeddings (optional)
        model_name: Sentence transformer model to use
        
    Returns:
        Path to saved embeddings file
    """
    generator = DynamicEmbeddingGenerator(model_name)
    generator.process_all_data(data_directory)
    
    if output_path is None:
        output_path = Path(data_directory) / "knowledge_graph_embeddings.pkl"
    
    generator.save_embeddings(str(output_path))
    return output_path


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data"
    
    print(f"Generating embeddings for directory: {data_dir}")
    embeddings_path = generate_embeddings_for_directory(data_dir)
    print(f"Embeddings saved to: {embeddings_path}") 