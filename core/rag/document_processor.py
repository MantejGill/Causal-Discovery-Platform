import os
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import asyncio

import PyPDF2
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports for newer LangChain versions
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to legacy imports
    from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

try:
    from langchain_community.vectorstores.utils import filter_complex_metadata
except ImportError:
    # Simple implementation if not available
    def filter_complex_metadata(metadata):
        filtered = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                filtered[k] = v
            elif isinstance(v, list):
                filtered[k] = str(v)
            else:
                filtered[k] = str(v)
        return filtered

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes and embeds documents for the RAG system"""
    
    def __init__(self, 
                embeddings_model: str = "openai",
                openai_api_key: Optional[str] = None,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                db_dir: str = "./data/rag_db"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_dir = db_dir
        
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize embeddings model
        if embeddings_model == "openai":
            if not openai_api_key:
                logger.warning("No OpenAI API key provided, defaulting to HuggingFace embeddings")
                embeddings_model = "huggingface"
            else:
                self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        if embeddings_model == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize vector DB
        try:
            self.db = Chroma(
                persist_directory=db_dir,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector database initialized with {self.db._collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            # Create a new database if loading fails
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=db_dir
            )
            logger.info("Created new vector database")
    
    def _safe_persist(self):
        """Safely persist the database, handling different Chroma versions"""
        try:
            # Try to call persist method (newer versions)
            if hasattr(self.db, 'persist'):
                self.db.persist()
            # For older versions that don't have the persist method
            elif hasattr(self.db, '_collection') and hasattr(self.db._collection, 'persist'):
                self.db._collection.persist()
            else:
                logger.warning("Could not find persist method on Chroma object")
        except Exception as e:
            logger.warning(f"Error persisting database: {str(e)}")
    
    async def process_document(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document and add to vector database"""
        document_id = str(uuid.uuid4())
        
        try:
            # Ensure the file exists
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            # Add document_id to metadata
            metadata["document_id"] = document_id
            metadata["source"] = os.path.basename(file_path)
            
            # Load document based on file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                documents = await self._load_pdf(file_path, metadata)
            elif file_ext == '.txt':
                documents = await self._load_text(file_path, metadata)
            elif file_ext == '.md':
                documents = await self._load_markdown(file_path, metadata)
            else:
                return {"status": "error", "message": f"Unsupported file format: {file_ext}"}
            
            # Skip if no documents were loaded
            if not documents or len(documents) == 0:
                return {"status": "error", "message": "No content could be extracted from the document"}
            
            # Split text into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Filter complex metadata to ensure compatibility with Chroma
            # Convert lists to strings and remove unsupported types
            for chunk in chunks:
                # Convert tags list to comma-separated string
                if 'tags' in chunk.metadata and isinstance(chunk.metadata['tags'], list):
                    chunk.metadata['tags'] = ','.join(str(tag) for tag in chunk.metadata['tags'])
                # Apply general filter for any other complex metadata
                chunk.metadata = filter_complex_metadata(chunk.metadata)
            
            # Add to vector database
            ids = [f"{document_id}-{i}" for i in range(len(chunks))]
            self.db.add_documents(chunks, ids=ids)
            self._safe_persist()
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks": len(chunks),
                "file_name": os.path.basename(file_path),
                "metadata": metadata
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _extract_title_from_pdf(self, file_path: Union[str, Path]) -> str:
        """Extract title from a PDF document
        
        Attempts to extract the title using several heuristics:
        1. Look for text that appears to be a title on the first page
        2. Use PDF metadata if available
        
        Returns:
            str: The extracted title or empty string if no title could be extracted
        """
        try:
            title = ""
            # Try to extract from PDF metadata first
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Check PDF metadata
                if pdf_reader.metadata and pdf_reader.metadata.title:
                    title = pdf_reader.metadata.title
                    if title and len(title.strip()) > 0:
                        return title.strip()
                
                # If no title in metadata, try to extract from first page content
                if len(pdf_reader.pages) > 0:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    
                    if text:
                        # Try to find title in first few lines
                        lines = text.split('\n')
                        non_empty_lines = [line.strip() for line in lines if line.strip()]
                        
                        # Look for title candidates: non-empty lines near the top
                        # that aren't too long and don't look like headers or footers
                        for i, line in enumerate(non_empty_lines[:5]):  # Check first 5 non-empty lines
                            # Basic heuristics for a title
                            if (len(line) > 10 and  # Not too short
                                len(line) < 200 and  # Not too long
                                not line.lower().startswith(('abstract', 'introduction', 'chapter')) and
                                not any(x in line.lower() for x in ['page', 'http', 'www', '@', 'published'])
                               ):
                                title = line
                                break
            
            return title.strip()
        except Exception as e:
            logger.warning(f"Error extracting title from PDF: {str(e)}")
            return ""
    
    async def _load_pdf(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> List[Document]:
        """Load content from a PDF file"""
        try:
            # Extract title if not already provided in metadata
            if not metadata.get("title") or metadata.get("title") == os.path.basename(file_path):
                title = self._extract_title_from_pdf(file_path)
                if title:
                    metadata["title"] = title
                    logger.info(f"Extracted title from PDF: {title}")
            
            # Use LangChain's PyPDFLoader for cleaner loading
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Update metadata for each document
            for doc in documents:
                doc.metadata.update(metadata)
                # Add page number to metadata
                if "page" in doc.metadata:
                    doc.metadata["page_num"] = doc.metadata["page"]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            
            # Fallback method using PyPDF2 directly
            try:
                logger.info(f"Attempting fallback PDF loading for {file_path}")
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    documents = []
                    
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():  # Skip empty pages
                            page_metadata = metadata.copy()
                            page_metadata["page_num"] = i
                            documents.append(Document(
                                page_content=text,
                                metadata=page_metadata
                            ))
                    
                    if not documents:
                        # If no documents were created, create at least one with empty content
                        documents = [Document(
                            page_content="[No extractable text found in this PDF]",
                            metadata=metadata
                        )]
                    
                    return documents
            except Exception as inner_e:
                logger.error(f"Fallback PDF loading also failed: {str(inner_e)}")
                # Create a minimal document to avoid the error
                return [Document(
                    page_content="[Error loading PDF]",
                    metadata=metadata
                )]
    
    async def _load_text(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> List[Document]:
        """Load content from a text file"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Update metadata for each document
            for doc in documents:
                doc.metadata.update(metadata)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            
            # Fallback method: read file directly
            try:
                logger.info(f"Attempting fallback text loading for {file_path}")
                with open(file_path, 'r', encoding='utf-8') as text_file:
                    text = text_file.read()
                    if text.strip():
                        return [Document(
                            page_content=text,
                            metadata=metadata
                        )]
                    else:
                        return []
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as text_file:
                        text = text_file.read()
                        if text.strip():
                            return [Document(
                                page_content=text,
                                metadata=metadata
                            )]
                        else:
                            return []
                except Exception as inner_e:
                    logger.error(f"Fallback text loading also failed: {str(inner_e)}")
                    return [Document(
                        page_content="[Error loading text file]",
                        metadata=metadata
                    )]
    
    async def _load_markdown(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> List[Document]:
        """Load content from a markdown file"""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            
            # Update metadata for each document
            for doc in documents:
                doc.metadata.update(metadata)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {str(e)}")
            
            # Fallback to text loader
            try:
                logger.info(f"Attempting fallback markdown loading as text for {file_path}")
                return await self._load_text(file_path, metadata)
            except Exception as inner_e:
                logger.error(f"Fallback markdown loading also failed: {str(inner_e)}")
                return [Document(
                    page_content="[Error loading markdown file]",
                    metadata=metadata
                )]
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and all its chunks from the vector database"""
        try:
            # Get document IDs to delete
            query = f"{document_id}-"
            results = self.db._collection.get(where={"$contains": query})
            
            if not results or not results['ids']:
                return {"status": "error", "message": f"Document not found: {document_id}"}
            
            # Delete all chunks for this document
            self.db._collection.delete(ids=results['ids'])
            self._safe_persist()
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_deleted": len(results['ids'])
            }
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the vector database"""
        try:
            # Get all documents
            results = self.db._collection.get()
            
            if not results or not results['ids']:
                return {"status": "success", "documents": [], "count": 0}
            
            # Extract unique document IDs from the chunk IDs
            doc_ids = set()
            doc_metadata = {}
            
            for i, doc_id in enumerate(results['ids']):
                # Extract document ID from chunk ID (format: doc_id-chunk_num)
                parts = doc_id.split('-')
                if len(parts) >= 2:
                    main_doc_id = '-'.join(parts[:-1])  # Handle UUIDs with hyphens
                    doc_ids.add(main_doc_id)
                    
                    # Get metadata from the first chunk of each document
                    if main_doc_id not in doc_metadata and 'metadatas' in results:
                        doc_metadata[main_doc_id] = results['metadatas'][i]
            
            # Format response
            documents = []
            for doc_id in doc_ids:
                doc_info = {
                    "document_id": doc_id,
                    "metadata": doc_metadata.get(doc_id, {})
                }
                documents.append(doc_info)
            
            return {
                "status": "success",
                "documents": documents,
                "count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """Get all chunks for a specific document"""
        try:
            # Query for chunks with this document ID
            query = f"{document_id}-"
            results = self.db._collection.get(where={"$contains": query})
            
            if not results or not results['ids']:
                return {"status": "error", "message": f"Document not found: {document_id}"}
            
            # Format chunks
            chunks = []
            for i, chunk_id in enumerate(results['ids']):
                chunk_info = {
                    "chunk_id": chunk_id,
                    "metadata": results['metadatas'][i] if 'metadatas' in results else {},
                    "content": results['documents'][i] if 'documents' in results else ""
                }
                chunks.append(chunk_info)
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks": chunks,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {document_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def clear_database(self) -> Dict[str, Any]:
        """Clear all documents from the vector database"""
        try:
            # Get count before deletion
            count = self.db._collection.count()
            
            if count > 0:
                try:
                    # Get all document IDs safely
                    results = self.db._collection.get()
                    
                    if results and 'ids' in results and results['ids']:
                        # Delete all documents
                        self.db._collection.delete(ids=results['ids'])
                        self._safe_persist()
                        logger.info(f"Successfully cleared database: {count} documents removed")
                        return {
                            "status": "success",
                            "message": f"Cleared {count} documents from the database"
                        }
                    else:
                        # Try alternative approach - delete without IDs
                        self.db._collection.delete()
                        self._safe_persist()
                        logger.info("Used alternative deletion method to clear database")
                        return {
                            "status": "success",
                            "message": f"Used alternative method to clear database"
                        }
                except Exception as inner_e:
                    logger.error(f"Error during deletion: {str(inner_e)}")
                    # Try last resort approach - recreate collection
                    try:
                        # Reinitialize the database
                        self.db = Chroma(
                            embedding_function=self.embeddings,
                            persist_directory=self.db_dir
                        )
                        logger.info("Reinitialized database as last resort")
                        return {
                            "status": "success",
                            "message": "Recreated database due to deletion failure"
                        }
                    except Exception as last_e:
                        return {
                            "status": "error",
                            "message": f"Multiple errors clearing database: {str(inner_e)}, {str(last_e)}"
                        }
            else:
                return {
                    "status": "success",
                    "message": "Database already empty (0 documents)"
                }
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}
