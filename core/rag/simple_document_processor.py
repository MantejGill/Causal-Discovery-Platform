"""
A simpler document processor implementation that avoids complex dependencies and failure modes.
This serves as a fallback when the main document processor encounters issues.
"""

import os
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import time

from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class SimpleDocumentProcessor:
    """A simplified document processor for handling documents more directly"""
    
    def __init__(self, db_dir: str = "./data/rag_db"):
        """Initialize with HuggingFace embeddings and Chroma DB"""
        self.db_dir = db_dir
        
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize embeddings with HuggingFace (no API key required)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
        
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
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Ensure metadata only contains simple types accepted by Chroma"""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                sanitized[k] = v
            elif isinstance(v, list):
                sanitized[k] = ','.join(str(item) for item in v)
            else:
                sanitized[k] = str(v)
        return sanitized
    
    def _extract_title_from_text(self, content: str) -> str:
        """Extract title from text content
        
        For text files, we'll use the first non-empty line as the title
        if it meets certain criteria
        
        Returns:
            str: The extracted title or empty string if no title could be extracted
        """
        if not content:
            return ""
            
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if non_empty_lines:
            # Use the first line as title if it's reasonable length
            first_line = non_empty_lines[0]
            if (len(first_line) > 5 and # Not too short
                len(first_line) < 200 and # Not too long
                not any(first_line.lower().startswith(x) for x in ['#', '<', '<!--', '/*', '```', 'title:', 'author:', 'date:']) and
                not any(x in first_line.lower() for x in ['http', 'www'])
               ):
                return first_line
                
        return ""
    
    def process_text_file(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text file directly"""
        document_id = str(uuid.uuid4())
        
        try:
            # Ensure the file exists
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            # Add document_id to metadata
            metadata["document_id"] = document_id
            metadata["source"] = os.path.basename(file_path)
            metadata = self._sanitize_metadata(metadata)
            
            # Read the text file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
            
            # Extract title if not already provided in metadata
            if not metadata.get("title") or metadata.get("title") == os.path.basename(file_path):
                extracted_title = self._extract_title_from_text(content)
                if extracted_title:
                    metadata["title"] = extracted_title
                    logger.info(f"Extracted title from text file: {extracted_title}")
            
            # Create chunks (simple splitting by paragraphs)
            chunks = []
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    chunks.append(Document(
                        page_content=para.strip(),
                        metadata=chunk_metadata
                    ))
            
            # If no chunks were created (empty file), create one placeholder chunk
            if not chunks:
                chunks = [Document(
                    page_content="[Empty file]",
                    metadata=metadata
                )]
            
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
            logger.error(f"Error processing text file: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_document(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process any document type by extracting text directly"""
        document_id = str(uuid.uuid4())
        file_path = str(file_path)  # Ensure it's a string
        
        try:
            # Ensure the file exists
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            # Add document_id to metadata
            metadata["document_id"] = document_id
            metadata["source"] = os.path.basename(file_path)
            metadata = self._sanitize_metadata(metadata)
            
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Process based on file type
            if file_ext == '.pdf':
                # Try to extract text from PDF
                try:
                    # Try to import PyPDF2
                    import PyPDF2
                    
                    # Extract title if not already provided in metadata
                    if not metadata.get("title") or metadata.get("title") == os.path.basename(file_path):
                        extracted_title = ""
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            
                            # Try to get title from PDF metadata
                            if pdf_reader.metadata and pdf_reader.metadata.title:
                                extracted_title = pdf_reader.metadata.title
                                if extracted_title and len(extracted_title.strip()) > 0:
                                    metadata["title"] = extracted_title.strip()
                                    logger.info(f"Extracted title from PDF metadata: {extracted_title}")
                            
                            # If no title in metadata, try to extract from first page content
                            if not extracted_title and len(pdf_reader.pages) > 0:
                                first_page = pdf_reader.pages[0]
                                text = first_page.extract_text()
                                
                                if text:
                                    # Try to find title in first few lines
                                    lines = text.split('\n')
                                    non_empty_lines = [line.strip() for line in lines if line.strip()]
                                    
                                    # Look for title candidates in first few lines
                                    for i, line in enumerate(non_empty_lines[:5]):
                                        # Basic heuristics for a title
                                        if (len(line) > 10 and  # Not too short
                                            len(line) < 200 and  # Not too long
                                            not line.lower().startswith(('abstract', 'introduction', 'chapter')) and
                                            not any(x in line.lower() for x in ['page', 'http', 'www', '@', 'published'])
                                        ):
                                            metadata["title"] = line.strip()
                                            logger.info(f"Extracted title from PDF content: {line}")
                                            break
                    
                    chunks = []
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        
                        for i, page in enumerate(pdf_reader.pages):
                            try:
                                text = page.extract_text()
                                if text and text.strip():
                                    chunk_metadata = metadata.copy()
                                    chunk_metadata["page_num"] = i
                                    chunks.append(Document(
                                        page_content=text.strip(),
                                        metadata=chunk_metadata
                                    ))
                            except Exception as e:
                                logger.warning(f"Error extracting text from PDF page {i}: {str(e)}")
                                # Create a minimal document for the page
                                chunk_metadata = metadata.copy()
                                chunk_metadata["page_num"] = i
                                chunk_metadata["error"] = str(e)
                                chunks.append(Document(
                                    page_content=f"[Error extracting text from page {i}]",
                                    metadata=chunk_metadata
                                ))
                    
                    # If no chunks were created, create a placeholder
                    if not chunks:
                        chunks = [Document(
                            page_content="[No extractable text found in PDF]",
                            metadata=metadata
                        )]
                
                except ImportError:
                    logger.error("PyPDF2 not available")
                    chunks = [Document(
                        page_content="[PDF processing not available - PyPDF2 missing]",
                        metadata=metadata
                    )]
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")
                    chunks = [Document(
                        page_content=f"[Error processing PDF: {str(e)}]",
                        metadata=metadata
                    )]
            
            elif file_ext in ['.txt', '.md']:
                # Process text file
                return self.process_text_file(file_path, metadata)
            
            else:
                # Unsupported file type - create a placeholder document
                chunks = [Document(
                    page_content=f"[Unsupported file format: {file_ext}]",
                    metadata=metadata
                )]
            
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
