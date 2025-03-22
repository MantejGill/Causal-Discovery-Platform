import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import json
import asyncio
import time

# Import RAG components
from core.rag.document_processor import DocumentProcessor
from core.rag.query_engine import RAGQueryEngine

# Import the simpler document processor as a fallback
from core.rag.simple_document_processor import SimpleDocumentProcessor

# Import LLM adapter type hint, but make it optional to avoid circular imports
try:
    from core.llm.adapter import LLMAdapter
except ImportError:
    # Create a type alias instead
    from typing import Protocol
    class LLMAdapter(Protocol):
        def get_name(self) -> str: ...
        def complete(self, prompt: str, **kwargs) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)

class RAGManager:
    """
    Central manager for the RAG functionality
    
    Coordinates between document processing, vector storage, 
    and query operations for the RAG system.
    """
    
    def __init__(self,
                llm_adapter: Optional[Any] = None,
                embeddings_model: str = "openai",
                openai_api_key: Optional[str] = None,
                db_dir: str = "./data/rag_db",
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                k: int = 5,
                score_threshold: float = 0.2,
                max_context_length: int = 4000,
                use_simple_processor: bool = False):
        """
        Initialize the RAG Manager
        
        Args:
            llm_adapter: Optional LLM adapter instance
            embeddings_model: Model to use for embeddings ("openai" or "huggingface")
            openai_api_key: API key for OpenAI (if using OpenAI embeddings)
            db_dir: Directory for vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score to include a document
            max_context_length: Maximum context length in tokens
            use_simple_processor: Use the simpler document processor (more stable but basic)
        """
        self.db_dir = db_dir
        
        # Default to HuggingFace embeddings if no OpenAI API key provided
        if embeddings_model == "openai" and not openai_api_key:
            logger.warning("No OpenAI API key provided, defaulting to HuggingFace embeddings")
            embeddings_model = "huggingface"
        
        # Initialize document processors
        try:
            if use_simple_processor:
                logger.info("Using simple document processor")
                self.document_processor = SimpleDocumentProcessor(db_dir=db_dir)
            else:
                logger.info(f"Using standard document processor with {embeddings_model} embeddings")
                self.document_processor = DocumentProcessor(
                    embeddings_model=embeddings_model,
                    openai_api_key=openai_api_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    db_dir=db_dir
                )
            
            # Also initialize a simple processor as backup
            self.backup_processor = SimpleDocumentProcessor(db_dir=db_dir)
            
        except Exception as e:
            logger.error(f"Error initializing document processor: {str(e)}")
            logger.warning("Falling back to simple document processor")
            self.document_processor = SimpleDocumentProcessor(db_dir=db_dir)
            self.backup_processor = self.document_processor
        
        # Create query engine
        self.query_engine = RAGQueryEngine(
            vector_store=self.document_processor.db,
            k=k,
            score_threshold=score_threshold,
            max_context_length=max_context_length
        )
        
        # Store the LLM adapter (if provided)
        self.llm_adapter = llm_adapter
        
        # Configuration
        self.config = {
            "embeddings_model": embeddings_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k": k,
            "score_threshold": score_threshold,
            "max_context_length": max_context_length,
            "db_dir": db_dir
        }
        
        # Status
        self.is_ready = self._check_readiness()
        self.last_updated = time.time() if self.is_ready else 0
    
    def _check_readiness(self) -> bool:
        """
        Check if the RAG system is ready (has documents)
        
        Returns:
            Boolean indicating if the system is ready
        """
        try:
            docs = self.document_processor.db._collection.count()
            return docs > 0
        except Exception as e:
            logger.error(f"Error checking RAG readiness: {str(e)}")
            return False
    
    async def add_document(self, 
                    file_path: Union[str, Path], 
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the knowledge base
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            
        Returns:
            Result dictionary with status and information
        """
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "added_at" not in metadata:
            metadata["added_at"] = time.time()
        
        # Ensure metadata only contains simple types
        filtered_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered_metadata[key] = value
            elif isinstance(value, list):
                filtered_metadata[key] = ','.join(str(item) for item in value)
            else:
                filtered_metadata[key] = str(value)
        
        # Process the document with the main processor
        try:
            result = await self.document_processor.process_document(file_path, filtered_metadata)
            
            # Check if successful
            if result["status"] == "success":
                # Update readiness status
                self.is_ready = self._check_readiness()
                if self.is_ready:
                    self.last_updated = time.time()
                
                return result
            else:
                logger.warning(f"Main processor failed: {result.get('message', 'Unknown error')}")
                # Fall back to the simple processor
                logger.info("Trying backup document processor")
                
                # The backup processor is synchronous, so we don't need to await it
                backup_result = self.backup_processor.process_document(file_path, filtered_metadata)
                
                # Update readiness status
                self.is_ready = self._check_readiness()
                if self.is_ready:
                    self.last_updated = time.time()
                
                return backup_result
                
        except Exception as e:
            logger.error(f"Error adding document with main processor: {str(e)}")
            logger.info("Trying backup document processor")
            
            try:
                # The backup processor is synchronous, so we don't need to await it
                backup_result = self.backup_processor.process_document(file_path, filtered_metadata)
                
                # Update readiness status
                self.is_ready = self._check_readiness()
                if self.is_ready:
                    self.last_updated = time.time()
                
                return backup_result
            except Exception as backup_e:
                logger.error(f"Backup processor also failed: {str(backup_e)}")
                return {
                    "status": "error",
                    "message": f"Both processors failed. Main error: {str(e)}. Backup error: {str(backup_e)}"
                }
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the knowledge base
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Result dictionary with status and information
        """
        result = self.document_processor.delete_document(document_id)
        
        # Update readiness status
        self.is_ready = self._check_readiness()
        
        return result
    
    def list_documents(self) -> Dict[str, Any]:
        """
        List all documents in the knowledge base
        
        Returns:
            Result dictionary with status and document list
        """
        return self.document_processor.list_documents()
    
    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: ID of the document
            
        Returns:
            Result dictionary with status and chunk list
        """
        return self.document_processor.get_document_chunks(document_id)
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        Clear all documents from the knowledge base
        
        Returns:
            Result dictionary with status and information
        """
        result = self.document_processor.clear_database()
        
        # Update readiness status
        self.is_ready = False
        
        return result
    
    def retrieve_context(self, 
                      query: str, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query string
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieved documents with content and metadata
        """
        return self.query_engine.retrieve_context(query, filter_metadata)
    
    def augment_prompt(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Augment a prompt with retrieved context
        
        Args:
            prompt: User query/prompt
            system_prompt: Optional system prompt to augment
            
        Returns:
            Dictionary with augmented prompts and metadata
        """
        if not self.is_ready:
            logger.warning("RAG system not ready, returning original prompt")
            return {
                "augmented_prompt": prompt,
                "augmented_system_prompt": system_prompt,
                "context_used": False,
                "documents": []
            }
        
        return self.query_engine.augment_prompt(prompt, system_prompt)
    
    def search_by_metadata(self, 
                        metadata_filter: Dict[str, Any], 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria
        
        Args:
            metadata_filter: Metadata filter dictionary
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        return self.query_engine.search_by_metadata(metadata_filter, limit)
    
    def keyword_search(self, 
                     query: str, 
                     fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword search on document content and metadata
        
        Args:
            query: Keyword search query
            fields: List of metadata fields to search in (default: all)
            
        Returns:
            List of matching documents
        """
        return self.query_engine.keyword_search(query, fields)
    
    def hybrid_search(self, 
                    query: str, 
                    keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            List of matching documents
        """
        return self.query_engine.hybrid_search(query, keyword_weight)
    
    def complete_with_rag(self,
                        prompt: str,
                        system_prompt: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Complete a prompt using the RAG system with the configured LLM
        
        Args:
            prompt: User query/prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM completion result with RAG metadata
        """
        if not self.llm_adapter:
            return {
                "error": "No LLM adapter configured for RAG manager",
                "status": "error"
            }
        
        if not self.is_ready:
            logger.warning("RAG system not ready, using original prompt with LLM")
            try:
                # Use the LLM without RAG
                result = self.llm_adapter.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
                result["rag_applied"] = False
                return result
            except Exception as e:
                logger.error(f"Error completing prompt with LLM: {str(e)}")
                return {
                    "error": str(e),
                    "status": "error",
                    "rag_applied": False
                }
        
        try:
            # Augment the prompt with RAG context
            augmented = self.augment_prompt(prompt, system_prompt)
            
            # Use the LLM with augmented prompt
            result = self.llm_adapter.complete(
                prompt=augmented["augmented_prompt"],
                system_prompt=augmented["augmented_system_prompt"],
                **kwargs
            )
            
            # Add RAG metadata to the result
            result["rag_applied"] = augmented["context_used"]
            result["rag_documents"] = augmented["documents"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error completing prompt with RAG: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system
        
        Returns:
            Status dictionary
        """
        try:
            doc_count = 0
            chunk_count = 0
            
            try:
                # Get document count from vector store
                chunk_count = self.document_processor.db._collection.count()
                
                # Get unique document count
                docs = self.list_documents()
                if docs["status"] == "success":
                    doc_count = docs["count"]
            except Exception as inner_e:
                logger.error(f"Error getting document counts: {str(inner_e)}")
            
            return {
                "status": "ready" if self.is_ready else "not_ready",
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "last_updated": self.last_updated,
                "embeddings_model": self.config["embeddings_model"],
                "db_dir": self.config["db_dir"],
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Error getting RAG status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export the RAG system configuration
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the RAG system configuration
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            Updated configuration
        """
        # Update configuration values
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Update query engine parameters
        if "k" in new_config:
            self.query_engine.k = new_config["k"]
        if "score_threshold" in new_config:
            self.query_engine.score_threshold = new_config["score_threshold"]
        if "max_context_length" in new_config:
            self.query_engine.max_context_length = new_config["max_context_length"]
        
        return self.config
    
    @classmethod
    async def create_with_documents(cls, 
                              file_paths: List[Union[str, Path]],
                              metadata_list: Optional[List[Dict[str, Any]]] = None,
                              **kwargs) -> 'RAGManager':
        """
        Create a RAG Manager and add documents in one operation
        
        Args:
            file_paths: List of document file paths
            metadata_list: Optional list of metadata dictionaries (one per document)
            **kwargs: Additional arguments for RAGManager initialization
            
        Returns:
            Initialized RAGManager with documents added
        """
        # Create RAG Manager
        manager = cls(**kwargs)
        
        # Process each document
        for i, file_path in enumerate(file_paths):
            # Get metadata for this document
            metadata = None
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]
            
            # Add document
            await manager.add_document(file_path, metadata)
        
        return manager
