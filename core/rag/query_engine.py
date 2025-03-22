from typing import Dict, List, Optional, Any, Tuple
import logging
import re
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """Retrieves relevant context for causal queries"""
    
    def __init__(self, 
                vector_store: Chroma,
                k: int = 5,
                score_threshold: float = 0.2,
                max_context_length: int = 4000):
        """
        Initialize the RAG Query Engine
        
        Args:
            vector_store: Initialized Chroma vector store
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score to include a document
            max_context_length: Maximum context length in tokens (approximate)
        """
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold
        self.max_context_length = max_context_length
    
    def retrieve_context(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query string
            filter_metadata: Optional metadata filter
        
        Returns:
            List of retrieved documents with content and metadata
        """
        try:
            # Use similarity_search_with_score to get relevance scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=self.k,
                filter=filter_metadata
            )
            
            # Format results and filter by score threshold
            results = []
            for doc, score in docs_with_scores:
                # Lower score is better in some embeddings models, higher in others
                # Normalize to ensure higher is always better (0-1 range)
                normalized_score = 1.0 - score if score <= 1.0 else 1.0/score
                
                if normalized_score >= self.score_threshold:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": normalized_score
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria
        
        Args:
            metadata_filter: Metadata filter dictionary
            limit: Maximum number of results
        
        Returns:
            List of matching documents
        """
        try:
            # Construct filter query
            filter_dict = {}
            for key, value in metadata_filter.items():
                filter_dict[key] = value
            
            # Execute search
            docs = self.vector_store.get(
                where=filter_dict,
                limit=limit
            )
            
            if not docs or 'documents' not in docs or not docs['documents']:
                return []
            
            # Format results
            results = []
            for i, content in enumerate(docs['documents']):
                result = {
                    "content": content,
                    "metadata": docs['metadatas'][i] if 'metadatas' in docs else {}
                }
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text
        Approximation: ~4 characters per token for English text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string and list of used documents
        """
        # Sort documents by relevance score (if available)
        sorted_docs = sorted(
            documents, 
            key=lambda x: x.get("relevance_score", 0), 
            reverse=True
        )
        
        context_parts = []
        used_docs = []
        total_tokens = 0
        
        for doc in sorted_docs:
            # Format document citation
            doc_source = doc['metadata'].get('source', 'Unknown Source')
            doc_title = doc['metadata'].get('title', doc_source)
            doc_author = doc['metadata'].get('author', 'Unknown Author')
            page_info = f"p.{doc['metadata'].get('page_num', '')}" if 'page_num' in doc['metadata'] else ""
            
            citation = f"{doc_title} ({doc_author}) {page_info}".strip()
            
            # Format document content with citation
            doc_text = f"Document: {citation}\n\n{doc['content']}"
            
            # Check if adding this document exceeds token limit
            doc_tokens = self._estimate_token_count(doc_text)
            if total_tokens + doc_tokens > self.max_context_length:
                break
                
            context_parts.append(doc_text)
            total_tokens += doc_tokens
            used_docs.append(doc)
        
        # Combine all context parts
        context = "\n\n---\n\n".join(context_parts)
        
        return context, used_docs
    
    def augment_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Augment a prompt with retrieved context
        
        Args:
            prompt: User query/prompt
            system_prompt: Optional system prompt to augment
            
        Returns:
            Dictionary with augmented prompts and metadata
        """
        try:
            # Get relevant documents
            docs = self.retrieve_context(prompt)
            
            if not docs:
                logger.info("No relevant context found for query")
                return {
                    "augmented_prompt": prompt, 
                    "augmented_system_prompt": system_prompt,
                    "context_used": False,
                    "documents": []
                }
            
            # Format context
            context, used_docs = self._format_context(docs)
            
            # Create augmented user prompt
            augmented_prompt = prompt
            
            # Create augmented system prompt
            if system_prompt:
                augmented_system_prompt = f"""
{system_prompt}

Use the following context from causal literature to help answer the query:

{context}

When using information from the context, refer to the source document by title.
If the context doesn't contain information needed to answer, acknowledge this and use your general knowledge.
"""
            else:
                augmented_system_prompt = f"""
Use the following context from causal literature to help answer the query:

{context}

When using information from the context, refer to the source document by title.
If the context doesn't contain information needed to answer, acknowledge this and use your general knowledge.
"""
            
            return {
                "augmented_prompt": augmented_prompt,
                "augmented_system_prompt": augmented_system_prompt,
                "context_used": True,
                "documents": [
                    {
                        "source": doc['metadata'].get('source', 'Unknown'),
                        "title": doc['metadata'].get('title', 'Unknown'),
                        "author": doc['metadata'].get('author', 'Unknown'),
                        "relevance": doc.get('relevance_score', 0)
                    } 
                    for doc in used_docs
                ]
            }
        
        except Exception as e:
            logger.error(f"Error augmenting prompt: {str(e)}")
            return {
                "augmented_prompt": prompt,
                "augmented_system_prompt": system_prompt,
                "context_used": False,
                "documents": [],
                "error": str(e)
            }
    
    def keyword_search(self, query: str, fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword search on document content and metadata
        
        Args:
            query: Keyword search query
            fields: List of metadata fields to search in (default: all)
            
        Returns:
            List of matching documents
        """
        try:
            # Basic keyword search implementation
            # For a more sophisticated implementation, consider using a hybrid search approach
            
            # First use vector search
            vector_results = self.retrieve_context(query)
            
            # Filter for keyword matches
            keywords = [k.lower() for k in re.split(r'\W+', query) if k]
            results = []
            
            for doc in vector_results:
                # Check content for keyword matches
                content_matches = any(keyword in doc['content'].lower() for keyword in keywords)
                
                # Check metadata fields for keyword matches
                metadata_matches = False
                if fields:
                    for field in fields:
                        if field in doc['metadata'] and any(
                            keyword in str(doc['metadata'][field]).lower() for keyword in keywords
                        ):
                            metadata_matches = True
                            break
                else:
                    # Search all metadata fields
                    for field, value in doc['metadata'].items():
                        if any(keyword in str(value).lower() for keyword in keywords):
                            metadata_matches = True
                            break
                
                if content_matches or metadata_matches:
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error performing keyword search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            List of matching documents
        """
        try:
            # Get semantic search results
            semantic_results = self.retrieve_context(query)
            
            # Get keyword search results
            keyword_results = self.keyword_search(query)
            
            # Combine results with score adjustment
            combined_results = {}
            
            # Add semantic results
            for doc in semantic_results:
                doc_id = doc['metadata'].get('document_id', '') + doc['content'][:50]
                combined_results[doc_id] = {
                    "doc": doc,
                    "score": doc.get('relevance_score', 0) * (1 - keyword_weight)
                }
            
            # Add/boost keyword results
            for doc in keyword_results:
                doc_id = doc['metadata'].get('document_id', '') + doc['content'][:50]
                if doc_id in combined_results:
                    # Boost existing score
                    combined_results[doc_id]['score'] += keyword_weight
                else:
                    # Add new document
                    combined_results[doc_id] = {
                        "doc": doc,
                        "score": keyword_weight
                    }
            
            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Return documents
            return [item['doc'] for item in sorted_results[:self.k]]
        
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            return []
