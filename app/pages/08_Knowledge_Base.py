import streamlit as st
import os
import asyncio
from datetime import datetime
import time
import pandas as pd
from pathlib import Path
import uuid
import json
import sys

# Import core modules
try:
    from core.rag.rag_manager import RAGManager
    from core.rag.async_utils import run_async
    RAG_AVAILABLE = True
except ImportError:
    st.error("RAG components not available. Please check your installation.")
    RAGManager = None
    run_async = None
    RAG_AVAILABLE = False

# Set page title and icon
st.set_page_config(
    page_title="Knowledge Base Management",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for RAG
if "rag_manager" not in st.session_state:
    st.session_state.rag_manager = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "rag_upload_status" not in st.session_state:
    st.session_state.rag_upload_status = {}
if "rag_documents" not in st.session_state:
    st.session_state.rag_documents = None
if "rag_last_refresh" not in st.session_state:
    st.session_state.rag_last_refresh = 0

# Create temporary directory for file uploads if it doesn't exist
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

st.title("📚 Knowledge Base Management")

# Check if RAG module is available
if RAGManager is None:
    st.warning("The RAG module is not available. Please make sure all dependencies are installed.")
    st.info("Run `pip install langchain langchain-community chromadb sentence-transformers pypdf2` to install required packages.")
    st.stop()

# Check if RAG is enabled
if not st.session_state.get("rag_enabled", False):
    st.warning("RAG is not enabled. Please enable it in the Settings page.")
    
    # Show a button to go to settings page
    if st.button("Go to Settings"):
        # Use Streamlit navigation to go to the Settings page
        st.experimental_set_query_params(page="06_Settings.py")
    
    st.stop()

# Initialize RAG manager if needed
if st.session_state.rag_manager is None and st.session_state.rag_enabled:
    try:
        openai_api_key = st.session_state.api_keys.get("openai", "")
        embeddings_model = st.session_state.get("embeddings_model", "huggingface")
        
        # Set the embeddings model to HuggingFace if OpenAI API key is missing and OpenAI was selected
        if embeddings_model == "openai" and not openai_api_key:
            st.warning("OpenAI API key is missing but OpenAI embeddings were selected. Falling back to HuggingFace embeddings.")
            embeddings_model = "huggingface"
            st.session_state.embeddings_model = "huggingface"
        
        with st.spinner("Initializing Knowledge Base..."):
            st.session_state.rag_manager = RAGManager(
                llm_adapter=st.session_state.llm_adapter,
                embeddings_model=embeddings_model,
                openai_api_key=openai_api_key,
                db_dir="./data/rag_db",
                use_simple_processor=True  # Use the simpler processor for better stability
            )
        
        st.success(f"Knowledge Base initialized successfully with {embeddings_model} embeddings")
    except Exception as e:
        st.error(f"Error initializing RAG manager: {str(e)}")
        st.stop()

# Helper function to process files asynchronously
@run_async
async def process_file(file_path, metadata):
    """Process a file and add it to the RAG system"""
    if st.session_state.rag_manager:
        return await st.session_state.rag_manager.add_document(file_path, metadata)
    return {"status": "error", "message": "RAG manager not initialized"}

# Helper function to refresh document list
def refresh_document_list():
    """Refresh the list of documents in the RAG system"""
    if st.session_state.rag_manager:
        st.session_state.rag_documents = st.session_state.rag_manager.list_documents()
        st.session_state.rag_last_refresh = time.time()

# Create tabs for different functions
tab1, tab2, tab3, tab4 = st.tabs(["Upload Documents", "Batch Import", "Manage Knowledge Base", "Settings"])

with tab1:
    st.header("Upload Documents")
    
    st.markdown("""
    Add causal literature to your knowledge base to enhance the LLM's expertise in causality.
    Supported formats: PDF, Text, and Markdown files.
    """)
    
    # Show current embedding model
    embeddings_model = st.session_state.get("embeddings_model", "huggingface")
    st.info(f"Using {embeddings_model.capitalize()} embeddings for document processing.")
    
    if embeddings_model == "openai" and not st.session_state.api_keys.get("openai"):
        st.error("OpenAI API key is missing but OpenAI embeddings were selected. Please add an OpenAI API key in the Settings page or switch to HuggingFace embeddings.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload causal literature", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Document metadata
        with st.expander("Document Metadata", expanded=True):
            st.info("The system will attempt to extract titles from PDF documents automatically. You can override this by entering a title below.")
            doc_title = st.text_input("Document Title", "", help="Leave blank to auto-extract from PDF metadata or content")
            doc_author = st.text_input("Author", "")
            doc_tags = st.text_input("Tags (comma-separated)", "")
            doc_description = st.text_area("Description", "", height=100)
            
            # Create base metadata
            metadata = {
                "title": doc_title,
                "author": doc_author,
                "tags": ','.join([tag.strip() for tag in doc_tags.split(",") if tag.strip()]),
                "description": doc_description,
                "uploaded_at": datetime.now().isoformat()
            }
        
        # Process button
        if st.button("Process Documents"):
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                # Process each file
                for uploaded_file in uploaded_files:
                    # Create filename with random suffix to prevent collisions
                    file_id = str(uuid.uuid4())[:8]
                    temp_file_path = TEMP_DIR / f"{file_id}_{uploaded_file.name}"
                    
                    # Save temp file
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Update metadata for this specific file
                    file_metadata = metadata.copy()
                    if not file_metadata["title"]:
                        file_metadata["title"] = uploaded_file.name
                    file_metadata["filename"] = uploaded_file.name
                    file_metadata["file_size"] = uploaded_file.size
                    
                    # Process the file - Handle it safely with the decorated function
                    try:
                        # The function is now synchronous thanks to the decorator
                        result = process_file(temp_file_path, file_metadata)
                        
                        # Store result in session state
                        st.session_state.rag_upload_status[uploaded_file.name] = result
                        
                        # Show success or error message
                        if result["status"] == "success":
                            title_display = f": '{result.get('metadata', {}).get('title', '')}'"
                            if not result.get('metadata', {}).get('title', ''):
                                title_display = ""
                            st.success(f"Processed {uploaded_file.name}{title_display}: {result['chunks']} chunks created")
                        else:
                            st.error(f"Error processing {uploaded_file.name}: {result.get('message', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        st.session_state.rag_upload_status[uploaded_file.name] = {
                            "status": "error", 
                            "message": str(e)
                        }
            
            # Refresh document list
            refresh_document_list()
    
    # Show upload history
    if st.session_state.rag_upload_status:
        with st.expander("Upload History", expanded=True):
            # Convert upload status to DataFrame for display
            upload_data = []
            for filename, result in st.session_state.rag_upload_status.items():
                upload_data.append({
                    "File": filename,
                    "Title": result.get("metadata", {}).get("title", ""),
                    "Status": result["status"],
                    "Document ID": result.get("document_id", "N/A"),
                    "Chunks": result.get("chunks", 0) if result["status"] == "success" else 0,
                    "Message": result.get("message", "")
                })
            
            # Display as table
            if upload_data:
                upload_df = pd.DataFrame(upload_data)
                st.dataframe(upload_df, use_container_width=True)
            else:
                st.info("No uploads yet")

with tab2:
    st.header("Batch Import from Folder")
    
    st.markdown("""
    Process all documents from a specified folder. This allows you to import large collections of documents at once.
    Supported formats: PDF, Text, and Markdown files.
    """)
    
    # Folder path input
    folder_path = st.text_input("Folder Path", "", help="Enter the absolute path to the folder containing your documents")
    
    # Options for batch processing
    with st.expander("Import Options", expanded=True):
        include_subfolders = st.checkbox("Include Subfolders", value=True, help="Process documents in subfolders as well")
        skip_existing = st.checkbox("Skip Existing Documents", value=True, help="Skip files that appear to have been processed already")
        max_files = st.number_input("Maximum Files to Process", min_value=1, max_value=1000, value=100, help="Limit the number of files to process")
        
        # Document metadata
        st.subheader("Default Metadata")
        doc_author = st.text_input("Default Author", "")
        doc_tags = st.text_input("Default Tags (comma-separated)", "")
        doc_description = st.text_area("Default Description", "", height=80)
        
        # Create base metadata
        default_metadata = {
            "author": doc_author,
            "tags": doc_tags,
            "description": doc_description,
            "imported_batch": True,
            "imported_at": datetime.now().isoformat()
        }
    
    # Process button
    if st.button("Process Folder"):
        if not folder_path or not os.path.exists(folder_path):
            st.error(f"The specified folder path does not exist: {folder_path}")
        else:
            # Get list of files to process
            files_to_process = []
            
            # Helper function to collect files
            def collect_files(dir_path):
                files = []
                try:
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        
                        if os.path.isfile(item_path):
                            # Check file extension
                            ext = os.path.splitext(item_path)[1].lower()
                            if ext in [".pdf", ".txt", ".md"]:
                                files.append(item_path)
                        elif os.path.isdir(item_path) and include_subfolders:
                            # Recursively collect files from subdirectories
                            files.extend(collect_files(item_path))
                except Exception as e:
                    st.warning(f"Error accessing directory {dir_path}: {str(e)}")
                return files
            
            # Collect files
            all_files = collect_files(folder_path)
            
            # Limit the number of files
            files_to_process = all_files[:max_files]
            
            if len(files_to_process) == 0:
                st.warning(f"No supported files found in {folder_path}")
            else:
                st.info(f"Found {len(files_to_process)} files to process (out of {len(all_files)} total)")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process files
                successful = 0
                failed = 0
                skipped = 0
                results = []
                
                for i, file_path in enumerate(files_to_process):
                    # Update progress
                    progress = int((i / len(files_to_process)) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing file {i+1} of {len(files_to_process)}: {os.path.basename(file_path)}")
                    
                    # Prepare metadata for this file
                    file_metadata = default_metadata.copy()
                    file_metadata["filename"] = os.path.basename(file_path)
                    file_metadata["title"] = os.path.splitext(os.path.basename(file_path))[0]
                    file_metadata["source_folder"] = folder_path
                    
                    # Check if file has been processed before
                    if skip_existing:
                        docs = st.session_state.rag_manager.list_documents()
                        existing_docs = [doc.get("metadata", {}).get("filename") for doc in docs.get("documents", [])]
                        if file_metadata["filename"] in existing_docs:
                            skipped += 1
                            results.append({
                                "file": file_metadata["filename"],
                                "status": "skipped",
                                "message": "File appears to have been processed before"
                            })
                            continue
                    
                    try:
                        # Process the file
                        result = process_file(file_path, file_metadata)
                        
                        # Store result
                        if result["status"] == "success":
                            successful += 1
                            title_display = f" (Title: '{result.get('metadata', {}).get('title', '')}')" if result.get('metadata', {}).get('title', '') else ""
                            results.append({
                                "file": file_metadata["filename"],
                                "title": result.get('metadata', {}).get('title', 'Untitled'),
                                "status": "success",
                                "chunks": result.get("chunks", 0),
                                "document_id": result.get("document_id", "")
                            })
                        else:
                            failed += 1
                            results.append({
                                "file": file_metadata["filename"],
                                "status": "error",
                                "message": result.get("message", "Unknown error")
                            })
                            
                    except Exception as e:
                        failed += 1
                        results.append({
                            "file": file_metadata["filename"],
                            "status": "error",
                            "message": str(e)
                        })
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Show summary
                st.success(f"Processed {len(files_to_process)} files: {successful} successful, {failed} failed, {skipped} skipped")
                
                # Show detailed results
                if results:
                    with st.expander("Detailed Results", expanded=True):
                        results_df = pd.DataFrame(results)
                        # Reorder columns to put title after file
                        if "title" in results_df.columns:
                            cols = results_df.columns.tolist()
                            # Move title after file
                            file_idx = cols.index("file")
                            cols.remove("title")
                            cols.insert(file_idx + 1, "title")
                            results_df = results_df[cols]
                        st.dataframe(results_df, use_container_width=True)
                
                # Refresh document list
                refresh_document_list()

with tab3:
    st.header("Manage Knowledge Base")
    
    # Refresh button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("🔄 Refresh"):
            refresh_document_list()
    
    # Auto-refresh if needed or if visiting first time
    if (st.session_state.rag_documents is None or 
        time.time() - st.session_state.rag_last_refresh > 30):  # Refresh every 30 seconds
        refresh_document_list()
    
    # Show document list
    if st.session_state.rag_documents:
        docs = st.session_state.rag_documents
        
        if docs["status"] == "success":
            if docs["count"] > 0:
                # Convert to DataFrame for better display
                doc_list = []
                for doc in docs["documents"]:
                    metadata = doc.get("metadata", {})
                    title = metadata.get("title", "Untitled")
                    filename = metadata.get("filename", metadata.get("source", "Unknown"))
                    
                    # Highlight if title was auto-extracted
                    title_source = ""
                    if title != "Untitled" and title != filename:
                        title_source = "📄 " # Indicate extracted title
                    
                    doc_list.append({
                        "Document ID": doc.get("document_id", "Unknown"),
                        "Title": f"{title_source}{title}",
                        "Author": metadata.get("author", "Unknown"),
                        "Filename": filename,
                        "Tags": metadata.get("tags", ""),
                        "Uploaded": metadata.get("uploaded_at", "Unknown")
                    })
                
                doc_df = pd.DataFrame(doc_list)
                st.dataframe(doc_df, use_container_width=True)
                
                # Document actions
                st.subheader("Document Actions")
                
                # Select document
                selected_doc_id = st.selectbox(
                    "Select Document", 
                    options=[doc["Document ID"] for doc in doc_list],
                    format_func=lambda x: f"{x} - {next((doc['Title'] for doc in doc_list if doc['Document ID'] == x), 'Unknown')}"
                )
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("View Document Details"):
                        with st.spinner("Loading document details..."):
                            # Get document chunks
                            chunks = st.session_state.rag_manager.get_document_chunks(selected_doc_id)
                            
                            if chunks["status"] == "success":
                                st.success(f"Document has {chunks['chunk_count']} chunks")
                                
                                # Show document metadata
                                if chunks["chunks"] and chunks["chunks"][0]["metadata"]:
                                    metadata = chunks["chunks"][0]["metadata"]
                                    st.subheader("Metadata")
                                    st.json(metadata)
                                
                                # Show sample chunks
                                st.subheader("Sample Chunks")
                                for i, chunk in enumerate(chunks["chunks"][:3]):  # Show first 3 chunks
                                    with st.expander(f"Chunk {i+1}"):
                                        st.text(chunk["content"])
                            else:
                                st.error(f"Error retrieving chunks: {chunks.get('message', 'Unknown error')}")
                
                with col2:
                    if st.button("Delete Document"):
                        # Confirm deletion
                        if st.checkbox("Confirm deletion"):
                            with st.spinner("Deleting document..."):
                                result = st.session_state.rag_manager.delete_document(selected_doc_id)
                                
                                if result["status"] == "success":
                                    st.success(f"Document deleted: {result['chunks_deleted']} chunks removed")
                                    # Refresh document list
                                    refresh_document_list()
                                else:
                                    st.error(f"Error deleting document: {result.get('message', 'Unknown error')}")
                
                with col3:
                    if st.button("Test Retrieval"):
                        # Get test query
                        test_query = st.text_input("Enter a test query:", 
                                                 "What are the key causal mechanisms described in this document?")
                        
                        if test_query:
                            with st.spinner("Retrieving relevant context..."):
                                # Build filter for just this document
                                filter_metadata = {"document_id": selected_doc_id}
                                
                                # Retrieve context
                                results = st.session_state.rag_manager.retrieve_context(
                                    test_query, 
                                    filter_metadata=filter_metadata
                                )
                                
                                if results:
                                    st.success(f"Found {len(results)} relevant chunks")
                                    
                                    # Show results
                                    for i, result in enumerate(results):
                                        with st.expander(f"Result {i+1} (Relevance: {result.get('relevance_score', 0):.2f})"):
                                            st.text(result["content"])
                                else:
                                    st.warning("No relevant chunks found for this query")
            else:
                st.info("No documents in the knowledge base. Upload documents in the 'Upload Documents' tab.")
        else:
            st.error(f"Error retrieving documents: {docs.get('message', 'Unknown error')}")
    else:
        st.info("Loading documents...")
    
    # Global knowledge base actions
    st.subheader("Knowledge Base Actions")
    
    # Warning about destructive actions
    st.warning("The following actions are destructive and cannot be undone.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Knowledge Base"):
            # Confirm clearing
            confirm = st.checkbox("I understand this will delete ALL documents")
            if confirm:
                with st.spinner("Clearing knowledge base..."):
                    # First, get document count to confirm what we're deleting
                    doc_count = len(docs.get("documents", [])) if docs and docs.get("status") == "success" else 0
                    
                    if doc_count > 0:
                        st.info(f"Attempting to delete {doc_count} documents...")
                        
                        # Perform the clear operation
                        result = st.session_state.rag_manager.clear_knowledge_base()
                        
                        if result["status"] == "success":
                            st.success(f"Knowledge base cleared: {result.get('message', '')}")
                            # Refresh document list
                            refresh_document_list()
                        else:
                            st.error(f"Error clearing knowledge base: {result.get('message', 'Unknown error')}")
                            # Offer alternative solution
                            if st.button("Try Alternative Reset"):
                                try:
                                    # Completely reinitialize the RAG manager
                                    embeddings_model = st.session_state.get("embeddings_model", "huggingface")
                                    openai_api_key = st.session_state.api_keys.get("openai", "")
                                    
                                    st.session_state.rag_manager = RAGManager(
                                        llm_adapter=st.session_state.llm_adapter,
                                        embeddings_model=embeddings_model,
                                        openai_api_key=openai_api_key,
                                        db_dir="./data/rag_db",
                                        use_simple_processor=True
                                    )
                                    st.success("RAG manager reinitialized successfully")
                                    refresh_document_list()
                                except Exception as e:
                                    st.error(f"Error reinitializing RAG manager: {str(e)}")
                    else:
                        st.success("Knowledge base is already empty.")
    
    with col2:
        if st.button("Export Knowledge Base Status"):
            # Get status
            status = st.session_state.rag_manager.get_status()
            
            # Display status
            st.json(status)
            
            # Create download link
            status_json = json.dumps(status, indent=2)
            st.download_button(
                label="Download Status JSON",
                data=status_json,
                file_name="rag_status.json",
                mime="application/json"
            )

with tab4:
    st.header("Knowledge Base Settings")
    
    # RAG Status
    status = st.session_state.rag_manager.get_status()
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Status", 
            value="Ready" if status["status"] == "ready" else "Not Ready",
            delta="✓" if status["status"] == "ready" else "✗"
        )
    
    with col2:
        st.metric(
            label="Documents", 
            value=status["document_count"]
        )
    
    with col3:
        st.metric(
            label="Chunks", 
            value=status["chunk_count"]
        )
    
    # RAG Configuration
    st.subheader("Configuration")
    
    # Get current config
    config = st.session_state.rag_manager.export_config()
    
    # Create form for updating config
    with st.form("rag_config_form"):
        # Embedding model selection
        embeddings_model = st.selectbox(
            "Embeddings Model",
            options=["huggingface", "openai"],
            index=0 if config["embeddings_model"] == "huggingface" else 1,
            help="Model to use for document embeddings. HuggingFace doesn't require an API key, but OpenAI may provide better quality for some use cases."
        )
        
        # OpenAI warning
        if embeddings_model == "openai" and not st.session_state.api_keys.get("openai"):
            st.warning("OpenAI embeddings require an OpenAI API key. Please add your API key in the Settings page or select HuggingFace embeddings.")
        
        # Chunk settings
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=config["chunk_size"],
                step=100,
                help="Size of document chunks in characters"
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=config["chunk_overlap"],
                step=50,
                help="Overlap between consecutive chunks in characters"
            )
        
        # Retrieval settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            k = st.number_input(
                "Number of Results (k)",
                min_value=1,
                max_value=20,
                value=config["k"],
                step=1,
                help="Number of documents to retrieve for each query"
            )
        
        with col2:
            score_threshold = st.slider(
                "Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config["score_threshold"],
                step=0.05,
                help="Minimum similarity score to include a document"
            )
        
        with col3:
            max_context_length = st.number_input(
                "Max Context Length",
                min_value=1000,
                max_value=10000,
                value=config["max_context_length"],
                step=500,
                help="Maximum context length in tokens (approximate)"
            )
        
        # Database directory
        db_dir = st.text_input(
            "Database Directory",
            value=config["db_dir"],
            help="Directory for vector database storage"
        )
        
        # Submit button
        submitted = st.form_submit_button("Update Configuration")
        
        if submitted:
            # Check for OpenAI embeddings without API key
            if embeddings_model == "openai" and not st.session_state.api_keys.get("openai"):
                st.error("Cannot use OpenAI embeddings without an API key. Please add your API key in the Settings page or select HuggingFace embeddings.")
            else:
                # Create new config
                new_config = {
                    "embeddings_model": embeddings_model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k": k,
                    "score_threshold": score_threshold,
                    "max_context_length": max_context_length,
                    "db_dir": db_dir
                }
                
                # Update configuration
                with st.spinner("Updating configuration..."):
                    updated_config = st.session_state.rag_manager.update_config(new_config)
                    st.session_state.embeddings_model = embeddings_model
                    st.success("Configuration updated successfully")
                    
                    # Show updated config
                    st.json(updated_config)
    
    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        st.warning("Changing these settings may require re-indexing your documents.")
        
        # Re-initialize RAG manager
        if st.button("Re-initialize RAG Manager"):
            # Get current embeddings model
            embeddings_model = st.session_state.get("embeddings_model", "huggingface")
            
            # Check OpenAI API key if using OpenAI embeddings
            if embeddings_model == "openai" and not st.session_state.api_keys.get("openai"):
                st.error("Cannot re-initialize with OpenAI embeddings without an API key. Please add your API key in the Settings page or switch to HuggingFace embeddings.")
            else:
                with st.spinner("Re-initializing RAG manager..."):
                    # Get current status and config
                    current_status = st.session_state.rag_manager.get_status()
                    current_config = st.session_state.rag_manager.export_config()
                    
                    # Re-initialize with current config
                    openai_api_key = st.session_state.api_keys.get("openai", "")
                    
                    st.session_state.rag_manager = RAGManager(
                        llm_adapter=st.session_state.llm_adapter,
                        embeddings_model=embeddings_model,
                        openai_api_key=openai_api_key,
                        db_dir=current_config["db_dir"],
                        chunk_size=current_config["chunk_size"],
                        chunk_overlap=current_config["chunk_overlap"],
                        k=current_config["k"],
                        score_threshold=current_config["score_threshold"],
                        max_context_length=current_config["max_context_length"],
                        use_simple_processor=True  # Use the simpler processor for better stability
                    )
                    
                    # Check if re-initialization was successful
                    new_status = st.session_state.rag_manager.get_status()
                    
                    st.success(f"RAG manager re-initialized successfully with {embeddings_model} embeddings. Documents: {new_status['document_count']}, Chunks: {new_status['chunk_count']}")

# Footer with documentation
st.divider()
st.markdown("""
### About Retrieval-Augmented Generation (RAG)

RAG combines the power of LLMs with the ability to retrieve context from your own knowledge base. Benefits include:

- **Domain-specific expertise**: Enhance LLM capabilities with knowledge from causal literature
- **Up-to-date information**: Include papers, books, and data not in the LLM's training
- **Accurate citations**: Get responses grounded in specific sources
- **Reduced hallucinations**: Provide factual context for more reliable responses

**Use this Knowledge Base Management interface to:**
1. Upload causal literature documents (PDF, text, markdown)
2. Process entire folders of documents at once with batch import
3. Manage your knowledge base (view, delete, test documents)
4. Configure retrieval parameters

**Embedding Models:**
- **HuggingFace**: Free, local embeddings that work without an API key
- **OpenAI**: Higher quality embeddings that require an OpenAI API key
""")
