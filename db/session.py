"""
Database session management for the causal discovery platform.
"""

import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from config.settings import get_settings
from db.models import Base


# Get database settings
settings = get_settings()
db_settings = settings.database

# Initialize the database engine
if db_settings.use_db:
    engine = create_engine(
        db_settings.db_url,
        poolclass=QueuePool,
        pool_size=db_settings.pool_size,
        max_overflow=db_settings.max_overflow,
        pool_pre_ping=True,  # Check connection before using it
        connect_args={"check_same_thread": False} if db_settings.db_url.startswith('sqlite') else {}
    )
else:
    # Create in-memory SQLite database if no DB is configured
    engine = create_engine(
        'sqlite:///:memory:',
        connect_args={"check_same_thread": False}
    )

# Create a session factory
SessionFactory = sessionmaker(bind=engine)

# Create a scoped session (thread-local)
Session = scoped_session(SessionFactory)


def get_session():
    """
    Get a database session.
    
    Returns:
        SQLAlchemy session
    """
    session = Session()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)


def close_db_connection():
    """Close the database connection."""
    Session.remove()
    if engine is not None:
        engine.dispose()


def get_engine():
    """
    Get the database engine.
    
    Returns:
        SQLAlchemy engine
    """
    return engine


def store_dataset(name: str, description: str, file_path: str, rows: int, columns: int) -> int:
    """
    Store dataset information in the database.
    
    Args:
        name: Dataset name
        description: Dataset description
        file_path: Path to the dataset file
        rows: Number of rows in the dataset
        columns: Number of columns in the dataset
        
    Returns:
        ID of the created dataset
    """
    from db.models import Dataset
    
    session = Session()
    try:
        dataset = Dataset(
            name=name,
            description=description,
            file_path=file_path,
            rows=rows,
            columns=columns
        )
        session.add(dataset)
        session.commit()
        return dataset.id
    finally:
        session.close()


def store_causal_graph(dataset_id: int, algorithm_name: str, parameters: dict,
                     nodes_count: int, edges_count: int, execution_time: float,
                     graph_data: dict) -> int:
    """
    Store causal graph information in the database.
    
    Args:
        dataset_id: ID of the associated dataset
        algorithm_name: Name of the algorithm used
        parameters: Algorithm parameters
        nodes_count: Number of nodes in the graph
        edges_count: Number of edges in the graph
        execution_time: Algorithm execution time in seconds
        graph_data: Graph data (serialized)
        
    Returns:
        ID of the created causal graph
    """
    from db.models import CausalGraph
    import json
    
    session = Session()
    try:
        causal_graph = CausalGraph(
            dataset_id=dataset_id,
            algorithm_name=algorithm_name,
            parameters=json.dumps(parameters),
            nodes_count=nodes_count,
            edges_count=edges_count,
            execution_time=execution_time,
            graph_data=json.dumps(graph_data)
        )
        session.add(causal_graph)
        session.commit()
        return causal_graph.id
    finally:
        session.close()


def store_graph_refinement(original_graph_id: int, llm_provider: str, llm_model: str,
                         domain_context: str, confidence_threshold: float, nodes_count: int,
                         edges_count: int, added_edges_count: int, removed_edges_count: int,
                         reversed_edges_count: int, hidden_variables_count: int,
                         refined_graph_data: dict, refinement_steps: list,
                         hidden_variables: list) -> int:
    """
    Store graph refinement information in the database.
    
    Args:
        original_graph_id: ID of the original causal graph
        llm_provider: LLM provider name
        llm_model: LLM model name
        domain_context: Domain context provided for refinement
        confidence_threshold: Confidence threshold for refinement
        nodes_count: Number of nodes in the refined graph
        edges_count: Number of edges in the refined graph
        added_edges_count: Number of edges added
        removed_edges_count: Number of edges removed
        reversed_edges_count: Number of edges reversed
        hidden_variables_count: Number of hidden variables discovered
        refined_graph_data: Refined graph data (serialized)
        refinement_steps: List of refinement steps
        hidden_variables: List of hidden variables
        
    Returns:
        ID of the created graph refinement
    """
    from db.models import GraphRefinement
    import json
    
    session = Session()
    try:
        refinement = GraphRefinement(
            original_graph_id=original_graph_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
            domain_context=domain_context,
            confidence_threshold=confidence_threshold,
            nodes_count=nodes_count,
            edges_count=edges_count,
            added_edges_count=added_edges_count,
            removed_edges_count=removed_edges_count,
            reversed_edges_count=reversed_edges_count,
            hidden_variables_count=hidden_variables_count,
            refined_graph_data=json.dumps(refined_graph_data),
            refinement_steps=json.dumps(refinement_steps),
            hidden_variables=json.dumps(hidden_variables)
        )
        session.add(refinement)
        session.commit()
        return refinement.id
    finally:
        session.close()


def get_recent_graphs(limit: int = 5):
    """
    Get most recent causal graphs.
    
    Args:
        limit: Maximum number of graphs to retrieve
        
    Returns:
        List of CausalGraph objects
    """
    from db.models import CausalGraph
    
    session = Session()
    try:
        graphs = session.query(CausalGraph).order_by(CausalGraph.created_at.desc()).limit(limit).all()
        return graphs
    finally:
        session.close()