"""
Database models for the causal discovery platform.
"""

import datetime
import json
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Dataset(Base):
    """Model for storing dataset information."""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(512), nullable=True)
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    causal_graphs = relationship("CausalGraph", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', rows={self.rows}, columns={self.columns})>"


class CausalGraph(Base):
    """Model for storing causal graph results."""
    __tablename__ = "causal_graphs"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    algorithm_name = Column(String(128), nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string of parameters
    nodes_count = Column(Integer, nullable=False)
    edges_count = Column(Integer, nullable=False)
    execution_time = Column(Float, nullable=True)
    graph_data = Column(Text, nullable=False)  # JSON string of NetworkX graph
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="causal_graphs")
    refinements = relationship("GraphRefinement", back_populates="original_graph")
    
    def __repr__(self):
        return f"<CausalGraph(id={self.id}, algorithm='{self.algorithm_name}', nodes={self.nodes_count}, edges={self.edges_count})>"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters as a dictionary."""
        if not self.parameters:
            return {}
        return json.loads(self.parameters)
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters from a dictionary."""
        self.parameters = json.dumps(params)
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data as a dictionary."""
        return json.loads(self.graph_data)
    
    def set_graph_data(self, graph_data: Dict[str, Any]) -> None:
        """Set graph data from a dictionary."""
        self.graph_data = json.dumps(graph_data)


class GraphRefinement(Base):
    """Model for storing graph refinement results."""
    __tablename__ = "graph_refinements"
    
    id = Column(Integer, primary_key=True)
    original_graph_id = Column(Integer, ForeignKey("causal_graphs.id"))
    llm_provider = Column(String(64), nullable=False)
    llm_model = Column(String(64), nullable=False)
    domain_context = Column(Text, nullable=True)
    confidence_threshold = Column(Float, nullable=False, default=0.5)
    nodes_count = Column(Integer, nullable=False)
    edges_count = Column(Integer, nullable=False)
    added_edges_count = Column(Integer, nullable=False, default=0)
    removed_edges_count = Column(Integer, nullable=False, default=0)
    reversed_edges_count = Column(Integer, nullable=False, default=0)
    hidden_variables_count = Column(Integer, nullable=False, default=0)
    refined_graph_data = Column(Text, nullable=False)  # JSON string of NetworkX graph
    refinement_steps = Column(Text, nullable=True)  # JSON string of refinement steps
    hidden_variables = Column(Text, nullable=True)  # JSON string of hidden variables
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    original_graph = relationship("CausalGraph", back_populates="refinements")
    
    def __repr__(self):
        return f"<GraphRefinement(id={self.id}, llm='{self.llm_provider}/{self.llm_model}', added={self.added_edges_count}, removed={self.removed_edges_count})>"
    
    def get_refined_graph_data(self) -> Dict[str, Any]:
        """Get refined graph data as a dictionary."""
        return json.loads(self.refined_graph_data)
    
    def set_refined_graph_data(self, graph_data: Dict[str, Any]) -> None:
        """Set refined graph data from a dictionary."""
        self.refined_graph_data = json.dumps(graph_data)
    
    def get_refinement_steps(self) -> List[Dict[str, Any]]:
        """Get refinement steps as a list of dictionaries."""
        if not self.refinement_steps:
            return []
        return json.loads(self.refinement_steps)
    
    def set_refinement_steps(self, steps: List[Dict[str, Any]]) -> None:
        """Set refinement steps from a list of dictionaries."""
        self.refinement_steps = json.dumps(steps)
    
    def get_hidden_variables(self) -> List[Dict[str, Any]]:
        """Get hidden variables as a list of dictionaries."""
        if not self.hidden_variables:
            return []
        return json.loads(self.hidden_variables)
    
    def set_hidden_variables(self, variables: List[Dict[str, Any]]) -> None:
        """Set hidden variables from a list of dictionaries."""
        self.hidden_variables = json.dumps(variables)


class UserSession(Base):
    """Model for storing user session data."""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False, unique=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    current_graph_id = Column(Integer, ForeignKey("causal_graphs.id"), nullable=True)
    settings = Column(Text, nullable=True)  # JSON string of user settings
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_active = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id='{self.session_id}')>"
    
    def get_settings(self) -> Dict[str, Any]:
        """Get settings as a dictionary."""
        if not self.settings:
            return {}
        return json.loads(self.settings)
    
    def set_settings(self, settings: Dict[str, Any]) -> None:
        """Set settings from a dictionary."""
        self.settings = json.dumps(settings)