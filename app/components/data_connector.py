# app/components/data_connector.py

import streamlit as st
import pandas as pd
import sqlite3
import mysql.connector
import psycopg2
import sqlalchemy
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Connects to various database types and loads data for causal analysis.
    Supports SQLite, MySQL, PostgreSQL, and generic SQLAlchemy connections.
    """
    
    def __init__(self):
        """Initialize the database connector"""
        self.connection = None
        self.engine = None
        self.metadata = {}
    
    def connect_sqlite(self, db_path: str) -> bool:
        """
        Connect to a SQLite database
        
        Args:
            db_path: Path to the SQLite database file
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(db_path)
            self.metadata = {
                "type": "sqlite",
                "path": db_path,
                "tables": self._get_sqlite_tables()
            }
            return True
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {str(e)}")
            return False
    
    def connect_mysql(self, host: str, user: str, password: str, database: str, port: int = 3306) -> bool:
        """
        Connect to a MySQL database
        
        Args:
            host: Database host
            user: Database user
            password: Database password
            database: Database name
            port: Database port (default: 3306)
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=port
            )
            self.metadata = {
                "type": "mysql",
                "host": host,
                "database": database,
                "user": user,
                "tables": self._get_mysql_tables()
            }
            return True
        except Exception as e:
            logger.error(f"Error connecting to MySQL database: {str(e)}")
            return False
    
    def connect_postgresql(self, host: str, user: str, password: str, database: str, port: int = 5432) -> bool:
        """
        Connect to a PostgreSQL database
        
        Args:
            host: Database host
            user: Database user
            password: Database password
            database: Database name
            port: Database port (default: 5432)
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(
                host=host,
                user=user,
                password=password,
                dbname=database,
                port=port
            )
            self.metadata = {
                "type": "postgresql",
                "host": host,
                "database": database,
                "user": user,
                "tables": self._get_postgresql_tables()
            }
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {str(e)}")
            return False
    
    def connect_sqlalchemy(self, connection_string: str) -> bool:
        """
        Connect using SQLAlchemy connection string
        
        Args:
            connection_string: SQLAlchemy connection string
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.engine = sqlalchemy.create_engine(connection_string)
            self.metadata = {
                "type": "sqlalchemy",
                "connection_string": connection_string.split("@")[-1] if "@" in connection_string else connection_string,
                "tables": self._get_sqlalchemy_tables()
            }
            return True
        except Exception as e:
            logger.error(f"Error connecting using SQLAlchemy: {str(e)}")
            return False
    
    def _get_sqlite_tables(self) -> List[Dict[str, Any]]:
        """Get tables from SQLite database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        result = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            result.append({
                "name": table_name,
                "columns": [col[1] for col in columns],
                "column_types": [col[2] for col in columns]
            })
        
        return result
    
    def _get_mysql_tables(self) -> List[Dict[str, Any]]:
        """Get tables from MySQL database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        
        result = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {table_name};")
            columns = cursor.fetchall()
            
            result.append({
                "name": table_name,
                "columns": [col[0] for col in columns],
                "column_types": [col[1] for col in columns]
            })
        
        return result
    
    def _get_postgresql_tables(self) -> List[Dict[str, Any]]:
        """Get tables from PostgreSQL database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        result = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}';
            """)
            columns = cursor.fetchall()
            
            result.append({
                "name": table_name,
                "columns": [col[0] for col in columns],
                "column_types": [col[1] for col in columns]
            })
        
        return result
    
    def _get_sqlalchemy_tables(self) -> List[Dict[str, Any]]:
        """Get tables from SQLAlchemy connection"""
        if not self.engine:
            return []
        
        inspector = sqlalchemy.inspect(self.engine)
        table_names = inspector.get_table_names()
        
        result = []
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            
            result.append({
                "name": table_name,
                "columns": [col["name"] for col in columns],
                "column_types": [str(col["type"]) for col in columns]
            })
        
        return result
    
    def load_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from a specific table
        
        Args:
            table_name: Name of the table to load
            limit: Maximum number of rows to load (None for all)
            
        Returns:
            DataFrame containing the data
        """
        if not self.connection and not self.engine:
            raise ValueError("Not connected to any database")
        
        # Construct query
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query based on connection type
        if self.engine:
            return pd.read_sql(query, self.engine)
        else:
            return pd.read_sql(query, self.connection)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing the query results
        """
        if not self.connection and not self.engine:
            raise ValueError("Not connected to any database")
        
        # Execute query based on connection type
        if self.engine:
            return pd.read_sql(query, self.engine)
        else:
            return pd.read_sql(query, self.connection)
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.metadata = {}
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current connection"""
        return self.metadata


def render_database_connector() -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Render the database connector interface in Streamlit
    
    Returns:
        Tuple of (success, dataframe, metadata)
    """
    st.subheader("Database Connection")
    
    # Create connector instance
    connector = DatabaseConnector()
    
    # Select database type
    db_type = st.selectbox(
        "Database Type",
        ["SQLite", "MySQL", "PostgreSQL", "SQLAlchemy"],
        index=0
    )
    
    # Connection form
    with st.form(key="db_connection_form"):
        if db_type == "SQLite":
            # SQLite options
            db_path = st.text_input("Database Path")
            
            # Submit button
            submit = st.form_submit_button("Connect")
            
            if submit:
                if not db_path:
                    st.error("Database path is required")
                    return False, None, {}
                
                success = connector.connect_sqlite(db_path)
                if not success:
                    st.error("Failed to connect to SQLite database")
                    return False, None, {}
        
        elif db_type == "MySQL":
            # MySQL options
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=3306, min_value=1, max_value=65535)
            database = st.text_input("Database")
            user = st.text_input("User")
            password = st.text_input("Password", type="password")
            
            # Submit button
            submit = st.form_submit_button("Connect")
            
            if submit:
                if not all([host, database, user]):
                    st.error("Host, database, and user are required")
                    return False, None, {}
                
                success = connector.connect_mysql(host, user, password, database, port)
                if not success:
                    st.error("Failed to connect to MySQL database")
                    return False, None, {}
        
        elif db_type == "PostgreSQL":
            # PostgreSQL options
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
            database = st.text_input("Database")
            user = st.text_input("User")
            password = st.text_input("Password", type="password")
            
            # Submit button
            submit = st.form_submit_button("Connect")
            
            if submit:
                if not all([host, database, user]):
                    st.error("Host, database, and user are required")
                    return False, None, {}
                
                success = connector.connect_postgresql(host, user, password, database, port)
                if not success:
                    st.error("Failed to connect to PostgreSQL database")
                    return False, None, {}
        
        elif db_type == "SQLAlchemy":
            # SQLAlchemy options
            connection_string = st.text_input("Connection String")
            
            # Submit button
            submit = st.form_submit_button("Connect")
            
            if submit:
                if not connection_string:
                    st.error("Connection string is required")
                    return False, None, {}
                
                success = connector.connect_sqlalchemy(connection_string)
                if not success:
                    st.error("Failed to connect using SQLAlchemy")
                    return False, None, {}
        
        else:
            st.error(f"Unsupported database type: {db_type}")
            return False, None, {}
            
    # If we get here without returning, we have a successful connection
    if connector.metadata:
        st.success(f"Connected to {db_type} database")
        
        # Show available tables
        tables = [table["name"] for table in connector.metadata.get("tables", [])]
        
        if not tables:
            st.warning("No tables found in the database")
            return True, None, connector.metadata
        
        # Let user select a table or enter a query
        use_query = st.checkbox("Use custom SQL query")
        
        if use_query:
            query = st.text_area("SQL Query", "SELECT * FROM table_name LIMIT 100")
            if st.button("Execute Query"):
                try:
                    df = connector.execute_query(query)
                    return True, df, connector.metadata
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
                    return False, None, {}
        else:
            table_name = st.selectbox("Select Table", tables)
            limit = st.number_input("Row Limit (0 for all)", value=1000, min_value=0)
            
            if st.button("Load Data"):
                try:
                    df = connector.load_data(table_name, limit if limit > 0 else None)
                    return True, df, connector.metadata
                except Exception as e:
                    st.error(f"Error loading table: {str(e)}")
                    return False, None, {}
    
    return False, None, {}


def component_executor(node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the database connector component
    
    Args:
        node_data: Node configuration data
        inputs: Input data from connected nodes
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Create connector instance
        connector = DatabaseConnector()
        
        # Get connection parameters from node data
        db_type = node_data.get("db_type", "sqlite")
        
        # Connect based on database type
        connection_success = False
        
        if db_type.lower() == "sqlite":
            db_path = node_data.get("db_path", "")
            if not db_path:
                return {
                    "status": "error",
                    "message": "SQLite database path not specified",
                    "data": {}
                }
            
            connection_success = connector.connect_sqlite(db_path)
        
        elif db_type.lower() == "mysql":
            host = node_data.get("host", "localhost")
            port = node_data.get("port", 3306)
            database = node_data.get("database", "")
            user = node_data.get("user", "")
            password = node_data.get("password", "")
            
            if not all([host, database, user]):
                return {
                    "status": "error",
                    "message": "Missing required MySQL connection parameters",
                    "data": {}
                }
            
            connection_success = connector.connect_mysql(host, user, password, database, port)
        
        elif db_type.lower() == "postgresql":
            host = node_data.get("host", "localhost")
            port = node_data.get("port", 5432)
            database = node_data.get("database", "")
            user = node_data.get("user", "")
            password = node_data.get("password", "")
            
            if not all([host, database, user]):
                return {
                    "status": "error",
                    "message": "Missing required PostgreSQL connection parameters",
                    "data": {}
                }
            
            connection_success = connector.connect_postgresql(host, user, password, database, port)
        
        elif db_type.lower() == "sqlalchemy":
            connection_string = node_data.get("connection_string", "")
            
            if not connection_string:
                return {
                    "status": "error",
                    "message": "SQLAlchemy connection string not specified",
                    "data": {}
                }
            
            connection_success = connector.connect_sqlalchemy(connection_string)
        
        else:
            return {
                "status": "error",
                "message": f"Unsupported database type: {db_type}",
                "data": {}
            }
        
        if not connection_success:
            return {
                "status": "error",
                "message": f"Failed to connect to {db_type} database",
                "data": {}
            }
        
        # Get data loading options
        use_query = node_data.get("use_query", False)
        
        if use_query:
            query = node_data.get("query", "")
            if not query:
                return {
                    "status": "error",
                    "message": "SQL query not specified",
                    "data": {}
                }
            
            try:
                df = connector.execute_query(query)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error executing query: {str(e)}",
                    "data": {}
                }
        else:
            table_name = node_data.get("table_name", "")
            if not table_name:
                return {
                    "status": "error",
                    "message": "Table name not specified",
                    "data": {}
                }
            
            limit = node_data.get("limit", 1000)
            
            try:
                df = connector.load_data(table_name, limit if limit > 0 else None)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error loading table: {str(e)}",
                    "data": {}
                }
        
        # Close the connection
        connector.close()
        
        # Return the results
        return {
            "status": "completed",
            "message": f"Successfully loaded data from {db_type} database",
            "data": {
                "data": df,
                "metadata": {
                    "source": "database",
                    "type": db_type,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in database connector: {str(e)}",
            "data": {}
        }