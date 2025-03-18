import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import core modules
from core.data.loader import DataLoader
from core.data.profiler import DataProfiler
from core.data.preprocessor import DataPreprocessor

# Ensure session state is initialized
for var in ['data_loaded', 'df', 'metadata', 'data_profile', 'preprocessor', 
            'causal_graphs', 'current_graph', 'refined_graph', 'llm_adapter', 'theme']:
    if var not in st.session_state:
        if var in ['data_loaded']:
            st.session_state[var] = False
        elif var in ['causal_graphs']:
            st.session_state[var] = {}
        elif var in ['theme']:
            st.session_state[var] = "light"
        else:
            st.session_state[var] = None

st.title("Data Loading")

tab1, tab2 = st.tabs(["Upload Data", "Sample Datasets"])

with tab1:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'])
    
    if uploaded_file is not None:
        try:
            # Determine file type and load
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext == '.csv':
                delimiter = st.text_input("Delimiter", ",")
                header = st.checkbox("First row as header", value=True)
                
                if st.button("Load CSV"):
                    data_loader = DataLoader()
                    df, metadata = data_loader.load_file(
                        uploaded_file, 
                        delimiter=delimiter, 
                        header=0 if header else None
                    )
                    
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.preprocessor = DataPreprocessor(df)
                    st.session_state.data_loaded = True
                    
                    # Profile data
                    profiler = DataProfiler()
                    st.session_state.data_profile = profiler.profile_data(df)
                    
                    st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            elif file_ext in ['.xlsx', '.xls']:
                sheet_name = st.text_input("Sheet Name (leave blank for first sheet)", "")
                header = st.checkbox("First row as header", value=True)
                
                if st.button("Load Excel"):
                    data_loader = DataLoader()
                    df, metadata = data_loader.load_file(
                        uploaded_file, 
                        sheet_name=sheet_name if sheet_name else 0, 
                        header=0 if header else None
                    )
                    
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.preprocessor = DataPreprocessor(df)
                    st.session_state.data_loaded = True
                    
                    # Profile data
                    profiler = DataProfiler()
                    st.session_state.data_profile = profiler.profile_data(df)
                    
                    st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            else:
                if st.button(f"Load {file_ext.upper()} File"):
                    data_loader = DataLoader()
                    df, metadata = data_loader.load_file(uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.preprocessor = DataPreprocessor(df)
                    st.session_state.data_loaded = True
                    
                    # Profile data
                    profiler = DataProfiler()
                    st.session_state.data_profile = profiler.profile_data(df)
                    
                    st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

with tab2:
    st.header("Sample Datasets")
    
    data_loader = DataLoader()
    sample_datasets = data_loader.list_sample_datasets()
    
    for dataset in sample_datasets:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(dataset["name"])
            st.write(dataset["description"])
            st.write(f"Variables: {dataset['n_variables']} | Samples: {dataset['n_samples']} | Domain: {dataset['domain']}")
        
        with col2:
            if st.button(f"Load {dataset['name']}", key=f"load_{dataset['id']}"):
                try:
                    df, metadata = data_loader.load_sample_dataset(dataset["id"])
                    
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.preprocessor = DataPreprocessor(df)
                    st.session_state.data_loaded = True
                    
                    # Profile data
                    profiler = DataProfiler()
                    st.session_state.data_profile = profiler.profile_data(df)
                    
                    st.success(f"Successfully loaded {dataset['name']} dataset")
                
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")
        
        st.divider()

# Display data preview if loaded
if st.session_state.data_loaded:
    st.header("Data Preview")
    st.dataframe(st.session_state.df.head(10))
    
    st.write(f"Shape: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
    
    # Show data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': st.session_state.df.dtypes.index,
        'Type': st.session_state.df.dtypes.values.astype(str)
    })
    st.dataframe(dtypes_df)
    
    # Show missing values
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Column': st.session_state.df.columns,
        'Missing Values': st.session_state.df.isna().sum().values,
        'Percentage': (st.session_state.df.isna().sum() / len(st.session_state.df) * 100).round(2).values
    })
    st.dataframe(missing_df)