# app/components/categorical_transformer.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoricalTransformer:
    """
    Transformer for categorical variables in causal discovery.
    Provides methods for encoding, binning, and handling categorical data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the categorical transformer
        
        Args:
            data: DataFrame with categorical data
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.transformations = []
        self.encoders = {}
    
    def detect_categorical_columns(self) -> List[str]:
        """
        Detect categorical columns in the data
        
        Returns:
            List of categorical column names
        """
        categorical_cols = []
        
        # Columns with object or category dtype
        cat_dtype_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols.extend(cat_dtype_cols)
        
        # Numeric columns with few unique values (potential categorical)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Count unique values
            n_unique = self.data[col].nunique()
            if n_unique <= 10 and n_unique < len(self.data) * 0.05:  # Consider numeric with <= 10 unique values as categorical
                categorical_cols.append(col)
        
        return categorical_cols
    
    def one_hot_encode(self, columns: List[str], drop_first: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical columns
        
        Args:
            columns: List of columns to encode
            drop_first: Whether to drop the first category (avoid multicollinearity)
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Create and fit encoder
        encoder = OneHotEncoder(sparse=False, drop='first' if drop_first else None)
        
        # Transform each column separately
        for col in columns:
            # Fit encoder
            encoded = encoder.fit_transform(self.data[[col]])
            
            # Get feature names
            if hasattr(encoder, 'get_feature_names_out'):
                feature_names = encoder.get_feature_names_out([col])
            else:
                # For older scikit-learn versions
                categories = encoder.categories_[0]
                if drop_first:
                    categories = categories[1:]
                feature_names = [f"{col}_{cat}" for cat in categories]
            
            # Add encoded columns to dataframe
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=self.data.index)
            self.data = pd.concat([self.data, encoded_df], axis=1)
            
            # Store encoder for this column
            self.encoders[col] = {
                'type': 'one_hot',
                'encoder': encoder,
                'feature_names': feature_names
            }
        
        # Record transformation
        self.transformations.append({
            "type": "one_hot_encode",
            "columns": columns,
            "drop_first": drop_first
        })
        
        return self.data
    
    def ordinal_encode(self, columns: List[str], categories: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Ordinal encode categorical columns
        
        Args:
            columns: List of columns to encode
            categories: Dictionary mapping column names to category orders
            
        Returns:
            DataFrame with ordinal encoded columns
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # If categories not provided, use existing values
        if categories is None:
            categories = {}
            for col in columns:
                categories[col] = list(self.data[col].dropna().unique())
        
        # Encode each column
        for col in columns:
            # Get categories for this column
            if col in categories:
                cats = categories[col]
            else:
                cats = list(self.data[col].dropna().unique())
            
            # Create encoder
            encoder = OrdinalEncoder(categories=[cats])
            
            # Fit and transform
            encoded = encoder.fit_transform(self.data[[col]])
            
            # Replace original column
            self.data[f"{col}_ordinal"] = encoded
            
            # Store encoder
            self.encoders[col] = {
                'type': 'ordinal',
                'encoder': encoder,
                'categories': cats
            }
        
        # Record transformation
        self.transformations.append({
            "type": "ordinal_encode",
            "columns": columns,
            "categories": categories
        })
        
        return self.data
    
    def label_encode(self, columns: List[str]) -> pd.DataFrame:
        """
        Label encode categorical columns
        
        Args:
            columns: List of columns to encode
            
        Returns:
            DataFrame with label encoded columns
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Encode each column
        for col in columns:
            # Create encoder
            encoder = LabelEncoder()
            
            # Fit and transform
            encoded = encoder.fit_transform(self.data[col].astype(str))
            
            # Replace original column
            self.data[f"{col}_label"] = encoded
            
            # Store encoder
            self.encoders[col] = {
                'type': 'label',
                'encoder': encoder,
                'classes': encoder.classes_
            }
        
        # Record transformation
        self.transformations.append({
            "type": "label_encode",
            "columns": columns
        })
        
        return self.data
    
    def bin_numeric(self, 
                   column: str, 
                   num_bins: int = 5, 
                   strategy: str = 'uniform',
                   labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Bin a numeric column into categories
        
        Args:
            column: Column to bin
            num_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            labels: Optional labels for bins
            
        Returns:
            DataFrame with binned column
        """
        # Validate column
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Ensure column is numeric
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        # Create bins
        if strategy == 'uniform':
            bins = pd.cut(self.data[column], bins=num_bins, labels=labels)
        elif strategy == 'quantile':
            bins = pd.qcut(self.data[column], q=num_bins, labels=labels)
        elif strategy == 'kmeans':
            from sklearn.cluster import KMeans
            
            # Use KMeans to find bin edges
            x = self.data[column].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(x)
            
            # Assign clusters
            clusters = kmeans.predict(x)
            
            # Create categorical bins
            if labels is not None:
                # Map clusters to labels
                cluster_map = {i: label for i, label in enumerate(labels)}
                bins = pd.Series([cluster_map[c] for c in clusters], index=self.data.index)
            else:
                bins = pd.Series(clusters, index=self.data.index)
        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'uniform', 'quantile', or 'kmeans'")
        
        # Add binned column
        self.data[f"{column}_binned"] = bins
        
        # Record transformation
        self.transformations.append({
            "type": "bin_numeric",
            "column": column,
            "num_bins": num_bins,
            "strategy": strategy
        })
        
        return self.data
    
    def create_dummy_variables(self, columns: List[str], drop_first: bool = False) -> pd.DataFrame:
        """
        Create dummy variables using pandas get_dummies
        
        Args:
            columns: List of columns to encode
            drop_first: Whether to drop the first category
            
        Returns:
            DataFrame with dummy variables
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Create dummy variables
        dummies = pd.get_dummies(self.data[columns], drop_first=drop_first)
        
        # Concatenate with original dataframe
        self.data = pd.concat([self.data, dummies], axis=1)
        
        # Record transformation
        self.transformations.append({
            "type": "create_dummy_variables",
            "columns": columns,
            "drop_first": drop_first
        })
        
        return self.data
    
    def collapse_categories(self, 
                          column: str, 
                          mapping: Dict[str, str],
                          new_column: Optional[str] = None) -> pd.DataFrame:
        """
        Collapse multiple categories into fewer categories
        
        Args:
            column: Column to transform
            mapping: Dictionary mapping original values to new values
            new_column: Name for new column (if None, replace original)
            
        Returns:
            DataFrame with collapsed categories
        """
        # Validate column
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Apply mapping
        if new_column:
            self.data[new_column] = self.data[column].map(mapping).fillna(self.data[column])
        else:
            self.data[column] = self.data[column].map(mapping).fillna(self.data[column])
        
        # Record transformation
        self.transformations.append({
            "type": "collapse_categories",
            "column": column,
            "mapping": mapping,
            "new_column": new_column
        })
        
        return self.data
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations applied
        
        Returns:
            Dictionary with transformation information
        """
        return {
            "original_shape": self.original_data.shape,
            "current_shape": self.data.shape,
            "transformations": self.transformations,
            "encoders": {k: {'type': v['type']} for k, v in self.encoders.items()}
        }
    
    def reset(self) -> pd.DataFrame:
        """
        Reset to original data
        
        Returns:
            Original DataFrame
        """
        self.data = self.original_data.copy()
        self.transformations = []
        self.encoders = {}
        return self.data


def render_categorical_transformer(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render the categorical transformer interface in Streamlit
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (processed_dataframe, transformation_summary)
    """
    st.subheader("Categorical Variable Transformer")
    
    # Initialize transformer
    transformer = CategoricalTransformer(data)
    
    # Detect categorical columns
    categorical_cols = transformer.detect_categorical_columns()
    
    st.markdown("### Detected Categorical Columns")
    if categorical_cols:
        st.write(", ".join(categorical_cols))
    else:
        st.write("No categorical columns detected")
    
    # Encoding options
    st.markdown("### Encoding Methods")
    
    encoding_method = st.selectbox(
        "Select encoding method",
        options=['One-Hot Encoding', 'Ordinal Encoding', 'Label Encoding', 'Dummy Variables', 'Bin Numeric'],
        index=0
    )
    
    if encoding_method == 'One-Hot Encoding':
        cols_to_encode = st.multiselect("Select columns to one-hot encode", categorical_cols)
        drop_first = st.checkbox("Drop first category (avoid multicollinearity)", value=False)
        
        if st.button("Apply One-Hot Encoding"):
            if cols_to_encode:
                transformer.one_hot_encode(cols_to_encode, drop_first)
                st.success(f"One-hot encoded {len(cols_to_encode)} columns")
            else:
                st.warning("No columns selected for encoding")
    
    elif encoding_method == 'Ordinal Encoding':
        cols_to_encode = st.multiselect("Select columns to ordinal encode", categorical_cols)
        
        # Option to specify order for one column
        if cols_to_encode:
            specify_order = st.checkbox("Specify category order for one column")
            
            if specify_order:
                col_for_order = st.selectbox("Select column", cols_to_encode)
                unique_values = data[col_for_order].dropna().unique().tolist()
                
                # Let user reorder values
                ordered_values = st.multiselect(
                    "Order categories (first = lowest, last = highest)",
                    options=unique_values,
                    default=unique_values
                )
                
                if st.button("Apply Ordinal Encoding"):
                    if ordered_values:
                        categories = {col_for_order: ordered_values}
                        transformer.ordinal_encode(cols_to_encode, categories)
                        st.success(f"Ordinal encoded {len(cols_to_encode)} columns with custom order for {col_for_order}")
                    else:
                        transformer.ordinal_encode(cols_to_encode)
                        st.success(f"Ordinal encoded {len(cols_to_encode)} columns")
            else:
                if st.button("Apply Ordinal Encoding"):
                    transformer.ordinal_encode(cols_to_encode)
                    st.success(f"Ordinal encoded {len(cols_to_encode)} columns")
        else:
            if st.button("Apply Ordinal Encoding"):
                st.warning("No columns selected for encoding")
    
    elif encoding_method == 'Label Encoding':
        cols_to_encode = st.multiselect("Select columns to label encode", categorical_cols)
        
        if st.button("Apply Label Encoding"):
            if cols_to_encode:
                transformer.label_encode(cols_to_encode)
                st.success(f"Label encoded {len(cols_to_encode)} columns")
            else:
                st.warning("No columns selected for encoding")
    
    elif encoding_method == 'Dummy Variables':
        cols_to_encode = st.multiselect("Select columns for dummy variables", categorical_cols)
        drop_first = st.checkbox("Drop first dummy (avoid multicollinearity)", value=False)
        
        if st.button("Create Dummy Variables"):
            if cols_to_encode:
                transformer.create_dummy_variables(cols_to_encode, drop_first)
                st.success(f"Created dummy variables for {len(cols_to_encode)} columns")
            else:
                st.warning("No columns selected for dummy variables")
    
    elif encoding_method == 'Bin Numeric':
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col_to_bin = st.selectbox("Select numeric column to bin", numeric_cols)
            num_bins = st.slider("Number of bins", min_value=2, max_value=20, value=5)
            
            strategy = st.selectbox(
                "Binning strategy",
                options=['uniform', 'quantile', 'kmeans'],
                index=0
            )
            
            if st.button("Bin Numeric Column"):
                transformer.bin_numeric(col_to_bin, num_bins, strategy)
                st.success(f"Binned column {col_to_bin} into {num_bins} bins using {strategy} strategy")
        else:
            st.warning("No numeric columns available for binning")
    
    # Category collapsing
    st.markdown("### Collapse Categories")
    
    if categorical_cols:
        col_to_collapse = st.selectbox("Select column to collapse categories", categorical_cols)
        
        if col_to_collapse:
            unique_values = data[col_to_collapse].dropna().unique().tolist()
            
            st.write("Define mapping for categories (leave blank to keep original):")
            
            # Create mapping inputs
            mapping = {}
            cols = st.columns(2)
            
            for i, val in enumerate(unique_values):
                with cols[i % 2]:
                    new_val = st.text_input(f"{val}", key=f"map_{i}")
                    if new_val:
                        mapping[val] = new_val
            
            new_col_name = st.text_input("New column name (leave blank to replace original)")
            
            if st.button("Collapse Categories"):
                if mapping:
                    transformer.collapse_categories(
                        col_to_collapse, 
                        mapping,
                        new_col_name if new_col_name else None
                    )
                    st.success(f"Collapsed categories for {col_to_collapse}")
                else:
                    st.warning("No mapping defined")
    else:
        st.write("No categorical columns available for collapsing")
    
    # Preview processed data
    st.markdown("### Preview Processed Data")
    st.write(transformer.data.head())
    
    # Get transformation summary
    transformation_summary = transformer.get_transformation_summary()
    
    with st.expander("Transformation Summary"):
        st.write(f"Original shape: {transformation_summary['original_shape']}")
        st.write(f"Current shape: {transformation_summary['current_shape']}")
        st.write("Transformations:")
        for i, t in enumerate(transformation_summary['transformations']):
            st.write(f"{i+1}. {t['type']} - {', '.join(t.get('columns', [t.get('column', '')]))}")
    
    # Option to reset
    if st.button("Reset to Original Data"):
        transformer.reset()
        st.success("Reset to original data")
    
    return transformer.data, transformation_summary


def component_executor(node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the categorical transformer component
    
    Args:
        node_data: Node configuration data
        inputs: Input data from connected nodes
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Get input data
        input_data = None
        for node_id, node_output in inputs.items():
            if "data" in node_output and isinstance(node_output["data"], pd.DataFrame):
                input_data = node_output["data"]
                break
        
        if input_data is None:
            return {
                "status": "error",
                "message": "No input dataframe provided",
                "data": {}
            }
        
        # Create transformer
        transformer = CategoricalTransformer(input_data)
        
        # Get transformations from node data
        transformations = node_data.get("transformations", [])
        
        # Execute each transformation in sequence
        for t in transformations:
            t_type = t.get("type")
            
            if t_type == "one_hot_encode":
                columns = t.get("columns", [])
                drop_first = t.get("drop_first", False)
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for one-hot encoding",
                        "data": {}
                    }
                
                try:
                    transformer.one_hot_encode(columns, drop_first)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error one-hot encoding columns: {str(e)}",
                        "data": {}
                    }
            
            elif t_type == "ordinal_encode":
                columns = t.get("columns", [])
                categories = t.get("categories")
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for ordinal encoding",
                        "data": {}
                    }
                
                try:
                    transformer.ordinal_encode(columns, categories)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error ordinal encoding columns: {str(e)}",
                        "data": {}
                    }
            
            elif t_type == "label_encode":
                columns = t.get("columns", [])
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for label encoding",
                        "data": {}
                    }
                
                try:
                    transformer.label_encode(columns)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error label encoding columns: {str(e)}",
                        "data": {}
                    }
            
            elif t_type == "create_dummy_variables":
                columns = t.get("columns", [])
                drop_first = t.get("drop_first", False)
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for dummy variables",
                        "data": {}
                    }
                
                try:
                    transformer.create_dummy_variables(columns, drop_first)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating dummy variables: {str(e)}",
                        "data": {}
                    }
            
            elif t_type == "bin_numeric":
                column = t.get("column")
                num_bins = t.get("num_bins", 5)
                strategy = t.get("strategy", "uniform")
                
                if not column:
                    return {
                        "status": "error",
                        "message": "No column specified for binning",
                        "data": {}
                    }
                
                try:
                    transformer.bin_numeric(column, num_bins, strategy)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error binning numeric column: {str(e)}",
                        "data": {}
                    }
            
            elif t_type == "collapse_categories":
                column = t.get("column")
                mapping = t.get("mapping", {})
                new_column = t.get("new_column")
                
                if not column:
                    return {
                        "status": "error",
                        "message": "No column specified for category collapsing",
                        "data": {}
                    }
                
                if not mapping:
                    return {
                        "status": "error",
                        "message": "No mapping provided for category collapsing",
                        "data": {}
                    }
                
                try:
                    transformer.collapse_categories(column, mapping, new_column)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error collapsing categories: {str(e)}",
                        "data": {}
                    }
        
        # Get transformation summary
        transformation_summary = transformer.get_transformation_summary()
        
        # Return the results
        return {
            "status": "completed",
            "message": f"Categorical transformations completed with {len(transformation_summary['transformations'])} transformations",
            "data": {
                "data": transformer.data,
                "transformation_summary": transformation_summary
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in categorical transformer: {str(e)}",
            "data": {}
        }