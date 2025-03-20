# core/data/loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import logging
import io
import urllib.request
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads datasets from files or sample datasets.
    Supports various file formats and provides built-in sample datasets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with sample datasets and data directory
        
        Args:
            data_dir: Directory for data files (default: 'data')
        """
        self.data_dir = data_dir
        self.sample_dir = os.path.join(data_dir, "samples")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize sample datasets
        self.sample_datasets = {
            "sachs": self._load_sachs,
            "boston_housing": self._load_boston_housing,
            "airfoil": self._load_airfoil,
            "galton": self._load_galton
        }
    
    def load_file(self, file_path: Union[str, Path, io.BytesIO], **kwargs) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Load data from a file
        
        Args:
            file_path: Path to the data file or file-like object
            **kwargs: Additional parameters for loading
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        # Handle file-like objects
        if isinstance(file_path, io.BytesIO):
            # Try to get filename if it exists (like from Streamlit's file_uploader)
            filename = getattr(file_path, "name", "uploaded_file")
            file_ext = os.path.splitext(filename)[1].lower()
        else:
            # Convert Path to string if needed
            file_path = str(file_path)
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Initialize metadata
            metadata = {
                "source": "file", 
                "filename": filename, 
                "format": file_ext[1:] if file_ext.startswith('.') else file_ext
            }
            
            if file_ext == '.csv':
                # Try to automatically detect delimiter
                try:
                    df = pd.read_csv(file_path, **kwargs)
                    metadata["format"] = "csv"
                    metadata["delimiter"] = kwargs.get("sep", ",")
                    
                    # Check if we might have the wrong delimiter
                    if df.shape[1] == 1 and df.iloc[0, 0].count(';') > 0:
                        logger.info("CSV file appears to use ';' as delimiter, trying again")
                        kwargs["sep"] = ";"
                        df = pd.read_csv(file_path, **kwargs)
                        metadata["delimiter"] = ";"
                except Exception as e:
                    logger.warning(f"Initial CSV parsing failed, trying alternative delimiters: {str(e)}")
                    # Try different delimiters
                    for delimiter in [';', '\t', '|']:
                        try:
                            kwargs["sep"] = delimiter
                            df = pd.read_csv(file_path, **kwargs)
                            metadata["delimiter"] = delimiter
                            break
                        except:
                            continue
                    else:
                        # If all delimiters fail, raise the original error
                        raise
            
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
                metadata["format"] = "excel"
                
                # Get sheet names if possible
                if not isinstance(file_path, io.BytesIO):
                    try:
                        metadata["sheets"] = pd.ExcelFile(file_path).sheet_names
                    except:
                        pass
            
            elif file_ext == '.json':
                # Try different JSON orientations
                orientations = ["records", "columns", "index", "split"]
                for orientation in orientations:
                    try:
                        curr_kwargs = {**kwargs, "orient": orientation}
                        df = pd.read_json(file_path, **curr_kwargs)
                        metadata["format"] = "json"
                        metadata["orientation"] = orientation
                        break
                    except:
                        continue
                else:
                    # If all orientations fail, try without specifying orientation
                    df = pd.read_json(file_path, **kwargs)
                    metadata["format"] = "json"
            
            elif file_ext == '.pkl':
                df = pd.read_pickle(file_path, **kwargs)
                metadata["format"] = "pickle"
            
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
                metadata["format"] = "parquet"
            
            elif file_ext == '.sav':
                import joblib
                df = joblib.load(file_path)
                if isinstance(df, pd.DataFrame):
                    metadata["format"] = "joblib"
                else:
                    raise ValueError("Loaded file is not a pandas DataFrame")
            
            elif file_ext in ['.txt', '.dat', '']:
                # Try to detect the format for text files
                try:
                    # Try CSV first
                    for delimiter in [',', '\t', ';', '|', ' ']:
                        try:
                            curr_kwargs = {**kwargs, "sep": delimiter}
                            df = pd.read_csv(file_path, **curr_kwargs)
                            if df.shape[1] > 1:  # We found a valid delimiter
                                metadata["format"] = "delimited_text"
                                metadata["delimiter"] = delimiter
                                break
                        except:
                            continue
                    else:
                        # If no delimiter works, try a more generic approach
                        df = pd.read_table(file_path, **kwargs)
                        metadata["format"] = "text"
                except Exception as e:
                    logger.error(f"Could not parse text file: {str(e)}")
                    raise ValueError(f"Could not determine format for file {filename}. Error: {str(e)}")
            
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
            
            # Enhance metadata with basic dataset information
            metadata.update({
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_missing_values": df.isna().any().any()
            })
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            raise ValueError(f"Failed to load file {filename}: {str(e)}") from e
    
    def load_from_url(self, url: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from a URL
        
        Args:
            url: URL to the data file
            **kwargs: Additional parameters for loading
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        try:
            # Extract filename from URL
            filename = os.path.basename(url)
            # Download the file to a temporary location
            with urllib.request.urlopen(url) as response:
                file_content = response.read()
            
            # Use BytesIO to create a file-like object
            file_obj = io.BytesIO(file_content)
            file_obj.name = filename  # Set name for format detection
            
            # Load using the file loader
            df, metadata = self.load_file(file_obj, **kwargs)
            
            # Update metadata
            metadata["source"] = "url"
            metadata["url"] = url
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading data from URL {url}: {str(e)}")
            raise ValueError(f"Failed to load data from URL {url}: {str(e)}") from e
    
    def load_sample_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a sample dataset
        
        Args:
            dataset_name: Name of the sample dataset
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        if dataset_name not in self.sample_datasets:
            raise ValueError(f"Unknown sample dataset: {dataset_name}. Available datasets: {list(self.sample_datasets.keys())}")
        
        try:
            return self.sample_datasets[dataset_name]()
        except Exception as e:
            logger.error(f"Error loading sample dataset {dataset_name}: {str(e)}")
            raise ValueError(f"Failed to load sample dataset {dataset_name}: {str(e)}") from e
    
    def list_sample_datasets(self) -> List[Dict[str, Any]]:
        """
        List available sample datasets
        
        Returns:
            List of dictionaries with dataset information
        """
        return [
            {
                "id": "sachs",
                "name": "Sachs Protein Signaling",
                "description": "Protein signaling network data with 11 variables and 7466 observations.",
                "n_variables": 11,
                "n_samples": 7466,
                "domain": "Biology/Proteomics",
                "has_ground_truth": True
            },
            {
                "id": "boston_housing",
                "name": "Boston Housing Dataset",
                "description": "Housing data with 14 variables and 506 observations.",
                "n_variables": 14,
                "n_samples": 506,
                "domain": "Real Estate/Economics",
                "has_ground_truth": False
            },
            {
                "id": "airfoil",
                "name": "NASA Airfoil Self-Noise",
                "description": "Airfoil self-noise data with 6 variables and 1503 observations.",
                "n_variables": 6,
                "n_samples": 1503,
                "domain": "Engineering/Aerodynamics",
                "has_ground_truth": False
            },
            {
                "id": "galton",
                "name": "Galton Height Data",
                "description": "Francis Galton's dataset on heights of parents and their children.",
                "n_variables": 5,
                "n_samples": 934,
                "domain": "Genetics/Heredity",
                "has_ground_truth": True
            }
        ]
    
    def save_dataframe(self, df: pd.DataFrame, file_path: str, format: str = None) -> Dict[str, Any]:
        """
        Save a DataFrame to a file
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            format: File format (if None, inferred from extension)
            
        Returns:
            Metadata about the saved file
        """
        try:
            # Make directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # If format not specified, infer from file extension
            if format is None:
                format = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # Save based on format
            if format in ['csv', '.csv']:
                df.to_csv(file_path, index=False)
                file_format = "csv"
            elif format in ['excel', 'xlsx', 'xls', '.xlsx', '.xls']:
                df.to_excel(file_path, index=False)
                file_format = "excel"
            elif format in ['json', '.json']:
                df.to_json(file_path, orient="records")
                file_format = "json"
            elif format in ['pickle', 'pkl', '.pickle', '.pkl']:
                df.to_pickle(file_path)
                file_format = "pickle"
            elif format in ['parquet', '.parquet']:
                df.to_parquet(file_path, index=False)
                file_format = "parquet"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Return metadata
            return {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "format": file_format,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "size_bytes": os.path.getsize(file_path)
            }
        
        except Exception as e:
            logger.error(f"Error saving DataFrame to {file_path}: {str(e)}")
            raise ValueError(f"Failed to save DataFrame to {file_path}: {str(e)}") from e
    
    def _load_sachs(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load Sachs protein signaling dataset"""
        try:
            # First try to load from causallearn
            try:
                from causallearn.utils.Dataset import load_dataset
                data, labels = load_dataset("sachs")
                df = pd.DataFrame(data, columns=labels)
                logger.info("Loaded Sachs dataset from causallearn")
            except:
                # If causallearn fails, try loading from a URL
                logger.info("Causallearn not available, loading Sachs dataset from URL")
                url = "https://github.com/kurowasan/causal-datasets/raw/main/sachs/sachs.csv"
                
                # Save to local file if not exists
                local_file = os.path.join(self.sample_dir, "sachs.csv")
                if not os.path.exists(local_file):
                    logger.info(f"Downloading Sachs dataset to {local_file}")
                    # Download the file
                    urllib.request.urlretrieve(url, local_file)
                
                # Load the dataset
                if os.path.exists(local_file):
                    df = pd.read_csv(local_file)
                else:
                    # Fallback to direct URL loading
                    df, _ = self.load_from_url(url)
            
            # Add ground truth edges
            ground_truth_edges = [
                ("Raf", "Mek"), ("Mek", "Erk"), ("PLCg", "PIP2"),
                ("PIP2", "PIP3"), ("PIP3", "Akt"), ("PKC", "Raf"),
                ("PKC", "Mek"), ("PKC", "Jnk"), ("PKA", "Raf"),
                ("PKA", "Mek"), ("PKA", "Erk"), ("PKA", "Akt"),
                ("PKA", "P38"), ("PKA", "Jnk")
            ]
            
            # Add metadata
            metadata = {
                "source": "sample",
                "name": "Sachs Protein Signaling",
                "description": "Protein signaling network data with interventions",
                "citation": "Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721), 523-529.",
                "variables": {
                    "Raf": {"description": "Proto-oncogene serine/threonine-protein kinase"},
                    "Mek": {"description": "Dual specificity mitogen-activated protein kinase"},
                    "Plcg": {"description": "Phospholipase C, gamma"},
                    "PIP2": {"description": "Phosphatidylinositol 4,5-bisphosphate"},
                    "PIP3": {"description": "Phosphatidylinositol (3,4,5)-trisphosphate"},
                    "Erk": {"description": "Extracellular signal-regulated kinases"},
                    "Akt": {"description": "Protein kinase B"},
                    "PKA": {"description": "Protein kinase A"},
                    "PKC": {"description": "Protein kinase C"},
                    "P38": {"description": "P38 mitogen-activated protein kinases"},
                    "JNK": {"description": "c-Jun N-terminal kinases"}
                },
                "ground_truth_available": True,
                "ground_truth_edges": ground_truth_edges,
                "domain": "Biology/Proteomics",
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading Sachs dataset: {str(e)}")
            
            # Fallback data if loading fails
            data = np.random.randn(100, 11)
            columns = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "JNK"]
            df = pd.DataFrame(data, columns=columns)
            
            metadata = {
                "source": "sample",
                "name": "Sachs Protein Signaling (Simulated)",
                "description": "Simulated data with similar structure to Sachs dataset (error loading original)",
                "variables": {col: {"description": f"Simulated {col}"} for col in columns},
                "ground_truth_available": False,
                "error": str(e),
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
    
    def _load_boston_housing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads the Boston Housing dataset.
        
        Note: This dataset is removed from scikit-learn due to ethical concerns, so
        we handle it carefully with appropriate fallbacks.
        """
        try:
            # First try to load from scikit-learn if available
            try:
                from sklearn.datasets import fetch_openml
                boston = fetch_openml(name="boston", version=1, as_frame=True)
                df = boston.frame
                logger.info("Loaded Boston Housing dataset from scikit-learn")
            except:
                # If scikit-learn fails, try loading from a URL or local file
                logger.info("Scikit-learn fetch failed, loading Boston Housing dataset from alternative source")
                
                # Local file path
                local_file = os.path.join(self.sample_dir, "boston_housing.csv")
                
                # If local file exists, load from it
                if os.path.exists(local_file):
                    df = pd.read_csv(local_file)
                else:
                    # Try to download from a reliable source
                    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
                    logger.info(f"Downloading Boston Housing dataset to {local_file}")
                    
                    try:
                        # Download and save locally
                        urllib.request.urlretrieve(url, local_file)
                        df = pd.read_csv(local_file)
                    except:
                        # Direct URL loading if download fails
                        df, _ = self.load_from_url(url)
            
            # Add metadata
            metadata = {
                "source": "sample",
                "name": "Boston Housing Dataset",
                "description": "Housing values in suburbs of Boston",
                "citation": "Harrison Jr, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. Journal of environmental economics and management, 5(1), 81-102.",
                "variables": {
                    "CRIM": {"description": "Per capita crime rate by town"},
                    "ZN": {"description": "Proportion of residential land zoned for lots over 25,000 sq.ft"},
                    "INDUS": {"description": "Proportion of non-retail business acres per town"},
                    "CHAS": {"description": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)"},
                    "NOX": {"description": "Nitric oxides concentration (parts per 10 million)"},
                    "RM": {"description": "Average number of rooms per dwelling"},
                    "AGE": {"description": "Proportion of owner-occupied units built prior to 1940"},
                    "DIS": {"description": "Weighted distances to Boston employment centers"},
                    "RAD": {"description": "Index of accessibility to radial highways"},
                    "TAX": {"description": "Full-value property-tax rate per $10,000"},
                    "PTRATIO": {"description": "Pupil-teacher ratio by town"},
                    "B": {"description": "1000(Bk - 0.63)² where Bk is the proportion of black residents"},
                    "LSTAT": {"description": "Percentage of lower status of the population"},
                    "MEDV": {"description": "Median value of owner-occupied homes in $1000s"}
                },
                "notes": "This dataset has been removed from scikit-learn due to ethical concerns about the 'B' variable. Use with caution.",
                "target_variable": "MEDV",
                "ground_truth_available": False,
                "domain": "Real Estate/Economics",
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading Boston Housing dataset: {str(e)}")
            
            # Fallback data
            data = np.random.randn(100, 14)
            columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
            df = pd.DataFrame(data, columns=columns)
            
            metadata = {
                "source": "sample",
                "name": "Boston Housing Dataset (Simulated)",
                "description": "Simulated data with similar structure to Boston Housing dataset (error loading original)",
                "variables": {col: {"description": f"Simulated {col}"} for col in columns},
                "ground_truth_available": False,
                "error": str(e),
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
    
    def _load_airfoil(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load NASA Airfoil Self-Noise dataset"""
        try:
            # URL for the dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
            
            # Local file path
            local_file = os.path.join(self.sample_dir, "airfoil.dat")
            
            # Column names
            columns = [
                "Frequency", 
                "AngleOfAttack", 
                "ChordLength", 
                "FreeStreamVelocity", 
                "SuctionSideDisplacement", 
                "ScaledSound"
            ]
            
            # If local file exists, load from it
            if os.path.exists(local_file):
                df = pd.read_csv(local_file, sep='\t', header=None, names=columns)
            else:
                # Try to download and save
                try:
                    logger.info(f"Downloading Airfoil dataset to {local_file}")
                    urllib.request.urlretrieve(url, local_file)
                    df = pd.read_csv(local_file, sep='\t', header=None, names=columns)
                except:
                    # Direct URL loading if download fails
                    try:
                        response = urllib.request.urlopen(url)
                        data = response.read().decode('utf-8')
                        df = pd.read_csv(io.StringIO(data), sep='\t', header=None, names=columns)
                    except:
                        # If both methods fail, create a custom request with specific headers
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        req = urllib.request.Request(url, headers=headers)
                        response = urllib.request.urlopen(req)
                        data = response.read().decode('utf-8')
                        df = pd.read_csv(io.StringIO(data), sep='\t', header=None, names=columns)
            
            # Add metadata
            metadata = {
                "source": "sample",
                "name": "NASA Airfoil Self-Noise Dataset",
                "description": "NASA data set, obtained from a series of aerodynamic and acoustic tests",
                "citation": "Brooks, T. F., Pope, D. S., & Marcolini, M. A. (1989). Airfoil self-noise and prediction (Vol. 1218). National Aeronautics and Space Administration, Office of Management.",
                "variables": {
                    "Frequency": {"description": "Frequency, in Hertz"},
                    "AngleOfAttack": {"description": "Angle of attack, in degrees"},
                    "ChordLength": {"description": "Chord length, in meters"},
                    "FreeStreamVelocity": {"description": "Free-stream velocity, in meters per second"},
                    "SuctionSideDisplacement": {"description": "Suction side displacement thickness, in meters"},
                    "ScaledSound": {"description": "Scaled sound pressure level, in decibels"}
                },
                "target_variable": "ScaledSound",
                "ground_truth_available": False,
                "domain": "Engineering/Aerodynamics",
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading Airfoil dataset: {str(e)}")
            
            # Fallback data
            data = np.random.randn(100, 6)
            columns = ["Frequency", "AngleOfAttack", "ChordLength", "FreeStreamVelocity", "SuctionSideDisplacement", "ScaledSound"]
            df = pd.DataFrame(data, columns=columns)
            
            metadata = {
                "source": "sample",
                "name": "NASA Airfoil Self-Noise Dataset (Simulated)",
                "description": "Simulated data with similar structure to Airfoil dataset (error loading original)",
                "variables": {col: {"description": f"Simulated {col}"} for col in columns},
                "ground_truth_available": False,
                "error": str(e),
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata

    def _load_galton(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load Galton height dataset"""
        try:
            # Try to load from a file if available
            local_file = os.path.join(self.sample_dir, "galton_heights.csv")
            
            if os.path.exists(local_file):
                df = pd.read_csv(local_file)
            else:
                # Try to download from a URL
                url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/HistData/GaltonFamilies.csv"
                
                try:
                    logger.info(f"Downloading Galton dataset to {local_file}")
                    urllib.request.urlretrieve(url, local_file)
                    df = pd.read_csv(local_file)
                except:
                    # Alternative method - try loading from embedded data
                    try:
                        embedded_path = os.path.join("core", "data", "local_datasets", "Galton_processed.txt")
                        if os.path.exists(embedded_path):
                            with open(embedded_path, 'r') as file:
                                galton_data = file.read()
                            df = pd.read_csv(io.StringIO(galton_data), sep='\t')
                        else:
                            # Direct URL loading if all else fails
                            df, _ = self.load_from_url(url)
                    except:
                        # Direct URL loading if all methods fail
                        df, _ = self.load_from_url(url)
            
            # Preprocess the dataset for causal discovery algorithms
            # Convert 'family' column to numeric by encoding as categorical
            if 'family' in df.columns:
                df['family'] = pd.Categorical(df['family']).codes
            
            # Ensure all columns are numeric
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        df[col] = pd.Categorical(df[col]).codes
            
            # Add metadata
            metadata = {
                "source": "sample",
                "name": "Galton Height Data",
                "description": "Francis Galton's dataset on heights of parents and their children",
                "citation": "Galton, F. (1886). Regression towards mediocrity in hereditary stature. Journal of the Anthropological Institute of Great Britain and Ireland, 15, 246-263.",
                "variables": {
                    "family": {"description": "Family identifier (categorical, encoded as numeric)"},
                    "father": {"description": "Father's height in inches"},
                    "mother": {"description": "Mother's height in inches"},
                    "Gender": {"description": "Child's gender (0 = male, 1 = female)"},
                    "Height": {"description": "Child's height in inches"}
                },
                "preprocessing": "Family ID encoded as categorical numeric to ensure compatibility with causal discovery algorithms",
                "ground_truth_available": True,
                "ground_truth_edges": [
                    ("father", "Height"), 
                    ("mother", "Height"),
                    ("Gender", "Height")
                ],
                "domain": "Genetics/Heredity",
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading Galton height dataset: {str(e)}")
            
            # Create fallback data
            data = np.random.randn(100, 5)
            columns = ["family", "father", "mother", "Gender", "Height"]
            df = pd.DataFrame(data, columns=columns)
            
            metadata = {
                "source": "sample",
                "name": "Galton Height Data (Simulated)",
                "description": "Simulated data with similar structure to Galton's height dataset (error loading original)",
                "variables": {col: {"description": f"Simulated {col}"} for col in columns},
                "ground_truth_available": False,
                "error": str(e),
                "shape": df.shape,
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
            
            return df, metadata