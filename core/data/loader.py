# core/data/loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads datasets from files or sample datasets.
    """
    
    def __init__(self):
        """Initialize DataLoader with sample datasets"""
        self.sample_datasets = {
            "sachs": self._load_sachs,
            "boston_housing": self._load_boston_housing,
            "airfoil": self._load_airfoil
        }
    
    def load_file(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Load data from a file
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters for loading
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path, **kwargs)
                return df, {"source": "file", "filename": os.path.basename(file_path), "format": "csv"}
            
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
                return df, {"source": "file", "filename": os.path.basename(file_path), "format": "excel"}
            
            elif file_ext == '.json':
                df = pd.read_json(file_path, **kwargs)
                return df, {"source": "file", "filename": os.path.basename(file_path), "format": "json"}
            
            elif file_ext == '.pkl':
                df = pd.read_pickle(file_path, **kwargs)
                return df, {"source": "file", "filename": os.path.basename(file_path), "format": "pickle"}
            
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
                return df, {"source": "file", "filename": os.path.basename(file_path), "format": "parquet"}
            
            elif file_ext == '.sav':
                import joblib
                df = joblib.load(file_path)
                if isinstance(df, pd.DataFrame):
                    return df, {"source": "file", "filename": os.path.basename(file_path), "format": "joblib"}
                else:
                    raise ValueError("Loaded file is not a pandas DataFrame")
            
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
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
            raise
    
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
                "domain": "Biology/Proteomics"
            },
            {
                "id": "boston_housing",
                "name": "Boston Housing Dataset",
                "description": "Housing data with 14 variables and 506 observations.",
                "n_variables": 14,
                "n_samples": 506,
                "domain": "Real Estate/Economics"
            },
            {
                "id": "airfoil",
                "name": "NASA Airfoil Self-Noise",
                "description": "Airfoil self-noise data with 6 variables and 1503 observations.",
                "n_variables": 6,
                "n_samples": 1503,
                "domain": "Engineering/Aerodynamics"
            }
        ]
    
    def _load_sachs(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load Sachs protein signaling dataset"""
        try:
            from causallearn.utils.Dataset import load_dataset
            data, labels = load_dataset("sachs")
            df = pd.DataFrame(data, columns=labels)
            
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
                "ground_truth_available": True
            }
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading Sachs dataset: {str(e)}")
            
            # Fallback data if causallearn's dataset loading fails
            data = np.random.randn(100, 11)
            columns = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "JNK"]
            df = pd.DataFrame(data, columns=columns)
            
            metadata = {
                "source": "sample",
                "name": "Sachs Protein Signaling (Simulated)",
                "description": "Simulated data with similar structure to Sachs dataset (error loading original)",
                "variables": {col: {"description": f"Simulated {col}"} for col in columns},
                "ground_truth_available": False,
                "error": str(e)
            }
            
            return df, metadata
    
    def _load_boston_housing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load Boston Housing dataset"""
        try:
            from sklearn.datasets import fetch_openml
            boston = fetch_openml(name="boston", version=1, as_frame=True)
            df = boston.frame
            
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
                "ground_truth_available": False
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
                "error": str(e)
            }
            
            return df, metadata
    
    def _load_airfoil(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load NASA Airfoil Self-Noise dataset"""
        try:
            # URL for the dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
            
            # Column names
            columns = [
                "Frequency", 
                "AngleOfAttack", 
                "ChordLength", 
                "FreeStreamVelocity", 
                "SuctionSideDisplacement", 
                "ScaledSound"
            ]
            
            # Load data
            df = pd.read_csv(url, sep='\t', header=None, names=columns)
            
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
                "ground_truth_available": False
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
                "error": str(e)
            }
            
            return df, metadata