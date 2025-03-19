# core/algorithms/executor.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import networkx as nx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmExecutor:
    """
    Executes causal discovery algorithms from causal-learn library.
    """
    
    def __init__(self):
        pass
        
    def execute_algorithm(self, algorithm_id: str, data: pd.DataFrame, 
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a specific causal discovery algorithm
        
        Args:
            algorithm_id: Identifier for the algorithm to run
            data: DataFrame containing the data
            params: Optional parameters for the algorithm
            
        Returns:
            Dictionary containing execution results including causal graph
        """
        # Convert to numpy array for causal-learn
        data_np = data.values
        
        # Default parameters if none provided
        if params is None:
            params = {}
        
        # Execute the selected algorithm
        try:
            if algorithm_id.startswith("pc_"):
                return self._execute_pc(algorithm_id, data_np, params)
            elif algorithm_id.startswith("fci_"):
                return self._execute_fci(algorithm_id, data_np, params)
            elif algorithm_id == "cdnod":
                return self._execute_cdnod(data_np, params)
            elif algorithm_id.startswith("ges_"):
                return self._execute_ges(algorithm_id, data_np, params)
            elif algorithm_id == "grasp":
                return self._execute_grasp(data_np, params)
            elif algorithm_id == "boss":
                return self._execute_boss(data_np, params)
            elif algorithm_id.startswith("exact_"):
                return self._execute_exact_search(algorithm_id, data_np, params)
            elif algorithm_id.startswith("lingam_"):
                return self._execute_lingam(algorithm_id, data_np, params)
            elif algorithm_id == "anm":
                return self._execute_anm(data_np, params)
            elif algorithm_id == "pnl":
                return self._execute_pnl(data_np, params)
            elif algorithm_id == "gin":
                return self._execute_gin(data_np, params)
            elif algorithm_id.startswith("granger_"):
                return self._execute_granger(algorithm_id, data_np, params)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_id}")
        except Exception as e:
            logger.error(f"Error executing algorithm {algorithm_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "algorithm_id": algorithm_id
            }
    
    def _execute_pc(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PC algorithm variants"""
        from causallearn.search.ConstraintBased.PC import pc
        
        # Default parameters
        alpha = params.get("alpha", 0.05)
        stable = params.get("stable", True)
        uc_rule = params.get("uc_rule", 0)
        uc_priority = params.get("uc_priority", 2)
        mvpc = params.get("mvpc", False)
        verbose = params.get("verbose", False)
        
        # Select independence test based on algorithm variant
        if algorithm_id == "pc_fisherz":
            indep_test = "fisherz"
        elif algorithm_id == "pc_chisq":
            indep_test = "chisq"
        elif algorithm_id == "pc_gsq":
            indep_test = "gsq"
        elif algorithm_id == "pc_kci":
            indep_test = "kci"
        else:
            raise ValueError(f"Unknown PC variant: {algorithm_id}")
        
        # Execute PC algorithm
        cg = pc(data, alpha=alpha, indep_test=indep_test, stable=stable, 
                uc_rule=uc_rule, uc_priority=uc_priority, mvpc=mvpc, 
                verbose=verbose)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(cg.G.graph)
        
        return {
            "status": "success",
            "algorithm_id": algorithm_id,
            "graph": G,
            "causal_learn_result": cg,
            "params": {
                "alpha": alpha,
                "indep_test": indep_test,
                "stable": stable,
                "uc_rule": uc_rule,
                "uc_priority": uc_priority,
                "mvpc": mvpc
            }
        }
    
    def _execute_fci(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FCI algorithm variants"""
        from causallearn.search.ConstraintBased.FCI import fci
        
        # Default parameters
        alpha = params.get("alpha", 0.05)
        depth = params.get("depth", -1)  # -1 for unlimited
        max_path_length = params.get("max_path_length", -1)  # -1 for unlimited
        verbose = params.get("verbose", False)
        
        # Select independence test based on algorithm variant
        if algorithm_id == "fci_fisherz":
            indep_test = "fisherz"
        elif algorithm_id == "fci_chisq":
            indep_test = "chisq"
        elif algorithm_id == "fci_kci":
            indep_test = "kci"
        else:
            raise ValueError(f"Unknown FCI variant: {algorithm_id}")
        
        # Execute FCI algorithm
        g, edges = fci(data, independence_test_method=indep_test, alpha=alpha, 
                        depth=depth, max_path_length=max_path_length,
                        verbose=verbose)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(g.graph)
        
        return {
            "status": "success",
            "algorithm_id": algorithm_id,
            "graph": G,
            "causal_learn_result": {
                "graph": g,
                "edges": edges
            },
            "params": {
                "alpha": alpha,
                "indep_test": indep_test,
                "depth": depth,
                "max_path_length": max_path_length
            }
        }
    
    def _execute_cdnod(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CD-NOD algorithm"""
        from causallearn.search.ConstraintBased.CDNOD import cdnod
        
        # Required parameter: time/domain index
        c_indx = params.get("c_indx")
        if c_indx is None:
            raise ValueError("c_indx parameter is required for CD-NOD")
        
        # Default parameters
        alpha = params.get("alpha", 0.05)
        indep_test = params.get("indep_test", "fisherz")
        stable = params.get("stable", True)
        uc_rule = params.get("uc_rule", 0)
        uc_priority = params.get("uc_priority", 2)
        mvcdnod = params.get("mvcdnod", False)
        verbose = params.get("verbose", False)
        
        # Execute CD-NOD algorithm
        cg = cdnod(data, c_indx, alpha=alpha, indep_test=indep_test, 
                   stable=stable, uc_rule=uc_rule, uc_priority=uc_priority, 
                   mvcdnod=mvcdnod, verbose=verbose)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(cg.G.graph)
        
        return {
            "status": "success",
            "algorithm_id": "cdnod",
            "graph": G,
            "causal_learn_result": cg,
            "params": {
                "c_indx": c_indx,
                "alpha": alpha,
                "indep_test": indep_test,
                "stable": stable,
                "uc_rule": uc_rule,
                "uc_priority": uc_priority,
                "mvcdnod": mvcdnod
            }
        }
    
    def _execute_ges(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GES algorithm variants"""
        from causallearn.search.ScoreBased.GES import ges
        
        # Default parameters
        maxP = params.get("maxP", None)  # Maximum number of parents
        parameters = {}
        
        # Set score function based on algorithm variant
        if algorithm_id == "ges_bic":
            score_func = "local_score_BIC"
        elif algorithm_id == "ges_bdeu":
            score_func = "local_score_BDeu"
            # Additional parameters for BDeu
            parameters = {
                "sample_prior": params.get("sample_prior", 1),
                "structure_prior": params.get("structure_prior", 1),
                "r_i_map": params.get("r_i_map", None)
            }
        elif algorithm_id == "ges_cv":
            score_func = "local_score_CV_general"
            # Additional parameters for CV scoring
            parameters = {
                "kfold": params.get("kfold", 10),
                "lambda": params.get("lambda", 0.01)
            }
            if "dlabel" in params:
                parameters["dlabel"] = params["dlabel"]
        else:
            raise ValueError(f"Unknown GES variant: {algorithm_id}")
        
        # Execute GES algorithm
        record = ges(data, score_func=score_func, maxP=maxP, parameters=parameters)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(record['G'].graph)
        
        return {
            "status": "success",
            "algorithm_id": algorithm_id,
            "graph": G,
            "causal_learn_result": record,
            "params": {
                "score_func": score_func,
                "maxP": maxP,
                "parameters": parameters
            }
        }
    
    def _execute_grasp(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GRaSP algorithm"""
        from causallearn.search.PermutationBased.GRaSP import grasp
        
        # Default parameters
        score_func = params.get("score_func", "local_score_BIC")
        depth = params.get("depth", None)
        
        # Parameters for score functions
        parameters = {}
        if score_func in ["local_score_CV_general", "local_score_CV_multi"]:
            parameters = {
                "kfold": params.get("kfold", 10),
                "lambda": params.get("lambda", 0.01)
            }
            if "dlabel" in params:
                parameters["dlabel"] = params["dlabel"]
        elif score_func == "local_score_BDeu":
            parameters = {
                "sample_prior": params.get("sample_prior", 1),
                "structure_prior": params.get("structure_prior", 1),
                "r_i_map": params.get("r_i_map", None)
            }
        
        # Execute GRaSP algorithm
        g = grasp(data, score_func=score_func, depth=depth, parameters=parameters)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(g.graph)
        
        return {
            "status": "success",
            "algorithm_id": "grasp",
            "graph": G,
            "causal_learn_result": g,
            "params": {
                "score_func": score_func,
                "depth": depth,
                "parameters": parameters
            }
        }
    
    def _execute_boss(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute BOSS algorithm"""
        from causallearn.search.PermutationBased.BOSS import boss
        
        # Default parameters
        score_func = params.get("score_func", "local_score_BIC")
        
        # Parameters for score functions
        parameters = {}
        if score_func in ["local_score_CV_general", "local_score_CV_multi"]:
            parameters = {
                "kfold": params.get("kfold", 10),
                "lambda": params.get("lambda", 0.01)
            }
            if "dlabel" in params:
                parameters["dlabel"] = params["dlabel"]
        elif score_func == "local_score_BDeu":
            parameters = {
                "sample_prior": params.get("sample_prior", 1),
                "structure_prior": params.get("structure_prior", 1),
                "r_i_map": params.get("r_i_map", None)
            }
        
        # Execute BOSS algorithm
        g = boss(data, score_func=score_func, parameters=parameters)
        
        # Convert to networkx for visualization and analysis
        G = self._convert_to_networkx(g.graph)
        
        return {
            "status": "success",
            "algorithm_id": "boss",
            "graph": G,
            "causal_learn_result": g,
            "params": {
                "score_func": score_func,
                "parameters": parameters
            }
        }
    
    def _execute_exact_search(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Exact Search algorithms"""
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
        
        # Check if valid algorithm variant
        if algorithm_id not in ["exact_dp", "exact_astar"]:
            raise ValueError(f"Unknown Exact Search variant: {algorithm_id}")
        
        # Default parameters
        super_graph = params.get("super_graph", None)
        use_path_extension = params.get("use_path_extension", True)
        use_k_cycle_heuristic = params.get("use_k_cycle_heuristic", False)
        k = params.get("k", 3)
        verbose = params.get("verbose", False)
        max_parents = params.get("max_parents", None)
        
        # Set search method
        search_method = "dp" if algorithm_id == "exact_dp" else "astar"
        
        # Execute Exact Search algorithm
        dag_est, search_stats = bic_exact_search(
            data, super_graph=super_graph, search_method=search_method,
            use_path_extension=use_path_extension, 
            use_k_cycle_heuristic=use_k_cycle_heuristic,
            k=k, verbose=verbose, max_parents=max_parents
        )
        
        # Convert to networkx for visualization and analysis
        G = nx.DiGraph()
        n_nodes = dag_est.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dag_est[i, j] == 1:
                    G.add_edge(i, j)
        
        return {
            "status": "success",
            "algorithm_id": algorithm_id,
            "graph": G,
            "causal_learn_result": {
                "dag_est": dag_est,
                "search_stats": search_stats
            },
            "params": {
                "search_method": search_method,
                "use_path_extension": use_path_extension,
                "use_k_cycle_heuristic": use_k_cycle_heuristic,
                "k": k,
                "max_parents": max_parents
            }
        }
    
    def _execute_lingam(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LiNGAM-based algorithms"""
        from causallearn.search.FCMBased import lingam
        
        if algorithm_id == "lingam_ica":
            # Parameters for ICA-based LiNGAM
            random_state = params.get("random_state", None)
            max_iter = params.get("max_iter", 1000)
            
            # Execute ICA-based LiNGAM
            model = lingam.ICALiNGAM(random_state=random_state, max_iter=max_iter)
            model.fit(data)
            
            # Create network graph from adjacency matrix
            G = nx.DiGraph()
            n_nodes = model.adjacency_matrix_.shape[0]
            G.add_nodes_from(range(n_nodes))
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if model.adjacency_matrix_[i, j] != 0:
                        G.add_edge(j, i, weight=model.adjacency_matrix_[i, j])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "adjacency_matrix": model.adjacency_matrix_,
                    "causal_order": model.causal_order_
                },
                "params": {
                    "random_state": random_state,
                    "max_iter": max_iter
                }
            }
            
        elif algorithm_id == "lingam_direct":
            # Parameters for DirectLiNGAM
            random_state = params.get("random_state", None)
            prior_knowledge = params.get("prior_knowledge", None)
            apply_prior_knowledge_softly = params.get("apply_prior_knowledge_softly", False)
            measure = params.get("measure", "pwling")
            
            # Execute DirectLiNGAM
            model = lingam.DirectLiNGAM(random_state=random_state, 
                                         prior_knowledge=prior_knowledge,
                                         apply_prior_knowledge_softly=apply_prior_knowledge_softly,
                                         measure=measure)
            model.fit(data)
            
            # Create network graph from adjacency matrix
            G = nx.DiGraph()
            n_nodes = model.adjacency_matrix_.shape[0]
            G.add_nodes_from(range(n_nodes))
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if model.adjacency_matrix_[i, j] != 0:
                        G.add_edge(j, i, weight=model.adjacency_matrix_[i, j])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "adjacency_matrix": model.adjacency_matrix_,
                    "causal_order": model.causal_order_
                },
                "params": {
                    "random_state": random_state,
                    "measure": measure
                }
            }
            
        elif algorithm_id == "lingam_var":
            # Parameters for VAR-LiNGAM
            lags = params.get("lags", 1)
            criterion = params.get("criterion", "bic")
            prune = params.get("prune", False)
            ar_coefs = params.get("ar_coefs", None)
            lingam_model = params.get("lingam_model", None)
            random_state = params.get("random_state", None)
            
            # Execute VAR-LiNGAM
            model = lingam.VARLiNGAM(lags=lags, criterion=criterion, prune=prune,
                                      ar_coefs=ar_coefs, lingam_model=lingam_model,
                                      random_state=random_state)
            model.fit(data)
            
            # Create network graphs for lag 0
            G = nx.DiGraph()
            n_nodes = model.adjacency_matrices_[0].shape[0]
            G.add_nodes_from(range(n_nodes))
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if model.adjacency_matrices_[0][i, j] != 0:
                        G.add_edge(j, i, weight=model.adjacency_matrices_[0][i, j])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "adjacency_matrices": model.adjacency_matrices_,
                    "causal_order": model.causal_order_,
                    "residuals": model.residuals_
                },
                "params": {
                    "lags": lags,
                    "criterion": criterion,
                    "prune": prune
                }
            }
            
        elif algorithm_id == "lingam_rcd":
            # Parameters for RCD
            max_explanatory_num = params.get("max_explanatory_num", 2)
            cor_alpha = params.get("cor_alpha", 0.01)
            ind_alpha = params.get("ind_alpha", 0.01)
            shapiro_alpha = params.get("shapiro_alpha", 0.01)
            MLHSICR = params.get("MLHSICR", False)
            bw_method = params.get("bw_method", "mdbs")
            
            # Execute RCD
            model = lingam.RCD(max_explanatory_num=max_explanatory_num,
                               cor_alpha=cor_alpha, ind_alpha=ind_alpha,
                               shapiro_alpha=shapiro_alpha, MLHSICR=MLHSICR,
                               bw_method=bw_method)
            model.fit(data)
            
            # Create network graph from adjacency matrix
            G = nx.DiGraph()
            n_nodes = model.adjacency_matrix_.shape[0]
            G.add_nodes_from(range(n_nodes))
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if model.adjacency_matrix_[i, j] != 0:
                        G.add_edge(j, i, weight=model.adjacency_matrix_[i, j])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "adjacency_matrix": model.adjacency_matrix_,
                    "ancestors_list": model.ancestors_list_
                },
                "params": {
                    "max_explanatory_num": max_explanatory_num,
                    "cor_alpha": cor_alpha,
                    "ind_alpha": ind_alpha,
                    "shapiro_alpha": shapiro_alpha,
                    "MLHSICR": MLHSICR,
                    "bw_method": bw_method
                }
            }
            
        elif algorithm_id == "lingam_camuv":
            # Parameters for CAM-UV
            alpha = params.get("alpha", 0.01)
            num_explanatory_vals = params.get("num_explanatory_vals", 2)
            
            # Execute CAM-UV
            from causallearn.search.FCMBased.lingam import CAMUV
            P, U = CAMUV.execute(data, alpha=alpha, 
                                  num_explanatory_vals=num_explanatory_vals)
            
            # Create network graph from parent lists
            G = nx.DiGraph()
            n_nodes = len(P)
            G.add_nodes_from(range(n_nodes))
            
            for i, parents in enumerate(P):
                for parent in parents:
                    G.add_edge(parent, i)
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "parents": P,
                    "undetermined_pairs": U
                },
                "params": {
                    "alpha": alpha,
                    "num_explanatory_vals": num_explanatory_vals
                }
            }
        
        else:
            raise ValueError(f"Unknown LiNGAM variant: {algorithm_id}")
    
    def _execute_anm(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Additive Noise Model (ANM) algorithm"""
        from causallearn.search.FCMBased.ANM.ANM import ANM
        
        # ANM is designed for pairwise causal discovery
        if data.shape[1] != 2:
            raise ValueError("ANM is designed for pairwise causal discovery (2 variables only)")
        
        # Execute ANM
        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(data[:, 0], data[:, 1])
        
        # Create network graph based on p-values
        G = nx.DiGraph()
        G.add_nodes_from([0, 1])
        
        # Determine direction based on p-values
        if p_value_forward > p_value_backward:
            # X -> Y (0 -> 1)
            G.add_edge(0, 1)
            direction = "0->1"
        else:
            # Y -> X (1 -> 0)
            G.add_edge(1, 0)
            direction = "1->0"
        
        return {
            "status": "success",
            "algorithm_id": "anm",
            "graph": G,
            "causal_learn_result": {
                "p_value_forward": p_value_forward,
                "p_value_backward": p_value_backward,
                "direction": direction
            },
            "params": {}
        }
    
    def _execute_pnl(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Post-Nonlinear (PNL) causal model algorithm"""
        from causallearn.search.FCMBased.PNL.PNL import PNL
        
        # PNL is designed for pairwise causal discovery
        if data.shape[1] != 2:
            raise ValueError("PNL is designed for pairwise causal discovery (2 variables only)")
        
        # Execute PNL
        pnl = PNL()
        p_value_forward, p_value_backward = pnl.cause_or_effect(data[:, 0], data[:, 1])
        
        # Create network graph based on p-values
        G = nx.DiGraph()
        G.add_nodes_from([0, 1])
        
        # Determine direction based on p-values
        if p_value_forward > p_value_backward:
            # X -> Y (0 -> 1)
            G.add_edge(0, 1)
            direction = "0->1"
        else:
            # Y -> X (1 -> 0)
            G.add_edge(1, 0)
            direction = "1->0"
        
        return {
            "status": "success",
            "algorithm_id": "pnl",
            "graph": G,
            "causal_learn_result": {
                "p_value_forward": p_value_forward,
                "p_value_backward": p_value_backward,
                "direction": direction
            },
            "params": {}
        }
    
    def _execute_gin(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Generalized Independent Noise (GIN) algorithm"""
        from causallearn.search.HiddenCausal.GIN.GIN import GIN
        
        # Execute GIN
        G, K = GIN(data)
        
        # Convert to networkx for visualization and analysis
        G_nx = self._convert_to_networkx(G.graph)
        
        return {
            "status": "success",
            "algorithm_id": "gin",
            "graph": G_nx,
            "causal_learn_result": {
                "graph": G,
                "causal_order": K
            },
            "params": {}
        }
    
    def _execute_granger(self, algorithm_id: str, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Granger causality algorithms"""
        from causallearn.search.Granger.Granger import Granger
        granger = Granger()
        
        if algorithm_id == "granger_test":
            # Granger test is designed for bivariate time series
            if data.shape[1] != 2:
                raise ValueError("granger_test is designed for bivariate time series (2 variables only)")
            
            # Execute Granger test
            p_value_matrix = granger.granger_test_2d(data)
            
            # Create network graph based on p-values
            G = nx.DiGraph()
            G.add_nodes_from([0, 1])
            
            # Add edges based on significance
            alpha = params.get("alpha", 0.05)
            
            if p_value_matrix[0, 1] < alpha:  # 0 Granger-causes 1
                G.add_edge(0, 1, p_value=p_value_matrix[0, 1])
            
            if p_value_matrix[1, 0] < alpha:  # 1 Granger-causes 0
                G.add_edge(1, 0, p_value=p_value_matrix[1, 0])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "p_value_matrix": p_value_matrix
                },
                "params": {
                    "alpha": alpha
                }
            }
            
        elif algorithm_id == "granger_lasso":
            # Execute Granger Lasso
            coef = granger.granger_lasso(data)
            
            # Create network graph from coefficient matrix
            G = nx.DiGraph()
            n_nodes = coef.shape[1]
            G.add_nodes_from(range(n_nodes))
            
            # Add edges where coefficients are non-zero
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and coef[i, j] != 0:
                        G.add_edge(j, i, weight=coef[i, j])
            
            return {
                "status": "success",
                "algorithm_id": algorithm_id,
                "graph": G,
                "causal_learn_result": {
                    "coef": coef
                },
                "params": {}
            }
        
        else:
            raise ValueError(f"Unknown Granger causality variant: {algorithm_id}")
    
    def _convert_to_networkx(self, graph: np.ndarray) -> nx.DiGraph:
        """Convert causal-learn graph matrix to networkx DiGraph"""
        G = nx.DiGraph()
        n_nodes = graph.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                # i --> j
                if graph[j, i] == 1 and graph[i, j] == -1:
                    G.add_edge(i, j)
                # i <-> j (bidirected edge)
                elif graph[j, i] == 1 and graph[i, j] == 1:
                    G.add_edge(i, j, bidirected=True)
                    G.add_edge(j, i, bidirected=True)
        
        return G
    
    # Add to core/algorithms/executor.py

def _execute_anm(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Additive Noise Model (ANM) algorithm"""
    from core.algorithms.nonlinear_models import AdditiveNoiseModel
    
    # Check if data has exactly 2 variables (pairwise causal discovery)
    if data.shape[1] != 2:
        raise ValueError("ANM is designed for pairwise causal discovery (2 variables only)")
    
    # Execute ANM
    anm = AdditiveNoiseModel(regression_method=params.get("regression_method", "gp"))
    direction_result = anm.test_direction(data[:, 0], data[:, 1])
    
    # Create network graph based on direction result
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    
    # Determine direction based on result
    if direction_result["direction"] == "0->1":
        # X -> Y (0 -> 1)
        G.add_edge(0, 1, weight=direction_result["confidence"])
        direction = "0->1"
    else:
        # Y -> X (1 -> 0)
        G.add_edge(1, 0, weight=direction_result["confidence"])
        direction = "1->0"
    
    return {
        "status": "success",
        "algorithm_id": "anm",
        "graph": G,
        "causal_learn_result": {
            "forward_score": direction_result.get("forward_score"),
            "backward_score": direction_result.get("backward_score"),
            "direction": direction,
            "confidence": direction_result.get("confidence")
        },
        "params": params
    }

def _execute_pnl(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Post-Nonlinear (PNL) causal model algorithm"""
    from core.algorithms.nonlinear_models import PostNonlinearModel
    
    # Check if data has exactly 2 variables (pairwise causal discovery)
    if data.shape[1] != 2:
        raise ValueError("PNL is designed for pairwise causal discovery (2 variables only)")
    
    # Execute PNL
    pnl = PostNonlinearModel(
        f1_degree=params.get("f1_degree", 3),
        f2_degree=params.get("f2_degree", 3),
        independence_test=params.get("independence_test", "hsic")
    )
    direction_result = pnl.test_direction(data[:, 0], data[:, 1])
    
    # Create network graph based on direction result
    G = nx.DiGraph()
    G.add_nodes_from([0, 1])
    
    # Determine direction based on result
    if direction_result["direction"] == "0->1":
        # X -> Y (0 -> 1)
        G.add_edge(0, 1, weight=direction_result["confidence"])
        direction = "0->1"
    else:
        # Y -> X (1 -> 0)
        G.add_edge(1, 0, weight=direction_result["confidence"])
        direction = "1->0"
    
    return {
        "status": "success",
        "algorithm_id": "pnl",
        "graph": G,
        "causal_learn_result": {
            "direction": direction,
            "confidence": direction_result.get("confidence")
        },
        "params": params
    }

def _execute_kernel_ci(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute kernel-based conditional independence tests"""
    from core.algorithms.kernel_methods import KernelCausalDiscovery
    
    # Parameters for kernel methods
    kernel_type = params.get("kernel_type", "rbf")
    kernel_params = params.get("kernel_params", {})
    alpha = params.get("alpha", 0.05)
    
    # Initialize kernel causal discovery
    kcd = KernelCausalDiscovery(kernel_type=kernel_type, kernel_params=kernel_params)
    
    # Select variables to test (if specified)
    x_idx = params.get("x_idx", 0)
    y_idx = params.get("y_idx", 1)
    z_idx = params.get("z_idx", None)
    
    # Extract variables
    x = data[:, x_idx]
    y = data[:, y_idx]
    z = data[:, z_idx] if z_idx is not None else None
    
    # Perform independence test
    independent, p_value = kcd.kernel_pc_independence_test(x, y, z, alpha)
    
    # Create simple graph representation of result
    G = nx.DiGraph()
    G.add_nodes_from([x_idx, y_idx])
    
    # Only add edge if not independent
    if not independent:
        G.add_edge(x_idx, y_idx, weight=1.0 - p_value, p_value=p_value)
    
    return {
        "status": "success",
        "algorithm_id": "kernel_ci",
        "graph": G,
        "causal_learn_result": {
            "independent": independent,
            "p_value": p_value
        },
        "params": params
    }

def _execute_nonstationary(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute causal discovery for nonstationary/heterogeneous data"""
    from core.algorithms.nonstationarity import NonStationaryCausalDiscovery
    
    # Get required parameters
    time_index = params.get("time_index")
    
    if time_index is None:
        raise ValueError("time_index parameter is required for nonstationary causal discovery")
    
    # Convert to numpy array if needed
    if isinstance(time_index, list):
        time_index = np.array(time_index)
    
    # Initialize nonstationary causal discovery
    nscd = NonStationaryCausalDiscovery()
    
    # Perform causal discovery
    G, additional_info = nscd.causal_discovery_nonstationary(
        data, 
        time_index, 
        alpha=params.get("alpha", 0.05)
    )
    
    return {
        "status": "success",
        "algorithm_id": "nonstationary",
        "graph": G,
        "causal_learn_result": additional_info,
        "params": params
    }

def _execute_timeseries(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute time series causal discovery"""
    from core.algorithms.timeseries import TimeSeriesCausalDiscovery
    
    # Get parameters
    method = params.get("method", "grangervar")
    lags = params.get("lags", None)
    var_names = params.get("var_names", None)
    
    # Initialize time series causal discovery
    tscd = TimeSeriesCausalDiscovery(method=method)
    
    # Perform causal discovery
    G, additional_info = tscd.discover_causal_graph(
        data,
        lags=lags,
        var_names=var_names,
        alpha=params.get("alpha", 0.05)
    )
    
    # If requested, also compute instantaneous effects
    if params.get("detect_instantaneous", False):
        G_inst = tscd.detect_instantaneous_effects(data)
        
        # Combine graphs if requested
        if params.get("combine_graphs", False):
            G = tscd.combine_temporal_instantaneous_graph(G, G_inst)
            additional_info["instantaneous_included"] = True
        else:
            additional_info["instantaneous_graph"] = G_inst
    
    return {
        "status": "success",
        "algorithm_id": f"timeseries_{method}",
        "graph": G,
        "causal_learn_result": additional_info,
        "params": params
    }

def _execute_var_lingam(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute VAR-LiNGAM for time series data"""
    from core.algorithms.timeseries import VARLiNGAM
    
    # Parameters
    lags = params.get("lags", 1)
    var_names = params.get("var_names", None)
    
    # Initialize and fit VAR-LiNGAM
    model = VARLiNGAM(lags=lags)
    result = model.fit(data)
    
    if result["success"]:
        # Convert to NetworkX graph
        G = model.to_networkx_graph(var_names=var_names)
        
        return {
            "status": "success",
            "algorithm_id": "var_lingam",
            "graph": G,
            "causal_learn_result": result,
            "params": params
        }
    else:
        return {
            "status": "error",
            "algorithm_id": "var_lingam",
            "error": result.get("error", "Unknown error fitting VAR-LiNGAM"),
            "params": params
        }
    
    # Modify the execute_algorithm method in core/algorithms/executor.py

def execute_algorithm(self, algorithm_id: str, data: pd.DataFrame, 
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a specific causal discovery algorithm
    
    Args:
        algorithm_id: Identifier for the algorithm to run
        data: DataFrame containing the data
        params: Optional parameters for the algorithm
        
    Returns:
        Dictionary containing execution results including causal graph
    """
    # Convert to numpy array for causal-learn
    data_np = data.values
    
    # Default parameters if none provided
    if params is None:
        params = {}
    
    # Execute the selected algorithm
    try:
        # Existing algorithm checks...
        if algorithm_id.startswith("pc_"):
            return self._execute_pc(algorithm_id, data_np, params)
        elif algorithm_id.startswith("fci_"):
            return self._execute_fci(algorithm_id, data_np, params)
        elif algorithm_id == "cdnod":
            return self._execute_cdnod(data_np, params)
        elif algorithm_id.startswith("ges_"):
            return self._execute_ges(algorithm_id, data_np, params)
        # ... other existing algorithms ...
        
        # New algorithms
        elif algorithm_id == "anm":
            return self._execute_anm(data_np, params)
        elif algorithm_id == "pnl":
            return self._execute_pnl(data_np, params)
        elif algorithm_id == "kernel_ci":
            return self._execute_kernel_ci(data_np, params)
        elif algorithm_id == "nonstationary":
            return self._execute_nonstationary(data_np, params)
        elif algorithm_id.startswith("timeseries_"):
            return self._execute_timeseries(data_np, params)
        elif algorithm_id == "var_lingam":
            return self._execute_var_lingam(data_np, params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")
    except Exception as e:
        logger.error(f"Error executing algorithm {algorithm_id}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "algorithm_id": algorithm_id
        }