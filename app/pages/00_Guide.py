import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def app():
    st.set_page_config(
        page_title="LLM-Augmented Causal Discovery Guide",
        page_icon="📚",
        layout="wide"
    )

    # Initialize session state variables if not exist
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'algorithm_results' not in st.session_state:
        st.session_state.algorithm_results = {}
    if 'refined_graph' not in st.session_state:
        st.session_state.refined_graph = None

    st.title("📚 Guide to LLM-Augmented Causal Discovery")
    
    st.markdown("""
    Welcome to the LLM-Augmented Causal Discovery Framework! This guide will help you understand:
    
    1. What causal discovery is and why it matters
    2. How this application works
    3. Choosing the right algorithms for your data
    4. Interpreting results and refining causal graphs
    5. Common questions and troubleshooting
    """)

    tabs = st.tabs([
        "Introduction", 
        "Getting Started", 
        "Algorithms Guide", 
        "Algorithm Selection", 
        "Graph Refinement",
        "Case Studies"
    ])

    # Introduction Tab
    with tabs[0]:
        st.header("Introduction to Causal Discovery")
        
        st.markdown("""
        ### What is Causal Discovery?
        
        **Causal discovery** is the process of identifying cause-and-effect relationships between variables in a dataset. 
        Unlike correlation, which only shows that variables are related, causation indicates that one variable directly 
        influences another.
        
        For example, while there might be a correlation between ice cream sales and drownings, neither causes the other. 
        Instead, a third variable (hot weather) causes both.
        
        ### Why Causal Discovery Matters
        
        Understanding causal relationships allows you to:
        
        - **Make better predictions** about what will happen when you intervene in a system
        - **Identify effective interventions** for achieving desired outcomes
        - **Avoid spurious correlations** that can lead to incorrect conclusions
        - **Understand mechanisms** behind observed phenomena
        
        ### Traditional Challenges
        
        Causal discovery has traditionally been challenging because:
        
        1. It requires specialized statistical knowledge
        2. Different methods make different assumptions
        3. Real-world data rarely satisfies all the assumptions
        4. Interpreting results requires domain expertise
        
        ### Our Solution: LLM-Augmented Causal Discovery
        
        This application combines established causal discovery algorithms with Large Language Models (LLMs) to:
        
        - **Democratize access** to causal discovery for users without specialized statistical training
        - **Automatically select** appropriate algorithms based on data characteristics
        - **Refine causal graphs** using domain knowledge via LLM integration
        - **Provide natural language explanations** of causal relationships
        """)

        st.subheader("How Causal Graphs Work")
        
        st.markdown("""
        A causal graph (or causal Bayesian network) is a directed graph where:
        
        - **Nodes** represent variables in your dataset
        - **Arrows** (→) represent direct causal relationships
        - The absence of an arrow means no direct causal effect

        The graph below shows a simple causal structure where:
        - A causes B
        - A causes C
        - B causes D
        - C causes D
        """)
        
        # Create a simple causal graph visualization
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_nodes_from(['A', 'B', 'C', 'D'])
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        
        pos = {'A': (0, 1), 'B': (1, 1), 'C': (1, 0), 'D': (2, 0.5)}
        
        fig, ax = plt.subplots(figsize=(6, 4))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, 
                arrowsize=20, ax=ax, font_size=14, font_weight='bold')
        
        # Convert matplotlib figure to image for Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        st.image(buf, caption="Example Causal Graph", use_column_width=False)
        
        st.markdown("""
        In causal discovery, our goal is to recover this graph structure from observational data alone.
        Different algorithms make different assumptions and have different strengths, which is why
        our framework helps you select and combine appropriate methods.
        """)

    # Getting Started Tab
    with tabs[1]:
        st.header("Getting Started")
        
        st.markdown("""
        ### Workflow Overview
        
        Working with the LLM-Augmented Causal Discovery Framework follows these steps:
        
        1. **Data Loading**: Upload your dataset or select from sample datasets
        2. **Data Exploration**: Understand your data's characteristics
        3. **Causal Discovery**: Run appropriate algorithms to discover causal relationships
        4. **Graph Refinement**: Use LLM capabilities to refine and improve your causal graph
        5. **Analysis & Explanation**: Get natural language explanations and insights
        
        Let's walk through each step in more detail.
        """)
        
        st.subheader("1. Data Loading")
        st.markdown("""
        On the **Data Loading** page, you can:
        
        - Upload your own CSV or Excel file
        - Choose from sample datasets
        - View basic statistics about your data
        
        **Data format requirements:**
        - Data should be in tabular format with variables as columns
        - Each row should represent an independent observation
        - For time series data, observations should be ordered chronologically
        - Missing values should be marked as NaN or left empty
        """)
        
        st.subheader("2. Data Exploration")
        st.markdown("""
        The **Data Exploration** page helps you understand your data before applying causal discovery:
        
        - View distributions of individual variables
        - Examine correlations between variables
        - Check for missing values and outliers
        - Profile your data to inform algorithm selection
        
        **Why this matters:** Different causal discovery algorithms have different assumptions about 
        data distribution, variable types, and sample size. Understanding your data characteristics 
        helps select appropriate algorithms.
        """)
        
        st.subheader("3. Causal Discovery")
        st.markdown("""
        On the **Causal Discovery** page:
        
        - Get algorithm recommendations based on your data profile
        - Select and run algorithms with customizable parameters
        - View resulting causal graphs and compare results from different algorithms
        - Optionally run an ensemble of methods for more robust results
        
        **Typical workflow:**
        1. Run the recommended primary algorithm first
        2. Examine the resulting graph
        3. Try alternative algorithms or parameter settings
        4. Compare results across multiple methods
        """)
        
        st.subheader("4. Graph Refinement")
        st.markdown("""
        The **Graph Refinement** page leverages LLM capabilities to:
        
        - Validate and refine discovered causal relationships
        - Incorporate domain knowledge into the causal graph
        - Identify potential hidden/latent variables
        - Resolve conflicts between different algorithm results
        
        This step is what makes our framework unique - it combines statistical methods with
        language model capabilities to improve graph quality and incorporate domain expertise.
        """)
        
        st.subheader("5. Analysis & Explanation")
        st.markdown("""
        Finally, the **Analysis & Explanation** page provides:
        
        - Natural language explanations of causal relationships
        - Interactive examination of specific causal pathways
        - Multi-level explanations tailored to your technical background
        - Confidence assessments for discovered relationships
        
        You can save your results, export graphs, and generate reports for further use.
        """)

    # Algorithms Guide Tab
    with tabs[2]:
        st.header("Algorithms Guide")
        
        st.markdown("""
        This application integrates algorithms from the causal-learn library, organized into several categories.
        Each algorithm has different assumptions, strengths, and limitations.
        """)
        
        algorithm_category = st.selectbox(
            "Select algorithm category to learn about:",
            ["Constraint-based Methods", "Score-based Methods", "FCM-based Methods", 
             "Hidden Causal Methods", "Granger Causality"]
        )
        
        if algorithm_category == "Constraint-based Methods":
            st.subheader("Constraint-based Methods")
            st.markdown("""
            Constraint-based methods use conditional independence tests to determine causal relationships. They:
            
            - Don't make strong assumptions about functional forms
            - Work well with both discrete and continuous data
            - Can identify the presence of hidden confounders (with certain algorithms)
            - May be sensitive to errors in independence tests
            """)
            
            st.markdown("#### PC Algorithm")
            st.markdown("""
            The Peter-Clark (PC) algorithm is a fundamental constraint-based method that:
            
            - Starts with a fully connected graph
            - Removes edges based on conditional independence tests
            - Orients edges based on identified v-structures (colliders)
            - Works well for Gaussian and discrete data
            
            **Variants in the application:**
            
            - **pc_fisherz**: PC with Fisher's Z test (for continuous, Gaussian data)
            - **pc_chisq**: PC with Chi-square test (for discrete data)
            - **pc_gsq**: PC with G-square test (for discrete data)
            - **pc_kci**: PC with Kernel CI test (for nonlinear dependencies)
            
            **When to use:**
            - When you have moderate sample sizes (n > 100)
            - When you believe there are no latent confounders
            - For both continuous and discrete data
            """)
            
            st.markdown("#### FCI Algorithm")
            st.markdown("""
            The Fast Causal Inference (FCI) algorithm extends PC to handle latent confounders:
            
            - Starts similarly to PC but allows for hidden common causes
            - Produces a richer graphical representation called a PAG (Partial Ancestral Graph)
            - Can distinguish direct relationships from those mediated by hidden variables
            
            **Variants in the application:**
            
            - **fci_fisherz**: FCI with Fisher's Z test
            - **fci_chisq**: FCI with Chi-square test
            - **fci_kci**: FCI with Kernel CI test
            
            **When to use:**
            - When you suspect latent confounders
            - When you need to distinguish direct from indirect causation
            - When you have sufficient sample size (larger than needed for PC)
            """)
            
            st.markdown("#### CD-NOD Algorithm")
            st.markdown("""
            Causal Discovery from Nonstationary/Heterogeneous Data (CD-NOD):
            
            - Designed for data with changing distributions
            - Leverages distribution changes to identify causal direction
            - Can handle both time series and grouped data
            
            **When to use:**
            - For time series with changing dynamics
            - For data collected under different conditions
            - When traditional methods give inconsistent results
            """)
            
        elif algorithm_category == "Score-based Methods":
            st.subheader("Score-based Methods")
            st.markdown("""
            Score-based methods assign a score to each possible graph and search for the graph with the optimal score:
            
            - Often computationally more efficient than constraint-based methods
            - Provide a natural ranking of alternative structures
            - Can incorporate prior knowledge easily
            - May get stuck in local optima
            """)
            
            st.markdown("#### GES Algorithm")
            st.markdown("""
            Greedy Equivalence Search (GES) is a score-based algorithm that:
            
            - Starts with an empty graph
            - Adds, removes, or reverses edges to maximize a score function
            - Is guaranteed to find the correct structure with infinite data (under assumptions)
            - Is more efficient than exhaustive search
            
            **Variants in the application:**
            
            - **ges_bic**: GES with BIC score (for Gaussian data)
            - **ges_bdeu**: GES with BDeu score (for discrete data)
            - **ges_cv**: GES with cross-validation score (for nonlinear relationships)
            
            **When to use:**
            - With moderate to large sample sizes
            - When efficiency is important
            - For both discrete and continuous data (with appropriate score)
            """)
            
            st.markdown("#### Permutation-based Methods")
            st.markdown("""
            **GRASP (Greedy Relaxation of the Sparsest Permutation):**
            
            - Searches over node orderings instead of directly over graphs
            - Often finds better solutions than direct structure search
            - Can be more efficient for larger problems
            
            **BOSS (Best Order Score Search):**
            
            - An optimized algorithm for finding the best causal ordering
            - Highly efficient for moderately sized problems
            - Often finds more accurate structures than other methods
            
            **When to use:**
            - For datasets with 5-30 variables
            - When other methods give inconsistent results
            - When you want a robust alternative to GES
            """)
            
            st.markdown("#### Exact Search")
            st.markdown("""
            **exact_dp (Dynamic Programming) and exact_astar (A* search):**
            
            - Guaranteed to find the globally optimal graph structure
            - Computationally intensive - only practical for small problems
            - Provides a benchmark for other algorithms
            
            **When to use:**
            - For small problems (< 10 variables)
            - When absolute optimality is required
            - For benchmarking other methods
            """)
            
        elif algorithm_category == "FCM-based Methods":
            st.subheader("Functional Causal Model (FCM) Based Methods")
            st.markdown("""
            FCM-based methods exploit asymmetries in the data generating process to identify causal direction:
            
            - Can often determine causal direction between pairs of variables
            - Make specific assumptions about the data generating process
            - Usually more powerful than constraint-based methods (when assumptions hold)
            - May fail if functional assumptions are violated
            """)
            
            st.markdown("#### LiNGAM-based Methods")
            st.markdown("""
            Linear, Non-Gaussian Acyclic Models (LiNGAM) exploits non-Gaussianity to identify causation:
            
            - **lingam_ica**: Original ICA-based LiNGAM algorithm
            - **lingam_direct**: DirectLiNGAM, more robust than the original
            - **lingam_var**: VAR-LiNGAM for time series data
            - **lingam_rcd**: RCD algorithm for latent confounders
            - **lingam_camuv**: CAM-UV for additive models with unobserved variables
            
            **When to use:**
            - For continuous, non-Gaussian data
            - When relationships are approximately linear
            - When you have sufficient sample size (n > 100)
            """)
            
            st.markdown("#### ANM and PNL")
            st.markdown("""
            Nonlinear methods for pairwise causal discovery:
            
            - **anm**: Additive Noise Models for nonlinear relationships
            - **pnl**: Post-Nonlinear Causal Models for more complex relationships
            
            **When to use:**
            - For pairwise causal discovery
            - When relationships may be nonlinear
            - With sufficient sample size (n > 200)
            """)
            
        elif algorithm_category == "Hidden Causal Methods":
            st.subheader("Hidden Causal Methods")
            st.markdown("""
            Methods specifically designed to discover latent variable structures:
            
            #### GIN (Generalized Independent Noise)
            
            - Discovers linear latent variable models
            - Based on generalized independent noise condition
            - Works well with non-Gaussian data
            
            **When to use:**
            - When you suspect latent variables
            - For linear relationships with non-Gaussian distributions
            - With moderate to large sample sizes
            """)
            
        elif algorithm_category == "Granger Causality":
            st.subheader("Granger Causality")
            st.markdown("""
            Time series methods based on predictive relevance:
            
            - **granger_test**: Basic bivariate Granger causality test
            - **granger_lasso**: Multivariate Granger causality with regularization
            
            **When to use:**
            - For time series data
            - When temporal ordering is clear
            - For forecasting and temporal causal analysis
            """)

    # Algorithm Selection Tab
    with tabs[3]:
        st.header("Algorithm Selection Guide")
        
        st.markdown("""
        Selecting the right algorithm is crucial for accurate causal discovery. Our framework analyzes 
        your data characteristics and recommends appropriate algorithms, but it's helpful to understand 
        the selection criteria.
        """)
        
        selection_criteria = st.selectbox(
            "Select a selection criteria to learn about:",
            ["Data Type", "Sample Size", "Latent Confounders", "Time Series Data", 
             "Distribution Characteristics", "Computational Constraints"]
        )
        
        if selection_criteria == "Data Type":
            st.subheader("Data Type")
            
            st.markdown("""
            Different algorithms are designed for different data types:
            
            **For continuous data:**
            - PC with Fisher's Z test (pc_fisherz)
            - GES with BIC score (ges_bic)
            - LiNGAM variants (for non-Gaussian data)
            
            **For discrete data:**
            - PC with Chi-square or G-square test (pc_chisq, pc_gsq)
            - GES with BDeu score (ges_bdeu)
            
            **For mixed data (both continuous and discrete):**
            - PC with Fisher's Z test (with limitations)
            - Consider transforming variables for consistency
            
            **For nonlinear relationships:**
            - PC or FCI with kernel-based tests (pc_kci, fci_kci)
            - ANM or PNL (for pairwise analysis)
            - GES with cross-validation score (ges_cv)
            """)
            
            st.info("""
            **Tip:** The data exploration page automatically analyzes your data types and 
            recommends appropriate algorithms, but you can override these recommendations 
            based on domain knowledge.
            """)
            
        elif selection_criteria == "Sample Size":
            st.subheader("Sample Size")
            
            st.markdown("""
            Sample size significantly impacts algorithm performance:
            
            **Small samples (n < 100):**
            - PC with Fisher's Z or Chi-square (with caution)
            - GES with appropriate score function
            - Exact search for very small problems (< 10 variables)
            
            **Moderate samples (100 < n < 500):**
            - PC algorithm variants
            - GES, GRaSP, BOSS
            - DirectLiNGAM (for non-Gaussian data)
            
            **Large samples (n > 500):**
            - Any algorithm, including computationally intensive ones
            - Kernel-based methods (pc_kci, fci_kci)
            - ANM, PNL for nonlinear relationships
            """)
            
            st.warning("""
            **Warning:** With small sample sizes, all causal discovery methods have limitations. 
            Results should be interpreted with caution and cross-validated with domain knowledge.
            """)
            
        elif selection_criteria == "Latent Confounders":
            st.subheader("Latent Confounders")
            
            st.markdown("""
            If you suspect unmeasured variables affecting multiple observed variables:
            
            **Methods that handle latent confounders:**
            - FCI algorithm variants (fci_fisherz, fci_chisq, fci_kci)
            - LiNGAM RCD (lingam_rcd)
            - CAM-UV (lingam_camuv)
            - GIN
            
            **When latent confounders are unlikely:**
            - PC algorithm variants
            - GES, GRaSP, BOSS
            - Standard LiNGAM variants
            
            Latent confounders can lead to spurious relationships in methods that don't account for them.
            """)
            
            st.info("""
            **Tip:** If you're unsure about latent confounders, run both FCI and PC algorithms 
            and compare the results. Major differences might indicate the presence of latent confounders.
            """)
            
        elif selection_criteria == "Time Series Data":
            st.subheader("Time Series Data")
            
            st.markdown("""
            For data with temporal ordering:
            
            **Time series specific methods:**
            - VAR-LiNGAM (lingam_var)
            - Granger causality methods (granger_test, granger_lasso)
            - CD-NOD (for nonstationary time series)
            
            **Considerations for time series:**
            - Ensure data is properly temporally aligned
            - Consider appropriate time lags
            - Check for stationarity and seasonality
            
            Traditional causal discovery methods assume i.i.d. samples and may not be appropriate for time series.
            """)
            
            st.warning("""
            **Warning:** Time series causality has additional complexities like temporal lags and 
            non-stationarity. Make sure to preprocess your data appropriately (e.g., differencing 
            for non-stationary series).
            """)
            
        elif selection_criteria == "Distribution Characteristics":
            st.subheader("Distribution Characteristics")
            
            st.markdown("""
            The distribution of your data affects algorithm choice:
            
            **For Gaussian (normal) distributions:**
            - PC with Fisher's Z test (pc_fisherz)
            - GES with BIC score (ges_bic)
            
            **For non-Gaussian continuous data:**
            - LiNGAM variants (lingam_ica, lingam_direct)
            - PC still works but may be less optimal
            
            **For discrete data with skewed distributions:**
            - PC with G-square test often performs better than Chi-square
            
            **For highly nonlinear relationships:**
            - Kernel-based methods (pc_kci, fci_kci)
            - ANM or PNL for pairwise analysis
            """)
            
            st.info("""
            **Tip:** The data exploration page includes distribution analysis to help identify 
            whether your data is Gaussian, non-Gaussian, or nonlinear.
            """)
            
        elif selection_criteria == "Computational Constraints":
            st.subheader("Computational Constraints")
            
            st.markdown("""
            Algorithm computational requirements increase with:
            - Number of variables
            - Sample size
            - Complexity of independence tests or score functions
            
            **Computationally efficient methods:**
            - PC with Fisher's Z or Chi-square tests
            - GES with BIC or BDeu score
            - DirectLiNGAM
            
            **Moderate computational requirements:**
            - FCI algorithm variants
            - GRaSP, BOSS
            
            **Computationally intensive methods:**
            - Exact search methods (only feasible for < 10 variables)
            - Kernel-based methods (pc_kci, fci_kci) for large samples
            - ANM, PNL with large samples
            """)
            
            st.warning("""
            **Warning:** For datasets with many variables (> 30), consider using a subset of variables 
            based on domain knowledge or using methods with strong sparsity assumptions.
            """)

    # Graph Refinement Tab
    with tabs[4]:
        st.header("LLM-based Graph Refinement")
        
        st.markdown("""
        One of the unique features of our framework is the ability to refine causal graphs using LLMs.
        This combines statistical algorithms with domain knowledge encoded in language models.
        """)
        
        st.subheader("What LLM-based Refinement Can Do")
        
        st.markdown("""
        The graph refinement process can help:
        
        1. **Validate discovered relationships** against general knowledge
        2. **Resolve conflicts** between different algorithm results
        3. **Identify potential hidden variables** not present in the original dataset
        4. **Incorporate domain-specific knowledge** into the causal graph
        5. **Assign confidence weights** to relationships based on evidence
        """)
        
        st.subheader("How to Use Graph Refinement")
        
        st.markdown("""
        On the Graph Refinement page:
        
        1. **Select a base graph** from your algorithm results
        2. **Choose refinement options:**
           - Relationship validation
           - Conflict resolution
           - Hidden variable discovery
           - Domain knowledge integration
        3. **Provide domain context** (optional but recommended)
        4. **Review and approve refinements** proposed by the LLM
        
        The system will walk you through each step with clear explanations.
        """)
        
        st.subheader("Best Practices for Graph Refinement")
        
        st.markdown("""
        For best results with LLM-based refinement:
        
        - **Start with a good base graph** from an appropriate algorithm
        - **Provide domain context** to help the LLM understand your specific domain
        - **Review all suggestions critically** before accepting them
        - **Use refinement iteratively** rather than trying to perfect the graph in one step
        - **Combine with ensemble methods** for more robust results
        
        Remember that while LLMs are powerful, they should complement, not replace, 
        your domain expertise and critical thinking.
        """)
        
        st.info("""
        **Tip:** You can experiment with different LLM providers and models by changing the settings 
        in the Settings page. More advanced models may provide better refinements but may be slower or more costly.
        """)

    # Case Studies Tab
    with tabs[5]:
        st.header("Case Studies")
        
        st.markdown("""
        Let's look at some practical examples to illustrate how to use the framework effectively.
        """)
        
        case_study = st.selectbox(
            "Select a case study:",
            ["Medical Diagnosis", "Marketing Attribution", "Environmental Factors", "Time Series Analysis"]
        )
        
        if case_study == "Medical Diagnosis":
            st.subheader("Case Study: Medical Diagnosis Factors")
            
            st.markdown("""
            **Scenario**: A dataset containing patient information (age, symptoms, test results) and diagnoses.
            
            **Goal**: Discover causal factors leading to specific medical conditions.
            
            **Approach**:
            
            1. **Data Loading and Exploration**:
               - Mixed data types (both continuous and categorical)
               - Moderate sample size (350 patients)
               - Some missing values in test results
            
            2. **Algorithm Selection**:
               - Primary: PC with Fisher's Z test (pc_fisherz)
               - Secondary: FCI with Fisher's Z test (fci_fisherz) to check for latent confounders
            
            3. **Initial Results**:
               - PC identified several symptom → diagnosis relationships
               - FCI suggested some hidden confounders
            
            4. **Graph Refinement**:
               - LLM identified a potential hidden variable: "genetic predisposition"
               - Several medically implausible links were removed
               - Confidence weights adjusted based on medical literature
            
            5. **Insights**:
               - Identified key causal pathways leading to diagnoses
               - Discovered unexpected relationships between symptoms
               - Highlighted factors that should be prioritized in screening
            """)
            
            st.info("""
            **Key Takeaway**: The combination of PC/FCI algorithms with LLM refinement allowed 
            identification of both direct causal factors and potential unmeasured variables in the 
            medical diagnosis process.
            """)
            
        elif case_study == "Marketing Attribution":
            st.subheader("Case Study: Marketing Attribution")
            
            st.markdown("""
            **Scenario**: Marketing data with advertising spend across channels and resulting sales.
            
            **Goal**: Determine which marketing channels truly drive sales vs. which are merely correlated.
            
            **Approach**:
            
            1. **Data Loading and Exploration**:
               - Continuous data with possible non-Gaussian distributions
               - Time-ordered observations (weekly data for 2 years)
               - Strong correlations between different marketing channels
            
            2. **Algorithm Selection**:
               - Primary: LiNGAM-based methods (lingam_direct)
               - Secondary: VAR-LiNGAM for time-series aspects
               - Comparative: Granger causality analysis
            
            3. **Initial Results**:
               - DirectLiNGAM identified social media and email as driving sales
               - VAR-LiNGAM suggested lagged effects from TV advertising
               - Significant correlations but no causal links for some channels
            
            4. **Graph Refinement**:
               - LLM suggested seasonal effects as potential confounders
               - Refined temporal relationships between channels
               - Added confidence weights based on consistency across methods
            
            5. **Insights**:
               - Identified true drivers of sales vs. correlated channels
               - Discovered optimal lag times for different marketing activities
               - Revealed potential synergies between channels
            """)
            
            st.info("""
            **Key Takeaway**: FCM-based methods combined with time-series analysis revealed true 
            causal dependencies in marketing data, helping optimize allocation of marketing budget.
            """)
            
        elif case_study == "Environmental Factors":
            st.subheader("Case Study: Environmental Factors on Crop Yield")
            
            st.markdown("""
            **Scenario**: Agricultural data on soil conditions, weather, farming practices, and crop yields.
            
            **Goal**: Identify the causal factors that most impact crop yield.
            
            **Approach**:
            
            1. **Data Loading and Exploration**:
               - Mixed data types (continuous and categorical)
               - Possible nonlinear relationships
               - Data from multiple regions with different conditions
            
            2. **Algorithm Selection**:
               - Primary: PC with kernel-based test (pc_kci) for nonlinear relationships
               - Secondary: GES with cross-validation score (ges_cv)
               - Comparative: CD-NOD to handle regional heterogeneity
            
            3. **Initial Results**:
               - PC-KCI identified complex nonlinear relationships
               - GES provided more conservative but clearer structure
               - CD-NOD revealed different causal patterns across regions
            
            4. **Graph Refinement**:
               - LLM integrated agricultural domain knowledge
               - Identified potential latent variables ("soil microbiome")
               - Resolved conflicts between algorithms based on plausibility
            
            5. **Insights**:
               - Discovered key causal drivers of crop yield
               - Identified region-specific factors that matter most
               - Found interactions between environmental factors
            """)
            
            st.info("""
            **Key Takeaway**: Nonlinear methods combined with algorithms that handle heterogeneity were 
            essential for this complex environmental system. LLM refinement added important domain context.
            """)
            
        elif case_study == "Time Series Analysis":
            st.subheader("Case Study: Economic Indicators Time Series")
            
            st.markdown("""
            **Scenario**: Macroeconomic time series data (GDP, inflation, unemployment, interest rates).
            
            **Goal**: Discover causal relationships between economic indicators.
            
            **Approach**:
            
            1. **Data Loading and Exploration**:
               - Continuous time series data (quarterly for 30 years)
               - Possible non-stationarity
               - Complex temporal dependencies
            
            2. **Algorithm Selection**:
               - Primary: VAR-LiNGAM for time series causal discovery
               - Secondary: Granger causality with Lasso regularization
               - Comparative: CD-NOD for handling regime changes
            
            3. **Initial Results**:
               - VAR-LiNGAM identified key monetary policy effects
               - Granger causality confirmed some relationships but missed others
               - CD-NOD revealed changing causal patterns during recessions
            
            4. **Graph Refinement**:
               - LLM integrated economic theory
               - Added confidence weights based on theoretical support
               - Identified potential regime changes not captured in the data
            
            5. **Insights**:
               - Mapped causal pathways in monetary and fiscal policy
               - Discovered how causal relationships change during economic cycles
               - Identified leading indicators for economic changes
            """)
            
            st.info("""
            **Key Takeaway**: Time-series specific methods were essential, and the LLM refinement step added 
            valuable economic domain knowledge that helped interpret the complex temporal dynamics.
            """)

if __name__ == "__main__":
    app()