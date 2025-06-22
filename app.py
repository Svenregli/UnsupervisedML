"""
Academic Research Analysis - Complete Fixed Streamlit Dashboard
============================================================

Interactive dashboard for exploring unsupervised learning results
on academic research data with robust error handling for loaded experiments.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
import umap
warnings.filterwarnings('ignore')

# Import your analysis modules
try:
    from ul_pipeline import AcademicResearchAnalyzer
    from visualisations_module import ResearchVisualizationSuite, AdvancedAnalysisSuite
    from results_loader import LocalResultsLoader, test_loaded_analyzer
except ImportError as e:
    st.error(f"Please ensure the analysis modules are in the same directory: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Academic Research Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .cluster-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    """Initialize session state variables for persistent storage."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'viz_suite' not in st.session_state:
        st.session_state.viz_suite = None
    if 'advanced_suite' not in st.session_state:
        st.session_state.advanced_suite = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'loaded_analyzer' not in st.session_state:
        st.session_state.loaded_analyzer = None
    if 'experiment_loaded' not in st.session_state:
        st.session_state.experiment_loaded = False

class StreamlitDashboard:
    """
    Main dashboard class for the Streamlit app.
    """
    
    def __init__(self):
        pass
        
    @property
    def analyzer(self):
        """Get analyzer from session state."""
        return st.session_state.analyzer
    
    @analyzer.setter
    def analyzer(self, value):
        """Set analyzer in session state."""
        st.session_state.analyzer = value
    
    @property
    def viz_suite(self):
        """Get viz_suite from session state."""
        return st.session_state.viz_suite
    
    @viz_suite.setter
    def viz_suite(self, value):
        """Set viz_suite in session state."""
        st.session_state.viz_suite = value
    
    @property
    def advanced_suite(self):
        """Get advanced_suite from session state."""
        return st.session_state.advanced_suite
    
    @advanced_suite.setter
    def advanced_suite(self, value):
        """Set advanced_suite in session state."""
        st.session_state.advanced_suite = value

def get_cluster_profiles_safely(analyzer, method):
    """
    Safely get cluster profiles with multiple fallback strategies.
    This handles the case where the loaded analyzer might be missing methods.
    """
    
    # Strategy 1: Check if we have pre-computed all_cluster_profiles
    if hasattr(analyzer, 'all_cluster_profiles') and method in analyzer.all_cluster_profiles:
        st.success(f"✅ Using pre-computed profiles for {method}")
        return analyzer.all_cluster_profiles[method]
    
    # Strategy 2: Check if current cluster_profiles match this method
    if hasattr(analyzer, 'cluster_profiles') and analyzer.cluster_profiles:
        # Verify profiles match the current method
        labels = analyzer.cluster_labels[method]
        current_cluster_ids = set(analyzer.cluster_profiles.keys())
        method_cluster_ids = set(np.unique(labels))
        method_cluster_ids.discard(-1)  # Remove noise cluster
        
        if current_cluster_ids == method_cluster_ids:
            st.info(f"ℹ️ Using existing profiles for {method}")
            return analyzer.cluster_profiles
    
    # Strategy 3: Try to use the analyzer's built-in methods
    if hasattr(analyzer, 'create_cluster_profiles'):
        try:
            st.info(f"🔄 Creating new profiles for {method}...")
            profiles = analyzer.create_cluster_profiles(method)
            return profiles
        except Exception as e:
            st.warning(f"⚠️ Built-in profile creation failed: {e}")
    
    # Strategy 4: Try the enhanced method if available
    if hasattr(analyzer, 'get_cluster_profiles_for_method'):
        try:
            st.info(f"🔄 Using enhanced profile method for {method}...")
            profiles = analyzer.get_cluster_profiles_for_method(method)
            return profiles
        except Exception as e:
            st.warning(f"⚠️ Enhanced profile method failed: {e}")
    
    # Strategy 5: Use fallback creation
    try:
        st.info(f"🔄 Using fallback profile creation for {method}...")
        profiles = create_cluster_profiles_fallback(analyzer, method)
        return profiles
    except Exception as e:
        st.error(f"❌ All profile creation methods failed: {e}")
        return None

def create_cluster_profiles_fallback(analyzer, method):
    """
    Fallback method to create cluster profiles when the analyzer's methods are unavailable.
    This recreates the functionality without relying on the analyzer's methods.
    """
    
    if method not in analyzer.cluster_labels:
        st.error(f"Method '{method}' not found in cluster labels")
        return None
    
    try:
        labels = analyzer.cluster_labels[method]
        df = analyzer.df
        
        # Create cluster profiles manually
        cluster_profiles = {}
        unique_clusters = np.unique(labels)
        
        # Remove noise cluster (-1) if present
        if -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            profile = {
                'size': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(df)) * 100,
                'numerical_features': {},
                'top_terms': [],
                'example_papers': []
            }
            
            # Calculate numerical feature statistics
            numerical_features = analyzer.numerical_features if hasattr(analyzer, 'numerical_features') else [
                'cited_by_count', 'publication_year', 'paper_age',
                'citations_per_year', 'author_count', 'concept_count',
                'avg_concept_score', 'referenced_works_count'
            ]
            
            for feature in numerical_features:
                if feature in df.columns:
                    feature_data = cluster_data[feature]
                    profile['numerical_features'][feature] = {
                        'mean': feature_data.mean(),
                        'median': feature_data.median(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max()
                    }
            
            # Extract top terms using TF-IDF (if available)
            if hasattr(analyzer, 'tfidf_vectorizer') and analyzer.tfidf_vectorizer is not None:
                if 'combined_text' in df.columns:
                    cluster_texts = cluster_data['combined_text'].tolist()
                elif 'title' in df.columns and 'abstract' in df.columns:
                    cluster_texts = (cluster_data['title'].fillna('') + ' ' + cluster_data['abstract'].fillna('')).tolist()
                else:
                    cluster_texts = []
                
                if cluster_texts:
                    try:
                        cluster_tfidf = analyzer.tfidf_vectorizer.transform(cluster_texts)
                        mean_tfidf = np.array(cluster_tfidf.mean(axis=0)).flatten()
                        feature_names = analyzer.tfidf_vectorizer.get_feature_names_out()
                        
                        # Get top terms
                        top_indices = mean_tfidf.argsort()[-20:][::-1]
                        profile['top_terms'] = [(feature_names[i], mean_tfidf[i]) for i in top_indices if mean_tfidf[i] > 0]
                    except Exception as e:
                        st.warning(f"Could not extract terms for cluster {cluster_id}: {e}")
                        profile['top_terms'] = []
            
            # Example papers (highest cited in cluster)
            if 'cited_by_count' in df.columns:
                top_papers = cluster_data.nlargest(5, 'cited_by_count')
                example_columns = ['title', 'cited_by_count']
                if 'publication_year' in df.columns:
                    example_columns.append('publication_year')
                
                profile['example_papers'] = top_papers[example_columns].to_dict('records')
            
            cluster_profiles[cluster_id] = profile
        
        # Store the profiles in the analyzer if possible
        if not hasattr(analyzer, 'cluster_profiles'):
            analyzer.cluster_profiles = {}
        analyzer.cluster_profiles = cluster_profiles
        
        if not hasattr(analyzer, 'all_cluster_profiles'):
            analyzer.all_cluster_profiles = {}
        analyzer.all_cluster_profiles[method] = cluster_profiles
        
        return cluster_profiles
        
    except Exception as e:
        st.error(f"Fallback profile creation failed: {e}")
        return None

def validate_analyzer_for_dashboard(analyzer):
    """
    Validate that the analyzer has all required components for the dashboard.
    """
    required_attributes = [
        'df', 'cluster_labels', 'numerical_features'
    ]
    
    optional_attributes = [
        'dimensionality_results', 'outlier_labels', 'clustering_metrics',
        'cluster_profiles', 'all_cluster_profiles', 'features_scaled', 'tfidf_vectorizer'
    ]
    
    missing_required = []
    missing_optional = []
    
    for attr in required_attributes:
        if not hasattr(analyzer, attr) or getattr(analyzer, attr) is None:
            missing_required.append(attr)
    
    for attr in optional_attributes:
        if not hasattr(analyzer, attr) or getattr(analyzer, attr) is None:
            missing_optional.append(attr)
    
    if missing_required:
        st.error(f"❌ Missing required attributes: {missing_required}")
        return False
    
    if missing_optional:
        st.warning(f"⚠️ Missing optional attributes (may affect some features): {missing_optional}")
    
    return True

def repair_loaded_analyzer(analyzer):
    """
    Attempt to repair a loaded analyzer that might be missing some functionality.
    """
    repairs_made = []
    
    # Ensure numerical_features exists
    if not hasattr(analyzer, 'numerical_features') or not analyzer.numerical_features:
        potential_features = [
            'cited_by_count', 'publication_year', 'paper_age',
            'citations_per_year', 'author_count', 'concept_count',
            'avg_concept_score', 'referenced_works_count'
        ]
        
        analyzer.numerical_features = [
            feat for feat in potential_features 
            if feat in analyzer.df.columns
        ]
        repairs_made.append('numerical_features')
    
    # Ensure dictionaries exist
    dict_attributes = [
        'cluster_labels', 'outlier_labels', 'dimensionality_results',
        'clustering_metrics', 'cluster_profiles', 'all_cluster_profiles'
    ]
    
    for attr in dict_attributes:
        if not hasattr(analyzer, attr):
            setattr(analyzer, attr, {})
            repairs_made.append(attr)
    
    if repairs_made:
        st.info(f"🔧 Repaired analyzer attributes: {', '.join(repairs_made)}")
    
    return analyzer

def show_cluster_analysis(dashboard):
    """
    Display cluster analysis results with robust profile handling.
    """
    
    analyzer = dashboard.analyzer
    df = analyzer.df
    
    st.header("🎯 Research Community Analysis")
    
    # Cluster method selection
    available_methods = []
    if hasattr(analyzer, 'cluster_labels') and analyzer.cluster_labels:
        available_methods = list(analyzer.cluster_labels.keys())
    
    if not available_methods:
        st.warning("No clustering results available. Please run analysis first.")
        return
    
    # Use session state to remember the selected method
    if 'selected_clustering_method' not in st.session_state:
        st.session_state.selected_clustering_method = 'kmeans' if 'kmeans' in available_methods else available_methods[0]
    
    selected_method = st.selectbox(
        "Select clustering method:",
        available_methods,
        index=available_methods.index(st.session_state.selected_clustering_method) if st.session_state.selected_clustering_method in available_methods else 0,
        key='clustering_method_selector'
    )
    
    # Update session state
    st.session_state.selected_clustering_method = selected_method
    
    cluster_column = f'cluster_{selected_method}'
    
    if cluster_column not in df.columns:
        st.error(f"Cluster labels for {selected_method} not found in data.")
        return
    
    # Show cluster information
    labels = analyzer.cluster_labels[selected_method]
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    n_noise = np.sum(labels == -1) if -1 in unique_clusters else 0
    
    if n_noise > 0:
        st.info(f"📊 **{selected_method.upper()}**: {n_clusters-1} clusters + {n_noise} noise points")
    else:
        st.info(f"📊 **{selected_method.upper()}**: {n_clusters} clusters")
    
    # Cluster overview visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Cluster Visualization")
        
        # Create UMAP visualization with clusters
        if hasattr(analyzer, 'dimensionality_results') and 'umap' in analyzer.dimensionality_results:
            umap_data = analyzer.dimensionality_results['umap']['transformed_data']
            
            # Create a copy of the cluster labels for visualization
            viz_labels = labels.copy()
            
            fig = px.scatter(
                x=umap_data[:, 0],
                y=umap_data[:, 1],
                color=viz_labels.astype(str),
                hover_data={
                    'Title': df['title'].str[:50] + '...' if 'title' in df.columns else df.index,
                    'Citations': df['cited_by_count'] if 'cited_by_count' in df.columns else 'N/A',
                    'Year': df['publication_year'] if 'publication_year' in df.columns else 'N/A'
                },
                title=f"Research Communities ({selected_method.upper()})",
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'color': 'Cluster'},
                width=600,
                height=500
            )
            
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("UMAP visualization not available. Please run dimensionality reduction first.")
    
    with col2:
        st.subheader("📊 Cluster Statistics")
        
        # Cluster size distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        # Handle noise cluster (-1) separately
        display_labels = []
        display_counts = []
        
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                display_labels.append("Noise")
            else:
                display_labels.append(f"Cluster {cluster_id}")
            display_counts.append(count)
        
        fig_bar = px.bar(
            x=display_labels,
            y=display_counts,
            title="Papers per Cluster",
            labels={'x': 'Cluster', 'y': 'Number of Papers'},
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Cluster quality metrics
        if hasattr(analyzer, 'clustering_metrics') and selected_method in analyzer.clustering_metrics:
            metrics = analyzer.clustering_metrics[selected_method]
            
            st.markdown("**Quality Metrics:**")
            st.markdown(f"- Silhouette Score: **{metrics['silhouette_score']:.3f}**")
            st.markdown(f"- Davies-Bouldin Score: **{metrics['davies_bouldin_score']:.3f}**")
            st.markdown(f"- Number of Clusters: **{metrics['n_clusters']}**")
            if metrics['n_noise'] > 0:
                st.markdown(f"- Noise Points: **{metrics['n_noise']}**")
    
    # FIXED: Cluster profiles with robust error handling
    st.subheader("📋 Cluster Profiles")
    
    profiles = get_cluster_profiles_safely(analyzer, selected_method)
    
    if profiles and len(profiles) > 0:
        # Show profiles for actual clusters (not noise)
        actual_clusters = [cid for cid in profiles.keys() if cid != -1]
        
        if not actual_clusters:
            st.warning("No cluster profiles available (only noise detected)")
            return
        
        st.markdown(f"**Found {len(actual_clusters)} cluster profiles:**")
        
        # Display each cluster
        for cluster_id in sorted(actual_clusters):
            profile = profiles[cluster_id]
            
            with st.expander(f"🔍 Cluster {cluster_id} ({profile['size']} papers, {profile['percentage']:.1f}%)", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Statistics:**")
                    
                    if 'cited_by_count' in profile['numerical_features']:
                        avg_cit = profile['numerical_features']['cited_by_count']['mean']
                        st.markdown(f"- Average Citations: **{avg_cit:.1f}**")
                    
                    if 'citations_per_year' in profile['numerical_features']:
                        avg_cit_year = profile['numerical_features']['citations_per_year']['mean']
                        st.markdown(f"- Citations per Year: **{avg_cit_year:.2f}**")
                    
                    if 'paper_age' in profile['numerical_features']:
                        avg_age = profile['numerical_features']['paper_age']['mean']
                        st.markdown(f"- Average Age: **{avg_age:.1f} years**")
                
                with col2:
                    st.markdown("**🔤 Key Terms:**")
                    if profile['top_terms']:
                        terms = [term for term, score in profile['top_terms'][:8]]
                        st.markdown(f"*{', '.join(terms)}*")
                
                # Example papers
                if profile['example_papers']:
                    st.markdown("**📄 Representative Papers:**")
                    for i, paper in enumerate(profile['example_papers'][:3], 1):
                        st.markdown(f"{i}. **{paper['title']}** ({paper['cited_by_count']} citations, {paper['publication_year']})")
    else:
        st.warning("⚠️ No cluster profiles available for this method")
        
        # Offer to create profiles using the fallback method
        if st.button(f"🔄 Generate Profiles for {selected_method.upper()}", key=f"generate_profiles_{selected_method}"):
            with st.spinner("Creating cluster profiles..."):
                try:
                    profiles = create_cluster_profiles_fallback(analyzer, selected_method)
                    if profiles:
                        st.success("✅ Cluster profiles created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create cluster profiles")
                except Exception as e:
                    st.error(f"❌ Error creating profiles: {e}")

def show_dimensionality_analysis_fixed(dashboard):
    """
    Fixed dimensionality analysis that handles missing data gracefully.
    """
    analyzer = dashboard.analyzer
    
    st.header("🔍 Dimensionality Reduction Analysis")
    
    if not hasattr(analyzer, 'dimensionality_results') or not analyzer.dimensionality_results:
        st.warning("Dimensionality reduction results not available in this experiment.")
        return
    
    results = analyzer.dimensionality_results
    
    # Show available methods
    available_methods = list(results.keys())
    st.info(f"Available methods: {', '.join(available_methods)}")
    
    # PCA Analysis
    if 'pca' in results:
        st.subheader("📊 Principal Component Analysis (PCA)")
        
        pca_results = results['pca']
        if 'explained_variance_ratio' in pca_results:
            explained_var = pca_results['explained_variance_ratio']
            cumulative_var = pca_results.get('cumulative_variance', np.cumsum(explained_var))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Explained variance plot
                fig = px.bar(
                    x=list(range(1, min(11, len(explained_var) + 1))),
                    y=explained_var[:10],
                    title="Explained Variance by Component",
                    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cumulative variance plot
                n_components = min(20, len(cumulative_var))
                fig = px.line(
                    x=list(range(1, n_components + 1)),
                    y=cumulative_var[:n_components],
                    title="Cumulative Explained Variance",
                    labels={'x': 'Number of Components', 'y': 'Cumulative Variance'}
                )
                fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                             annotation_text="80% Variance")
                fig.add_hline(y=0.9, line_dash="dash", line_color="green", 
                             annotation_text="90% Variance")
                st.plotly_chart(fig, use_container_width=True)
    
    # 2D Visualizations
    st.subheader("🗺️ 2D Projections")
    
    # Get available visualization methods
    viz_methods = []
    for method in ['umap', 'tsne', 'pca']:
        if method in results and 'transformed_data' in results[method]:
            viz_methods.append(method.upper())
    
    if not viz_methods:
        st.warning("No 2D projection data available.")
        return
    
    selected_viz = st.selectbox("Select visualization method:", viz_methods)
    
    method_key = selected_viz.lower()
    if method_key in results:
        projected_data = results[method_key]['transformed_data']
        
        # Color options based on available columns
        color_options = []
        color_mapping = {}
        
        if 'cited_by_count' in analyzer.df.columns:
            color_options.append("Citation Count")
            color_mapping["Citation Count"] = "cited_by_count"
        
        if 'publication_year' in analyzer.df.columns:
            color_options.append("Publication Year")
            color_mapping["Publication Year"] = "publication_year"
        
        if 'paper_age' in analyzer.df.columns:
            color_options.append("Paper Age")
            color_mapping["Paper Age"] = "paper_age"
        
        if 'citations_per_year' in analyzer.df.columns:
            color_options.append("Citations per Year")
            color_mapping["Citations per Year"] = "citations_per_year"
        
        if color_options:
            color_by = st.selectbox("Color points by:", color_options)
            color_col = color_mapping[color_by]
            
            fig = px.scatter(
                x=projected_data[:, 0],
                y=projected_data[:, 1],
                color=analyzer.df[color_col],
                hover_data={
                    'Title': analyzer.df['title'].str[:50] + '...' if 'title' in analyzer.df.columns else analyzer.df.index,
                    'Citations': analyzer.df['cited_by_count'] if 'cited_by_count' in analyzer.df.columns else 'N/A',
                    'Year': analyzer.df['publication_year'] if 'publication_year' in analyzer.df.columns else 'N/A'
                },
                title=f"{selected_viz} Projection (colored by {color_by})",
                labels={'x': f'{selected_viz} Dimension 1', 
                       'y': f'{selected_viz} Dimension 2',
                       'color': color_by},
                height=600
            )
            
            fig.update_traces(marker=dict(size=4, opacity=0.6))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No suitable columns found for coloring the visualization.")

def show_outlier_analysis_fixed(dashboard):
    """
    Fixed outlier analysis that handles missing data gracefully with additional plots.
    """
    analyzer = dashboard.analyzer
    df = analyzer.df
    
    st.header("⚡ Outlier Detection Analysis")
    
    if not hasattr(analyzer, 'outlier_labels') or not analyzer.outlier_labels:
        st.warning("Outlier detection results not available in this experiment.")
        
        # Check if we have outlier consensus in the DataFrame
        if 'outlier_consensus_score' in df.columns:
            st.info("Found outlier consensus scores in the data.")
            
            consensus_scores = df['outlier_consensus_score']
            strong_outliers = (consensus_scores >= 3).sum()
            moderate_outliers = (consensus_scores == 2).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strong Outliers", f"{strong_outliers:,}")
            with col2:
                st.metric("Moderate Outliers", f"{moderate_outliers:,}")
            with col3:
                outlier_rate = ((strong_outliers + moderate_outliers) / len(df)) * 100
                st.metric("Outlier Rate", f"{outlier_rate:.1f}%")
        
        return
    
    # Original outlier analysis if data is available
    st.subheader("📊 Outlier Detection Overview")
    
    outlier_methods = list(analyzer.outlier_labels.keys())
    
    if outlier_methods:
        # Method comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Outlier counts by method
            outlier_counts = {}
            for method in outlier_methods:
                outliers = (analyzer.outlier_labels[method] == -1).sum()
                outlier_counts[method] = outliers
            
            fig = px.bar(
                x=list(outlier_counts.keys()),
                y=list(outlier_counts.values()),
                title="Outliers Detected by Method",
                labels={'x': 'Detection Method', 'y': 'Number of Outliers'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Detection Statistics:**")
            total_outliers = sum(outlier_counts.values())
            outlier_rate = (total_outliers / (len(df) * len(outlier_methods))) * 100
            
            st.metric("Total Detections", f"{total_outliers:,}")
            st.metric("Average Rate", f"{outlier_rate:.1f}%")
            
            if 'outlier_consensus_score' in df.columns:
                strong_outliers = (df['outlier_consensus_score'] >= 3).sum()
                st.metric("Strong Outliers", f"{strong_outliers:,}")
    
    # NEW PLOT 1: Outlier Distribution in UMAP Space
    st.subheader("🗺️ Outlier Distribution in 2D Space")
    
    if hasattr(analyzer, 'dimensionality_results') and 'umap' in analyzer.dimensionality_results:
        umap_data = analyzer.dimensionality_results['umap']['transformed_data']
        
        # Select outlier method to visualize
        selected_method = st.selectbox(
            "Select outlier detection method for visualization:",
            outlier_methods,
            key="outlier_method_selector"
        )
        
        if selected_method in analyzer.outlier_labels:
            outlier_labels = analyzer.outlier_labels[selected_method]
            
            # Create outlier vs normal classification
            is_outlier = (outlier_labels == -1)
            
            # Create the scatter plot
            plot_data = pd.DataFrame({
                'UMAP_1': umap_data[:, 0],
                'UMAP_2': umap_data[:, 1],
                'Is_Outlier': is_outlier,
                'Title': df['title'].str[:50] + '...' if 'title' in df.columns else df.index,
                'Citations': df['cited_by_count'] if 'cited_by_count' in df.columns else 'N/A',
                'Year': df['publication_year'] if 'publication_year' in df.columns else 'N/A'
            })
            
            fig = px.scatter(
                plot_data,
                x='UMAP_1',
                y='UMAP_2',
                color='Is_Outlier',
                color_discrete_map={True: 'red', False: 'blue'},
                hover_data=['Title', 'Citations', 'Year'],
                title=f"Outlier Distribution in UMAP Space ({selected_method.upper()})",
                labels={'UMAP_1': 'UMAP Dimension 1', 'UMAP_2': 'UMAP Dimension 2', 'Is_Outlier': 'Outlier Status'}
            )
            
            # Update marker sizes
            fig.update_traces(
                marker=dict(
                    size=[8 if outlier else 4 for outlier in is_outlier],
                    opacity=[0.9 if outlier else 0.6 for outlier in is_outlier]
                )
            )
            
            # Update legend
            fig.for_each_trace(lambda t: t.update(name="Outlier" if t.name == "True" else "Normal"))
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("UMAP visualization not available for outlier distribution.")
    
    # NEW PLOT 2: Citation vs Age Scatter with Outlier Highlighting
    st.subheader("📈 Citation Patterns: Outliers vs Normal Papers")
    
    if 'cited_by_count' in df.columns and 'paper_age' in df.columns and outlier_methods:
        
        # Method selection for this plot
        selected_method_2 = st.selectbox(
            "Select outlier method for citation analysis:",
            outlier_methods,
            key="outlier_citation_method"
        )
        
        if selected_method_2 in analyzer.outlier_labels:
            outlier_labels_2 = analyzer.outlier_labels[selected_method_2]
            is_outlier_2 = (outlier_labels_2 == -1)
            
            # Create the citation vs age plot
            citation_data = pd.DataFrame({
                'Paper_Age': df['paper_age'],
                'Citations': df['cited_by_count'],
                'Is_Outlier': is_outlier_2,
                'Title': df['title'].str[:50] + '...' if 'title' in df.columns else df.index,
                'Year': df['publication_year'] if 'publication_year' in df.columns else 'N/A'
            })
            
            # Add citations per year if available
            if 'citations_per_year' in df.columns:
                citation_data['Citations_Per_Year'] = df['citations_per_year']
            
            fig = px.scatter(
                citation_data,
                x='Paper_Age',
                y='Citations',
                color='Is_Outlier',
                color_discrete_map={True: 'red', False: 'lightblue'},
                hover_data=['Title', 'Year'] + (['Citations_Per_Year'] if 'citations_per_year' in df.columns else []),
                title=f"Citations vs Paper Age - Outlier Detection ({selected_method_2.upper()})",
                labels={'Paper_Age': 'Paper Age (years)', 'Citations': 'Citation Count', 'Is_Outlier': 'Outlier Status'}
            )
            
            # Update marker properties
            fig.update_traces(
                marker=dict(
                    size=[10 if outlier else 5 for outlier in is_outlier_2],
                    opacity=[0.9 if outlier else 0.6 for outlier in is_outlier_2]
                )
            )
            
            # Update legend
            fig.for_each_trace(lambda t: t.update(name="Outlier" if t.name == "True" else "Normal"))
            
            # Add log scale option
            log_scale = st.checkbox("Use logarithmic scale for citations", key="citation_log_scale")
            if log_scale:
                fig.update_yaxis(type="log")
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Outlier Characteristics:**")
                outlier_papers = df[is_outlier_2]
                normal_papers = df[~is_outlier_2]
                
                if len(outlier_papers) > 0:
                    avg_citations_outlier = outlier_papers['cited_by_count'].mean()
                    avg_age_outlier = outlier_papers['paper_age'].mean()
                    st.markdown(f"- Average citations: **{avg_citations_outlier:.1f}**")
                    st.markdown(f"- Average age: **{avg_age_outlier:.1f} years**")
                    
                    if 'citations_per_year' in df.columns:
                        avg_cit_per_year_outlier = outlier_papers['citations_per_year'].mean()
                        st.markdown(f"- Citations per year: **{avg_cit_per_year_outlier:.2f}**")
            
            with col2:
                st.markdown("**📊 Normal Paper Characteristics:**")
                if len(normal_papers) > 0:
                    avg_citations_normal = normal_papers['cited_by_count'].mean()
                    avg_age_normal = normal_papers['paper_age'].mean()
                    st.markdown(f"- Average citations: **{avg_citations_normal:.1f}**")
                    st.markdown(f"- Average age: **{avg_age_normal:.1f} years**")
                    
                    if 'citations_per_year' in df.columns:
                        avg_cit_per_year_normal = normal_papers['citations_per_year'].mean()
                        st.markdown(f"- Citations per year: **{avg_cit_per_year_normal:.2f}**")
    else:
        st.info("Citation vs age analysis requires citation and age data.")

def show_saved_experiments_sidebar():
    """Fixed experiment selection with proper analyzer loading."""
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Saved Experiments")
    
    # Create loader instance
    loader = LocalResultsLoader()
    
    # Get experiments list
    experiments = loader.list_available_experiments()
    
    if experiments:
        # Create options for selectbox
        experiment_options = {
            f"{exp['experiment_name']} ({exp.get('n_papers', 'N/A')} papers)": exp['experiment_name']
            for exp in experiments
        }
        
        # Add empty option at the start
        display_options = ['Select an experiment...'] + list(experiment_options.keys())
        
        selected_display = st.sidebar.selectbox(
            "Load saved experiment:",
            options=display_options,
            key='saved_experiment_selector'
        )
        
        if selected_display != 'Select an experiment...':
            selected_experiment = experiment_options[selected_display]
            
            # Show experiment info
            exp_data = next(exp for exp in experiments if exp['experiment_name'] == selected_experiment)
            st.sidebar.markdown("**📊 Info:**")
            st.sidebar.markdown(f"- Papers: {exp_data.get('n_papers', 'N/A'):,}")
            st.sidebar.markdown(f"- Date: {exp_data.get('completion_time', 'N/A')[:10]}")
            st.sidebar.markdown(f"- Methods: {', '.join(exp_data.get('methods', []))}")
            
            # Load button
            if st.sidebar.button("🚀 Load Experiment", key="load_experiment_btn"):
                # Show loading state
                with st.spinner("Loading saved experiment..."):
                    
                    # Load experiment using the FIXED loader
                    analyzer = loader.load_experiment(selected_experiment)
                
                if analyzer:
                    # Test that the analyzer works properly
                    if test_loaded_analyzer(analyzer):
                        # Store in session state
                        st.session_state.loaded_analyzer = analyzer
                        st.session_state.experiment_loaded = True
                        st.session_state.experiment_name = selected_experiment
                        
                        # Clear any existing analysis
                        st.session_state.analysis_complete = False
                        
                        st.success(f"✅ Loaded: {selected_experiment}")
                        st.rerun()
                    else:
                        st.error("❌ Loaded analyzer is missing required functionality")
                else:
                    st.error("❌ Failed to load experiment")
            
            # Optional: Delete button
            if st.sidebar.button("🗑️ Delete Experiment", key="delete_experiment_btn"):
                if st.sidebar.button("⚠️ Confirm Delete", key="confirm_delete_btn"):
                    success, message = loader.delete_experiment(selected_experiment)
                    if success:
                        st.sidebar.success("✅ Experiment deleted")
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ {message}")
    else:
        st.sidebar.info("No saved experiments found")
        st.sidebar.markdown("💡 Run an analysis with auto-save to create experiments")
    
    # Option to run new analysis with persistence
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 New Analysis")
    
    if st.sidebar.button("🚀 Run New Analysis with Auto-Save", key="new_analysis_btn"):
        st.session_state.show_new_analysis = True
        st.rerun()

def show_new_analysis_interface():
    """Interface to run new analysis with auto-save."""
    st.header("🚀 Run New Analysis with Auto-Save")
    
    # Data input
    uploaded_file = st.file_uploader("Upload JSON data", type=['json'])
    experiment_name = st.text_input(
        "Experiment name (optional):",
        value=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    
    if uploaded_file and st.button("🔄 Run Complete Analysis"):
        # Save uploaded file temporarily
        with open("temp_data.json", "wb") as f:
            f.write(uploaded_file.read())
        
        # Run analysis with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing analyzer...")
            progress_bar.progress(10)
            
            analyzer = AcademicResearchAnalyzer()
            
            status_text.text("🧠 Running complete analysis with auto-save...")
            progress_bar.progress(30)
            
            # Use the new method with auto-save
            result_experiment = analyzer.perform_full_analysis_with_auto_save("temp_data.json", experiment_name)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed and saved!")
            
            if result_experiment:
                st.success(f"""
                🎉 **Analysis completed successfully!**
                
                - **Experiment:** {result_experiment}
                - **Papers analyzed:** {len(analyzer.df):,}
                - **Results saved and ready for instant loading!**
                """)
                
                # Clear the analysis interface
                st.session_state.show_new_analysis = False
                st.rerun()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def show_analysis_dashboard_fixed(dashboard):
    """
    Enhanced dashboard display that properly handles loaded experiments.
    """
    
    analyzer = dashboard.analyzer
    
    # Validate and repair analyzer if needed
    if not validate_analyzer_for_dashboard(analyzer):
        st.error("❌ Analyzer validation failed. Cannot display dashboard.")
        return
    
    analyzer = repair_loaded_analyzer(analyzer)
    df = analyzer.df
    
    # Overview metrics
    st.subheader("📊 Analysis Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📚 Total Papers", f"{len(df):,}")
    
    with col2:
        if hasattr(analyzer, 'cluster_labels') and analyzer.cluster_labels:
            # Get the number of clusters from any available method
            method = list(analyzer.cluster_labels.keys())[0]
            n_clusters = len(np.unique(analyzer.cluster_labels[method]))
            if -1 in analyzer.cluster_labels[method]:  # Remove noise cluster from count
                n_clusters -= 1
            st.metric("🎯 Research Communities", n_clusters)
        else:
            st.metric("🎯 Research Communities", "N/A")
    
    with col3:
        if 'cited_by_count' in df.columns:
            avg_citations = df['cited_by_count'].mean()
            st.metric("📈 Avg Citations", f"{avg_citations:.1f}")
        else:
            st.metric("📈 Avg Citations", "N/A")
    
    with col4:
        if 'outlier_consensus_score' in df.columns:
            outliers = (df['outlier_consensus_score'] > 0).sum()
            st.metric("⚡ Outlier Papers", f"{outliers:,}")
        else:
            st.metric("⚡ Outlier Papers", "N/A")
    
    with col5:
        if 'publication_year' in df.columns:
            year_range = f"{df['publication_year'].min():.0f}-{df['publication_year'].max():.0f}"
            st.metric("📅 Year Range", year_range)
        else:
            st.metric("📅 Year Range", "N/A")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Cluster Analysis", 
        "🔍 Dimensionality Reduction", 
        "⚡ Outlier Detection", 
        "📈 Advanced Analytics", 
        "📄 Research Explorer"
    ])
    
    # Tab 1: Cluster Analysis (using the fixed function)
    with tab1:
        show_cluster_analysis(dashboard)
    
    # Tab 2: Dimensionality Reduction
    with tab2:
        show_dimensionality_analysis_fixed(dashboard)
    
    # Tab 3: Outlier Detection  
    with tab3:
        show_outlier_analysis_fixed(dashboard)
    
    # Tab 4: Advanced Analytics
    with tab4:
        show_advanced_analytics(dashboard)
    
    # Tab 5: Research Explorer
    with tab5:
        show_research_explorer_fixed(dashboard)

def show_advanced_analytics(dashboard):
    """
    Display advanced analytics including TF-IDF analysis.
    """
    
    analyzer = dashboard.analyzer
    df = analyzer.df
    
    st.header("📈 Advanced Analytics")
    
    # Add TF-IDF Analysis Section
    st.subheader("📝 TF-IDF Text Analysis")
    
    if hasattr(analyzer, 'text_features') and hasattr(analyzer, 'tfidf_vectorizer'):
        # TF-IDF Matrix Overview
        tfidf_matrix = analyzer.text_features
        feature_names = analyzer.tfidf_vectorizer.get_feature_names_out()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", tfidf_matrix.shape[0])
        with col2:
            st.metric("Features", tfidf_matrix.shape[1])
        with col3:
            sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
            st.metric("Sparsity", f"{sparsity:.1f}%")
        with col4:
            st.metric("Non-zero Elements", f"{tfidf_matrix.nnz:,}")
        
        # TF-IDF Analysis Options
        analysis_type = st.selectbox(
            "Select TF-IDF Analysis:",
            ["Top Terms by Frequency", "Term-Document Heatmap", "Cluster-Specific Terms", "Document Similarity Matrix"]
        )
        
        if analysis_type == "Top Terms by Frequency":
            # Show top terms across all documents
            n_top_terms = st.slider("Number of top terms to display:", 10, 100, 20)
            
            # Calculate term frequencies
            term_frequencies = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = np.argsort(term_frequencies)[-n_top_terms:][::-1]
            
            top_terms_data = []
            for idx in top_indices:
                top_terms_data.append({
                    'Term': feature_names[idx],
                    'Total TF-IDF Score': term_frequencies[idx],
                    'Document Frequency': (tfidf_matrix[:, idx] > 0).sum()
                })
            
            top_terms_df = pd.DataFrame(top_terms_data)
            
            # Visualization
            fig = px.bar(
                top_terms_df.head(20), 
                x='Total TF-IDF Score', 
                y='Term',
                title=f"Top {min(20, n_top_terms)} Terms by TF-IDF Score",
                orientation='h'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(top_terms_df, use_container_width=True)
            
        elif analysis_type == "Term-Document Heatmap":
            # Show heatmap for top terms and sample documents
            st.info("Showing TF-IDF scores for top terms across sample documents")
            
            n_terms = st.slider("Number of top terms:", 10, 50, 20)
            n_docs = st.slider("Number of sample documents:", 10, 100, 20)
            
            # Get top terms
            term_frequencies = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_term_indices = np.argsort(term_frequencies)[-n_terms:][::-1]
            
            # Sample documents
            sample_doc_indices = np.random.choice(tfidf_matrix.shape[0], min(n_docs, tfidf_matrix.shape[0]), replace=False)
            
            # Create heatmap data
            heatmap_data = tfidf_matrix[sample_doc_indices][:, top_term_indices].toarray()
            
            # Create figure with document titles
            doc_titles = []
            for i in sample_doc_indices:
                if 'title' in df.columns:
                    title = str(df.iloc[i]['title'])
                    # Truncate long titles for better display
                    if len(title) > 40:
                        title = title[:37] + "..."
                    doc_titles.append(title)
                else:
                    doc_titles.append(f"Doc {i}")
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[feature_names[i] for i in top_term_indices],
                y=doc_titles,
                colorscale='Viridis',
                hovertemplate='Term: %{x}<br>Document: %{y}<br>TF-IDF: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"TF-IDF Heatmap: Top {n_terms} Terms vs {len(sample_doc_indices)} Sample Documents",
                xaxis_title="Terms",
                yaxis_title="Document Titles",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Cluster-Specific Terms" and 'cluster_kmeans' in df.columns:
            # Show top terms for each cluster
            st.info("Analyzing characteristic terms for each research cluster")
            
            n_terms_per_cluster = st.slider("Terms per cluster:", 5, 20, 10)
            
            clusters = sorted(df['cluster_kmeans'].unique())
            cluster_terms_data = []
            
            for cluster_id in clusters:
                cluster_docs = df[df['cluster_kmeans'] == cluster_id].index
                
                if len(cluster_docs) > 0:
                    # Calculate mean TF-IDF for this cluster
                    cluster_tfidf = tfidf_matrix[cluster_docs].mean(axis=0).A1
                    top_indices = np.argsort(cluster_tfidf)[-n_terms_per_cluster:][::-1]
                    
                    for idx in top_indices:
                        cluster_terms_data.append({
                            'Cluster': f'Cluster {cluster_id}',
                            'Term': feature_names[idx],
                            'Mean TF-IDF': cluster_tfidf[idx],
                            'Cluster Size': len(cluster_docs)
                        })
            
            cluster_terms_df = pd.DataFrame(cluster_terms_data)
            
            # Create subplot for each cluster
            fig = px.bar(
                cluster_terms_df, 
                x='Mean TF-IDF', 
                y='Term',
                facet_col='Cluster',
                facet_col_wrap=3,
                title="Top Terms by Cluster",
                orientation='h'
            )
            
            fig.update_layout(height=800)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table
            st.subheader("Detailed Cluster Terms")
            selected_cluster = st.selectbox("Select cluster for detailed view:", clusters)
            cluster_data = cluster_terms_df[cluster_terms_df['Cluster'] == f'Cluster {selected_cluster}']
            st.dataframe(cluster_data, use_container_width=True)
            
        elif analysis_type == "Document Similarity Matrix":
            # Show document similarity based on TF-IDF
            st.info("Computing cosine similarity between documents based on TF-IDF vectors")
            
            n_docs = st.slider("Number of documents for similarity analysis:", 10, 100, 20)
            
            # Sample documents for similarity analysis
            sample_indices = np.random.choice(tfidf_matrix.shape[0], min(n_docs, tfidf_matrix.shape[0]), replace=False)
            sample_matrix = tfidf_matrix[sample_indices]
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(sample_matrix)
            
            # Create document titles for similarity matrix
            doc_titles = []
            for i in sample_indices:
                if 'title' in df.columns:
                    title = str(df.iloc[i]['title'])
                    # Truncate long titles for better display
                    if len(title) > 30:
                        title = title[:27] + "..."
                    doc_titles.append(title)
                else:
                    doc_titles.append(f"Doc {i}")
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=doc_titles,
                y=doc_titles,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Document Similarity Matrix (Cosine Similarity)",
                xaxis_title="Document Titles",
                yaxis_title="Document Titles",
                height=600,
                xaxis={'tickangle': 45},  # Rotate x-axis labels for better readability
                yaxis={'tickangle': 0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show most similar document pairs with actual titles
            st.subheader("Most Similar Document Pairs")
            
            # Get upper triangle of similarity matrix (excluding diagonal)
            similarities = []
            
            for i in range(len(sample_indices)):
                for j in range(i+1, len(sample_indices)):
                    doc1_title = df.iloc[sample_indices[i]]['title'] if 'title' in df.columns else f"Document {sample_indices[i]}"
                    doc2_title = df.iloc[sample_indices[j]]['title'] if 'title' in df.columns else f"Document {sample_indices[j]}"
                    
                    similarities.append({
                        'Doc 1 Index': sample_indices[i],
                        'Doc 2 Index': sample_indices[j],
                        'Similarity': similarity_matrix[i, j],
                        'Document 1 Title': doc1_title[:80] + "..." if len(str(doc1_title)) > 80 else str(doc1_title),
                        'Document 2 Title': doc2_title[:80] + "..." if len(str(doc2_title)) > 80 else str(doc2_title)
                    })
            
            similarities_df = pd.DataFrame(similarities).sort_values('Similarity', ascending=False)
            
            # Display with better column names
            display_similarities = similarities_df[['Document 1 Title', 'Document 2 Title', 'Similarity']].head(10)
            st.dataframe(display_similarities, use_container_width=True)
    
    else:
        st.warning("TF-IDF analysis not available. Please ensure text features are computed during analysis.")
    
    # Temporal analysis
    st.subheader("⏰ Temporal Research Evolution")
    
    if 'cluster_kmeans' in df.columns and 'publication_year' in df.columns:
        # Cluster evolution over time
        temporal_data = []
        years = sorted(df['publication_year'].unique())
        clusters = sorted(df['cluster_kmeans'].unique())
        
        for year in years:
            year_papers = df[df['publication_year'] == year]
            total_year_papers = len(year_papers)
            
            for cluster in clusters:
                cluster_count = len(year_papers[year_papers['cluster_kmeans'] == cluster])
                
                temporal_data.append({
                    'Year': year,
                    'Cluster': f'Cluster {cluster}',
                    'Count': cluster_count
                })
        
        temporal_df = pd.DataFrame(temporal_data)
        
        # Plot temporal evolution (absolute counts only)
        fig = px.line(
            temporal_df, 
            x='Year', 
            y='Count', 
            color='Cluster',
            title="Research Community Evolution (Absolute Counts)",
            markers=True
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Temporal analysis requires clustering results and publication year data.")
    
    # Feature Statistics
    st.subheader("📊 Feature Statistics")
    
    if hasattr(analyzer, 'numerical_features') and analyzer.numerical_features:
        feature_stats = []
        for feature in analyzer.numerical_features:
            if feature in df.columns:
                feature_stats.append({
                    'Feature': feature,
                    'Mean': df[feature].mean(),
                    'Std': df[feature].std(),
                    'Min': df[feature].min(),
                    'Max': df[feature].max()
                })
        
        if feature_stats:
            stats_df = pd.DataFrame(feature_stats)
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("Feature statistics not available.")

def show_research_explorer_fixed(dashboard):
    """
    Fixed research explorer that works with loaded experiments.
    """
    st.header("📄 Research Explorer")
    
    analyzer = dashboard.analyzer
    df = analyzer.df
    
    # Search and filter interface
    st.subheader("🔍 Search & Filter Papers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text search
        search_term = st.text_input("Search in titles:", placeholder="Enter keywords...")
        
        # Citation filter
        if 'cited_by_count' in df.columns:
            min_citations = st.number_input(
                "Minimum citations:", 
                min_value=0, 
                max_value=int(df['cited_by_count'].max()), 
                value=0
            )
        else:
            min_citations = 0
    
    with col2:
        # Year filter
        if 'publication_year' in df.columns:
            year_range = st.slider(
                "Publication year range:",
                min_value=int(df['publication_year'].min()),
                max_value=int(df['publication_year'].max()),
                value=(int(df['publication_year'].min()), int(df['publication_year'].max()))
            )
        else:
            year_range = None
        
        # Cluster filter
        if hasattr(analyzer, 'cluster_labels') and analyzer.cluster_labels:
            available_methods = list(analyzer.cluster_labels.keys())
            selected_method = st.selectbox("Filter by cluster (method):", ['All'] + available_methods)
            
            if selected_method != 'All':
                cluster_column = f'cluster_{selected_method}'
                if cluster_column in df.columns:
                    unique_clusters = sorted(df[cluster_column].unique())
                    selected_cluster = st.selectbox("Select cluster:", ['All'] + [str(c) for c in unique_clusters])
                else:
                    selected_cluster = 'All'
            else:
                selected_cluster = 'All'
        else:
            selected_method = 'All'
            selected_cluster = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term and 'title' in df.columns:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
    
    if 'cited_by_count' in df.columns:
        filtered_df = filtered_df[filtered_df['cited_by_count'] >= min_citations]
    
    if year_range and 'publication_year' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['publication_year'] >= year_range[0]) & 
            (filtered_df['publication_year'] <= year_range[1])
        ]
    
    if selected_method != 'All' and selected_cluster != 'All':
        cluster_column = f'cluster_{selected_method}'
        if cluster_column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[cluster_column] == int(selected_cluster)]
    
    # Display results
    st.subheader(f"📋 Results ({len(filtered_df):,} papers)")
    
    if len(filtered_df) > 0:
        # Show top papers
        display_columns = ['title']
        if 'cited_by_count' in filtered_df.columns:
            display_columns.append('cited_by_count')
        if 'publication_year' in filtered_df.columns:
            display_columns.append('publication_year')
        
        # Sort by citations if available
        if 'cited_by_count' in filtered_df.columns:
            display_df = filtered_df.nlargest(50, 'cited_by_count')[display_columns]
        else:
            display_df = filtered_df.head(50)[display_columns]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Summary statistics
        if len(filtered_df) > 1:
            st.subheader("📊 Filtered Data Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'cited_by_count' in filtered_df.columns:
                    avg_citations = filtered_df['cited_by_count'].mean()
                    st.metric("Average Citations", f"{avg_citations:.1f}")
            
            with col2:
                if 'publication_year' in filtered_df.columns:
                    median_year = filtered_df['publication_year'].median()
                    st.metric("Median Year", f"{median_year:.0f}")
            
            with col3:
                st.metric("Total Papers", f"{len(filtered_df):,}")
    else:
        st.info("No papers match the current filters. Try adjusting your search criteria.")

def main():
    """
    Main Streamlit app function with complete fixed functionality.
    """
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Academic Research Landscape Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Unsupervised Machine Learning Analysis Dashboard**
    
    Explore research communities, discover breakthrough papers, and analyze academic trends 
    using advanced clustering and outlier detection techniques.
    """)
    
    # Initialize dashboard
    dashboard = StreamlitDashboard()
    
    # Sidebar for data loading and controls
    with st.sidebar:
        st.header("🔧 Analysis Controls")
        
        # Data loading options
        st.subheader("📁 Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload New Data", "Use Sample Data"]
        )
        
        if data_source == "Upload New Data":
            uploaded_file = st.file_uploader(
                "Upload your OpenAlex JSON data",
                type=['json', 'csv'],
                help="Upload the JSON file from your Azure Functions pipeline"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_data.json", "wb") as f:
                    f.write(uploaded_file.read())
                
                if st.button("🚀 Run Analysis"):
                    success = run_new_analysis_simple(dashboard, "temp_data.json")
                    if success:
                        st.rerun()  # Refresh the app to show results
        
        elif data_source == "Use Sample Data":
            if st.button("📊 Generate Sample Data"):
                st.info("Sample data generation not available in this demo.")
        
        # Add a reset button
        if st.session_state.analysis_complete or st.session_state.experiment_loaded:
            st.markdown("---")
            if st.button("🔄 Reset Analysis", type="secondary"):
                # Clear session state
                for key in ['analyzer', 'viz_suite', 'advanced_suite', 'analysis_complete', 
                           'data_loaded', 'loaded_analyzer', 'experiment_loaded', 'experiment_name']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Add saved experiments to sidebar
    show_saved_experiments_sidebar()
    
    # Main dashboard content
    # Check if we have a loaded experiment from persistence
    if st.session_state.experiment_loaded and st.session_state.loaded_analyzer:
        # Show dashboard with loaded data
        analyzer = st.session_state.loaded_analyzer
        
        # Success message
        st.success(f"📊 **Experiment loaded: {st.session_state.get('experiment_name', 'Unknown')}** - All graphs ready instantly!")
        
        # Create mock dashboard object for existing functions
        class MockDashboard:
            def __init__(self, analyzer):
                self.analyzer = analyzer
                try:
                    self.viz_suite = ResearchVisualizationSuite(analyzer)
                    self.advanced_suite = AdvancedAnalysisSuite(analyzer)
                except Exception as e:
                    st.warning(f"Visualization suites could not be initialized: {e}")
                    self.viz_suite = None
                    self.advanced_suite = None
        
        dashboard = MockDashboard(analyzer)
        show_analysis_dashboard_fixed(dashboard)
        
    elif st.session_state.analysis_complete:
        # Show dashboard with current analysis
        show_analysis_dashboard_fixed(dashboard)
        
    elif getattr(st.session_state, 'show_new_analysis', False):
        # Show new analysis interface
        show_new_analysis_interface()
        
    else:
        # Show welcome screen
        st.info("👆 Please select a data source from the sidebar to begin analysis, or load a saved experiment.")
        
        # Show example visualizations or instructions
        st.subheader("🎯 What You'll Discover")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Research Communities**
            - Identify distinct research clusters
            - Analyze community characteristics
            - Track evolution over time
            """)
        
        with col2:
            st.markdown("""
            **⚡ Breakthrough Papers**
            - Detect high-impact outliers
            - Find undervalued research
            - Predict future trends
            """)
        
        with col3:
            st.markdown("""
            **📈 Advanced Analytics**
            - Interactive visualizations
            - Network analysis
            - Temporal patterns
            """)

def run_new_analysis_simple(dashboard, data_path):
    """
    Simplified new analysis runner for the dashboard.
    """
    try:
        with st.spinner("🔄 Running unsupervised learning analysis..."):
            # Initialize analyzer
            dashboard.analyzer = AcademicResearchAnalyzer(random_state=42)
            
            # Load and prepare data
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📊 Loading and preprocessing data...")
            df = dashboard.analyzer.load_and_prepare_data(data_path)
            progress_bar.progress(20)
            
            # Sample data if too large for real-time analysis
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                dashboard.analyzer.df = df
                st.info(f"📝 Using sample of 10,000 papers for real-time analysis")
            
            status_text.text("🔧 Scaling features...")
            dashboard.analyzer.scale_features(method='robust')
            progress_bar.progress(40)
            
            status_text.text("🎯 Performing dimensionality reduction...")
            dashboard.analyzer.perform_dimensionality_reduction()
            progress_bar.progress(60)
            
            status_text.text("🔍 Clustering analysis...")
            optimal_clusters = dashboard.analyzer.analyze_optimal_clusters(max_clusters=12)
            dashboard.analyzer.perform_clustering_analysis(n_clusters=optimal_clusters)
            progress_bar.progress(80)
            
            status_text.text("⚡ Detecting outliers...")
            dashboard.analyzer.detect_outliers()
            dashboard.analyzer.create_cluster_profiles(method='kmeans')
            progress_bar.progress(100)
            
            # Initialize suites
            try:
                dashboard.viz_suite = ResearchVisualizationSuite(dashboard.analyzer)
                dashboard.advanced_suite = AdvancedAnalysisSuite(dashboard.analyzer)
            except:
                pass  # Suites are optional
            
            status_text.text("✅ Analysis complete!")
            
            # Update session state
            st.session_state.analysis_complete = True
            st.session_state.data_loaded = True
            
            st.success(f"🎉 Analysis completed successfully! Analyzed {len(df):,} papers")
            return True
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return False

if __name__ == "__main__":
    main()