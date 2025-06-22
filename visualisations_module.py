"""
Visualization and Advanced Analysis Module
========================================

This module provides comprehensive visualization and advanced analysis
capabilities for the academic research clustering project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples
import umap

class ResearchVisualizationSuite:
    """
    Comprehensive visualization suite for academic research analysis.
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
        
    def plot_dimensionality_analysis(self, save_path=None):
        """
        Create comprehensive dimensionality reduction visualizations.
        Similar to PCA analysis in the house price project.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. PCA Explained Variance
        pca_results = self.analyzer.dimensionality_results['pca']
        explained_var = pca_results['explained_variance_ratio']
        cumulative_var = pca_results['cumulative_variance']
        
        axes[0, 0].bar(range(1, min(11, len(explained_var) + 1)), explained_var[:10])
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA: Explained Variance by Component')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA Cumulative Variance
        axes[0, 1].plot(range(1, min(21, len(cumulative_var) + 1)), cumulative_var[:20], 'bo-')
        axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
        axes[0, 1].axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('PCA: Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PCA 2D Visualization with citations
        pca_2d = pca_results['transformed_data'][:, :2]
        scatter = axes[0, 2].scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                   c=self.df['cited_by_count'], 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[0, 2].set_xlabel('First Principal Component')
        axes[0, 2].set_ylabel('Second Principal Component')
        axes[0, 2].set_title('PCA: 2D Projection (colored by citations)')
        plt.colorbar(scatter, ax=axes[0, 2], label='Citation Count')
        
        # 4. UMAP Visualization
        umap_2d = self.analyzer.dimensionality_results['umap']['transformed_data']
        scatter = axes[1, 0].scatter(umap_2d[:, 0], umap_2d[:, 1], 
                                   c=self.df['cited_by_count'], 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[1, 0].set_xlabel('UMAP Dimension 1')
        axes[1, 0].set_ylabel('UMAP Dimension 2')
        axes[1, 0].set_title('UMAP: 2D Projection (colored by citations)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Citation Count')
        
        # 5. t-SNE Visualization
        tsne_2d = self.analyzer.dimensionality_results['tsne']['transformed_data']
        scatter = axes[1, 1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], 
                                   c=self.df['cited_by_count'], 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[1, 1].set_xlabel('t-SNE Dimension 1')
        axes[1, 1].set_ylabel('t-SNE Dimension 2')
        axes[1, 1].set_title('t-SNE: 2D Projection (colored by citations)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Citation Count')
        
        # 6. Comparison of methods with clusters (if available)
        if 'kmeans' in self.analyzer.cluster_labels:
            cluster_labels = self.analyzer.cluster_labels['kmeans']
            scatter = axes[1, 2].scatter(umap_2d[:, 0], umap_2d[:, 1], 
                                       c=cluster_labels, 
                                       cmap='tab10', alpha=0.7, s=20)
            axes[1, 2].set_xlabel('UMAP Dimension 1')
            axes[1, 2].set_ylabel('UMAP Dimension 2')
            axes[1, 2].set_title('UMAP with K-Means Clusters')
        else:
            axes[1, 2].text(0.5, 0.5, 'Clusters not computed yet', 
                          transform=axes[1, 2].transAxes, ha='center', va='center')
            axes[1, 2].set_title('Clustering Analysis Pending')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clustering_analysis(self, save_path=None):
        """
        Create comprehensive clustering analysis visualizations.
        """
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Elbow Analysis', 'Silhouette Analysis', 'Davies-Bouldin Score',
                'K-Means Clusters', 'Hierarchical Clusters', 'GMM Clusters',
                'DBSCAN Clusters', 'Cluster Size Distribution', 'Silhouette Plot'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get cluster analysis results
        results = self.analyzer.cluster_analysis_results
        cluster_range = results['cluster_range']
        
        # 1. Elbow Analysis
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['inertias'], 
                      mode='lines+markers', name='Inertia',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Silhouette Analysis
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['silhouette_scores'], 
                      mode='lines+markers', name='Silhouette Score',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # 3. Davies-Bouldin Score
        fig.add_trace(
            go.Scatter(x=cluster_range, y=results['davies_bouldin_scores'], 
                      mode='lines+markers', name='Davies-Bouldin Score',
                      line=dict(color='red', width=2)),
            row=1, col=3
        )
        
        # Get UMAP coordinates for cluster visualization
        umap_2d = self.analyzer.dimensionality_results['umap']['transformed_data']
        
        # 4-7. Cluster visualizations
        cluster_methods = ['kmeans', 'hierarchical', 'gmm', 'dbscan']
        positions = [(2, 1), (2, 2), (2, 3), (3, 1)]
        
        for method, (row, col) in zip(cluster_methods, positions):
            if method in self.analyzer.cluster_labels:
                labels = self.analyzer.cluster_labels[method]
                
                # Create scatter plot for each cluster
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    label_name = f'Cluster {label}' if label != -1 else 'Noise'
                    color = 'black' if label == -1 else None
                    
                    fig.add_trace(
                        go.Scatter(x=umap_2d[mask, 0], y=umap_2d[mask, 1],
                                  mode='markers', name=label_name,
                                  marker=dict(size=4, color=color),
                                  showlegend=False),
                        row=row, col=col
                    )
        
        # 8. Cluster Size Distribution
        if 'kmeans' in self.analyzer.cluster_labels:
            labels = self.analyzer.cluster_labels['kmeans']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            fig.add_trace(
                go.Bar(x=[f'Cluster {i}' for i in unique_labels], y=counts,
                       name='Cluster Sizes', showlegend=False),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Clustering Analysis",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=3)
        
        for row, col in positions:
            fig.update_xaxes(title_text="UMAP Dimension 1", row=row, col=col)
            fig.update_yaxes(title_text="UMAP Dimension 2", row=row, col=col)
        
        fig.update_xaxes(title_text="Clusters", row=3, col=2)
        fig.update_yaxes(title_text="Number of Papers", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def plot_outlier_analysis(self, save_path=None):
        """
        Create comprehensive outlier analysis visualizations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Get UMAP coordinates
        umap_2d = self.analyzer.dimensionality_results['umap']['transformed_data']
        
        # 1. Isolation Forest Outliers
        if 'isolation_forest' in self.analyzer.outlier_labels:
            outliers = self.analyzer.outlier_labels['isolation_forest'] == -1
            axes[0, 0].scatter(umap_2d[~outliers, 0], umap_2d[~outliers, 1], 
                             c='blue', alpha=0.6, s=10, label='Normal')
            axes[0, 0].scatter(umap_2d[outliers, 0], umap_2d[outliers, 1], 
                             c='red', alpha=0.8, s=30, label='Outlier')
            axes[0, 0].set_title('Isolation Forest Outliers')
            axes[0, 0].legend()
        
        # 2. LOF Outliers
        if 'lof' in self.analyzer.outlier_labels:
            outliers = self.analyzer.outlier_labels['lof'] == -1
            axes[0, 1].scatter(umap_2d[~outliers, 0], umap_2d[~outliers, 1], 
                             c='blue', alpha=0.6, s=10, label='Normal')
            axes[0, 1].scatter(umap_2d[outliers, 0], umap_2d[outliers, 1], 
                             c='red', alpha=0.8, s=30, label='Outlier')
            axes[0, 1].set_title('Local Outlier Factor')
            axes[0, 1].legend()
        
        # 3. Citation-based Outliers
        if 'high_impact_outlier' in self.df.columns:
            high_impact = self.df['high_impact_outlier']
            undervalued = self.df['undervalued_outlier']
            normal = ~(high_impact | undervalued)
            
            axes[0, 2].scatter(umap_2d[normal, 0], umap_2d[normal, 1], 
                             c='gray', alpha=0.5, s=10, label='Normal')
            axes[0, 2].scatter(umap_2d[high_impact, 0], umap_2d[high_impact, 1], 
                             c='gold', alpha=0.8, s=30, label='High Impact')
            axes[0, 2].scatter(umap_2d[undervalued, 0], umap_2d[undervalued, 1], 
                             c='purple', alpha=0.8, s=30, label='Undervalued')
            axes[0, 2].set_title('Citation-based Outliers')
            axes[0, 2].legend()
        
        # 4. Outlier Consensus
        if 'outlier_consensus_score' in self.df.columns:
            consensus_scores = self.df['outlier_consensus_score']
            scatter = axes[1, 0].scatter(umap_2d[:, 0], umap_2d[:, 1], 
                                       c=consensus_scores, cmap='Reds', 
                                       alpha=0.7, s=20)
            axes[1, 0].set_title('Outlier Consensus Score')
            plt.colorbar(scatter, ax=axes[1, 0], label='Number of Methods')
        
        # 5. Citation vs Age scatter with outliers
        outlier_mask = self.df.get('is_strong_outlier', pd.Series([False] * len(self.df)))
        axes[1, 1].scatter(self.df.loc[~outlier_mask, 'paper_age'], 
                          self.df.loc[~outlier_mask, 'cited_by_count'],
                          alpha=0.6, s=10, label='Normal', color='blue')
        axes[1, 1].scatter(self.df.loc[outlier_mask, 'paper_age'], 
                          self.df.loc[outlier_mask, 'cited_by_count'],
                          alpha=0.8, s=30, label='Strong Outlier', color='red')
        axes[1, 1].set_xlabel('Paper Age (years)')
        axes[1, 1].set_ylabel('Citation Count')
        axes[1, 1].set_title('Citations vs Age with Outliers')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        # 6. Outlier Method Comparison
        if len(self.analyzer.outlier_labels) > 1:
            method_names = list(self.analyzer.outlier_labels.keys())
            outlier_counts = [
                (self.analyzer.outlier_labels[method] == -1).sum() 
                for method in method_names
            ]
            
            axes[1, 2].bar(method_names, outlier_counts)
            axes[1, 2].set_xlabel('Detection Method')
            axes[1, 2].set_ylabel('Number of Outliers')
            axes[1, 2].set_title('Outliers by Detection Method')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cluster_heatmap(self, method='kmeans', save_path=None):
        """
        Create a heatmap showing cluster characteristics.
        Similar to the biplot analysis in the house price project.
        """
        if method not in self.analyzer.cluster_labels:
            raise ValueError(f"Method {method} not found in cluster labels")
        
        # Get cluster profiles
        if not hasattr(self.analyzer, 'cluster_profiles'):
            self.analyzer.create_cluster_profiles(method)
        
        profiles = self.analyzer.cluster_profiles
        
        # Create feature matrix for heatmap
        features = self.analyzer.numerical_features
        cluster_ids = sorted(profiles.keys())
        
        heatmap_data = []
        for cluster_id in cluster_ids:
            row = []
            for feature in features:
                if feature in profiles[cluster_id]['numerical_features']:
                    # Use standardized values (z-score relative to overall mean)
                    cluster_mean = profiles[cluster_id]['numerical_features'][feature]['mean']
                    overall_mean = self.df[feature].mean()
                    overall_std = self.df[feature].std()
                    z_score = (cluster_mean - overall_mean) / (overall_std + 1e-8)
                    row.append(z_score)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=[f'Cluster {i}' for i in cluster_ids],
                                 columns=features)
        
        sns.heatmap(heatmap_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.2f', ax=ax, cbar_kws={'label': 'Z-Score'})
        
        ax.set_title(f'Cluster Characteristics Heatmap ({method.upper()})')
        ax.set_xlabel('Features')
        ax.set_ylabel('Clusters')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hierarchical_dendrogram(self, sample_size=1000, save_path=None):
        """
        Create hierarchical clustering dendrogram.
        Similar to the dendrogram analysis in the house price project.
        """
        # Sample data for computational efficiency
        if len(self.analyzer.features_scaled) > sample_size:
            indices = np.random.choice(len(self.analyzer.features_scaled), 
                                     sample_size, replace=False)
            sample_data = self.analyzer.features_scaled[indices]
        else:
            sample_data = self.analyzer.features_scaled
        
        # Compute linkage matrix
        linkage_matrix = linkage(sample_data, method='ward')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(15, 8))
        
        dendrogram(linkage_matrix, ax=ax, leaf_rotation=90, leaf_font_size=8)
        
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_cluster_explorer(self, method='kmeans'):
        """
        Create an interactive plot for exploring clusters.
        """
        if method not in self.analyzer.cluster_labels:
            raise ValueError(f"Method {method} not found")
        
        # Get data for visualization
        umap_2d = self.analyzer.dimensionality_results['umap']['transformed_data']
        cluster_labels = self.analyzer.cluster_labels[method]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'UMAP_1': umap_2d[:, 0],
            'UMAP_2': umap_2d[:, 1],
            'Cluster': cluster_labels,
            'Title': self.df['title'].str[:60] + '...',
            'Citations': self.df['cited_by_count'],
            'Year': self.df['publication_year'],
            'Citations_per_Year': self.df['citations_per_year']
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            plot_df, 
            x='UMAP_1', 
            y='UMAP_2', 
            color='Cluster',
            size='Citations',
            hover_data=['Title', 'Citations', 'Year'],
            title=f'Interactive Cluster Explorer ({method.upper()})',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            hovermode='closest'
        )
        
        fig.show()
        
        return fig

class AdvancedAnalysisSuite:
    """
    Advanced analysis capabilities for deeper insights.
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
    
    def temporal_analysis(self):
        """
        Analyze how research communities evolve over time.
        """
        if 'kmeans' not in self.analyzer.cluster_labels:
            print("Clustering must be performed first")
            return
        
        cluster_labels = self.analyzer.cluster_labels['kmeans']
        
        # Create temporal analysis
        temporal_data = []
        
        years = sorted(self.df['publication_year'].unique())
        clusters = sorted(np.unique(cluster_labels))
        
        for year in years:
            year_mask = self.df['publication_year'] == year
            year_clusters = cluster_labels[year_mask]
            
            for cluster in clusters:
                cluster_count = (year_clusters == cluster).sum()
                total_year_papers = year_mask.sum()
                
                temporal_data.append({
                    'year': year,
                    'cluster': cluster,
                    'count': cluster_count,
                    'percentage': (cluster_count / total_year_papers) * 100 if total_year_papers > 0 else 0
                })
        
        temporal_df = pd.DataFrame(temporal_data)
        
        # Plot temporal evolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Absolute counts
        for cluster in clusters:
            cluster_data = temporal_df[temporal_df['cluster'] == cluster]
            ax1.plot(cluster_data['year'], cluster_data['count'], 
                    marker='o', label=f'Cluster {cluster}')
        
        ax1.set_xlabel('Publication Year')
        ax1.set_ylabel('Number of Papers')
        ax1.set_title('Cluster Evolution Over Time (Absolute Counts)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Percentage
        for cluster in clusters:
            cluster_data = temporal_df[temporal_df['cluster'] == cluster]
            ax2.plot(cluster_data['year'], cluster_data['percentage'], 
                    marker='o', label=f'Cluster {cluster}')
        
        ax2.set_xlabel('Publication Year')
        ax2.set_ylabel('Percentage of Papers')
        ax2.set_title('Cluster Evolution Over Time (Percentage)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return temporal_df
    
    def citation_network_analysis(self, max_papers=1000):
        """
        Analyze citation networks within and between clusters.
        """
        # This would require citation relationship data
        # For now, we'll create a placeholder for the structure
        print("Citation network analysis requires citation relationship data")
        print("This would analyze:")
        print("- Inter-cluster citation patterns")
        print("- Network centrality measures")
        print("- Community detection in citation networks")
        print("- Identification of bridge papers between communities")
        
        # Placeholder for actual implementation
        return None
    
    def identify_breakthrough_papers(self, method='combined'):
        """
        Identify potentially breakthrough papers using multiple criteria.
        """
        breakthrough_scores = pd.DataFrame(index=self.df.index)
        
        # 1. High citation velocity (citations per year)
        citation_velocity_score = (
            self.df['citations_per_year'] - self.df['citations_per_year'].mean()
        ) / self.df['citations_per_year'].std()
        
        # 2. Outlier consensus score
        if 'outlier_consensus_score' in self.df.columns:
            outlier_score = self.df['outlier_consensus_score'] / self.df['outlier_consensus_score'].max()
        else:
            outlier_score = pd.Series(0, index=self.df.index)
        
        # 3. Cross-cluster influence (papers that might bridge communities)
        # This would require more sophisticated analysis
        cross_cluster_score = pd.Series(0, index=self.df.index)
        
        # 4. Recent papers with unusual citation patterns
        recent_mask = self.df['paper_age'] <= 3
        recent_citation_score = pd.Series(0, index=self.df.index)
        if recent_mask.any():
            recent_citations = self.df.loc[recent_mask, 'cited_by_count']
            recent_percentile = recent_citations.rank(pct=True)
            recent_citation_score.loc[recent_mask] = recent_percentile
        
        # Combine scores
        breakthrough_scores['citation_velocity'] = citation_velocity_score
        breakthrough_scores['outlier_score'] = outlier_score
        breakthrough_scores['cross_cluster_score'] = cross_cluster_score
        breakthrough_scores['recent_citation_score'] = recent_citation_score
        
        # Combined score
        weights = {
            'citation_velocity': 0.4,
            'outlier_score': 0.3,
            'cross_cluster_score': 0.1,
            'recent_citation_score': 0.2
        }
        
        breakthrough_scores['combined_score'] = sum(
            breakthrough_scores[col] * weight 
            for col, weight in weights.items()
        )
        
        # Rank papers
        breakthrough_scores['rank'] = breakthrough_scores['combined_score'].rank(ascending=False)
        
        # Add paper information
        result_df = pd.concat([
            self.df[['title', 'cited_by_count', 'publication_year', 'citations_per_year']],
            breakthrough_scores
        ], axis=1)
        
        # Get top breakthrough candidates
        top_breakthrough = result_df.nsmallest(20, 'rank')
        
        print("Top 10 Potential Breakthrough Papers:")
        print("=" * 60)
        for idx, (_, paper) in enumerate(top_breakthrough.head(10).iterrows(), 1):
            print(f"{idx}. {paper['title'][:60]}...")
            print(f"   Year: {paper['publication_year']}, Citations: {paper['cited_by_count']}, "
                  f"Score: {paper['combined_score']:.3f}")
            print()
        
        return top_breakthrough
    
    def generate_research_report(self, output_path=None):
        """
        Generate a comprehensive research report.
        """
        report = []
        
        # Header
        report.append("ACADEMIC RESEARCH LANDSCAPE ANALYSIS")
        report.append("=" * 50)
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        report.append(f"Total Papers Analyzed: {len(self.df):,}")
        report.append("")
        
        # Data Overview
        report.append("DATA OVERVIEW")
        report.append("-" * 20)
        report.append(f"Publication Years: {self.df['publication_year'].min():.0f} - {self.df['publication_year'].max():.0f}")
        report.append(f"Average Citation Count: {self.df['cited_by_count'].mean():.1f}")
        report.append(f"Average Citations per Year: {self.df['citations_per_year'].mean():.2f}")
        report.append(f"Average Paper Age: {self.df['paper_age'].mean():.1f} years")
        report.append("")
        
        # Clustering Results
        if hasattr(self.analyzer, 'clustering_metrics'):
            report.append("CLUSTERING ANALYSIS RESULTS")
            report.append("-" * 30)
            for method, metrics in self.analyzer.clustering_metrics.items():
                report.append(f"{method.upper()}:")
                report.append(f"  Number of Clusters: {metrics['n_clusters']}")
                report.append(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
                report.append(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
                if metrics['n_noise'] > 0:
                    report.append(f"  Noise Points: {metrics['n_noise']}")
                report.append("")
        
        # Cluster Profiles
        if hasattr(self.analyzer, 'cluster_profiles'):
            report.append("CLUSTER PROFILES")
            report.append("-" * 20)
            for cluster_id, profile in self.analyzer.cluster_profiles.items():
                report.append(f"Cluster {cluster_id} ({profile['size']} papers, {profile['percentage']:.1f}%):")
                
                # Key characteristics
                if 'cited_by_count' in profile['numerical_features']:
                    avg_citations = profile['numerical_features']['cited_by_count']['mean']
                    report.append(f"  Average Citations: {avg_citations:.1f}")
                
                if 'citations_per_year' in profile['numerical_features']:
                    avg_cit_per_year = profile['numerical_features']['citations_per_year']['mean']
                    report.append(f"  Average Citations/Year: {avg_cit_per_year:.2f}")
                
                # Top terms
                if profile['top_terms']:
                    top_terms = [term for term, score in profile['top_terms'][:5]]
                    report.append(f"  Key Terms: {', '.join(top_terms)}")
                
                # Example paper
                if profile['example_papers']:
                    example = profile['example_papers'][0]
                    report.append(f"  Example: {example['title'][:50]}... ({example['cited_by_count']} citations)")
                
                report.append("")
        
        # Outlier Analysis
        if 'outlier_consensus_score' in self.df.columns:
            strong_outliers = self.df['is_strong_outlier'].sum()
            moderate_outliers = self.df['is_moderate_outlier'].sum()
            
            report.append("OUTLIER DETECTION RESULTS")
            report.append("-" * 25)
            report.append(f"Strong Outliers (3+ methods): {strong_outliers}")
            report.append(f"Moderate Outliers (2 methods): {moderate_outliers}")
            report.append(f"Total Outlier Rate: {((strong_outliers + moderate_outliers) / len(self.df) * 100):.1f}%")
            report.append("")
        
        # Recommendations
        report.append("KEY FINDINGS & RECOMMENDATIONS")
        report.append("-" * 35)
        report.append("1. Research Community Structure:")
        if hasattr(self.analyzer, 'cluster_profiles'):
            n_clusters = len(self.analyzer.cluster_profiles)
            report.append(f"   - Identified {n_clusters} distinct research communities")
            report.append("   - Each community shows unique citation and topical patterns")
        
        report.append("2. Outlier Papers:")
        report.append("   - High-impact outliers may represent breakthrough research")
        report.append("   - Undervalued papers may represent future opportunities")
        
        report.append("3. Temporal Trends:")
        report.append("   - Research communities show evolution over time")
        report.append("   - Some communities may be emerging or declining")
        
        # Join report and optionally save
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        print(report_text)
        return report_text