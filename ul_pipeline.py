import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
import umap
import os
import pickle  
import json
from datetime import datetime


class AcademicResearchAnalyzer: 
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.df = None
        self.feature_matrix = None
        self.features_scaled = None
        self.text_features = None
        self.numerical_features = []
        self.scaler = None
        self.tfidf_vectorizer = None
        
        # Analysis results storage
        self.dimensionality_results = {}
        self.cluster_labels = {}
        self.clustering_models = {}
        self.clustering_metrics = {}
        self.cluster_analysis_results = {}
        self.outlier_labels = {}
        self.cluster_profiles = {}
        self.all_cluster_profiles = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
    
    def load_and_prepare_data(self, data_path):
        """
        Load and prepare academic papers data for analysis.
        """
        print(f"Loading data from {data_path}...")
        
        # Load data (assuming JSON format from OpenAlex)
        if data_path.endswith('.json'):
            self.df = pd.read_json(data_path)
        elif data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        print(f"Loaded {len(self.df)} papers")
        
        # Data cleaning and feature engineering
        self._clean_and_engineer_features()
        
        # Create feature matrix
        self._create_feature_matrix()
        
        return self.df
    
    def _clean_and_engineer_features(self):
        """Clean data and engineer features for analysis."""
        
        # Handle missing values
        self.df['cited_by_count'] = self.df['cited_by_count'].fillna(0)
        self.df['publication_year'] = self.df['publication_year'].fillna(self.df['publication_year'].median())
        self.df['abstract'] = self.df['abstract'].fillna('')
        self.df['title'] = self.df['title'].fillna('Unknown Title')
        
        # Feature engineering
        current_year = 2024
        self.df['paper_age'] = current_year - self.df['publication_year']
        self.df['citations_per_year'] = self.df['cited_by_count'] / (self.df['paper_age'] + 1)
        
        # Handle edge cases
        self.df['citations_per_year'] = self.df['citations_per_year'].replace([np.inf, -np.inf], 0)
        
        # Extract author count
        self.df['author_count'] = self.df['authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Extract concept count and average concept score
        def extract_concept_features(concepts):
            if isinstance(concepts, list) and concepts:
                return len(concepts), np.mean([c.get('score', 0) for c in concepts])
            return 0, 0
        
        concept_features = self.df['concepts'].apply(extract_concept_features)
        self.df['concept_count'] = concept_features.apply(lambda x: x[0])
        self.df['avg_concept_score'] = concept_features.apply(lambda x: x[1])
        
        # Reference counts
        self.df['referenced_works_count'] = self.df['referenced_works_count'].fillna(0)
        
        print("Feature engineering completed")
    
    def _create_feature_matrix(self):
        """Create numerical feature matrix for ML analysis."""
        
        # Define numerical features to use
        self.numerical_features = [
            'cited_by_count', 'publication_year', 'paper_age',
            'citations_per_year', 'author_count', 'concept_count',
            'avg_concept_score', 'referenced_works_count'
        ]
        
        # Create numerical feature matrix
        self.feature_matrix = self.df[self.numerical_features].copy()
        
        # Handle any remaining missing values
        self.feature_matrix = self.feature_matrix.fillna(0)
        
        print(f"Created feature matrix with {self.feature_matrix.shape[1]} numerical features")
        
        # Create text features using TF-IDF
        self._create_text_features()
    
    def _create_text_features(self):
        """Create TF-IDF features from title and abstract."""
        
        # Combine title and abstract for better text representation
        self.df['combined_text'] = self.df['title'] + ' ' + self.df['abstract']
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features for computational efficiency
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Include bigrams
            min_df=5,  # Ignore terms that appear in less than 5 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        self.text_features = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])
        print(f"Created TF-IDF matrix with {self.text_features.shape[1]} features")
    
    def scale_features(self, method='standard'):
        """Scale numerical features for clustering."""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        self.features_scaled = self.scaler.fit_transform(self.feature_matrix)
        print(f"Scaled features using {method} scaling")
    
    def perform_dimensionality_reduction(self):
        """
        Apply multiple dimensionality reduction techniques.
        Similar to PCA analysis in the house price project.
        """
        print("Performing dimensionality reduction analysis...")
        
        # 1. PCA Analysis
        pca_full = PCA(random_state=self.random_state)
        pca_transformed = pca_full.fit_transform(self.features_scaled)
        
        # Store PCA results
        self.dimensionality_results['pca'] = {
            'transformer': pca_full,
            'transformed_data': pca_transformed,
            'explained_variance_ratio': pca_full.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca_full.explained_variance_ratio_)
        }
        
        # 2. UMAP for non-linear dimensionality reduction
        umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=self.random_state,
            n_neighbors=15,
            min_dist=0.1
        )
        umap_transformed = umap_reducer.fit_transform(self.features_scaled)
        
        self.dimensionality_results['umap'] = {
            'transformer': umap_reducer,
            'transformed_data': umap_transformed
        }
        
        # 3. t-SNE for visualization
        tsne_reducer = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=30,
            max_iter=1000
        )
        tsne_transformed = tsne_reducer.fit_transform(self.features_scaled)
        
        self.dimensionality_results['tsne'] = {
            'transformer': tsne_reducer,
            'transformed_data': tsne_transformed
        }
        
        print("Dimensionality reduction completed")
    
    def analyze_optimal_clusters(self, max_clusters=15):
        """
        Determine optimal number of clusters using multiple methods.
        Similar to elbow method in the house price project.
        """
        print("Analyzing optimal number of clusters...")
        
        cluster_range = range(2, max_clusters + 1)
        
        # Methods to evaluate
        results = {
            'inertias': [],
            'silhouette_scores': [],
            'davies_bouldin_scores': [],
            'aic_scores': [],  # For GMM
            'bic_scores': []   # For GMM
        }
        
        for n_clusters in cluster_range:
            # K-Means analysis
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            kmeans_labels = kmeans.fit_predict(self.features_scaled)
            
            results['inertias'].append(kmeans.inertia_)
            results['silhouette_scores'].append(silhouette_score(self.features_scaled, kmeans_labels))
            results['davies_bouldin_scores'].append(davies_bouldin_score(self.features_scaled, kmeans_labels))
            
            # GMM analysis for AIC/BIC
            gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
            gmm.fit(self.features_scaled)
            
            results['aic_scores'].append(gmm.aic(self.features_scaled))
            results['bic_scores'].append(gmm.bic(self.features_scaled))
        
        self.cluster_analysis_results = {
            'cluster_range': list(cluster_range),
            **results
        }
        
        # Determine optimal clusters
        optimal_clusters = self._determine_optimal_clusters()
        print(f"Suggested optimal number of clusters: {optimal_clusters}")
        
        return optimal_clusters
    
    def _determine_optimal_clusters(self):
        """Determine optimal number of clusters using multiple criteria."""
        scores = self.cluster_analysis_results
        
        # Elbow method for inertia
        inertias = scores['inertias']
        elbow_point = self._find_elbow_point(inertias)
        
        # Best silhouette score
        silhouette_optimal = scores['cluster_range'][np.argmax(scores['silhouette_scores'])]
        
        # Best Davies-Bouldin score (lower is better)
        db_optimal = scores['cluster_range'][np.argmin(scores['davies_bouldin_scores'])]
        
        # BIC optimal (lower is better)
        bic_optimal = scores['cluster_range'][np.argmin(scores['bic_scores'])]
        
        # Return most common suggestion or median
        suggestions = [elbow_point, silhouette_optimal, db_optimal, bic_optimal]
        optimal = int(np.median(suggestions))
        
        print(f"Cluster suggestions - Elbow: {elbow_point}, Silhouette: {silhouette_optimal}, "
              f"Davies-Bouldin: {db_optimal}, BIC: {bic_optimal}")
        
        return optimal
    
    def _find_elbow_point(self, values):
        """Find elbow point using the maximum curvature method."""
        # Normalize values to [0, 1]
        values = np.array(values)
        values_norm = (values - values.min()) / (values.max() - values.min())
        
        # Calculate curvature
        n_points = len(values_norm)
        x = np.arange(n_points)
        
        # Fit line from first to last point
        line_x = np.array([x[0], x[-1]])
        line_y = np.array([values_norm[0], values_norm[-1]])
        
        # Calculate distances from points to line
        distances = []
        for i in range(n_points):
            point = np.array([x[i], values_norm[i]])
            # Distance from point to line
            dist = np.abs(np.cross(line_y - line_x, line_x - point)) / np.linalg.norm(line_y - line_x)
            distances.append(dist)
        
        # Return the point with maximum distance
        elbow_idx = np.argmax(distances)
        
        # FIXED: Ensure elbow_idx is within valid range for cluster_range
        cluster_range = self.cluster_analysis_results['cluster_range']
        if elbow_idx < len(cluster_range):
            return cluster_range[elbow_idx]
        else:
            # Fallback to last valid cluster number
            return cluster_range[-1]
    
    def perform_clustering_analysis(self, n_clusters=None):
        """
        Perform comprehensive clustering analysis using multiple algorithms.
        Similar to the house price project's multiple clustering approaches.
        """
        if n_clusters is None:
            n_clusters = self.analyze_optimal_clusters()
        
        print(f"Performing clustering analysis with {n_clusters} clusters...")
        
        # 1. K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels['kmeans'] = kmeans.fit_predict(self.features_scaled)
        
        # 2. Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.cluster_labels['hierarchical'] = hierarchical.fit_predict(self.features_scaled)
        
        # 3. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        self.cluster_labels['gmm'] = gmm.fit_predict(self.features_scaled)
        
        # 4. DBSCAN (automatic cluster detection)
        # Determine eps using k-distance graph
        eps = self._estimate_dbscan_eps()
        dbscan = DBSCAN(eps=eps, min_samples=5)
        self.cluster_labels['dbscan'] = dbscan.fit_predict(self.features_scaled)
        
        # Store clustering models
        self.clustering_models = {
            'kmeans': kmeans,
            'hierarchical': hierarchical,
            'gmm': gmm,
            'dbscan': dbscan
        }
        
        # Evaluate clustering quality
        self._evaluate_clustering_quality()

        # Add cluster labels to dataframe
        self.add_cluster_labels_to_dataframe()
        
        print("Clustering analysis completed")
    
    def add_cluster_labels_to_dataframe(self):
        """
        Add all cluster labels to the main DataFrame.
        Call this after clustering if you haven't called create_cluster_profiles().
        """
        for method, labels in self.cluster_labels.items():
            column_name = f'cluster_{method}'
            self.df[column_name] = labels
            print(f"Added {column_name} to DataFrame")
    
    def _estimate_dbscan_eps(self):
        """Estimate optimal eps parameter for DBSCAN."""
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate distances to k-th nearest neighbor
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.features_scaled)
        distances, indices = nbrs.kneighbors(self.features_scaled)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find elbow point in k-distance graph
        eps_idx = self._find_elbow_point_for_eps(distances[::-1])  # Reverse for elbow finding
        eps_value = distances[len(distances) - eps_idx - 1] if eps_idx < len(distances) else distances[-1]
        
        return eps_value
    
    def _find_elbow_point_for_eps(self, values):
        """Find elbow point for eps estimation - separate from cluster elbow method."""
        values = np.array(values)
        values_norm = (values - values.min()) / (values.max() - values.min())
        
        n_points = len(values_norm)
        x = np.arange(n_points)
        
        # Fit line from first to last point
        line_x = np.array([x[0], x[-1]])
        line_y = np.array([values_norm[0], values_norm[-1]])
        
        # Calculate distances from points to line
        distances = []
        for i in range(n_points):
            point = np.array([x[i], values_norm[i]])
            dist = np.abs(np.cross(line_y - line_x, line_x - point)) / np.linalg.norm(line_y - line_x)
            distances.append(dist)
        
        # Return the index with maximum distance
        return np.argmax(distances)
    
    def _evaluate_clustering_quality(self):
        """Evaluate the quality of different clustering methods."""
        self.clustering_metrics = {}
        
        for method, labels in self.cluster_labels.items():
            # Skip evaluation for DBSCAN if all points are noise
            if method == 'dbscan' and len(np.unique(labels)) <= 1:
                continue
                
            # Only evaluate if we have more than 1 cluster
            if len(np.unique(labels)) > 1:
                try:
                    silhouette = silhouette_score(self.features_scaled, labels)
                    davies_bouldin = davies_bouldin_score(self.features_scaled, labels)
                    
                    self.clustering_metrics[method] = {
                        'silhouette_score': silhouette,
                        'davies_bouldin_score': davies_bouldin,
                        'n_clusters': len(np.unique(labels)),
                        'n_noise': np.sum(labels == -1) if method == 'dbscan' else 0
                    }
                except:
                    print(f"Could not evaluate {method} clustering")
        
        # Print clustering quality summary
        print("\nClustering Quality Summary:")
        print("=" * 50)
        for method, metrics in self.clustering_metrics.items():
            print(f"{method.upper()}:")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
            print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
            print(f"  Number of Clusters: {metrics['n_clusters']}")
            if metrics['n_noise'] > 0:
                print(f"  Noise Points: {metrics['n_noise']}")
            print()
    
    def detect_outliers(self):
        """
        Detect outliers using multiple methods.
        """
        print("Detecting outliers using multiple methods...")
        
        # 1. Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.05,  # Expect 5% outliers
            random_state=self.random_state
        )
        self.outlier_labels['isolation_forest'] = isolation_forest.fit_predict(self.features_scaled)
        
        # 2. Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.05)
        self.outlier_labels['local_outlier_factor'] = lof.fit_predict(self.features_scaled)
        
        # 3. One-Class SVM
        svm = OneClassSVM(gamma='scale', nu=0.05)
        self.outlier_labels['one_class_svm'] = svm.fit_predict(self.features_scaled)
        
        # 4. Statistical outliers (Z-score based)
        z_scores = np.abs((self.features_scaled - np.mean(self.features_scaled, axis=0)) / np.std(self.features_scaled, axis=0))
        statistical_outliers = np.any(z_scores > 3, axis=1)
        self.outlier_labels['statistical'] = np.where(statistical_outliers, -1, 1)
        
        # Create consensus outlier score
        self._create_outlier_consensus()
        
        print("Outlier detection completed")
    
    def _create_outlier_consensus(self):
        """Create consensus outlier detection based on multiple methods."""
        # Convert outlier labels to binary (1 = normal, -1 = outlier)
        outlier_matrix = np.array([
            (labels == -1).astype(int) for labels in self.outlier_labels.values()
        ]).T
        
        # Calculate consensus score (how many methods agree)
        consensus_scores = outlier_matrix.sum(axis=1)
        
        # Papers identified as outliers by multiple methods
        strong_outliers = consensus_scores >= 3
        moderate_outliers = consensus_scores == 2
        
        self.df['outlier_consensus_score'] = consensus_scores
        self.df['is_strong_outlier'] = strong_outliers
        self.df['is_moderate_outlier'] = moderate_outliers
        
        print(f"Outlier consensus analysis:")
        print(f"  Strong outliers (3+ methods): {strong_outliers.sum()}")
        print(f"  Moderate outliers (2 methods): {moderate_outliers.sum()}")
        print(f"  Total outliers detected: {(consensus_scores > 0).sum()}")
    
    def create_cluster_profiles(self, method='kmeans'):
        """
        Create detailed profiles for each cluster.
        Similar to the house price project's cluster characterization.
        """
        if method not in self.cluster_labels:
            raise ValueError(f"Clustering method '{method}' not found. Available: {list(self.cluster_labels.keys())}")
        
        labels = self.cluster_labels[method]
        
        # Add cluster labels to dataframe
        self.df[f'cluster_{method}'] = labels
        
        # Create cluster profiles
        cluster_profiles = {}
        
        unique_clusters = np.unique(labels)
        if -1 in unique_clusters:  # Remove noise cluster for DBSCAN
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_data = self.df[cluster_mask]
            
            profile = {
                'size': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(self.df)) * 100,
                'numerical_features': {},
                'top_terms': [],
                'example_papers': []
            }
            
            # Numerical feature statistics
            for feature in self.numerical_features:
                if feature in self.df.columns:
                    feature_data = cluster_data[feature]
                    profile['numerical_features'][feature] = {
                        'mean': feature_data.mean(),
                        'median': feature_data.median(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max()
                    }
            
            # Extract top terms using TF-IDF
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                cluster_texts = cluster_data['combined_text'].tolist()
                if cluster_texts:
                    try:
                        cluster_tfidf = self.tfidf_vectorizer.transform(cluster_texts)
                        mean_tfidf = np.array(cluster_tfidf.mean(axis=0)).flatten()
                        feature_names = self.tfidf_vectorizer.get_feature_names_out()
                        
                        # Get top terms
                        top_indices = mean_tfidf.argsort()[-20:][::-1]
                        profile['top_terms'] = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
                    except:
                        profile['top_terms'] = []
            
            # Example papers (highest cited in cluster)
            top_papers = cluster_data.nlargest(5, 'cited_by_count')
            profile['example_papers'] = top_papers[['title', 'cited_by_count', 'publication_year']].to_dict('records')
            
            cluster_profiles[cluster_id] = profile
        
        self.cluster_profiles = cluster_profiles
        
        print(f"Created profiles for {len(cluster_profiles)} clusters using {method} method")
        return cluster_profiles
    
    def create_all_cluster_profiles(self):
        """
        Create cluster profiles for ALL clustering methods.
        This ensures profiles are available for every method.
        """
        print("🔄 Creating cluster profiles for all methods...")
        
        if not hasattr(self, 'cluster_labels') or not self.cluster_labels:
            print("❌ No cluster labels available. Run clustering analysis first.")
            return
        
        # Store all profiles in a nested dictionary
        all_profiles = {}
        
        for method in self.cluster_labels.keys():
            try:
                print(f"  📊 Creating profiles for {method.upper()}...")
                
                # Create profiles for this method
                profiles = self.create_cluster_profiles(method)
                all_profiles[method] = profiles
                
                # Also store in the main cluster_profiles attribute for the current method
                if not hasattr(self, 'cluster_profiles'):
                    self.cluster_profiles = {}
                
                # Store method-specific profiles
                if not hasattr(self, 'all_cluster_profiles'):
                    self.all_cluster_profiles = {}
                
                self.all_cluster_profiles[method] = profiles
                
                print(f"    ✅ Created {len(profiles)} profiles for {method}")
                
            except Exception as e:
                print(f"    ❌ Failed to create profiles for {method}: {e}")
                continue
        
        print(f"✅ Cluster profiles created for {len(all_profiles)} methods")
        return all_profiles

    def get_cluster_profiles_for_method(self, method='kmeans'):
        """
        Get cluster profiles for a specific method.
        Creates them if they don't exist.
        """
        
        # Check if we have method-specific profiles
        if hasattr(self, 'all_cluster_profiles') and method in self.all_cluster_profiles:
            return self.all_cluster_profiles[method]
        
        # Check if current cluster_profiles matches the requested method
        if hasattr(self, 'cluster_profiles') and self.cluster_profiles:
            # Check if the current profiles are for this method by looking at cluster labels
            if method in self.cluster_labels:
                current_labels = self.cluster_labels[method]
                profile_cluster_ids = set(self.cluster_profiles.keys())
                method_cluster_ids = set(np.unique(current_labels))
                
                # Remove noise cluster (-1) from comparison
                method_cluster_ids.discard(-1)
                
                # If they match, return current profiles
                if profile_cluster_ids == method_cluster_ids:
                    return self.cluster_profiles
        
        # Need to create profiles for this method
        print(f"🔄 Creating missing cluster profiles for {method}...")
        try:
            profiles = self.create_cluster_profiles(method)
            
            # Store in method-specific storage
            if not hasattr(self, 'all_cluster_profiles'):
                self.all_cluster_profiles = {}
            self.all_cluster_profiles[method] = profiles
            
            return profiles
        except Exception as e:
            print(f"❌ Error creating profiles for {method}: {e}")
            return {}

    def perform_clustering_analysis_with_all_profiles(self, n_clusters=None):
        """
        Enhanced clustering analysis that creates profiles for ALL methods.
        """
        if n_clusters is None:
            n_clusters = self.analyze_optimal_clusters()
        
        print(f"Performing clustering analysis with {n_clusters} clusters...")
        
        # 1. K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels['kmeans'] = kmeans.fit_predict(self.features_scaled)
        
        # 2. Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.cluster_labels['hierarchical'] = hierarchical.fit_predict(self.features_scaled)
        
        # 3. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        self.cluster_labels['gmm'] = gmm.fit_predict(self.features_scaled)
        
        # 4. DBSCAN (automatic cluster detection)
        eps = self._estimate_dbscan_eps()
        dbscan = DBSCAN(eps=eps, min_samples=5)
        self.cluster_labels['dbscan'] = dbscan.fit_predict(self.features_scaled)
        
        # Store clustering models
        self.clustering_models = {
            'kmeans': kmeans,
            'hierarchical': hierarchical,
            'gmm': gmm,
            'dbscan': dbscan
        }
        
        # Evaluate clustering quality
        self._evaluate_clustering_quality()
        
        # Add all cluster labels to DataFrame
        self.add_cluster_labels_to_dataframe()
        
        # CREATE PROFILES FOR ALL METHODS - This is the key fix!
        self.create_all_cluster_profiles()
        
        print("Clustering analysis completed with profiles for all methods")

    def save(self, path="analyzer_state.pkl"):
        """Save the analyzer state to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Analyzer state saved to {path}")

    @staticmethod
    def load(path="analyzer_state.pkl"):
        """Load an analyzer state from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def save_everything_for_dashboard(self, experiment_name=None):
        """
        Save all analysis results for instant dashboard loading.
        """
        if experiment_name is None:
            experiment_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        results_base_dir = "saved_results"
        results_dir = os.path.join(results_base_dir, experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"💾 Saving all results to: {results_dir}")
        
        # 1. Save main DataFrame with all cluster labels
        print("  📄 Saving complete DataFrame...")
        df_path = os.path.join(results_dir, 'complete_results.csv')
        self.df.to_csv(df_path, index=False)
        
        # 2. Save all analysis objects
        analysis_objects = {
            'cluster_labels': self.cluster_labels,
            'dimensionality_results': self.dimensionality_results,
            'outlier_labels': self.outlier_labels,
            'clustering_metrics': getattr(self, 'clustering_metrics', {}),
            'cluster_profiles': getattr(self, 'cluster_profiles', {}),
            'all_cluster_profiles': getattr(self, 'all_cluster_profiles', {}),
            'numerical_features': self.numerical_features,
            'features_scaled': self.features_scaled,
            'text_features': self.text_features,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        for name, obj in analysis_objects.items():
            if obj is not None:
                print(f"  💾 Saving {name}...")
                with open(os.path.join(results_dir, f'{name}.pkl'), 'wb') as f:
                    pickle.dump(obj, f)
        
        # 3. Save metadata
        print("  📋 Saving metadata...")
        metadata = {
            'experiment_name': experiment_name,
            'analysis_date': datetime.now().isoformat(),
            'n_papers': len(self.df),
            'available_methods': list(self.cluster_labels.keys()),
            'random_state': self.random_state,
            'has_dimensionality_results': bool(self.dimensionality_results),
            'has_outlier_detection': bool(self.outlier_labels),
            'has_cluster_profiles': hasattr(self, 'cluster_profiles'),
            'has_all_cluster_profiles': hasattr(self, 'all_cluster_profiles'),
            'cluster_counts': {
                method: dict(pd.Series(labels).value_counts().to_dict())
                for method, labels in self.cluster_labels.items()
            }
        }
        
        with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # 4. Update experiments index for easy discovery
        self._update_experiments_index(experiment_name, results_dir, results_base_dir)
        
        print(f"✅ All results saved successfully!")
        print(f"🌐 Ready for instant dashboard loading!")
        return experiment_name

    def _update_experiments_index(self, experiment_name, results_dir, results_base_dir):
        """Helper method to maintain experiments index."""
        summary = {
            'experiment_name': experiment_name,
            'path': results_dir,
            'status': 'completed',
            'completion_time': datetime.now().isoformat(),
            'n_papers': len(self.df),
            'methods': list(self.cluster_labels.keys())
        }
        
        summary_path = os.path.join(results_base_dir, 'experiments_index.json')
        
        # Load existing index or create new
        try:
            with open(summary_path, 'r') as f:
                experiments_index = json.load(f)
        except FileNotFoundError:
            experiments_index = []
        
        # Add current experiment
        experiments_index = [exp for exp in experiments_index if exp['experiment_name'] != experiment_name]
        experiments_index.append(summary)
        experiments_index.sort(key=lambda x: x['completion_time'], reverse=True)
        
        with open(summary_path, 'w') as f:
            json.dump(experiments_index, f, indent=2)

    def perform_full_analysis_with_auto_save(self, data_path, experiment_name=None):
        """
        Perform complete analysis and automatically save everything.
        This is your new main analysis method.
        """
        print("🚀 Starting complete analysis with auto-save...")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(data_path)
        
        # Step 2: Scale features  
        self.scale_features()
        
        # Step 3: Run all analysis steps
        self.perform_dimensionality_reduction()
        self.perform_clustering_analysis_with_all_profiles()
        self.detect_outliers()
        
        # Step 4: Auto-save everything
        saved_experiment = self.save_everything_for_dashboard(experiment_name)
        
        print(f"🎉 Analysis completed and saved: {saved_experiment}")
        return saved_experiment


# Utility function for loading saved experiments
def load_saved_experiment(experiment_name, results_base_dir="saved_results"):
    """
    Load a previously saved experiment for dashboard use.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment to load
    results_base_dir : str
        Base directory where results are stored
    
    Returns:
    --------
    dict
        Dictionary containing all loaded analysis results
    """
    experiment_dir = os.path.join(results_base_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found in {results_base_dir}")
    
    print(f"📂 Loading experiment: {experiment_name}")
    
    # Load metadata
    with open(os.path.join(experiment_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load DataFrame
    df = pd.read_csv(os.path.join(experiment_dir, 'complete_results.csv'))
    
    # Load analysis objects
    analysis_objects = {}
    analysis_files = [
        'cluster_labels.pkl', 'dimensionality_results.pkl', 'outlier_labels.pkl',
        'clustering_metrics.pkl', 'cluster_profiles.pkl', 'all_cluster_profiles.pkl',
        'numerical_features.pkl', 'features_scaled.pkl', 'text_features.pkl',
        'tfidf_vectorizer.pkl'
    ]
    
    for filename in analysis_files:
        filepath = os.path.join(experiment_dir, filename)
        if os.path.exists(filepath):
            object_name = filename.replace('.pkl', '')
            print(f"  📦 Loading {object_name}...")
            with open(filepath, 'rb') as f:
                analysis_objects[object_name] = pickle.load(f)
    
    print(f"✅ Experiment loaded successfully!")
    
    return {
        'metadata': metadata,
        'df': df,
        **analysis_objects
    }


def list_available_experiments(results_base_dir="saved_results"):
    """
    List all available saved experiments.
    
    Parameters:
    -----------
    results_base_dir : str
        Base directory where results are stored
    
    Returns:
    --------
    list
        List of experiment metadata dictionaries
    """
    index_path = os.path.join(results_base_dir, 'experiments_index.json')
    
    try:
        with open(index_path, 'r') as f:
            experiments = json.load(f)
        return experiments
    except FileNotFoundError:
        print(f"No experiments index found at {index_path}")
        return []

