# arXiv Topic Clustering and Researcher Network Analysis
# Project Setup and Data Collection

import arxiv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
import re
from collections import defaultdict, Counter

# Text processing and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Network analysis
import networkx as nx

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class ArxivDataCollector:
    """Collects and preprocesses arXiv papers data"""
    
    def __init__(self):
        self.papers_data = []
        self.client = arxiv.Client()
        
    def collect_papers(self, categories=['cs.AI', 'cs.LG', 'cs.CL'], 
                      max_papers=1000, days_back=365):
        """
        Collect papers from arXiv
        
        Args:
            categories: List of arXiv categories to search
            max_papers: Maximum number of papers to collect
            days_back: How many days back to search
        """
        
        print(f"Collecting up to {max_papers} papers from categories: {categories}")
        
        # Build search query
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        search = arxiv.Search(
            query=category_query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers_collected = 0
        
        try:
            for paper in self.client.results(search):
                # Extract paper information
                paper_data = {
                    'id': paper.entry_id.split('/')[-1],
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'categories': paper.categories,
                    'published': paper.published,
                    'updated': paper.updated,
                    'primary_category': paper.primary_category
                }
                
                self.papers_data.append(paper_data)
                papers_collected += 1
                
                if papers_collected % 100 == 0:
                    print(f"Collected {papers_collected} papers...")
                    
        except Exception as e:
            print(f"Error collecting papers: {e}")
            
        print(f"Successfully collected {len(self.papers_data)} papers")
        return self.papers_data
    
    def to_dataframe(self):
        """Convert collected data to pandas DataFrame"""
        if not self.papers_data:
            print("No data collected yet. Run collect_papers() first.")
            return None
            
        df = pd.DataFrame(self.papers_data)
        
        # Add derived features
        df['num_authors'] = df['authors'].apply(len)
        df['abstract_length'] = df['abstract'].apply(len)
        df['year'] = pd.to_datetime(df['published']).dt.year
        
        return df

class TextPreprocessor:
    """Handles text preprocessing for abstracts"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stop words
        self.stop_words.update([
            'paper', 'method', 'approach', 'model', 'algorithm', 'technique',
            'result', 'show', 'present', 'propose', 'study', 'research',
            'work', 'analysis', 'system', 'data', 'performance', 'arxiv',
            'preprint', 'submitted'
        ])
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess_abstracts(self, abstracts):
        """Preprocess a list of abstracts"""
        print("Preprocessing abstracts...")
        processed = [self.clean_text(abstract) for abstract in abstracts]
        print("Preprocessing complete!")
        return processed

class TopicClusterer:
    """Handles topic clustering of papers"""
    
    def __init__(self):
        self.vectorizer = None
        self.feature_matrix = None
        self.cluster_labels = None
        
    def create_features(self, processed_abstracts, max_features=5000):
        """Create TF-IDF features from processed abstracts"""
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8  # Maximum document frequency
        )
        
        self.feature_matrix = self.vectorizer.fit_transform(processed_abstracts)
        print(f"Created feature matrix: {self.feature_matrix.shape}")
        
        return self.feature_matrix
    
    def find_optimal_clusters(self, max_clusters=20):
        """Find optimal number of clusters using elbow method"""
        if self.feature_matrix is None:
            print("Create features first!")
            return
            
        print("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.feature_matrix)
            
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.feature_matrix, cluster_labels)
            silhouette_scores.append(sil_score)
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Return best k based on silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"Recommended number of clusters: {best_k}")
        
        return best_k, inertias, silhouette_scores
    
    def cluster_papers(self, n_clusters=10, algorithm='kmeans'):
        """Cluster papers using specified algorithm"""
        if self.feature_matrix is None:
            print("Create features first!")
            return
            
        print(f"Clustering papers with {algorithm}...")
        
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError("Algorithm must be 'kmeans' or 'dbscan'")
            
        self.cluster_labels = clusterer.fit_predict(self.feature_matrix)
        
        # Print cluster statistics
        unique_labels = np.unique(self.cluster_labels)
        print(f"Found {len(unique_labels)} clusters")
        
        for label in unique_labels:
            count = np.sum(self.cluster_labels == label)
            if label == -1:  # DBSCAN noise points
                print(f"Noise points: {count}")
            else:
                print(f"Cluster {label}: {count} papers")
                
        return self.cluster_labels
    
    def get_top_terms_per_cluster(self, n_terms=10):
        """Get top terms for each cluster"""
        if self.cluster_labels is None or self.vectorizer is None:
            print("Cluster papers first!")
            return
            
        feature_names = self.vectorizer.get_feature_names_out()
        
        cluster_terms = {}
        
        for cluster_id in np.unique(self.cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get papers in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_center = self.feature_matrix[cluster_mask].mean(axis=0).A1
            
            # Get top terms
            top_indices = cluster_center.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = cluster_center[top_indices]
            
            cluster_terms[cluster_id] = list(zip(top_terms, top_scores))
            
        return cluster_terms

# Example usage and initialization
def main():
    """Main function to demonstrate the pipeline"""
    
    print("=== arXiv Topic Clustering Project ===\n")
    
    # Step 1: Collect data
    collector = ArxivDataCollector()
    papers = collector.collect_papers(
        categories=['cs.AI', 'cs.LG', 'cs.CL'], 
        max_papers=500,  # Start small for testing
        days_back=180
    )
    
    # Convert to DataFrame
    df = collector.to_dataframe()
    print(f"\nDataset shape: {df.shape}")
    print(f"Categories: {df['primary_category'].value_counts()}")
    
    # Step 2: Preprocess text
    preprocessor = TextPreprocessor()
    processed_abstracts = preprocessor.preprocess_abstracts(df['abstract'].tolist())
    
    # Step 3: Create features and cluster
    clusterer = TopicClusterer()
    features = clusterer.create_features(processed_abstracts)
    
    # Find optimal clusters
    optimal_k, inertias, silhouette_scores = clusterer.find_optimal_clusters(max_clusters=15)
    
    # Perform clustering
    cluster_labels = clusterer.cluster_papers(n_clusters=optimal_k)
    
    # Get top terms per cluster
    cluster_terms = clusterer.get_top_terms_per_cluster()
    
    print("\n=== Cluster Topics ===")
    for cluster_id, terms in cluster_terms.items():
        print(f"\nCluster {cluster_id}:")
        top_words = ', '.join([term for term, score in terms[:5]])
        print(f"  Top terms: {top_words}")
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    print(f"\nProject initialization complete!")
    print(f"Next steps:")
    print("1. Analyze cluster quality and interpretability")
    print("2. Build researcher collaboration network")
    print("3. Create visualizations")
    print("4. Implement autoencoders for enhanced representations")
    
    return df, clusterer, preprocessor

if __name__ == "__main__":
    # Run the main pipeline
    df, clusterer, preprocessor = main()