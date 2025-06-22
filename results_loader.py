

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Import the full analyzer class
try:
    from ul_pipeline import AcademicResearchAnalyzer
except ImportError:
    print("Warning: Could not import AcademicResearchAnalyzer. Make sure ul_pipeline.py is available.")


class LocalResultsLoader:
    """
    Enhanced results loader that properly reconstructs the full analyzer object.
    """
    
    def __init__(self, results_base_dir="saved_results"):
        self.results_base_dir = results_base_dir
    
    def list_available_experiments(self):
        """
        List all available saved experiments.
        """
        index_path = os.path.join(self.results_base_dir, 'experiments_index.json')
        
        try:
            with open(index_path, 'r') as f:
                experiments = json.load(f)
            return experiments
        except FileNotFoundError:
            # Fallback: scan directories if index doesn't exist
            return self._scan_for_experiments()
    
    def _scan_for_experiments(self):
        """
        Fallback method to scan for experiments if index doesn't exist.
        """
        experiments = []
        
        if not os.path.exists(self.results_base_dir):
            return experiments
        
        for item in os.listdir(self.results_base_dir):
            experiment_dir = os.path.join(self.results_base_dir, item)
            
            if os.path.isdir(experiment_dir):
                metadata_path = os.path.join(experiment_dir, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        experiments.append({
                            'experiment_name': item,
                            'path': experiment_dir,
                            'status': 'completed',
                            'completion_time': metadata.get('analysis_date', 'Unknown'),
                            'n_papers': metadata.get('n_papers', 'N/A'),
                            'methods': metadata.get('available_methods', [])
                        })
                    except:
                        continue
        
        return sorted(experiments, key=lambda x: x['completion_time'], reverse=True)
    
    def load_experiment(self, experiment_name):
        """
        Load a saved experiment and reconstruct the full analyzer object.
        
        This is the key fix - we properly reconstruct the analyzer with all methods.
        """
        experiment_dir = os.path.join(self.results_base_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            print(f"❌ Experiment '{experiment_name}' not found")
            return None
        
        try:
            print(f"📂 Loading experiment: {experiment_name}")
            
            # 1. Load metadata
            metadata_path = os.path.join(experiment_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 2. Create a new full analyzer instance (this is the key fix!)
            analyzer = AcademicResearchAnalyzer(
                random_state=metadata.get('random_state', 42)
            )
            
            # 3. Load the DataFrame
            df_path = os.path.join(experiment_dir, 'complete_results.csv')
            analyzer.df = pd.read_csv(df_path)
            
            # 4. Load all the analysis results and restore them to the analyzer
            self._restore_analyzer_attributes(analyzer, experiment_dir)
            
            # 5. Restore essential attributes for the analyzer to work properly
            self._restore_essential_attributes(analyzer, experiment_dir)
            
            print(f"✅ Successfully loaded experiment with full functionality!")
            print(f"   📊 Papers: {len(analyzer.df):,}")
            print(f"   🎯 Methods: {', '.join(analyzer.cluster_labels.keys() if analyzer.cluster_labels else [])}")
            
            return analyzer
            
        except Exception as e:
            print(f"❌ Error loading experiment: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _restore_analyzer_attributes(self, analyzer, experiment_dir):
        """
        Restore all the analysis results to the analyzer object.
        """
        # List of attributes to restore
        attributes_to_restore = [
            'cluster_labels',
            'dimensionality_results', 
            'outlier_labels',
            'clustering_metrics',
            'cluster_profiles',
            'all_cluster_profiles',
            'numerical_features',
            'features_scaled',
            'text_features',
            'tfidf_vectorizer'
        ]
        
        for attr_name in attributes_to_restore:
            attr_path = os.path.join(experiment_dir, f'{attr_name}.pkl')
            
            if os.path.exists(attr_path):
                try:
                    with open(attr_path, 'rb') as f:
                        attr_value = pickle.load(f)
                    
                    setattr(analyzer, attr_name, attr_value)
                    print(f"  ✅ Restored {attr_name}")
                    
                except Exception as e:
                    print(f"  ⚠️ Could not restore {attr_name}: {e}")
                    # Set to default value
                    if attr_name in ['cluster_labels', 'outlier_labels', 'dimensionality_results', 
                                   'clustering_metrics', 'cluster_profiles', 'all_cluster_profiles']:
                        setattr(analyzer, attr_name, {})
                    elif attr_name == 'numerical_features':
                        setattr(analyzer, attr_name, [])
                    else:
                        setattr(analyzer, attr_name, None)
    
    def _restore_essential_attributes(self, analyzer, experiment_dir):
        """
        Restore essential attributes needed for the analyzer to function properly.
        """
        # If numerical_features is empty, try to infer from DataFrame
        if not analyzer.numerical_features:
            potential_features = [
                'cited_by_count', 'publication_year', 'paper_age',
                'citations_per_year', 'author_count', 'concept_count',
                'avg_concept_score', 'referenced_works_count'
            ]
            
            analyzer.numerical_features = [
                feat for feat in potential_features 
                if feat in analyzer.df.columns
            ]
            
            print(f"  🔧 Inferred numerical features: {analyzer.numerical_features}")
        
        # Create feature matrix if it doesn't exist
        if analyzer.numerical_features and hasattr(analyzer, 'features_scaled') and analyzer.features_scaled is None:
            try:
                analyzer.feature_matrix = analyzer.df[analyzer.numerical_features].fillna(0)
                print(f"  🔧 Recreated feature matrix")
            except:
                pass
        
        # Ensure clustering models dict exists (even if empty)
        if not hasattr(analyzer, 'clustering_models'):
            analyzer.clustering_models = {}
        
        # Ensure cluster_analysis_results exists
        if not hasattr(analyzer, 'cluster_analysis_results'):
            analyzer.cluster_analysis_results = {}
    
    def delete_experiment(self, experiment_name):
        """
        Delete a saved experiment.
        """
        experiment_dir = os.path.join(self.results_base_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            return False, f"Experiment '{experiment_name}' not found"
        
        try:
            import shutil
            shutil.rmtree(experiment_dir)
            
            # Update experiments index
            self._update_experiments_index_after_deletion(experiment_name)
            
            return True, f"Experiment '{experiment_name}' deleted successfully"
            
        except Exception as e:
            return False, f"Error deleting experiment: {e}"
    
    def _update_experiments_index_after_deletion(self, deleted_experiment):
        """
        Update the experiments index after deleting an experiment.
        """
        index_path = os.path.join(self.results_base_dir, 'experiments_index.json')
        
        try:
            with open(index_path, 'r') as f:
                experiments = json.load(f)
            
            # Remove the deleted experiment
            experiments = [
                exp for exp in experiments 
                if exp['experiment_name'] != deleted_experiment
            ]
            
            with open(index_path, 'w') as f:
                json.dump(experiments, f, indent=2)
                
        except FileNotFoundError:
            # Index doesn't exist, nothing to update
            pass
        except Exception as e:
            print(f"Warning: Could not update experiments index: {e}")
    
    def get_experiment_info(self, experiment_name):
        """
        Get detailed information about a specific experiment.
        """
        experiment_dir = os.path.join(self.results_base_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            return None
        
        try:
            # Load metadata
            metadata_path = os.path.join(experiment_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get file sizes
            files_info = {}
            for filename in os.listdir(experiment_dir):
                filepath = os.path.join(experiment_dir, filename)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    files_info[filename] = f"{size_mb:.2f} MB"
            
            return {
                'metadata': metadata,
                'files': files_info,
                'total_size': sum(
                    os.path.getsize(os.path.join(experiment_dir, f)) 
                    for f in os.listdir(experiment_dir)
                    if os.path.isfile(os.path.join(experiment_dir, f))
                ) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"Error getting experiment info: {e}")
            return None


# Utility functions for easier usage

def quick_load_experiment(experiment_name, results_base_dir="saved_results"):
    """
    Quick function to load an experiment.
    
    Usage:
        analyzer = quick_load_experiment("my_analysis_20240101_1200")
    """
    loader = LocalResultsLoader(results_base_dir)
    return loader.load_experiment(experiment_name)


def list_experiments(results_base_dir="saved_results"):
    """
    Quick function to list all experiments.
    
    Usage:
        experiments = list_experiments()
        for exp in experiments:
            print(f"{exp['experiment_name']} - {exp['n_papers']} papers")
    """
    loader = LocalResultsLoader(results_base_dir)
    return loader.list_available_experiments()


def test_loaded_analyzer(analyzer):
    """
    Test that a loaded analyzer has all the necessary methods and attributes.
    """
    print("🧪 Testing loaded analyzer...")
    
    # Test basic attributes
    required_attrs = [
        'df', 'cluster_labels', 'dimensionality_results', 
        'numerical_features', 'features_scaled'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(analyzer, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"❌ Missing attributes: {missing_attrs}")
        return False
    
    # Test key methods
    required_methods = [
        'create_cluster_profiles', 'create_all_cluster_profiles',
        'get_cluster_profiles_for_method', 'add_cluster_labels_to_dataframe'
    ]
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(analyzer, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing methods: {missing_methods}")
        return False
    
    print("✅ Analyzer loaded successfully with all required functionality!")
    print(f"   📊 Papers: {len(analyzer.df):,}")
    print(f"   🎯 Clustering methods: {list(analyzer.cluster_labels.keys())}")
    print(f"   📐 Features: {len(analyzer.numerical_features)}")
    
    return True
