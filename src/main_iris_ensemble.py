"""
Main Ensemble Learning Application for Iris Dataset
===================================================

This is the main application that orchestrates the ensemble learning
analysis on the Iris dataset, including data processing, model training,
evaluation, and comprehensive reporting.

Author: Francesco
Date: October 2025
Course: Artificial Intelligence - Assignment 3
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import IrisDataProcessor
from ensemble_core import EnsembleLearningCore
from visualization import EnsembleVisualization


class IrisEnsembleLearningApplication:
    """
    Main application for ensemble learning on Iris dataset.
    """
    
    def __init__(self, output_dir="results", random_state=42):
        """
        Initialize the ensemble learning application.
        
        Args:
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = IrisDataProcessor(random_state=random_state)
        self.ensemble_core = EnsembleLearningCore(random_state=random_state)
        self.visualizer = None
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Results storage
        self.performance_results = {}
        self.importance_results = {}
        self.optimization_results = {}
        
        print("🚀 Iris Ensemble Learning Application Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎲 Random state: {self.random_state}")
        print(f"⏰ Session timestamp: {self.timestamp}")
    
    def load_and_prepare_data(self):
        """Load and prepare the Iris dataset."""
        print("\n" + "="*60)
        print("📊 STEP 1: DATA LOADING AND PREPARATION")
        print("="*60)
        
        # Load data
        X, y, analysis = self.data_processor.load_and_analyze_data()
        
        # Create train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_processor.create_train_test_split(X, y)
        
        # Apply feature scaling
        self.X_train_scaled, self.X_test_scaled = \
            self.data_processor.apply_feature_scaling(self.X_train, self.X_test)
        
        # Initialize visualizer with feature and target names
        self.visualizer = EnsembleVisualization(
            feature_names=self.data_processor.feature_names,
            target_names=self.data_processor.target_names
        )
        
        # Analyze feature importance
        importance = self.data_processor.analyze_feature_importance(X, y)
        
        # Save data analysis results
        self._save_data_analysis(analysis, importance)
        
        print("✅ Data preparation completed successfully!")
        return X, y, analysis
    
    def create_ensemble_models(self):
        """Create all ensemble learning models."""
        print("\n" + "="*60)
        print("🎯 STEP 2: ENSEMBLE MODEL CREATION")
        print("="*60)
        
        # Create all ensemble methods
        ensemble_models = self.ensemble_core.create_all_ensembles()
        
        print(f"✅ Created {len(ensemble_models)} ensemble models")
        
        return ensemble_models
    
    def evaluate_ensemble_performance(self, ensemble_models):
        """Evaluate performance of all ensemble methods."""
        print("\n" + "="*60)
        print("📈 STEP 3: PERFORMANCE EVALUATION")
        print("="*60)
        
        # Evaluate on scaled data for fair comparison
        self.performance_results = self.ensemble_core.evaluate_ensemble_performance(
            self.X_train_scaled, self.X_test_scaled, 
            self.y_train, self.y_test, cv_folds=5
        )
        
        # Get feature importance analysis
        self.importance_results = self.ensemble_core.get_feature_importance_analysis()
        
        # Print performance summary
        self._print_performance_summary()
        
        # Save performance results
        self._save_performance_results()
        
        print("✅ Performance evaluation completed!")
        
        return self.performance_results
    
    def optimize_hyperparameters(self):
        """Perform hyperparameter optimization for selected models."""
        print("\n" + "="*60)
        print("⚙️ STEP 4: HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        # Define parameter grids for optimization
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'bagging': {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.8, 1.0],
                'max_features': [0.5, 0.8, 1.0]
            }
        }
        
        # Optimize selected models
        for model_name, param_grid in param_grids.items():
            if model_name in self.ensemble_core.ensemble_models:
                print(f"🔧 Optimizing {model_name}...")
                
                try:
                    opt_result = self.ensemble_core.hyperparameter_optimization(
                        self.X_train_scaled, self.y_train, model_name, param_grid
                    )
                    self.optimization_results[model_name] = opt_result
                    
                    print(f"   Best params: {opt_result['best_params']}")
                    print(f"   Best score: {opt_result['best_score']:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Optimization failed: {str(e)}")
        
        # Save optimization results
        self._save_optimization_results()
        
        print("✅ Hyperparameter optimization completed!")
    
    def create_visualizations(self, X, y, ensemble_models):
        """Create comprehensive visualizations."""
        print("\n" + "="*60)
        print("🎨 STEP 5: VISUALIZATION CREATION")
        print("="*60)
        
        # Create comprehensive visualization report
        self.visualizer.create_comprehensive_report(
            X, y, self.X_test_scaled, self.y_test,
            ensemble_models, self.performance_results, 
            self.importance_results, self.output_dir
        )
        
        print("✅ Visualizations created successfully!")
    
    def generate_final_report(self):
        """Generate final analysis report."""
        print("\n" + "="*60)
        print("📋 STEP 6: FINAL REPORT GENERATION")
        print("="*60)
        
        report_path = os.path.join(self.output_dir, f"ensemble_analysis_report_{self.timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("IRIS ENSEMBLE LEARNING ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random State: {self.random_state}\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training samples: {len(self.y_train)}\n")
            f.write(f"Test samples: {len(self.y_test)}\n")
            f.write(f"Features: {len(self.data_processor.feature_names)}\n")
            f.write(f"Classes: {len(self.data_processor.target_names)}\n\n")
            
            # Performance results
            f.write("PERFORMANCE RESULTS\n")
            f.write("-" * 20 + "\n")
            for method, metrics in self.performance_results.items():
                f.write(f"\n{method.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # Best performing method
            best_method = max(self.performance_results.items(), 
                            key=lambda x: x[1]['test_accuracy'])
            f.write(f"\nBEST PERFORMING METHOD: {best_method[0]}\n")
            f.write(f"Test Accuracy: {best_method[1]['test_accuracy']:.4f}\n")
            
            # Feature importance
            if self.importance_results:
                f.write("\nFEATURE IMPORTANCE\n")
                f.write("-" * 20 + "\n")
                for method, importances in self.importance_results.items():
                    f.write(f"\n{method}:\n")
                    for feature, importance in zip(self.data_processor.feature_names, importances):
                        f.write(f"  {feature}: {importance:.4f}\n")
            
            # Optimization results
            if self.optimization_results:
                f.write("\nHYPERPARAMETER OPTIMIZATION\n")
                f.write("-" * 30 + "\n")
                for method, result in self.optimization_results.items():
                    f.write(f"\n{method}:\n")
                    f.write(f"  Best parameters: {result['best_params']}\n")
                    f.write(f"  Best CV score: {result['best_score']:.4f}\n")
        
        print(f"📄 Final report saved to: {report_path}")
        
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete ensemble learning analysis pipeline."""
        start_time = time.time()
        
        print("🚀 STARTING COMPLETE ENSEMBLE LEARNING ANALYSIS")
        print("=" * 70)
        
        try:
            # Step 1: Load and prepare data
            X, y, analysis = self.load_and_prepare_data()
            
            # Step 2: Create ensemble models
            ensemble_models = self.create_ensemble_models()
            
            # Step 3: Evaluate performance
            performance_results = self.evaluate_ensemble_performance(ensemble_models)
            
            # Step 4: Optimize hyperparameters (optional)
            self.optimize_hyperparameters()
            
            # Step 5: Create visualizations
            self.create_visualizations(X, y, ensemble_models)
            
            # Step 6: Generate final report
            report_path = self.generate_final_report()
            
            # Summary
            elapsed_time = time.time() - start_time
            
            print("\n" + "="*70)
            print("✅ ENSEMBLE LEARNING ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📋 Final report: {report_path}")
            
            # Best method summary
            best_method = max(performance_results.items(), 
                            key=lambda x: x[1]['test_accuracy'])
            print(f"🏆 Best performing method: {best_method[0]}")
            print(f"🎯 Best test accuracy: {best_method[1]['test_accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Analysis failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_data_analysis(self, analysis, importance):
        """Save data analysis results to files."""
        # Save basic analysis
        analysis_path = os.path.join(self.output_dir, f"data_analysis_{self.timestamp}.txt")
        with open(analysis_path, 'w') as f:
            f.write("IRIS DATASET ANALYSIS\n")
            f.write("=" * 25 + "\n\n")
            f.write(f"Dataset shape: {analysis['dataset_shape']}\n")
            f.write(f"Features: {analysis['feature_names']}\n")
            f.write(f"Classes: {analysis['target_names']}\n")
            f.write(f"Class distribution: {analysis['class_distribution'].to_dict()}\n")
        
        # Save feature statistics
        stats_path = os.path.join(self.output_dir, f"feature_statistics_{self.timestamp}.csv")
        analysis['feature_statistics'].to_csv(stats_path)
        
        # Save correlation matrix
        corr_path = os.path.join(self.output_dir, f"correlation_matrix_{self.timestamp}.csv")
        analysis['correlation_matrix'].to_csv(corr_path)
    
    def _save_performance_results(self):
        """Save performance results to CSV."""
        results_path = os.path.join(self.output_dir, f"performance_results_{self.timestamp}.csv")
        df = pd.DataFrame(self.performance_results).T
        df.to_csv(results_path)
        print(f"💾 Performance results saved to: {results_path}")
    
    def _save_optimization_results(self):
        """Save optimization results."""
        if self.optimization_results:
            opt_path = os.path.join(self.output_dir, f"optimization_results_{self.timestamp}.txt")
            with open(opt_path, 'w') as f:
                f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
                f.write("=" * 40 + "\n\n")
                for method, result in self.optimization_results.items():
                    f.write(f"{method}:\n")
                    f.write(f"  Best parameters: {result['best_params']}\n")
                    f.write(f"  Best score: {result['best_score']:.4f}\n\n")
    
    def _print_performance_summary(self):
        """Print a summary of performance results."""
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        # Sort by test accuracy
        sorted_results = sorted(
            self.performance_results.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        for rank, (method, metrics) in enumerate(sorted_results, 1):
            print(f"{rank}. {method:15} | "
                  f"Test Acc: {metrics['test_accuracy']:.4f} | "
                  f"CV Acc: {metrics['cv_mean_accuracy']:.4f}±{metrics['cv_std_accuracy']:.4f}")


if __name__ == "__main__":
    # Run the complete analysis
    app = IrisEnsembleLearningApplication(
        output_dir="results",
        random_state=42
    )
    
    success = app.run_complete_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("📁 Check the 'results' directory for all outputs")
    else:
        print("\n❌ Analysis failed. Check error messages above.")
        sys.exit(1)