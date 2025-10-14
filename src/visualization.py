"""
Visualization and Analysis Tools for Ensemble Learning
======================================================

This module provides comprehensive visualization and analysis capabilities
for ensemble learning experiments on the Iris dataset.

Author: Francesco
Date: October 2025
Course: Artificial Intelligence - Assignment 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnsembleVisualization:
    """
    Comprehensive visualization toolkit for ensemble learning analysis.
    """
    
    def __init__(self, feature_names=None, target_names=None, figsize=(12, 8)):
        """
        Initialize visualization toolkit.
        
        Args:
            feature_names: List of feature names
            target_names: List of target class names
            figsize: Default figure size
        """
        self.feature_names = feature_names if feature_names is not None else ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        self.target_names = target_names if target_names is not None else ['Class 0', 'Class 1', 'Class 2']
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C', '#FFB6C1', '#20B2AA']
        
    def plot_dataset_overview(self, X, y, save_path=None):
        """
        Create comprehensive dataset overview plots.
        
        Args:
            X: Feature matrix
            y: Target vector
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Iris Dataset Overview', fontsize=20, fontweight='bold')
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(X, columns=self.feature_names)
        df['species'] = [self.target_names[i] for i in y]
        
        # 1. Pairplot
        ax1 = plt.subplot(2, 3, 1)
        sns.scatterplot(data=df, x=self.feature_names[0], y=self.feature_names[1], 
                       hue='species', s=80, alpha=0.8)
        plt.title('Sepal Length vs Sepal Width', fontweight='bold')
        
        # 2. Feature distributions
        ax2 = plt.subplot(2, 3, 2)
        for i, species in enumerate(self.target_names):
            species_data = df[df['species'] == species][self.feature_names[0]]
            plt.hist(species_data, alpha=0.7, label=species, bins=15)
        plt.xlabel(self.feature_names[0])
        plt.ylabel('Frequency')
        plt.title('Feature Distribution', fontweight='bold')
        plt.legend()
        
        # 3. Correlation heatmap
        ax3 = plt.subplot(2, 3, 3)
        correlation_matrix = df[self.feature_names].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontweight='bold')
        
        # 4. Box plots
        ax4 = plt.subplot(2, 3, 4)
        df_melted = df.melt(id_vars=['species'], value_vars=self.feature_names,
                           var_name='feature', value_name='value')
        sns.boxplot(data=df_melted, x='feature', y='value', hue='species')
        plt.xticks(rotation=45)
        plt.title('Feature Distributions by Species', fontweight='bold')
        
        # 5. PCA visualization
        ax5 = plt.subplot(2, 3, 5)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        for i, species in enumerate(self.target_names):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=species, alpha=0.8, s=80)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization', fontweight='bold')
        plt.legend()
        
        # 6. Class distribution
        ax6 = plt.subplot(2, 3, 6)
        class_counts = pd.Series(y).value_counts().sort_index()
        bars = plt.bar(range(len(self.target_names)), class_counts.values, 
                      color=self.colors[:len(self.target_names)], alpha=0.8)
        plt.xticks(range(len(self.target_names)), self.target_names)
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution', fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dataset overview saved to {save_path}")
        
        plt.show()
        
    def plot_ensemble_performance_comparison(self, performance_results, save_path=None):
        """
        Create comprehensive performance comparison plots.
        
        Args:
            performance_results: Dictionary with performance metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Methods Performance Comparison', 
                    fontsize=20, fontweight='bold')
        
        # Extract data for plotting
        methods = list(performance_results.keys())
        metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
        
        # 1. Accuracy comparison
        ax1 = plt.subplot(2, 2, 1)
        train_acc = [performance_results[method]['train_accuracy'] for method in methods]
        test_acc = [performance_results[method]['test_accuracy'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, train_acc, width, label='Training', alpha=0.8)
        bars2 = plt.bar(x + width/2, test_acc, width, label='Testing', alpha=0.8)
        
        plt.xlabel('Ensemble Methods')
        plt.ylabel('Accuracy')
        plt.title('Training vs Testing Accuracy', fontweight='bold')
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0.8, 1.0)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Cross-validation scores with error bars
        ax2 = plt.subplot(2, 2, 2)
        cv_means = [performance_results[method]['cv_mean_accuracy'] for method in methods]
        cv_stds = [performance_results[method]['cv_std_accuracy'] for method in methods]
        
        bars = plt.bar(methods, cv_means, yerr=cv_stds, capsize=5, alpha=0.8,
                      color=self.colors[:len(methods)])
        plt.xlabel('Ensemble Methods')
        plt.ylabel('CV Accuracy')
        plt.title('Cross-Validation Performance', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add value labels
        for bar, mean, std in zip(bars, cv_means, cv_stds):
            plt.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Multi-metric comparison
        ax3 = plt.subplot(2, 2, 3)
        metric_data = []
        for metric in metrics:
            metric_values = [performance_results[method][metric] for method in methods]
            metric_data.append(metric_values)
        
        x = np.arange(len(methods))
        bar_width = 0.2
        
        for i, (metric, values) in enumerate(zip(metrics, metric_data)):
            plt.bar(x + i * bar_width, values, bar_width, 
                   label=metric.replace('_', ' ').title(), alpha=0.8)
        
        plt.xlabel('Ensemble Methods')
        plt.ylabel('Score')
        plt.title('Multi-Metric Performance', fontweight='bold')
        plt.xticks(x + bar_width * 1.5, methods, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0.8, 1.0)
        
        # 4. Performance ranking
        ax4 = plt.subplot(2, 2, 4)
        test_accuracies = [(method, performance_results[method]['test_accuracy']) 
                          for method in methods]
        test_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        ranked_methods, ranked_scores = zip(*test_accuracies)
        colors_ranked = [self.colors[i % len(self.colors)] for i in range(len(ranked_methods))]
        
        bars = plt.barh(range(len(ranked_methods)), ranked_scores, 
                       color=colors_ranked, alpha=0.8)
        plt.yticks(range(len(ranked_methods)), ranked_methods)
        plt.xlabel('Test Accuracy')
        plt.title('Performance Ranking', fontweight='bold')
        plt.xlim(0.8, 1.0)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, ranked_scores)):
            plt.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Performance comparison saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, importance_results, save_path=None):
        """
        Visualize feature importance from ensemble methods.
        
        Args:
            importance_results: Dictionary with feature importance arrays
            save_path: Path to save the plot
        """
        if not importance_results:
            print("⚠️ No feature importance data available")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Heatmap of feature importance
        ax1 = plt.subplot(1, 2, 1)
        
        # Create DataFrame for heatmap
        importance_df = pd.DataFrame(importance_results, index=self.feature_names)
        
        sns.heatmap(importance_df, annot=True, cmap='viridis', 
                   fmt='.3f', cbar_kws={'label': 'Importance'})
        plt.title('Feature Importance Heatmap', fontweight='bold')
        plt.ylabel('Features')
        plt.xlabel('Ensemble Methods')
        
        # 2. Grouped bar chart
        ax2 = plt.subplot(1, 2, 2)
        
        n_features = len(self.feature_names)
        n_methods = len(importance_results)
        x = np.arange(n_features)
        bar_width = 0.8 / n_methods
        
        for i, (method, importances) in enumerate(importance_results.items()):
            plt.bar(x + i * bar_width, importances, bar_width, 
                   label=method, alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance by Method', fontweight='bold')
        plt.xticks(x + bar_width * (n_methods - 1) / 2, self.feature_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔍 Feature importance plot saved to {save_path}")
        
        plt.show()
        
    def plot_confusion_matrices(self, ensemble_models, X_test, y_test, save_path=None):
        """
        Plot confusion matrices for all ensemble methods.
        
        Args:
            ensemble_models: Dictionary of trained ensemble models
            X_test: Test features
            y_test: Test targets
            save_path: Path to save the plot
        """
        n_models = len(ensemble_models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Confusion Matrices for Ensemble Methods', 
                    fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, model) in enumerate(ensemble_models.items()):
            row = idx // cols
            col = idx % cols
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            ax = axes[row, col] if rows > 1 else axes[col]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.target_names,
                       yticklabels=self.target_names,
                       ax=ax)
            ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎯 Confusion matrices saved to {save_path}")
        
        plt.show()
        
    def create_comprehensive_report(self, X, y, X_test, y_test, ensemble_models, 
                                  performance_results, importance_results, 
                                  output_dir="results"):
        """
        Create comprehensive visualization report.
        
        Args:
            X, y: Full dataset
            X_test, y_test: Test data
            ensemble_models: Dictionary of ensemble models
            performance_results: Performance metrics
            importance_results: Feature importance results
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Creating comprehensive visualization report...")
        
        # 1. Dataset overview
        self.plot_dataset_overview(X, y, 
                                  save_path=f"{output_dir}/iris_dataset_overview.png")
        
        # 2. Performance comparison
        self.plot_ensemble_performance_comparison(performance_results,
                                                save_path=f"{output_dir}/ensemble_performance_comparison.png")
        
        # 3. Feature importance
        if importance_results:
            self.plot_feature_importance(importance_results,
                                        save_path=f"{output_dir}/feature_importance_analysis.png")
        
        # 4. Confusion matrices
        self.plot_confusion_matrices(ensemble_models, X_test, y_test,
                                    save_path=f"{output_dir}/confusion_matrices.png")
        
        print(f"✅ Comprehensive report saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("🎨 Visualization module ready for ensemble learning analysis!")