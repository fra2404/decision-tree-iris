"""
Iris Dataset Processor for Ensemble Learning
============================================

This module handles data loading, preprocessing, and feature engineering
for the Iris dataset used in ensemble learning experiments.

Author: Francesco
Date: October 2025
Course: Artificial Intelligence - Assignment 3
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class IrisDataProcessor:
    """
    Advanced data processor for the Iris dataset with comprehensive
    preprocessing and analysis capabilities.
    """
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize the Iris data processor.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        self.class_distribution = None
        
    def load_and_analyze_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load Iris dataset and perform comprehensive analysis.
        
        Returns:
            X: Feature matrix
            y: Target vector
            analysis: Dictionary with dataset analysis
        """
        print("🌸 Loading Iris Dataset...")
        
        # Load the dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Create DataFrame for analysis
        df = pd.DataFrame(X, columns=self.feature_names)
        df['species'] = [self.target_names[i] for i in y]
        
        # Comprehensive analysis
        analysis = {
            'dataset_shape': X.shape,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'class_distribution': pd.Series(y).value_counts().sort_index(),
            'feature_statistics': df[self.feature_names].describe(),
            'correlation_matrix': df[self.feature_names].corr(),
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes
        }
        
        self.class_distribution = analysis['class_distribution']
        
        print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Classes: {', '.join(self.target_names)}")
        print(f"⚖️ Class balance: {analysis['class_distribution'].to_dict()}")
        
        return X, y, analysis
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                              stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train-test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            stratify: Whether to stratify the split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"🔄 Creating train-test split (test_size={self.test_size})...")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📉 Test set: {X_test.shape[0]} samples")
        
        # Verify class distribution
        train_dist = pd.Series(y_train).value_counts().sort_index()
        test_dist = pd.Series(y_test).value_counts().sort_index()
        
        print("🎯 Class distribution:")
        for i, class_name in enumerate(self.target_names):
            print(f"   {class_name}: Train={train_dist[i]}, Test={test_dist[i]}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_feature_scaling(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply standardization to features.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Scaled X_train, X_test
        """
        print("📏 Applying feature standardization...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Print scaling statistics
        print("📊 Feature scaling statistics:")
        for i, feature in enumerate(self.feature_names):
            mean_val = self.scaler.mean_[i]
            std_val = self.scaler.scale_[i]
            print(f"   {feature}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        return X_train_scaled, X_test_scaled
    
    def create_cv_folds(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> StratifiedKFold:
        """
        Create stratified cross-validation folds.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            
        Returns:
            StratifiedKFold object
        """
        print(f"🔄 Creating {n_splits}-fold stratified cross-validation...")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Verify fold balance
        fold_distributions = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_dist = pd.Series(y[val_idx]).value_counts().sort_index()
            fold_distributions.append(fold_dist)
            print(f"   Fold {fold_idx + 1}: {fold_dist.to_dict()}")
        
        return cv
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze feature importance using mutual information.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.feature_selection import mutual_info_classif
        
        print("🔍 Analyzing feature importance...")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Create importance dictionary
        importance_dict = dict(zip(self.feature_names, mi_scores))
        
        print("📊 Feature importance (Mutual Information):")
        for feature, score in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {score:.4f}")
        
        return importance_dict
    
    def create_visualization_data(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Create DataFrame optimized for visualization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with features and target
        """
        df = pd.DataFrame(X, columns=self.feature_names)
        df['species'] = [self.target_names[i] for i in y]
        df['species_numeric'] = y
        
        return df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing information
        """
        summary = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'scaling_applied': self.scaler is not None,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_classes': len(self.target_names) if self.target_names else 0,
            'class_distribution': self.class_distribution.to_dict() if self.class_distribution is not None else None
        }
        
        return summary


if __name__ == "__main__":
    # Example usage and testing
    processor = IrisDataProcessor(test_size=0.3, random_state=42)
    
    # Load and analyze data
    X, y, analysis = processor.load_and_analyze_data()
    
    # Create train-test split
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)
    
    # Apply scaling
    X_train_scaled, X_test_scaled = processor.apply_feature_scaling(X_train, X_test)
    
    # Create CV folds
    cv = processor.create_cv_folds(X_train, y_train)
    
    # Analyze feature importance
    importance = processor.analyze_feature_importance(X, y)
    
    # Get preprocessing summary
    summary = processor.get_preprocessing_summary()
    
    print("\n✅ Data preprocessing completed successfully!")
    print(f"📋 Summary: {summary}")