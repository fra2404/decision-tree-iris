"""
Ensemble Learning Core Implementation
=====================================

This module implements various ensemble learning methods for classification
including Voting, Bagging, Random Forest, and Stacking approaches.

Author: Francesco
Date: October 2025
Course: Artificial Intelligence - Assignment 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Stacking Classifier implementation.
    """
    
    def __init__(self, base_classifiers: List, meta_classifier, cv_folds: int = 5):
        """
        Initialize stacking classifier.
        
        Args:
            base_classifiers: List of base classifiers
            meta_classifier: Meta-learner classifier
            cv_folds: Number of CV folds for meta-features
        """
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.cv_folds = cv_folds
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the stacking classifier."""
        from sklearn.model_selection import StratifiedKFold
        
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        n_classifiers = len(self.base_classifiers)
        
        # Create meta-features using cross-validation
        meta_features = np.zeros((n_samples, n_classifiers))
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, classifier in enumerate(self.base_classifiers):
                classifier.fit(X_train_fold, y_train_fold)
                meta_features[val_idx, i] = classifier.predict_proba(X_val_fold)[:, 1] if len(self.classes_) == 2 else classifier.predict(X_val_fold)
        
        # Train meta-classifier
        self.meta_classifier.fit(meta_features, y)
        
        # Retrain base classifiers on full dataset
        for classifier in self.base_classifiers:
            classifier.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using stacking."""
        meta_features = np.zeros((X.shape[0], len(self.base_classifiers)))
        
        for i, classifier in enumerate(self.base_classifiers):
            if len(self.classes_) == 2:
                meta_features[:, i] = classifier.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = classifier.predict(X)
        
        return self.meta_classifier.predict(meta_features)
    
    def predict_proba(self, X):
        """Predict class probabilities using stacking."""
        meta_features = np.zeros((X.shape[0], len(self.base_classifiers)))
        
        for i, classifier in enumerate(self.base_classifiers):
            if hasattr(classifier, 'predict_proba'):
                if len(self.classes_) == 2:
                    meta_features[:, i] = classifier.predict_proba(X)[:, 1]
                else:
                    meta_features[:, i] = classifier.predict(X)
            else:
                meta_features[:, i] = classifier.predict(X)
        
        if hasattr(self.meta_classifier, 'predict_proba'):
            return self.meta_classifier.predict_proba(meta_features)
        else:
            # Convert predictions to probabilities
            predictions = self.meta_classifier.predict(meta_features)
            proba = np.zeros((len(predictions), len(self.classes_)))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba


class EnsembleLearningCore:
    """
    Core ensemble learning implementation with multiple methods.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble learning core.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.base_classifiers = {}
        self.ensemble_models = {}
        self.performance_results = {}
        
    def create_base_classifiers(self) -> Dict[str, Any]:
        """
        Create diverse base classifiers for ensemble methods.
        
        Returns:
            Dictionary of base classifiers
        """
        print("🔧 Creating base classifiers...")
        
        self.base_classifiers = {
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'naive_bayes': GaussianNB()
        }
        
        print(f"✅ Created {len(self.base_classifiers)} base classifiers")
        return self.base_classifiers
    
    def create_voting_classifier(self, voting_type: str = 'soft') -> VotingClassifier:
        """
        Create voting classifier ensemble.
        
        Args:
            voting_type: 'hard' or 'soft' voting
            
        Returns:
            Configured VotingClassifier
        """
        print(f"🗳️ Creating {voting_type} voting classifier...")
        
        # Prepare estimators list
        estimators = [(name, clf) for name, clf in self.base_classifiers.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting_type
        )
        
        self.ensemble_models[f'voting_{voting_type}'] = voting_clf
        return voting_clf
    
    def create_bagging_classifier(self, base_estimator=None, n_estimators: int = 100) -> BaggingClassifier:
        """
        Create bagging classifier ensemble.
        
        Args:
            base_estimator: Base estimator (default: DecisionTree)
            n_estimators: Number of base estimators
            
        Returns:
            Configured BaggingClassifier
        """
        print(f"🎒 Creating bagging classifier with {n_estimators} estimators...")
        
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        
        bagging_clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=self.random_state,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=True
        )
        
        self.ensemble_models['bagging'] = bagging_clf
        return bagging_clf
    
    def create_random_forest(self, n_estimators: int = 100, max_features: str = 'sqrt') -> RandomForestClassifier:
        """
        Create random forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_features: Number of features to consider at each split
            
        Returns:
            Configured RandomForestClassifier
        """
        print(f"🌲 Creating random forest with {n_estimators} trees...")
        
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=self.random_state,
            bootstrap=True,
            oob_score=True,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.ensemble_models['random_forest'] = rf_clf
        return rf_clf
    
    def create_stacking_classifier(self) -> StackingClassifier:
        """
        Create stacking classifier ensemble.
        
        Returns:
            Configured StackingClassifier
        """
        print("🏗️ Creating stacking classifier...")
        
        # Select base classifiers for stacking
        base_clfs = [
            self.base_classifiers['decision_tree'],
            self.base_classifiers['svm'],
            self.base_classifiers['knn']
        ]
        
        # Use logistic regression as meta-classifier
        meta_clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        stacking_clf = StackingClassifier(
            base_classifiers=base_clfs,
            meta_classifier=meta_clf,
            cv_folds=5
        )
        
        self.ensemble_models['stacking'] = stacking_clf
        return stacking_clf
    
    def create_boosting_classifiers(self) -> Dict[str, Any]:
        """
        Create boosting-based ensemble classifiers.
        
        Returns:
            Dictionary of boosting classifiers
        """
        print("🚀 Creating boosting classifiers...")
        
        # AdaBoost
        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=self.random_state),
            n_estimators=100,
            learning_rate=1.0,
            random_state=self.random_state
        )
        
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state
        )
        
        boosting_clfs = {
            'adaboost': ada_clf,
            'gradient_boosting': gb_clf
        }
        
        self.ensemble_models.update(boosting_clfs)
        return boosting_clfs
    
    def evaluate_ensemble_performance(self, X_train, X_test, y_train, y_test, cv_folds=5) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of all ensemble methods.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with performance metrics for each ensemble
        """
        print("📊 Evaluating ensemble performance...")
        
        results = {}
        
        for name, model in self.ensemble_models.items():
            print(f"   Evaluating {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Calculate metrics
            results[name] = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'precision': precision_score(y_test, test_pred, average='weighted'),
                'recall': recall_score(y_test, test_pred, average='weighted'),
                'f1_score': f1_score(y_test, test_pred, average='weighted')
            }
            
            # Add OOB score if available
            if hasattr(model, 'oob_score_'):
                results[name]['oob_score'] = model.oob_score_
        
        self.performance_results = results
        return results
    
    def get_feature_importance_analysis(self) -> Dict[str, np.ndarray]:
        """
        Extract feature importance from tree-based ensembles.
        
        Returns:
            Dictionary of feature importance arrays
        """
        print("🔍 Analyzing feature importance...")
        
        importance_results = {}
        
        for name, model in self.ensemble_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_results[name] = model.feature_importances_
            elif name == 'voting_soft' or name == 'voting_hard':
                # Average importance from tree-based estimators in voting
                importances = []
                for est_name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    importance_results[name] = np.mean(importances, axis=0)
        
        return importance_results
    
    def hyperparameter_optimization(self, X_train, y_train, model_name: str, param_grid: Dict) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for specified ensemble.
        
        Args:
            X_train, y_train: Training data
            model_name: Name of ensemble model to optimize
            param_grid: Parameter grid for optimization
            
        Returns:
            Best parameters and scores
        """
        print(f"⚙️ Optimizing hyperparameters for {model_name}...")
        
        if model_name not in self.ensemble_models:
            raise ValueError(f"Model {model_name} not found in ensemble models")
        
        model = self.ensemble_models[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid,
            cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        optimization_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        # Update model with best parameters
        self.ensemble_models[model_name] = grid_search.best_estimator_
        
        return optimization_results
    
    def create_all_ensembles(self) -> Dict[str, Any]:
        """
        Create all ensemble methods in one call.
        
        Returns:
            Dictionary of all created ensemble models
        """
        print("🎯 Creating all ensemble methods...")
        
        # Create base classifiers first
        self.create_base_classifiers()
        
        # Create all ensemble types
        self.create_voting_classifier('soft')
        self.create_voting_classifier('hard')
        self.create_bagging_classifier()
        self.create_random_forest()
        self.create_stacking_classifier()
        self.create_boosting_classifiers()
        
        print(f"✅ Created {len(self.ensemble_models)} ensemble models:")
        for name in self.ensemble_models.keys():
            print(f"   - {name}")
        
        return self.ensemble_models


if __name__ == "__main__":
    # Example usage and testing
    from data_processor import IrisDataProcessor
    
    # Load data
    processor = IrisDataProcessor()
    X, y, analysis = processor.load_and_analyze_data()
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)
    X_train_scaled, X_test_scaled = processor.apply_feature_scaling(X_train, X_test)
    
    # Create ensemble core
    ensemble_core = EnsembleLearningCore()
    
    # Create all ensemble methods
    ensembles = ensemble_core.create_all_ensembles()
    
    # Evaluate performance
    results = ensemble_core.evaluate_ensemble_performance(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Print results
    print("\n📈 Performance Results:")
    for name, metrics in results.items():
        print(f"{name}: Test Accuracy = {metrics['test_accuracy']:.4f}")
    
    print("\n✅ Ensemble learning core tested successfully!")