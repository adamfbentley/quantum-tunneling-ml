"""
Machine Learning Models for Quantum Dot Spin Classification
===========================================================

This module implements machine learning algorithms for classifying quantum dot
spin states from transport measurements.

Author: Adam Bentley
Institution: Victoria University of Wellington
Connection: Michele Governale's research group
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
import pickle
import warnings
warnings.filterwarnings('ignore')

class QuantumDotMLClassifier:
    """
    Machine learning pipeline for quantum dot spin state classification.
    
    This class provides a complete ML workflow including:
    - Data preprocessing and feature scaling
    - Multiple algorithm training and evaluation
    - Model comparison and selection
    - Physics-based interpretation of results
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ML classifier with multiple algorithms.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.class_names = None
        
        # Initialize algorithms
        self._setup_algorithms()
    
    def _setup_algorithms(self):
        """Setup machine learning algorithms for comparison."""
        self.algorithms = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=self.random_state
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', 
                gamma='scale',
                random_state=self.random_state
            ),
            'SVM (Linear)': SVC(
                kernel='linear',
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'Bagging': BaggingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
    
    def prepare_data(self, features, labels, feature_names=None, test_size=0.2):
        """
        Prepare data for machine learning training.
        
        Parameters:
        -----------
        features : array
            Feature matrix (n_samples, n_features)
        labels : array
            Target labels
        feature_names : list
            Names of features
        test_size : float
            Fraction of data for testing
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Split and scaled training/testing data
        """
        self.feature_names = feature_names
        
        # Create class names from unique labels
        unique_labels = np.unique(labels)
        spin_map = {0: 'Singlet (S=0)', 1: 'Doublet (S=1/2)', 
                   2: 'Triplet (S=1)', 3: 'Quartet (S=3/2)'}
        self.class_names = [spin_map.get(label, f'S={label/2}') for label in unique_labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=self.random_state, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Testing set: {X_test_scaled.shape[0]} samples")
        print(f"Features: {X_train_scaled.shape[1]}")
        print(f"Classes: {len(np.unique(labels))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_all_models(self, X_train=None, y_train=None, cv_folds=5):
        """
        Train all algorithms and evaluate performance.
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data (optional, uses stored data if None)
        cv_folds : int
            Number of cross-validation folds
        """
        if X_train is None:
            X_train, y_train = self.X_train, self.y_train
        
        print("Training machine learning models...")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, algorithm in self.algorithms.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(algorithm, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Train on full training set
            algorithm.fit(X_train, y_train)
            
            # Store model and results
            self.models[name] = algorithm
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nTraining completed!")
    
    def evaluate_models(self, X_test=None, y_test=None):
        """
        Evaluate all trained models on test set.
        
        Parameters:
        -----------
        X_test, y_test : arrays
            Test data (optional, uses stored data if None)
        """
        if X_test is None:
            X_test, y_test = self.X_test, self.y_test
        
        print("Evaluating models on test set...")
        print("=" * 40)
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.results[name]['test_accuracy'] = test_accuracy
            self.results[name]['predictions'] = y_pred
            
            print(f"{name:20s}: {test_accuracy:.4f}")
        
        # Find best model
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['test_accuracy'])
        print(f"\nBest model: {best_model} "
              f"(Accuracy: {self.results[best_model]['test_accuracy']:.4f})")
        
        return best_model
    
    def detailed_evaluation(self, model_name, X_test=None, y_test=None):
        """
        Provide detailed evaluation for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of model to evaluate
        """
        if X_test is None:
            X_test, y_test = self.X_test, self.y_test
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        print(f"Detailed Evaluation: {model_name}")
        print("=" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def feature_importance_analysis(self, model_name='Random Forest', top_k=10):
        """
        Analyze feature importance for interpretability.
        
        Parameters:
        -----------
        model_name : str
            Name of model to analyze (must support feature importance)
        top_k : int
            Number of top features to display
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]
        
        # Check if model supports feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            method = "Built-in"
        else:
            # Use permutation importance for models without built-in importance
            print("Using permutation importance (this may take a moment)...")
            perm_importance = permutation_importance(
                model, self.X_test, self.y_test, 
                n_repeats=10, random_state=self.random_state
            )
            importances = perm_importance.importances_mean
            method = "Permutation"
        
        # Sort features by importance
        if self.feature_names is not None:
            feature_importance = list(zip(self.feature_names, importances))
        else:
            feature_importance = list(zip([f'Feature_{i}' for i in range(len(importances))], 
                                        importances))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Feature Importance Analysis - {model_name} ({method})")
        print("=" * 60)
        
        for i, (feature, importance) in enumerate(feature_importance[:top_k]):
            print(f"{i+1:2d}. {feature:20s}: {importance:.4f}")
        
        # Visualization
        features, importances = zip(*feature_importance[:top_k])
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def physics_interpretation(self, feature_importance_list):
        """
        Provide physics-based interpretation of ML results.
        
        Parameters:
        -----------
        feature_importance_list : list
            List of (feature_name, importance) tuples
        """
        print("Physics Interpretation of ML Results")
        print("=" * 50)
        
        # Group features by physics category
        peak_features = ['peak_height', 'peak_position', 'peak_width', 'peak_asymmetry']
        area_features = ['total_area', 'baseline_level']
        noise_features = ['noise_level']
        shape_features = ['peak_sharpness', 'left_tail_slope', 'right_tail_slope']
        statistical_features = ['second_moment', 'skewness']
        
        categories = {
            'Peak Properties': peak_features,
            'Integral Properties': area_features,
            'Noise Characteristics': noise_features,
            'Shape Analysis': shape_features,
            'Statistical Moments': statistical_features
        }
        
        # Calculate category importance
        category_importance = {}
        for category, features in categories.items():
            total_importance = sum(imp for feat, imp in feature_importance_list 
                                 if feat in features)
            category_importance[category] = total_importance
        
        print("Importance by Physics Category:")
        for category, importance in sorted(category_importance.items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"  {category:20s}: {importance:.4f}")
        
        print("\nPhysics Insights:")
        
        # Most important individual features
        top_features = feature_importance_list[:3]
        for i, (feature, importance) in enumerate(top_features):
            print(f"\n{i+1}. {feature} (importance: {importance:.4f})")
            
            # Physics interpretation
            if feature == 'peak_height':
                print("   → Related to tunnel coupling strength and level degeneracy")
            elif feature == 'peak_position':
                print("   → Reflects chemical potential and charging energy")
            elif feature == 'peak_width':
                print("   → Temperature broadening and tunnel-coupling effects")
            elif feature == 'peak_asymmetry':
                print("   → Asymmetric tunnel barriers or energy-dependent coupling")
            elif feature == 'total_area':
                print("   → Total spectral weight and tunnel-coupling strength")
            elif feature == 'peak_sharpness':
                print("   → Temperature effects and coherent vs. incoherent transport")
            elif 'slope' in feature:
                print("   → Energy dependence of density of states")
            elif feature in statistical_features:
                print("   → Higher-order correlations and many-body effects")
            else:
                print("   → Complex many-body signature requiring further analysis")
        
        # Visualization of category importance
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        plt.figure(figsize=(10, 6))
        plt.pie(importances, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.title('Feature Importance by Physics Category')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filepath):
        """
        Save trained models and results.
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        save_data = {
            'models': self.models,
            'results': self.results,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath):
        """
        Load previously saved models and results.
        
        Parameters:
        -----------
        filepath : str
            Path to load results from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.results = save_data['results']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.class_names = save_data['class_names']
        
        print(f"Results loaded from {filepath}")
    
    def compare_models(self):
        """
        Generate comprehensive model comparison visualization.
        """
        if not self.results:
            print("No models trained yet!")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        cv_means = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]
        test_accs = [self.results[model].get('test_accuracy', 0) for model in models]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cross-validation results
        x_pos = np.arange(len(models))
        ax1.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_title('Cross-Validation Performance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Test set results
        ax2.bar(x_pos, test_accs, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Test Set Accuracy')
        ax2.set_title('Test Set Performance')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Performance summary table
        print("\nModel Performance Summary:")
        print("=" * 70)
        print(f"{'Model':<20} {'CV Accuracy':<15} {'CV Std':<10} {'Test Accuracy':<15}")
        print("-" * 70)
        
        for model in models:
            cv_mean = self.results[model]['cv_mean']
            cv_std = self.results[model]['cv_std']
            test_acc = self.results[model].get('test_accuracy', 0)
            
            print(f"{model:<20} {cv_mean:<15.4f} {cv_std:<10.4f} {test_acc:<15.4f}")


def run_complete_analysis(dataset, samples_per_class=200, test_size=0.2):
    """
    Run complete machine learning analysis pipeline.
    
    Parameters:
    -----------
    dataset : dict
        Dataset from quantum dot simulator
    samples_per_class : int
        Number of samples per spin class for balanced dataset
    test_size : float
        Fraction of data for testing
        
    Returns:
    --------
    classifier : QuantumDotMLClassifier
        Trained classifier with results
    """
    from quantum_simulation.quantum_dot_simulator import extract_features, create_balanced_dataset
    
    print("Quantum Dot Spin State Classification Analysis")
    print("=" * 60)
    
    # Create balanced dataset
    print("Creating balanced dataset...")
    if 'conductance' not in dataset:
        print("Error: Dataset must contain 'conductance' and 'labels' keys")
        return None
    
    # Extract features
    print("Extracting features from conductance data...")
    features, feature_names = extract_features(dataset['conductance'])
    labels = dataset['labels']
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Initialize classifier
    classifier = QuantumDotMLClassifier(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        features, labels, feature_names, test_size=test_size
    )
    
    # Train all models
    classifier.train_all_models(cv_folds=5)
    
    # Evaluate models
    best_model = classifier.evaluate_models()
    
    # Model comparison
    classifier.compare_models()
    
    # Detailed evaluation of best model
    print(f"\nDetailed evaluation of best model: {best_model}")
    classifier.detailed_evaluation(best_model)
    
    # Feature importance analysis
    print("\nFeature importance analysis...")
    importance_list = classifier.feature_importance_analysis(best_model, top_k=10)
    
    # Physics interpretation
    classifier.physics_interpretation(importance_list)
    
    return classifier


if __name__ == "__main__":
    # Demonstration of ML pipeline
    print("Quantum Dot ML Classification Module")
    print("=" * 40)
    
    # This would typically be run with data from the quantum simulator
    print("To use this module:")
    print("1. Generate data with quantum_dot_simulator.py")
    print("2. Run: classifier = run_complete_analysis(dataset)")
    print("3. Analyze results and physics insights")
    
    print("\nModule ready for quantum dot spin classification!")