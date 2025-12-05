#!/usr/bin/env python3
"""
K-Nearest Neighbors Training Script for NeuroFlow
Supports Classification and Regression
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress sklearn warnings specifically
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)

# Auto-install missing libraries
def check_and_install_libraries():
    required = {
        'pandas': 'pandas', 'numpy': 'numpy', 'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib', 'seaborn': 'seaborn', 'imbalanced-learn': 'imblearn', 
        'joblib': 'joblib', 'scipy': 'scipy'
    }
    missing = []
    for lib_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(lib_name)
    if missing:
        print(f"📦 Installing: {', '.join(missing)}")
        import subprocess
        for lib in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "-q"])
        print("✅ All libraries installed!\n")

check_and_install_libraries()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
import joblib
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', required=True)
    parser.add_argument('--test_csv_path', default=None)
    parser.add_argument('--test_split_ratio', type=float, default=0.2)
    parser.add_argument('--train_columns', required=True)
    parser.add_argument('--output_column', required=True)
    parser.add_argument('--selected_graphs', default='[]')
    parser.add_argument('--selected_missingval_tech', default='[]')
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--encoding_type', default='one-hot')
    parser.add_argument('--feature_scaling', default='standard')
    parser.add_argument('--selected_explorations', default='[]')
    
    # KNN specific
    parser.add_argument('--k_value', type=int, default=5)
    parser.add_argument('--enable_auto_k', default='false')
    parser.add_argument('--k_range_start', type=int, default=1)
    parser.add_argument('--k_range_end', type=int, default=20)
    parser.add_argument('--distance_metric', default='euclidean')
    parser.add_argument('--weights', default='uniform')
    parser.add_argument('--algorithm', default='auto')
    parser.add_argument('--leaf_size', type=int, default=30)
    parser.add_argument('--p_value', type=int, default=2)
    
    # Outlier detection
    parser.add_argument('--enable_outlier_detection', default='false')
    parser.add_argument('--outlier_method', default='Z-score')
    parser.add_argument('--z_score_threshold', type=float, default=3.0)
    
    # Advanced
    parser.add_argument('--enable_cv', default='false')
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--enable_dim_reduction', default='false')
    parser.add_argument('--dim_reduction_method', default='PCA')
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--enable_imbalance', default='false')
    parser.add_argument('--imbalance_method', default=None)
    
    # Effect features
    parser.add_argument('--effect_features', default='[]')
    
    return parser.parse_args()

def normalize_path(path):
    """Convert path separators to forward slashes for consistent handling"""
    return path.replace('\\', '/')

def sanitize_filename(name):
    """Clean filename for safe file operations"""
    import re
    name = re.sub(r'[^\w\s-]', '_', name)
    name = re.sub(r'[\s_]+', '_', name)
    return name.strip('_')

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("K-NEAREST NEIGHBORS TRAINING")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 80)
    
    # Parse JSON parameters
    train_columns = json.loads(args.train_columns)
    output_column = args.output_column
    selected_graphs = json.loads(args.selected_graphs)
    selected_explorations = json.loads(args.selected_explorations)
    missing_val_tech = json.loads(args.selected_missingval_tech)
    effect_features = json.loads(args.effect_features) if args.effect_features else []
    
    # Clean column names by removing extra quotes
    train_columns = [col.strip().strip('"').strip("'") for col in train_columns]
    output_column = output_column.strip().strip('"').strip("'")
    
    enable_cv = args.enable_cv.lower() == 'true'
    enable_auto_k = args.enable_auto_k.lower() == 'true'
    enable_outlier = args.enable_outlier_detection.lower() == 'true'
    enable_dim_reduction = args.enable_dim_reduction.lower() == 'true'
    enable_imbalance = args.enable_imbalance.lower() == 'true'
    
    # Load data
    print("\n📂 Loading Training Data...")
    print(f"   Path: {args.train_csv_path}")
    
    # Validate file exists
    if not os.path.exists(args.train_csv_path):
        print(f"❌ Training dataset not found at {args.train_csv_path}")
        sys.exit(1)
    
    try:
        if args.train_csv_path.endswith('.csv'):
            df = pd.read_csv(args.train_csv_path)
        elif args.train_csv_path.endswith('.xlsx'):
            df = pd.read_excel(args.train_csv_path)
        else:
            print("❌ Unsupported format. Use CSV or XLSX")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load dataset: {str(e)}")
        print("   File may be corrupted or inaccessible")
        sys.exit(1)
    
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data Exploration
    if selected_explorations:
        print("\n" + "=" * 80)
        print("📊 DATA EXPLORATION")
        print("=" * 80)
        for technique in selected_explorations:
            if technique == "First 5 Rows":
                print("\n📋 First 5 Rows:\n", df.head())
            elif technique == "Last 5 Rows":
                print("\n📋 Last 5 Rows:\n", df.tail())
            elif technique == "Dataset Shape":
                print(f"\n📐 Shape: {df.shape}")
            elif technique == "Data Types":
                print("\n📊 Data Types:\n", df.dtypes)
            elif technique == "Summary Statistics":
                print("\n📈 Summary Stats:\n", df.describe())
            elif technique == "Missing Values":
                missing = df.isnull().sum()
                print("\n❓ Missing Values:\n", missing[missing > 0] if missing.sum() > 0 else "None")
            elif technique == "Duplicate Rows":
                print(f"\n🔄 Duplicates: {df.duplicated().sum()}")
    
    # Handle missing values
    if missing_val_tech:
        print("\n" + "=" * 80)
        print("🧹 DATA CLEANING")
        print("=" * 80)
        for tech in missing_val_tech:
            if tech == "Drop Missing Rows":
                before = len(df)
                df = df.dropna()
                print(f"✅ Dropped {before - len(df)} rows with missing values")
            elif tech == "Fill with Mean":
                df = df.fillna(df.mean(numeric_only=True))
                print("✅ Filled missing values with mean")
            elif tech == "Fill with Median":
                df = df.fillna(df.median(numeric_only=True))
                print("✅ Filled missing values with median")
            elif tech == "Fill with Mode":
                for col in df.columns:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                print("✅ Filled missing values with mode")
    
    # Remove duplicates
    if args.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        print(f"✅ Removed {before - len(df)} duplicate rows")
    
    # Prepare features and target
    X = df[train_columns]
    y = df[output_column]
    
    # Handle any remaining NaN values in features before encoding
    # Fill numeric columns with median, categorical columns will be filled during encoding
    numeric_features = X.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 0 and X[numeric_features].isnull().any().any():
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
        print(f"✅ Filled NaN in {len(numeric_features)} numeric feature columns with median")
    
    # Initialize encoded_feature_names (will be updated after encoding)
    encoded_feature_names = list(X.columns)
    
    # Determine problem type and store original class count
    unique_classes = len(y.unique())
    is_classification = unique_classes < 20 or y.dtype == 'object'
    is_binary = is_classification and unique_classes == 2
    problem_type = "Classification" if is_classification else "Regression"
    print(f"\n🎯 Problem Type: {problem_type}")
    print(f"   Target: {output_column}")
    print(f"   Features: {len(train_columns)}")
    print(f"   Classes: {unique_classes}" if is_classification else f"   Range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Encode categorical variables
    encoding_type_lower = args.encoding_type.lower().replace('-', '')  # Normalize: 'one-hot' -> 'onehot'
    if encoding_type_lower != 'none':
        print(f"\n🔤 Encoding: {args.encoding_type}")
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0:
            if encoding_type_lower == 'onehot':
                # Handle NaN in categorical columns before one-hot encoding
                for col in categorical_cols:
                    X[col] = X[col].fillna('Missing')
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                print(f"✅ Encoded {len(categorical_cols)} categorical columns")
                print(f"   Features after encoding: {X.shape[1]}")
            elif encoding_type_lower == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
                print(f"✅ Encoded {len(categorical_cols)} categorical columns")
        
        # Verify no object columns remain
        remaining_objects = X.select_dtypes(include=['object']).columns.tolist()
        if len(remaining_objects) > 0:
            print(f"⚠️ Warning: Still have object columns: {remaining_objects}")
            # Convert any remaining object columns to string then label encode
            for col in remaining_objects:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
        
        # Update feature names after encoding
        encoded_feature_names = list(X.columns)
        print(f"   Encoded feature names ({len(encoded_feature_names)}): {encoded_feature_names[:5]}...")
    
    # Encode target if classification
    label_encoder = None
    if is_classification and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"✅ Encoded target column")
    
    # Split data
    test_size = args.test_split_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if is_classification else None
    )
    print(f"\n📊 Train/Test Split: {1-test_size:.0%}/{test_size:.0%}")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Outlier detection
    if enable_outlier:
        print(f"\n🔍 Outlier Detection: {args.outlier_method}")
        if args.outlier_method == 'Z-score':
            z_scores = np.abs(stats.zscore(X_train.select_dtypes(include=[np.number])))
            mask = (z_scores < args.z_score_threshold).all(axis=1)
            before = len(X_train)
            X_train = X_train[mask]
            y_train = y_train[mask]
            print(f"✅ Removed {before - len(X_train)} outliers")
    
    # Feature Scaling (REQUIRED for KNN)
    print(f"\n⚖️ Feature Scaling: {args.feature_scaling}")
    if args.feature_scaling == 'standard':
        scaler = StandardScaler()
    elif args.feature_scaling == 'minmax':
        scaler = MinMaxScaler()
    elif args.feature_scaling == 'robust':
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Initialize PCA as None (will be set if dimensionality reduction is used)
    pca = None
    
    # Handle class imbalance
    if enable_imbalance and is_classification:
        print(f"\n⚖️ Handling Imbalance: {args.imbalance_method}")
        if args.imbalance_method == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        elif args.imbalance_method == 'Random Oversampling':
            ros = RandomOverSampler(random_state=42)
            X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)
        elif args.imbalance_method == 'Random Undersampling':
            rus = RandomUnderSampler(random_state=42)
            X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
        print(f"✅ Training samples: {X_train_scaled.shape[0]}")
    
    # Dimensionality Reduction
    if enable_dim_reduction:
        print(f"\n📉 Dimensionality Reduction: {args.dim_reduction_method}")
        if args.dim_reduction_method == 'PCA':
            pca = PCA(n_components=min(args.n_components, X_train_scaled.shape[1]))
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            print(f"✅ Reduced to {X_train_scaled.shape[1]} components")
            print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Auto K-finder
    k_value = args.k_value
    if enable_auto_k:
        print(f"\n🔍 Finding Optimal K (range: {args.k_range_start}-{args.k_range_end})...")
        k_scores = []
        k_range = range(args.k_range_start, args.k_range_end + 1)
        
        for k in k_range:
            if is_classification:
                temp_model = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=args.distance_metric,
                    weights=args.weights,
                    algorithm=args.algorithm,
                    leaf_size=args.leaf_size,
                    p=args.p_value if args.distance_metric == 'minkowski' else 2
                )
            else:
                temp_model = KNeighborsRegressor(
                    n_neighbors=k,
                    metric=args.distance_metric,
                    weights=args.weights,
                    algorithm=args.algorithm,
                    leaf_size=args.leaf_size,
                    p=args.p_value if args.distance_metric == 'minkowski' else 2
                )
            
            temp_model.fit(X_train_scaled, y_train)
            y_pred = temp_model.predict(X_test_scaled)
            
            if is_classification:
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)
            
            k_scores.append(score)
        
        optimal_k = k_range[np.argmax(k_scores)]
        k_value = optimal_k
        print(f"✅ Optimal K: {k_value} (score: {max(k_scores):.4f})")
    
    # Train KNN Model
    print("\n" + "=" * 80)
    print("🤖 TRAINING KNN MODEL")
    print("=" * 80)
    print(f"K (neighbors): {k_value}")
    print(f"Distance Metric: {args.distance_metric}")
    print(f"Weights: {args.weights}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Leaf Size: {args.leaf_size}")
    if args.distance_metric == 'minkowski':
        print(f"P-value: {args.p_value}")
    
    if is_classification:
        model = KNeighborsClassifier(
            n_neighbors=k_value,
            metric=args.distance_metric,
            weights=args.weights,
            algorithm=args.algorithm,
            leaf_size=args.leaf_size,
            p=args.p_value if args.distance_metric == 'minkowski' else 2
        )
    else:
        model = KNeighborsRegressor(
            n_neighbors=k_value,
            metric=args.distance_metric,
            weights=args.weights,
            algorithm=args.algorithm,
            leaf_size=args.leaf_size,
            p=args.p_value if args.distance_metric == 'minkowski' else 2
        )
    
    model.fit(X_train_scaled, y_train)
    print("✅ Model trained successfully!")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("📊 MODEL EVALUATION")
    print("=" * 80)
    
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'problem_type': 'classification',
            'k_value': k_value
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save metrics
        metrics = {
            'r2_score': float(r2),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'problem_type': 'regression',
            'k_value': k_value
        }
    
    # Create output directory with proper naming convention
    base_path = os.path.dirname(args.train_csv_path)
    csv_filename = os.path.basename(args.train_csv_path).split('.')[0]
    output_dir = os.path.join(base_path, f"knn-{csv_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Cross-Validation with fold model saving
    available_models = []
    cv_fold_models = []
    if enable_cv:
        print(f"\n🔄 Cross-Validation ({args.cv_folds} folds)...")
        print("Training and saving individual fold models...")
        
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Choose appropriate CV splitter
        if is_classification:
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), 1):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Adjust K if needed for small folds
            fold_k = min(k_value, len(X_fold_train) - 1)
            
            # Train model on this fold
            if is_classification:
                fold_model = KNeighborsClassifier(
                    n_neighbors=fold_k,
                    metric=args.distance_metric,
                    weights=args.weights,
                    algorithm=args.algorithm,
                    leaf_size=args.leaf_size,
                    p=args.p_value if args.distance_metric == 'minkowski' else 2
                )
            else:
                fold_model = KNeighborsRegressor(
                    n_neighbors=fold_k,
                    metric=args.distance_metric,
                    weights=args.weights,
                    algorithm=args.algorithm,
                    leaf_size=args.leaf_size,
                    p=args.p_value if args.distance_metric == 'minkowski' else 2
                )
            
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            y_fold_pred = fold_model.predict(X_fold_val)
            
            if is_classification:
                fold_score = accuracy_score(y_fold_val, y_fold_pred)
                fold_precision = precision_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
                fold_recall = recall_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
                fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
                
                print(f"  Fold {fold_idx}: Accuracy={fold_score:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}, F1={fold_f1:.4f}")
                
                cv_fold_models.append({
                    'fold': fold_idx,
                    'model': fold_model,
                    'accuracy': fold_score,
                    'precision': fold_precision,
                    'recall': fold_recall,
                    'f1_score': fold_f1
                })
            else:
                fold_score = r2_score(y_fold_val, y_fold_pred)
                fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
                fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
                
                print(f"  Fold {fold_idx}: R²={fold_score:.4f}, MSE={fold_mse:.4f}, MAE={fold_mae:.4f}")
                
                cv_fold_models.append({
                    'fold': fold_idx,
                    'model': fold_model,
                    'r2_score': fold_score,
                    'mse': fold_mse,
                    'mae': fold_mae
                })
            
            fold_scores.append(fold_score)
            
            # Save fold model
            try:
                fold_model_path = os.path.join(output_dir, f"model_fold_{fold_idx}.pkl")
                joblib.dump(fold_model, fold_model_path)
            except Exception as e:
                print(f"⚠️ Warning: Failed to save fold {fold_idx} model: {str(e)}")
            
            # Add to available models with all metrics
            fold_model_info = {
                'name': f"CV Fold {fold_idx}",
                'filename': f"model_fold_{fold_idx}.pkl",
                'type': 'cv_fold',
                'fold_number': fold_idx
            }
            
            if is_classification:
                fold_model_info.update({
                    'accuracy': float(cv_fold_models[-1]['accuracy']),
                    'precision': float(cv_fold_models[-1]['precision']),
                    'recall': float(cv_fold_models[-1]['recall']),
                    'f1_score': float(cv_fold_models[-1]['f1_score'])
                })
            else:
                fold_model_info.update({
                    'r2_score': float(cv_fold_models[-1]['r2_score']),
                    'mse': float(cv_fold_models[-1]['mse']),
                    'mae': float(cv_fold_models[-1]['mae']),
                    'rmse': float(np.sqrt(cv_fold_models[-1]['mse']))
                })
            
            available_models.append(fold_model_info)
        
        print(f"\n✅ Mean CV Score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores) * 2:.4f})")
        print(f"✅ Saved {len(cv_fold_models)} fold models")
        
        # Print comprehensive results table for all CV folds
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE RESULTS TABLE")
        print("=" * 80)
        
        if is_classification:
            # Classification results table
            print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 80)
            for fold_info in cv_fold_models:
                # Calculate metrics for this fold (stored in fold_info)
                if 'accuracy' in fold_info:
                    print(f"CV Fold {fold_info['fold']:<17} {fold_info.get('accuracy', 0):<12.4f} "
                          f"{fold_info.get('precision', 0):<12.4f} {fold_info.get('recall', 0):<12.4f} "
                          f"{fold_info.get('f1_score', 0):<12.4f}")
            print("-" * 80)
            print(f"{'Final Model (Full Data)':<25} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
            print("=" * 80)
        else:
            # Regression results table
            print(f"{'Model':<25} {'R² Score':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
            print("-" * 80)
            for fold_info in cv_fold_models:
                fold_rmse = np.sqrt(fold_info['mse'])
                print(f"CV Fold {fold_info['fold']:<17} {fold_info['r2_score']:<12.4f} "
                      f"{fold_info['mse']:<12.4f} {fold_info['mae']:<12.4f} {fold_rmse:<12.4f}")
            print("-" * 80)
            print(f"{'Final Model (Full Data)':<25} {r2:<12.4f} {mse:<12.4f} {mae:<12.4f} {rmse:<12.4f}")
            print("=" * 80)
        
        metrics['cv_scores'] = [float(s) for s in fold_scores]
        metrics['cv_mean'] = float(np.mean(fold_scores))
        metrics['cv_std'] = float(np.std(fold_scores))
    
    # Save metrics
    try:
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved: {metrics_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save metrics: {str(e)}")
    
    # Prepare categorical info for predictions
    categorical_cols = []
    categorical_values = {}
    numeric_cols = list(X.columns)
    
    if encoding_type_lower != 'none':
        original_categorical_cols = df[train_columns].select_dtypes(include=['object']).columns.tolist()
        for col in original_categorical_cols:
            categorical_cols.append(col)
            categorical_values[col] = sorted(df[col].dropna().unique().tolist())
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Determine final feature names after all transformations
    if enable_dim_reduction and args.dim_reduction_method == 'PCA':
        # After PCA, features are named PCA_0, PCA_1, etc.
        final_feature_names = [f'PCA_{i}' for i in range(X_train_scaled.shape[1])]
    else:
        # No dimensionality reduction, use encoded feature names
        final_feature_names = encoded_feature_names
    
    print(f"\n📋 Final feature names for model: {len(final_feature_names)} features")
    print(f"   First 5: {final_feature_names[:5]}")
    
    # Save preprocessing info
    preprocessing_info = {
        'original_train_columns': train_columns,
        'feature_names': list(X.columns),
        'final_feature_names': final_feature_names,  # Add final feature names for predict.py
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'categorical_values': categorical_values,
        'label_encoder': label_encoder,
        'label_encoders': {},  # For compatibility with predict.py (not used in KNN)
        'target_means': {},  # For compatibility with predict.py (not used in KNN)
        'encoding_added_cols': {},  # For compatibility with predict.py (not used in KNN)
        'is_binary_classification': is_binary,  # Use the stored is_binary variable
        'target_name': output_column,
        'is_classification': is_classification,
        'k_value': k_value,
        'distance_metric': args.distance_metric,
        'weights': args.weights,
        'encoding_type': args.encoding_type,
        'feature_scaling': args.feature_scaling,
        'pca_transformer': pca,  # Save PCA transformer if used
        'enable_dim_reduction': enable_dim_reduction,
        'available_models': available_models
    }
    
    try:
        preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')
        joblib.dump(preprocessing_info, preprocessing_path)
        print(f"💾 Preprocessing info saved: {preprocessing_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save preprocessing info: {str(e)}")
    
    # Save scaler separately
    try:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"💾 Scaler saved: {scaler_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save scaler: {str(e)}")
    
    # Save main model
    try:
        model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"💾 Main model saved: {model_path}")
    except Exception as e:
        print(f"❌ Error: Failed to save main model: {str(e)}")
        sys.exit(1)
    
    # Add main model to available models with all metrics
    main_model_info = {
        'name': f"Final Model ({'CV-Trained on Full Data' if enable_cv else 'Train/Test Split'})",
        'filename': 'model.pkl',
        'type': 'final',
        'trained_on_full': enable_cv
    }
    
    if is_classification:
        main_model_info.update({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        })
    else:
        main_model_info.update({
            'r2_score': float(metrics['r2_score']),
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse'])
        })
    
    available_models.insert(0, main_model_info)
    
    # Update preprocessing_info with complete available_models list
    preprocessing_info['available_models'] = available_models
    try:
        preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')
        joblib.dump(preprocessing_info, preprocessing_path)
        print(f"💾 Preprocessing info updated with all models: {preprocessing_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to update preprocessing info: {str(e)}")
    
    # Save available models list
    try:
        models_list_path = os.path.join(output_dir, 'available_models.json')
        with open(models_list_path, 'w') as f:
            json.dump(available_models, f, indent=2)
        print(f"💾 Available models list saved: {models_list_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save available models list: {str(e)}")
    
    # Generate graphs
    generated_graphs = []
    if selected_graphs:
        print("\n" + "=" * 80)
        print("📈 GENERATING GRAPHS")
        print("=" * 80)
        
        for graph_name in selected_graphs:
            try:
                # Skip graphs that aren't applicable
                if graph_name == "K vs Accuracy" and not enable_auto_k:
                    print(f"⚠️ Skipped {graph_name}: Auto K-finder not enabled")
                    continue
                
                if graph_name in ["Confusion Matrix", "ROC Curve", "Precision-Recall"] and not is_classification:
                    print(f"⚠️ Skipped {graph_name}: Only for classification problems")
                    continue
                
                if graph_name in ["Actual vs Predicted", "Predicted vs Actual", "Residual Plot", "Error Distribution"] and is_classification:
                    print(f"⚠️ Skipped {graph_name}: Only for regression problems")
                    continue
                
                # Skip advanced visualizations that need 2 features
                if graph_name in ["Distance Distribution", "Decision Boundary", "Neighbor Analysis", "Feature Impact"]:
                    if X_train_scaled.shape[1] != 2:
                        print(f"⚠️ Skipped {graph_name}: Requires exactly 2 features")
                        continue
                
                plt.figure(figsize=(10, 6))
                
                if graph_name == "Confusion Matrix":
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                
                elif graph_name == "K vs Accuracy":
                    plt.plot(k_range, k_scores, marker='o', linewidth=2)
                    plt.axvline(k_value, color='r', linestyle='--', linewidth=2, label=f'Selected K={k_value}')
                    plt.xlabel('K Value (Number of Neighbors)')
                    plt.ylabel('Accuracy' if is_classification else 'R² Score')
                    plt.title('K Value vs Model Performance')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "ROC Curve":
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_curve, auc
                    n_classes = len(np.unique(y_test))
                    
                    if n_classes == 2:
                        y_score = model.predict_proba(X_test_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_score)
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    else:
                        y_score = model.predict_proba(X_test_scaled)
                        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, linewidth=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "Precision-Recall":
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    n_classes = len(np.unique(y_test))
                    
                    if n_classes == 2:
                        y_score = model.predict_proba(X_test_scaled)[:, 1]
                        precision, recall, _ = precision_recall_curve(y_test, y_score)
                        avg_precision = average_precision_score(y_test, y_score)
                        plt.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avg_precision:.2f})')
                    else:
                        y_score = model.predict_proba(X_test_scaled)
                        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                        for i in range(n_classes):
                            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                            avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
                            plt.plot(recall, precision, linewidth=2, label=f'Class {i} (AP = {avg_precision:.2f})')
                    
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name in ["Actual vs Predicted", "Predicted vs Actual"]:
                    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.title('Predicted vs Actual Values')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "Residual Plot":
                    residuals = y_test - y_pred
                    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
                    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "Error Distribution":
                    errors = np.abs(y_test - y_pred)
                    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
                    plt.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean Error = {errors.mean():.2f}')
                    plt.xlabel('Absolute Error')
                    plt.ylabel('Frequency')
                    plt.title('Error Distribution')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "Correlation Heatmap":
                    corr = pd.DataFrame(X_train_scaled, columns=X.columns).corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                    plt.title('Feature Correlation Heatmap')
                
                elif graph_name == "PCA Visualization":
                    if X_train_scaled.shape[1] > 2:
                        from sklearn.decomposition import PCA as PCAViz
                        pca_viz = PCAViz(n_components=2)
                        X_pca = pca_viz.fit_transform(X_train_scaled)
                        
                        if is_classification:
                            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6, edgecolors='k')
                            plt.colorbar(scatter, label='Class')
                        else:
                            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, edgecolors='k')
                            plt.colorbar(scatter, label='Target Value')
                        
                        plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)')
                        plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)')
                        plt.title('PCA Visualization (2D Projection)')
                        plt.grid(True, alpha=0.3)
                    else:
                        print(f"⚠️ Skipped {graph_name}: Need >2 features for PCA visualization")
                        plt.close()
                        continue
                
                elif graph_name == "Box Plots":
                    df_plot = pd.DataFrame(X_train_scaled, columns=X.columns)
                    df_plot.boxplot(figsize=(12, 6), rot=45)
                    plt.ylabel('Scaled Values')
                    plt.title('Feature Distribution (Box Plots)')
                    plt.tight_layout()
                
                elif graph_name == "Decision Boundary":
                    # Only works with exactly 2 features
                    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
                    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                         np.arange(y_min, y_max, 0.02))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
                    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                                        c=y_train, cmap='viridis', edgecolors='k', s=50)
                    plt.colorbar(scatter, label='Class' if is_classification else 'Value')
                    plt.xlabel(X.columns[0])
                    plt.ylabel(X.columns[1])
                    plt.title('KNN Decision Boundary')
                
                elif graph_name == "Distance Distribution":
                    # Show distribution of distances to K-nearest neighbors
                    distances, _ = model.kneighbors(X_test_scaled)
                    mean_distances = distances.mean(axis=1)
                    plt.hist(mean_distances, bins=20, edgecolor='black', alpha=0.7)
                    plt.axvline(mean_distances.mean(), color='r', linestyle='--', 
                               linewidth=2, label=f'Mean = {mean_distances.mean():.2f}')
                    plt.xlabel('Mean Distance to K Neighbors')
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of Distances (K={k_value})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif graph_name == "Neighbor Analysis":
                    # Scatter plot showing test points colored by prediction confidence
                    distances, indices = model.kneighbors(X_test_scaled)
                    if is_classification:
                        proba = model.predict_proba(X_test_scaled)
                        confidence = proba.max(axis=1)
                        scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                                            c=confidence, cmap='RdYlGn', edgecolors='k', s=50)
                        plt.colorbar(scatter, label='Prediction Confidence')
                    else:
                        scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                                            c=y_pred, cmap='coolwarm', edgecolors='k', s=50)
                        plt.colorbar(scatter, label='Predicted Value')
                    plt.xlabel(X.columns[0])
                    plt.ylabel(X.columns[1])
                    plt.title('Test Set Predictions with Neighbor Analysis')
                
                elif graph_name == "Feature Impact":
                    # Show how predictions vary as we move along each feature axis
                    feature_ranges = []
                    for i in range(2):
                        feature_vals = np.linspace(X_train_scaled[:, i].min(), 
                                                   X_train_scaled[:, i].max(), 100)
                        other_feature = np.median(X_train_scaled[:, 1-i])
                        if i == 0:
                            test_points = np.column_stack([feature_vals, 
                                                          np.full(100, other_feature)])
                        else:
                            test_points = np.column_stack([np.full(100, other_feature), 
                                                          feature_vals])
                        predictions = model.predict(test_points)
                        plt.plot(feature_vals, predictions, label=X.columns[i], linewidth=2)
                    plt.xlabel('Feature Value (Scaled)')
                    plt.ylabel('Prediction')
                    plt.title('Feature Impact on Predictions')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                elif "Learning Curve" in graph_name:
                    # Adjust K for learning curve if dataset is too small
                    n_samples = X_train_scaled.shape[0]
                    cv_folds = min(args.cv_folds if enable_cv else 5, n_samples)
                    
                    # Calculate safe K value (max K that works with smallest fold)
                    min_fold_size = n_samples // cv_folds
                    safe_k = min(k_value, max(1, min_fold_size - 1))
                    
                    # Create temporary model with safe K
                    if is_classification:
                        temp_model = KNeighborsClassifier(
                            n_neighbors=safe_k,
                            metric=args.distance_metric,
                            weights=args.weights,
                            algorithm=args.algorithm,
                            leaf_size=args.leaf_size,
                            p=args.p_value if args.distance_metric == 'minkowski' else 2
                        )
                    else:
                        temp_model = KNeighborsRegressor(
                            n_neighbors=safe_k,
                            metric=args.distance_metric,
                            weights=args.weights,
                            algorithm=args.algorithm,
                            leaf_size=args.leaf_size,
                            p=args.p_value if args.distance_metric == 'minkowski' else 2
                        )
                    
                    # Suppress warnings during learning curve computation
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        train_sizes, train_scores, val_scores = learning_curve(
                            temp_model, X_train_scaled, y_train,
                            cv=cv_folds,
                            scoring='accuracy' if is_classification else 'r2',
                            n_jobs=-1,
                            train_sizes=np.linspace(0.1, 1.0, 5),
                            error_score='raise'
                        )
                    
                    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', marker='o')
                    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score', marker='s')
                    plt.fill_between(train_sizes, 
                                   train_scores.mean(axis=1) - train_scores.std(axis=1),
                                   train_scores.mean(axis=1) + train_scores.std(axis=1), 
                                   alpha=0.1)
                    plt.fill_between(train_sizes,
                                   val_scores.mean(axis=1) - val_scores.std(axis=1),
                                   val_scores.mean(axis=1) + val_scores.std(axis=1),
                                   alpha=0.1)
                    plt.xlabel('Training Set Size')
                    plt.ylabel('Score')
                    plt.title(f'Learning Curve (K={safe_k})')
                    plt.legend()
                    plt.grid(True)
                
                safe_name = sanitize_filename(graph_name)
                graph_path = os.path.join(output_dir, f'{safe_name}.png')
                plt.tight_layout()
                plt.savefig(graph_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"✅ {graph_name}")
                generated_graphs.append(normalize_path(graph_path))
            
            except Exception as e:
                print(f"⚠️ Skipped {graph_name}: {str(e)}")
                plt.close()
    
    # Generate individual learning curves for each CV fold if requested
    if enable_cv and cv_folds > 0 and "Learning Curve - All Folds" in selected_graphs:
        print(f"\n📊 Generating {cv_folds} individual learning curves (one per CV fold)...")
        
        from sklearn.model_selection import StratifiedKFold, KFold
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_classification else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_idx = 1
        for train_idx, val_idx in kfold.split(X_train_scaled, y_train if is_classification else None):
            try:
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                # Handle both DataFrame and numpy array for y_train
                if hasattr(y_train, 'iloc'):
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Calculate safe K for this fold
                min_fold_size = len(y_fold_train) // max(2, cv_folds // 2)
                safe_k = min(k_value, max(1, min_fold_size - 1))
                
                # Create temporary model
                if is_classification:
                    fold_model = KNeighborsClassifier(
                        n_neighbors=safe_k,
                        metric=args.distance_metric,
                        weights=args.weights,
                        algorithm=args.algorithm,
                        leaf_size=args.leaf_size,
                        p=args.p_value if args.distance_metric == 'minkowski' else 2
                    )
                else:
                    fold_model = KNeighborsRegressor(
                        n_neighbors=safe_k,
                        metric=args.distance_metric,
                        weights=args.weights,
                        algorithm=args.algorithm,
                        leaf_size=args.leaf_size,
                        p=args.p_value if args.distance_metric == 'minkowski' else 2
                    )
                
                # Calculate learning curve for this fold
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    train_sizes, train_scores, val_scores = learning_curve(
                        fold_model, X_fold_train, y_fold_train,
                        cv=min(3, len(y_fold_train) // 10),
                        scoring='accuracy' if is_classification else 'r2',
                        n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        error_score='raise'
                    )
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', marker='o', linewidth=2)
                plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score', marker='s', linewidth=2)
                plt.fill_between(train_sizes,
                               train_scores.mean(axis=1) - train_scores.std(axis=1),
                               train_scores.mean(axis=1) + train_scores.std(axis=1),
                               alpha=0.1)
                plt.fill_between(train_sizes,
                               val_scores.mean(axis=1) - val_scores.std(axis=1),
                               val_scores.mean(axis=1) + val_scores.std(axis=1),
                               alpha=0.1)
                plt.xlabel('Training Set Size')
                plt.ylabel('Accuracy' if is_classification else 'R² Score')
                plt.title(f'Learning Curve - CV Fold {fold_idx} (K={safe_k})', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save
                graph_path = os.path.join(output_dir, f'Learning_Curve_Fold_{fold_idx}.png')
                plt.tight_layout()
                plt.savefig(graph_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"✅ Learning Curve - CV Fold {fold_idx}")
                generated_graphs.append(normalize_path(graph_path))
                
                fold_idx += 1
            except Exception as e:
                print(f"⚠️ Skipped Learning Curve Fold {fold_idx}: {str(e)}")
                plt.close()
                fold_idx += 1
    
    # Output generated graphs JSON for frontend
    print(f"__GENERATED_GRAPHS_JSON__{json.dumps(generated_graphs)}__END_GRAPHS__")
    sys.stdout.flush()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("FINISHED SUCCESSFULLY")

if __name__ == '__main__':
    main()
