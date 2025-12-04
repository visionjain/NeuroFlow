#!/usr/bin/env python3
"""
Logistic Regression Training Script for NeuroFlow
Supports Binary and Multi-class Classification
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Auto-install missing libraries
def check_and_install_libraries():
    required = {
        'pandas': 'pandas', 'numpy': 'numpy', 'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib', 'seaborn': 'seaborn', 'imbalanced-learn': 'imblearn', 
        'joblib': 'joblib', 'shap': 'shap'
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, log_loss)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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
    parser.add_argument('--feature_scaling', default=None)
    parser.add_argument('--selected_explorations', default='[]')
    
    # Logistic Regression specific
    parser.add_argument('--solver', default='lbfgs')
    parser.add_argument('--penalty', default='l2')
    parser.add_argument('--c_value', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--l1_ratio', type=float, default=0.5)
    
    # Class imbalance
    parser.add_argument('--enable_imbalance', default='false')
    parser.add_argument('--imbalance_method', default=None)
    parser.add_argument('--class_weight', default=None)
    
    # Advanced
    parser.add_argument('--probability_threshold', type=float, default=0.5)
    parser.add_argument('--use_stratified_split', default='true')
    parser.add_argument('--multi_class_strategy', default='auto')
    
    # CV
    parser.add_argument('--enable_cv', default='false')
    parser.add_argument('--cv_folds', type=int, default=5)
    
    # Effect features
    parser.add_argument('--effect_features', default='[]')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("=" * 60)
    print("LOGISTIC REGRESSION TRAINING")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 60)
    
    # Parse JSON parameters
    train_columns = json.loads(args.train_columns)
    output_column = args.output_column
    selected_graphs = json.loads(args.selected_graphs)
    selected_explorations = json.loads(args.selected_explorations)
    missing_val_tech = json.loads(args.selected_missingval_tech)
    effect_features = json.loads(args.effect_features) if args.effect_features else []
    
    enable_cv = args.enable_cv.lower() == 'true'
    enable_imbalance = args.enable_imbalance.lower() == 'true'
    use_stratified = args.use_stratified_split.lower() == 'true'
    
    # Load data
    print("\n📂 Loading Training Data...")
    print(f"   Path: {args.train_csv_path}")
    
    if args.train_csv_path.endswith('.csv'):
        df = pd.read_csv(args.train_csv_path)
    elif args.train_csv_path.endswith('.xlsx'):
        df = pd.read_excel(args.train_csv_path)
    else:
        print("❌ Unsupported format. Use CSV or XLSX")
        sys.exit(1)
    
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data Exploration
    if selected_explorations:
        print("\n" + "=" * 60)
        print("📊 DATA EXPLORATION")
        print("=" * 60)
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
            elif technique in ["Target Column Distribution", "Class Distribution"]:
                print(f"\n🎯 Class Distribution ({output_column}):\n", df[output_column].value_counts())
                print("\nPercentages:\n", df[output_column].value_counts(normalize=True) * 100)
    
    # Preprocessing
    print("\n" + "=" * 60)
    print("🧹 PREPROCESSING")
    print("=" * 60)
    
    # Handle missing values
    if missing_val_tech == "Drop Rows with Missing Values":
        before = len(df)
        df = df.dropna()
        print(f"✅ Dropped {before - len(df)} rows with missing values")
    elif missing_val_tech == "Fill with Mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("✅ Filled missing with mean")
    elif missing_val_tech == "Fill with Median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print("✅ Filled missing with median")
    elif missing_val_tech == "Fill with Mode":
        for col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("✅ Filled missing with mode")
    
    # Remove duplicates
    if args.remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            print(f"✅ Removed {removed} duplicates")
    
    # Prepare features and target
    X = df[train_columns].copy()
    y = df[output_column].copy()
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"\n🔤 Encoding categorical columns: {categorical_cols}")
        if args.encoding_type == 'one-hot':
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            print(f"   One-Hot Encoding applied → {X.shape[1]} features")
        elif args.encoding_type == 'label':
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            print(f"   Label Encoding applied")
    
    # Encode target if categorical
    label_encoder = None
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"🎯 Target encoded: {label_encoder.classes_}")
    
    n_classes = len(np.unique(y))
    is_binary = n_classes == 2
    print(f"\n🎯 Classification Type: {'Binary' if is_binary else 'Multi-class'} ({n_classes} classes)")
    
    # Train-test split
    print("\n📊 Splitting Data...")
    if use_stratified:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split_ratio, random_state=args.random_seed, stratify=y
        )
        print(f"✅ Stratified split: {len(X_train)} train, {len(X_test)} test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split_ratio, random_state=args.random_seed
        )
        print(f"✅ Random split: {len(X_train)} train, {len(X_test)} test")
    
    # Handle class imbalance
    if enable_imbalance and args.imbalance_method:
        print(f"\n⚖️ Handling class imbalance: {args.imbalance_method}")
        if args.imbalance_method == 'smote':
            sampler = SMOTE(random_state=args.random_seed)
        elif args.imbalance_method == 'random-over':
            sampler = RandomOverSampler(random_state=args.random_seed)
        elif args.imbalance_method == 'random-under':
            sampler = RandomUnderSampler(random_state=args.random_seed)
        else:
            sampler = None
        
        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"✅ Resampled: {len(X_train)} samples")
    
    # Feature scaling
    scaler = None
    if args.feature_scaling and args.feature_scaling != "none":
        print(f"\n📏 Applying feature scaling: {args.feature_scaling}")
        if 'Min-Max' in args.feature_scaling:
            scaler = MinMaxScaler()
        elif 'Standard' in args.feature_scaling:
            scaler = StandardScaler()
        elif 'Robust' in args.feature_scaling:
            scaler = RobustScaler()
        
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print("✅ Scaling applied")
    
    # Train model
    print("\n" + "=" * 60)
    print("🤖 TRAINING MODEL")
    print("=" * 60)
    print(f"Solver: {args.solver}, Penalty: {args.penalty}, C: {args.c_value}")
    print(f"Max Iter: {args.max_iter if args.max_iter > 0 else 'default (100)'}, Random Seed: {args.random_seed}")
    
    model_params = {
        'solver': args.solver,
        'random_state': args.random_seed
    }
    
    # Only add max_iter if specified (otherwise use sklearn default)
    if args.max_iter > 0:
        model_params['max_iter'] = args.max_iter
    
    # Only add multi_class if not 'auto' (deprecated parameter)
    if args.multi_class_strategy != 'auto':
        model_params['multi_class'] = args.multi_class_strategy
    
    # Handle penalty and C
    if args.penalty != 'none':
        model_params['penalty'] = args.penalty
        model_params['C'] = args.c_value
        if args.penalty == 'elasticnet':
            model_params['l1_ratio'] = args.l1_ratio
    else:
        model_params['penalty'] = None
    
    # Handle class weight
    if enable_imbalance and args.class_weight == 'balanced':
        model_params['class_weight'] = 'balanced'
        print("Class weight: balanced")
    
    model = LogReg(**model_params)
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    if is_binary:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"ROC-AUC:   {roc_auc:.4f}")
        except:
            pass
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save outputs directory
    output_dir = os.path.join(os.path.dirname(args.train_csv_path), 
                              f"logistic-{os.path.basename(args.train_csv_path).split('.')[0]}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Cross-validation with model saving
    cv_fold_models = []
    if enable_cv:
        print("\n" + "=" * 60)
        print(f"🔄 CROSS-VALIDATION ({args.cv_folds}-Fold)")
        print("=" * 60)
        print(f"Training and saving individual fold models...")
        
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
        
        # Combine train and test data
        if isinstance(X_train, np.ndarray):
            X_full = np.vstack([X_train, X_test])
        else:
            X_full = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)
        y_full = np.concatenate([y_train, y_test])
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full, y_full), 1):
            # Split data for this fold
            if isinstance(X_full, np.ndarray):
                X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
            else:
                X_fold_train, X_fold_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_fold_train, y_fold_val = y_full[train_idx], y_full[val_idx]
            
            # Train model on this fold
            fold_model = LogReg(**model_params)
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            y_fold_pred = fold_model.predict(X_fold_val)
            fold_accuracy = accuracy_score(y_fold_val, y_fold_pred)
            fold_precision = precision_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
            fold_recall = recall_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
            fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0)
            
            fold_scores.append(fold_accuracy)
            
            # Save fold model
            fold_model_path = os.path.join(output_dir, f"logistic_model_fold_{fold_idx}.pkl")
            joblib.dump(fold_model, fold_model_path)
            
            # Store model info
            cv_fold_models.append({
                'fold': fold_idx,
                'model': fold_model,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'f1_score': fold_f1,
                'filename': f"logistic_model_fold_{fold_idx}.pkl",
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            
            print(f"  Fold {fold_idx}: Accuracy={fold_accuracy:.4f}, F1={fold_f1:.4f} (trained on {len(train_idx)} samples)")
        
        print(f"\nCV Scores: {fold_scores}")
        print(f"Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores) * 2:.4f})")
        print(f"✓ Saved {len(cv_fold_models)} CV fold models")
        
        # Print comprehensive CV results table
        print("\n" + "=" * 110)
        print("COMPREHENSIVE RESULTS TABLE")
        print("=" * 110)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Train Size':<12}")
        print("-" * 110)
        
        for fold_info in cv_fold_models:
            print(f"CV Fold {fold_info['fold']:<13} {fold_info['accuracy']:<12.4f} {fold_info['precision']:<12.4f} {fold_info['recall']:<12.4f} {fold_info['f1_score']:<12.4f} {fold_info['train_size']:<12}")
        
        # Final model metrics will be added after training
        print(f"Final Model (Full)   {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {len(y_train):<12}")
        print("-" * 110)
        
        # CV Statistics
        mean_acc = np.mean([f['accuracy'] for f in cv_fold_models])
        std_acc = np.std([f['accuracy'] for f in cv_fold_models])
        print(f"\nCV Statistics:")
        print(f"  Mean Accuracy: {mean_acc:.4f} (± {std_acc:.4f})")
        print(f"  Model Stability: {'High' if std_acc < 0.05 else 'Moderate' if std_acc < 0.10 else 'Low'}")
        print("=" * 110)
    
    # Save final model
    model_path = os.path.join(output_dir, "logistic_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n💾 Final model saved: {model_path}")
    
    if scaler:
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
    
    # Generate graphs
    generated_graphs = []
    if selected_graphs:
        # Count total graphs to generate
        total_graphs = 0
        graph_list = []
        
        if "Confusion Matrix" in selected_graphs:
            total_graphs += 1
            graph_list.append("Confusion Matrix")
        if "ROC Curve" in selected_graphs and is_binary:
            total_graphs += 1
            graph_list.append("ROC Curve")
        if "Precision-Recall Curve" in selected_graphs and is_binary:
            total_graphs += 1
            graph_list.append("Precision-Recall Curve")
        if "Feature Importance" in selected_graphs:
            total_graphs += 1
            graph_list.append("Feature Importance")
        if "Classification Report" in selected_graphs:
            total_graphs += 1
            graph_list.append("Classification Report")
        if "Probability Distribution" in selected_graphs and is_binary:
            total_graphs += 1
            graph_list.append("Probability Distribution")
        if "Calibration Curve" in selected_graphs and is_binary:
            total_graphs += 1
            graph_list.append("Calibration Curve")
        if "Decision Boundary (2D)" in selected_graphs and X_test.shape[1] >= 2:
            total_graphs += 1
            graph_list.append("Decision Boundary (2D)")
        if "Class Separation (PCA)" in selected_graphs:
            total_graphs += 1
            graph_list.append("Class Separation (PCA)")
        if "Correlation Heatmap" in selected_graphs:
            total_graphs += 1
            graph_list.append("Correlation Heatmap")
        if "Box Plot" in selected_graphs:
            total_graphs += 1
            graph_list.append("Box Plot")
        if "Histogram Distribution" in selected_graphs:
            total_graphs += 1
            graph_list.append("Histogram Distribution")
        if effect_features:
            if "Individual Effect Plot" in selected_graphs:
                total_graphs += 1
                graph_list.append("Individual Effect Plot")
            if "Mean Effect Plot" in selected_graphs:
                total_graphs += 1
                graph_list.append("Mean Effect Plot")
            if "Trend Effect Plot" in selected_graphs:
                total_graphs += 1
                graph_list.append("Trend Effect Plot")
        if "Shap Summary Plot" in selected_graphs:
            total_graphs += 1
            graph_list.append("Shap Summary Plot")
        if enable_cv and "Learning Curve - All Folds" in selected_graphs:
            total_graphs += 1
            graph_list.append("Learning Curve - All Folds")
        if "Learning Curve - Overall" in selected_graphs:
            total_graphs += 1
            graph_list.append("Learning Curve - Overall")
        
        print("\n📊 Generating graphs...")
        print(f"Total graphs to generate: {total_graphs}")
        print("=" * 60)
        
        current_graph = 0
        
        # 1. Confusion Matrix
        if "Confusion Matrix" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Confusion Matrix...")
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 2. ROC Curve (Binary)
        if "ROC Curve" in selected_graphs and is_binary:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating ROC Curve...")
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba[:, 1]):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 3. Precision-Recall Curve (Binary only)
        if "Precision-Recall Curve" in selected_graphs and is_binary:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Precision-Recall Curve...")
            plt.figure(figsize=(8, 6))
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            plt.plot(recall_vals, precision_vals)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            path = os.path.join(output_dir, 'precision_recall_curve.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 4. Feature Importance
        if "Feature Importance" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Feature Importance...")
            plt.figure(figsize=(10, 6))
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) == 2 and model.coef_.shape[0] == 1 else np.abs(model.coef_).mean(axis=0)
                feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(len(importance))]
                indices = np.argsort(importance)[-20:]
                plt.barh(range(len(indices)), importance[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Absolute Coefficient Value')
                plt.title('Top 20 Feature Importance')
                path = os.path.join(output_dir, 'feature_importance.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
        
        # 5. Classification Report (Visual)
        if "Classification Report" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Classification Report...")
            plt.figure(figsize=(10, 6))
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Classification Report Heatmap')
            path = os.path.join(output_dir, 'classification_report.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 6. Probability Distribution
        if "Probability Distribution" in selected_graphs and is_binary:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Probability Distribution...")
            plt.figure(figsize=(10, 6))
            pos_probs = y_pred_proba[:, 1]
            plt.hist([pos_probs[y_test == 0], pos_probs[y_test == 1]], bins=30, label=['Class 0', 'Class 1'], alpha=0.7)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Frequency')
            plt.title('Probability Distribution by Class')
            plt.legend()
            path = os.path.join(output_dir, 'probability_distribution.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 7. Calibration Curve (Binary only)
        if "Calibration Curve" in selected_graphs and is_binary:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Calibration Curve...")
            from sklearn.calibration import calibration_curve
            plt.figure(figsize=(8, 6))
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='Logistic')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            path = os.path.join(output_dir, 'calibration_curve.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 8. Decision Boundary (2D) - Only for 2 features
        if "Decision Boundary (2D)" in selected_graphs and X_test.shape[1] >= 2:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Decision Boundary (2D)...")
            try:
                from sklearn.decomposition import PCA
                plt.figure(figsize=(10, 6))
                if X_test.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_test)
                else:
                    X_2d = X_test
                
                h = 0.02
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                # Create a temporary model for 2D
                temp_model = LogReg(**model_params)
                if X_test.shape[1] > 2:
                    temp_model.fit(pca.transform(X_train), y_train)
                    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
                else:
                    temp_model.fit(X_train, y_train)
                    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
                
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=0.3)
                plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test, edgecolors='k', marker='o')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.title('Decision Boundary (PCA 2D)' if X_test.shape[1] > 2 else 'Decision Boundary')
                path = os.path.join(output_dir, 'decision_boundary.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
            except Exception as e:
                print(f"⚠️ Could not generate decision boundary: {e}")
        
        # 9. Class Separation (PCA)
        if "Class Separation (PCA)" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Class Separation (PCA)...")
            try:
                from sklearn.decomposition import PCA
                plt.figure(figsize=(10, 6))
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_test)
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', alpha=0.6)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                plt.title('Class Separation (PCA)')
                plt.colorbar(scatter, label='Class')
                path = os.path.join(output_dir, 'class_separation_pca.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
            except Exception as e:
                print(f"⚠️ Could not generate PCA plot: {e}")
        
        # 10. Correlation Heatmap
        if "Correlation Heatmap" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Correlation Heatmap...")
            plt.figure(figsize=(12, 10))
            corr = pd.DataFrame(X_train).corr()
            sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
            plt.title('Feature Correlation Heatmap')
            path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 11. Box Plot
        if "Box Plot" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Box Plot...")
            plt.figure(figsize=(12, 6))
            df_plot = pd.DataFrame(X_test)
            df_plot['target'] = y_test
            cols_to_plot = df_plot.columns[:min(10, len(df_plot.columns)-1)]
            df_plot[list(cols_to_plot)].boxplot()
            plt.xticks(rotation=45)
            plt.title('Feature Distribution (Box Plot)')
            path = os.path.join(output_dir, 'box_plot.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 12. Histogram Distribution
        if "Histogram Distribution" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Histogram Distribution...")
            plt.figure(figsize=(12, 8))
            df_plot = pd.DataFrame(X_test)
            cols_to_plot = df_plot.columns[:min(9, len(df_plot.columns))]
            df_plot[list(cols_to_plot)].hist(bins=20, figsize=(12, 8))
            plt.suptitle('Feature Distributions')
            plt.tight_layout()
            path = os.path.join(output_dir, 'histogram_distribution.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # 13-15. Effect Plots (requires effect_features)
        if effect_features and len(effect_features) > 0:
            # Individual Effect Plot
            if "Individual Effect Plot" in selected_graphs:
                current_graph += 1
                print(f"[{current_graph}/{total_graphs}] 🎨 Generating Individual Effect Plot...")
                fig, axes = plt.subplots(len(effect_features), 1, figsize=(10, 4*len(effect_features)))
                if len(effect_features) == 1:
                    axes = [axes]
                for idx, feature in enumerate(effect_features):
                    if feature in X.columns:
                        feat_idx = X.columns.get_loc(feature)
                        X_copy = X_test.copy()
                        feature_range = np.linspace(X_copy[:, feat_idx].min(), X_copy[:, feat_idx].max(), 50)
                        effects = []
                        for val in feature_range:
                            X_temp = X_copy.copy()
                            X_temp[:, feat_idx] = val
                            effects.append(model.predict_proba(X_temp)[:, 1].mean() if is_binary else model.predict(X_temp).mean())
                        axes[idx].plot(feature_range, effects)
                        axes[idx].set_xlabel(feature)
                        axes[idx].set_ylabel('Predicted Effect')
                        axes[idx].set_title(f'Effect of {feature}')
                plt.tight_layout()
                path = os.path.join(output_dir, 'individual_effect_plot.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
            
            # Mean Effect Plot
            if "Mean Effect Plot" in selected_graphs:
                current_graph += 1
                print(f"[{current_graph}/{total_graphs}] 🎨 Generating Mean Effect Plot...")
                plt.figure(figsize=(10, 6))
                mean_effects = []
                for feature in effect_features:
                    if feature in X.columns:
                        feat_idx = X.columns.get_loc(feature)
                        X_copy = X_test.copy()
                        feature_range = np.linspace(X_copy[:, feat_idx].min(), X_copy[:, feat_idx].max(), 20)
                        effects = []
                        for val in feature_range:
                            X_temp = X_copy.copy()
                            X_temp[:, feat_idx] = val
                            effects.append(model.predict_proba(X_temp)[:, 1].mean() if is_binary else model.predict(X_temp).mean())
                        mean_effects.append(np.mean(effects))
                plt.barh(effect_features, mean_effects)
                plt.xlabel('Mean Effect')
                plt.title('Mean Effect of Features')
                path = os.path.join(output_dir, 'mean_effect_plot.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
            
            # Trend Effect Plot
            if "Trend Effect Plot" in selected_graphs:
                current_graph += 1
                print(f"[{current_graph}/{total_graphs}] 🎨 Generating Trend Effect Plot...")
                plt.figure(figsize=(12, 6))
                for feature in effect_features:
                    if feature in X.columns:
                        feat_idx = X.columns.get_loc(feature)
                        X_copy = X_test.copy()
                        feature_range = np.linspace(X_copy[:, feat_idx].min(), X_copy[:, feat_idx].max(), 50)
                        effects = []
                        for val in feature_range:
                            X_temp = X_copy.copy()
                            X_temp[:, feat_idx] = val
                            effects.append(model.predict_proba(X_temp)[:, 1].mean() if is_binary else model.predict(X_temp).mean())
                        plt.plot(feature_range, effects, label=feature)
                plt.xlabel('Feature Value')
                plt.ylabel('Predicted Effect')
                plt.title('Trend Effect Plot')
                plt.legend()
                path = os.path.join(output_dir, 'trend_effect_plot.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
        
        # 16. SHAP Summary Plot (requires shap library)
        if "Shap Summary Plot" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating SHAP Summary Plot...")
            try:
                import shap
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test[:100])  # Limit to 100 samples for speed
                plt.figure()
                shap.summary_plot(shap_values, X_test[:100], show=False)
                path = os.path.join(output_dir, 'shap_summary_plot.png')
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                generated_graphs.append(path)
            except Exception as e:
                print(f"⚠️ SHAP plot requires 'shap' library: {e}")
        
        # 17-18. Learning Curves
        if enable_cv and "Learning Curve - All Folds" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Learning Curve - All Folds...")
            from sklearn.model_selection import learning_curve
            plt.figure(figsize=(12, 8))
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=args.cv_folds, 
                scoring='accuracy', n_jobs=-1
            )
            for i in range(args.cv_folds):
                plt.plot(train_sizes, train_scores[:, i], alpha=0.3, color='blue')
                plt.plot(train_sizes, val_scores[:, i], alpha=0.3, color='red')
            plt.xlabel('Training Size')
            plt.ylabel('Score')
            plt.title('Learning Curves - All Folds')
            plt.legend(['Train', 'Validation'])
            path = os.path.join(output_dir, 'learning_curve_all_folds.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        if "Learning Curve - Overall" in selected_graphs:
            current_graph += 1
            print(f"[{current_graph}/{total_graphs}] 🎨 Generating Learning Curve - Overall...")
            from sklearn.model_selection import learning_curve
            plt.figure(figsize=(10, 6))
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, 
                scoring='accuracy', n_jobs=-1
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, label='Train', color='blue')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
            plt.plot(train_sizes, val_mean, label='Validation', color='red')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.title('Learning Curve - Overall')
            plt.legend()
            path = os.path.join(output_dir, 'learning_curve_overall.png')
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            generated_graphs.append(path)
        
        # Graph generation completed
        if total_graphs > 0:
            print("=" * 60)
            print(f"✅ Successfully generated {len(generated_graphs)}/{total_graphs} graphs!")
            print("=" * 60)
    
    # Save preprocessing info for prediction
    preprocessing_path = os.path.join(output_dir, "preprocessing.pkl")
    
    # Build available models list
    available_models = []
    
    # Add CV fold models if CV was enabled
    if enable_cv and len(cv_fold_models) > 0:
        for fold_info in cv_fold_models:
            available_models.append({
                'name': f"CV Fold {fold_info['fold']}",
                'filename': fold_info['filename'],
                'accuracy': float(fold_info['accuracy']),
                'precision': float(fold_info['precision']),
                'recall': float(fold_info['recall']),
                'f1_score': float(fold_info['f1_score']),
                'type': 'cv_fold',
                'fold_number': fold_info['fold'],
                'train_size': fold_info['train_size'],
                'val_size': fold_info['val_size']
            })
    
    # Add final model
    available_models.insert(0, {
        'name': f"Final Model ({'CV-Trained' if enable_cv else 'Train/Test Split'})",
        'filename': 'logistic_model.pkl',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'type': 'final',
        'trained_on_full': enable_cv,
        'train_size': len(y_train)
    })
    
    # Get final feature names after encoding (BEFORE scaling)
    # We need to get this from the DataFrame BEFORE train_test_split and scaling
    # Let's reload and get feature names properly
    if args.train_csv_path.endswith('.csv'):
        df_temp = pd.read_csv(args.train_csv_path)
    elif args.train_csv_path.endswith('.xlsx'):
        df_temp = pd.read_excel(args.train_csv_path)
    X_temp = df_temp[train_columns].copy()
    
    # Store original categorical columns BEFORE encoding
    original_categorical_cols = X_temp.select_dtypes(include=['object']).columns.tolist()
    categorical_values_dict = {}
    for col in original_categorical_cols:
        categorical_values_dict[col] = sorted(X_temp[col].unique().tolist())
    
    # Apply same encoding to get final feature names
    if original_categorical_cols:
        if args.encoding_type == 'one-hot':
            X_temp = pd.get_dummies(X_temp, columns=original_categorical_cols, drop_first=True)
        elif args.encoding_type == 'label':
            for col in original_categorical_cols:
                X_temp[col] = LabelEncoder().fit_transform(X_temp[col].astype(str))
    
    # Get final feature names (after encoding, before scaling)
    final_feature_names = list(X_temp.columns) if hasattr(X_temp, 'columns') else train_columns
    
    # Numeric columns in ORIGINAL data (before encoding)
    original_numeric_cols = [col for col in train_columns if col not in original_categorical_cols]
    
    # Prepare preprocessing information with ALL required fields
    preprocessing_info = {
        'original_train_columns': train_columns,
        'final_feature_names': final_feature_names,  # REQUIRED by predict.py
        'encoding_type': args.encoding_type,
        'categorical_cols': original_categorical_cols,
        'categorical_values': categorical_values_dict,
        'numeric_cols': original_numeric_cols,
        'is_binary_classification': is_binary,
        'target_name': args.output_column,
        'available_models': available_models,
        'cv_enabled': enable_cv,
        'cv_folds': args.cv_folds if enable_cv else None,
        'label_encoders': {},  # Empty for now, can add if needed
        'target_means': {},  # Empty for now, can add if needed
        'encoding_added_cols': {}  # Empty for now, can add if needed
    }
    
    joblib.dump(preprocessing_info, preprocessing_path)
    print(f"\n💾 Preprocessing info saved: {preprocessing_path}")
    print(f"✓ Total models available for prediction: {len(available_models)}")
    
    # Output graphs JSON
    print(f"\n__GENERATED_GRAPHS_JSON__{json.dumps(generated_graphs)}__END_GRAPHS__")
    
    print("\n" + "=" * 60)
    print("✅ FINISHED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
