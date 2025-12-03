import subprocess
import sys
import ast
import os
import argparse
import logging
import json
import re



def log_and_print(message):
    """Logs and prints the message with extra blank lines and a separator."""
    print(message)
    print("-" * 200)
    sys.stdout.flush()


def sanitize_filename(filename):
    """Remove or replace characters that are problematic in filenames."""
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[^\w\s-]', '_', filename)
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[\s_]+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename





# ====== Function to Run Commands and Capture Logs ======
def run_command(command):
    """Executes a shell command and logs output."""
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            log_and_print(line.strip())
        for line in process.stderr:
            log_and_print(line.strip())
        process.wait()
        if process.returncode != 0:
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        log_and_print(f"Command failed: {command}\\nError: {e}")
        sys.exit(1)






# ====== Check Python Installation ======
try:
    run_command("python --version")
except FileNotFoundError:
    log_and_print("Python is not installed. Please install Python first.")
    sys.exit(1)

# ====== Upgrade pip ======
run_command("python -m pip install --upgrade pip")

# ====== Install Required Packages ======
def install_package(package):
    """Installs a package and logs the process."""
    try:
        log_and_print(f"Installing {package}...")
        run_command(f"python -m pip install {package}")
        log_and_print(f"Successfully installed {package}")
    except Exception as e:
        log_and_print(f"Failed to install {package}: {e}")
        sys.exit(1)

packages = ["pandas", "category_encoders", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "six", "scipy"]
for package in packages:
    try:
        __import__(package)
    except ImportError:
        install_package(package)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from category_encoders import TargetEncoder
import joblib
from scipy import stats









# ====== Parse Command-Line Arguments ======
parser = argparse.ArgumentParser(description="Run Linear Regression Model")
parser.add_argument("--train_csv_path", required=True, help="Path to the training dataset CSV")
parser.add_argument("--test_csv_path", required=False, help="Path to the test dataset CSV (optional)")
parser.add_argument("--test_split_ratio", type=float, help="Test split ratio if test dataset is not provided")
parser.add_argument("--train_columns", required=True, help="Comma-separated column names for training features")
parser.add_argument("--output_column", required=True, help="Name of the target output column")
parser.add_argument("--selected_graphs", required=False, help="Comma-separated list of graph filenames to generate (optional)")
parser.add_argument("--selected_missingval_tech", required=True, help="Selected missing value handling technique")
parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicate rows from the dataset")
parser.add_argument("--encoding_type", required=True, choices=["one-hot", "label", "target", "none"], help="Encoding type for categorical variables")
parser.add_argument("--selected_explorations", type=str, default="[]")  # Default empty list

# Outlier Handling Arguments
parser.add_argument("--enable_outlier_detection", type=str, help="Enable outlier detection (true/false)", default="false")
parser.add_argument("--outlier_method", type=str, help="Selected outlier removal method")
parser.add_argument("--z_score_threshold", type=float, help="Threshold for Z-score method", default=3.0)
parser.add_argument("--iqr_lower", type=float, help="Multiplier for lower bound in IQR", default=1.5)
parser.add_argument("--iqr_upper", type=float, help="Multiplier for upper bound in IQR", default=1.5)
parser.add_argument("--winsor_lower", type=int, help="Lower percentile for Winsorization", default=1)
parser.add_argument("--winsor_upper", type=int, help="Upper percentile for Winsorization", default=99)

# ✅ Feature Scaling
parser.add_argument("--feature_scaling", type=str, help="Selected feature scaling method")

# ✅ Effect Features for Comparison
parser.add_argument("--effect_features", type=str, help="Features to generate effect plots for (JSON array)")

# ✅ Regularization Parameters
parser.add_argument("--regularization_type", type=str, default="none", choices=["none", "ridge", "lasso", "elasticnet"], help="Type of regularization to use")
parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength (alpha parameter)")

# ✅ Cross-Validation Parameters
parser.add_argument("--enable_cv", type=str, default="false", help="Enable cross-validation (true/false)")
parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validation")

args = parser.parse_args()

train_csv_path = args.train_csv_path
test_csv_path = args.test_csv_path if args.test_csv_path and args.test_csv_path.lower() != "none" else None
selected_missingval_tech = args.selected_missingval_tech.strip('"')  # Clean extra quotes if present

# This will be refined via ast.literal_eval further below
train_columns = args.train_columns.split(",")  
output_column = args.output_column
# Convert JSON string back to Python list
selected_explorations = json.loads(args.selected_explorations)

# ✅ Process Feature Scaling
feature_scaling = args.feature_scaling if args.feature_scaling else None

# ✅ Process Effect Features
effect_features = json.loads(args.effect_features) if args.effect_features else None

# ✅ Process Regularization Parameters
regularization_type = args.regularization_type.lower()
alpha_value = args.alpha

# ✅ Process Cross-Validation Parameters
enable_cv = args.enable_cv.lower() == "true"
cv_folds = args.cv_folds

# Process selected_graphs argument into a list (if provided)
selected_graphs = None
if args.selected_graphs:
    try:
        # If the string starts with '[' then attempt to literal_eval, otherwise split by comma
        if args.selected_graphs.strip().startswith("["):
            selected_graphs = ast.literal_eval(args.selected_graphs)
        else:
            selected_graphs = [g.strip() for g in args.selected_graphs.split(",")]
    except Exception as e:
        logging.error(f"Error parsing selected_graphs: {e}")
        sys.exit(1)








# ====== Function to Create Log File in Output Directory ======
def get_output_dir(csv_path):
    """Creates and returns an output directory based on the CSV filename."""
    base_path = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path).split('.')[0]
    output_dir = os.path.join(base_path, f"linearregression-{csv_filename}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ====== Initialize Logging ======
output_dir = get_output_dir(train_csv_path)
log_file = os.path.join(output_dir, "setup_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format=" %(message)s",
    filemode="w",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

log_and_print("========== STARTED ==========")

# ====== Track Generated Graphs ======
generated_graphs = []

# ====== Normalize Paths for Cross-Platform Compatibility ======
def normalize_path(path):
    """Convert path separators to forward slashes for consistent handling"""
    return path.replace('\\', '/')




# ====== Check If Dataset Exists ======
if not os.path.exists(train_csv_path):
    log_and_print(f"Training dataset not found at {train_csv_path}. Exiting.")
    sys.exit(1)

df_train_original = pd.read_csv(train_csv_path)
df_train_original.columns = df_train_original.columns.str.strip().str.replace('"', '').str.replace("'", "")

if test_csv_path and os.path.exists(test_csv_path):
    df_test_original = pd.read_csv(test_csv_path)
    df_test_original.columns = df_test_original.columns.str.strip().str.replace('"', '').str.replace("'", "")
else:
    df_test_original = None

log_and_print(f"Dataset Loaded Successfully! Shape: {df_train_original.shape}")







# ========== Data Exploration ==========
logging.info("========== Data Exploration ==========")

# Display first few rows
if "First 5 Rows" in selected_explorations:
    logging.info(f"\n📌 First 5 rows:\n{df_train_original.head()}")

if "Last 5 Rows" in selected_explorations:
    logging.info(f"\n📌 Last 5 rows:\n{df_train_original.tail()}")

# Dataset Shape
if "Dataset Shape" in selected_explorations:
    logging.info(f"\n📏 Dataset Shape: {df_train_original.shape} (Rows: {df_train_original.shape[0]}, Columns: {df_train_original.shape[1]})")

# Data Types
if "Data Types" in selected_explorations:
    logging.info(f"\n📊 Data Types:\n{df_train_original.dtypes}")

# Summary Statistics (including categorical)
if "Summary Statistics" in selected_explorations:
    logging.info(f"\n📈 Summary Statistics:\n{df_train_original.describe(include='all')}")

# Missing Values
if "Missing Values" in selected_explorations:
    missing_values = df_train_original.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    logging.info(f"\n🚨 Missing Values:\n{missing_values if not missing_values.empty else 'No missing values found.'}")

# Unique Values in Each Column
if "Unique Values Per Column" in selected_explorations:
    unique_values = df_train_original.nunique()
    logging.info(f"\n🔢 Unique Values Per Column:\n{unique_values}")

# Checking for duplicate rows
if "Duplicate Rows" in selected_explorations:
    duplicate_rows = df_train_original.duplicated().sum()
    logging.info(f"\n🔄 Duplicate Rows: {duplicate_rows}")

# Min & Max Values for Each Numeric Column
numeric_cols = df_train_original.select_dtypes(include=[np.number]).columns
if "Min & Max Values" in selected_explorations and not numeric_cols.empty:
    min_max_values = df_train_original[numeric_cols].agg(["min", "max"])
    logging.info(f"\n📌 Min & Max Values:\n{min_max_values}")

# Correlation Matrix (Numerical Features)
if "Correlation Matrix" in selected_explorations and len(numeric_cols) > 1:
    numeric_corr = df_train_original[numeric_cols].corr()
    logging.info(f"\n🔗 Correlation Matrix:\n{numeric_corr}")

# Skewness of numerical columns
if "Skewness" in selected_explorations and not numeric_cols.empty:
    skewness = df_train_original[numeric_cols].skew()
    logging.info(f"\n⚠️ Skewness of Numerical Features:\n{skewness}")

# Checking class imbalance (if target column exists)
if "Target Column Distribution" in selected_explorations and output_column in df_train_original.columns:
    class_distribution = df_train_original[output_column].value_counts()
    logging.info(f"\n📊 Target Column Distribution:\n{class_distribution}")

logging.info("✅ Data Exploration Completed.")



# ========== TRAIN DATA PROCESSING ==========
logging.info("========== Data Cleaning ==========")
# Ensure train_columns is a proper list (especially if passed as a string)
try:
    const_columns = ast.literal_eval(args.train_columns)
    if isinstance(const_columns, list):
        train_columns = [col.strip() for col in const_columns]
    else:
        raise ValueError("Parsed train_columns is not a list.")
except Exception as e:
    train_columns = [col.strip() for col in args.train_columns.split(",")]

# Select only the required columns
all_needed_columns = list(set(train_columns + [output_column]))
df_train = df_train_original[all_needed_columns].copy()

# Identify numeric and non-numeric columns
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
non_numeric_cols = df_train.select_dtypes(exclude=[np.number]).columns

logging.info("Starting missing value handling for training dataset...")
initial_rows = df_train.shape[0]
log_and_print(f"Initial dataset: {initial_rows} rows")

# ========== Handle Missing Values Based on User Selection ==========
if selected_missingval_tech == "Mean Imputation":
    df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].mean())
    logging.info("Missing values in numeric columns filled using Mean Imputation.")
elif selected_missingval_tech == "Median Imputation":
    df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
    logging.info("Missing values in numeric columns filled using Median Imputation.")
elif selected_missingval_tech == "Mode Imputation":
    for col in numeric_cols:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    for col in non_numeric_cols:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    logging.info("Missing values filled using Mode Imputation.")
elif selected_missingval_tech == "Forward/Backward Fill":
    df_train.ffill(inplace=True)
    df_train.bfill(inplace=True)
    logging.info("Missing values filled using Forward/Backward Fill.")
elif selected_missingval_tech == "Drop Rows with Missing Values":
    before_rows = df_train.shape[0]
    df_train.dropna(inplace=True)
    after_rows = df_train.shape[0]
    log_and_print(f"⚠️ Dropped {before_rows - after_rows} rows with missing values. Remaining: {after_rows}")
    logging.info(f"Dropped {before_rows - after_rows} rows due to missing values.")
else:
    logging.error(f"Invalid missing value handling technique: {selected_missingval_tech}")
    sys.exit(1)

# ========== Remove Duplicates ==========
if args.remove_duplicates:
    before_rows = df_train.shape[0]
    df_train.drop_duplicates(inplace=True)
    after_rows = df_train.shape[0]
    removed = before_rows - after_rows
    if removed > 0:
        log_and_print(f"Removed {removed} duplicate rows. Remaining: {after_rows}")
    logging.info(f"Removed {removed} duplicate rows from the training dataset.")

# ========== Outlier Removal ==========
if args.enable_outlier_detection.lower() == "true":
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    logging.info(f"🛠️ Outlier removal enabled using method: {args.outlier_method}")
    method = str(args.outlier_method).strip().lower().strip('"').strip("'")
    if method == "z-score":
        logging.info(f"🔹 Applying Z-score method (threshold={args.z_score_threshold})...")
        if len(numeric_cols) == 0:
            logging.warning("⚠️ No numeric columns found. Skipping Z-score outlier removal.")
        else:
            try:
                before_rows = df_train.shape[0]
                z_scores = np.abs(stats.zscore(df_train[numeric_cols]))
                df_train = df_train[(z_scores < args.z_score_threshold).all(axis=1)]
                removed = before_rows - df_train.shape[0]
                log_and_print(f"Z-score outlier removal: Removed {removed} rows. Remaining: {df_train.shape[0]}")
                logging.info(f"✅ Outliers removed using Z-score method. New shape: {df_train.shape}")
            except Exception as e:
                logging.error(f"⚠️ Error in Z-score computation: {e}")
    elif method == "iqr":
        logging.info("🔹 Applying IQR method...")
        Q1 = df_train[numeric_cols].quantile(0.25)
        Q3 = df_train[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (args.iqr_lower * IQR)
        upper_bound = Q3 + (args.iqr_upper * IQR)
        df_train = df_train[~((df_train[numeric_cols] < lower_bound) | (df_train[numeric_cols] > upper_bound)).any(axis=1)]
        logging.info(f"✅ Outliers removed using IQR method. New shape: {df_train.shape}")
    elif method == "winsorization":
        logging.info("🔹 Applying Winsorization...")
        from scipy.stats.mstats import winsorize
        for col in numeric_cols:
            df_train[col] = winsorize(df_train[col], limits=(args.winsor_lower / 100, (100 - args.winsor_upper) / 100))
        logging.info("✅ Outliers capped using Winsorization.")
else:
    logging.info("⚠️ Outlier removal is disabled. Skipping outlier detection.")

# ========== Feature Scaling ==========
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
if feature_scaling:
    logging.info(f"✅ Applying Feature Scaling: {feature_scaling}")
    scaler = None
    if feature_scaling == "Min-Max Scaling":
        scaler = MinMaxScaler()
    elif feature_scaling == "Standard Scaling (Z-score Normalization)":
        scaler = StandardScaler()
    elif feature_scaling == "Robust Scaling":
        scaler = RobustScaler()
    else:
        logging.error(f"⚠️ Invalid Feature Scaling technique: {feature_scaling}")
        sys.exit(1)
    if scaler:
        df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        logging.info(f"✅ Feature Scaling applied using {feature_scaling}.")
else:
    logging.info(" -- No feature scaling selected. Skipping feature scaling. --")

# ========== Convert Categorical Columns Based on Encoding Method ==========
feature_cols = [col for col in df_train.columns if col != output_column]
categorical_cols = [col for col in feature_cols if df_train[col].dtype == 'object']

# IMPORTANT: Extract unique values BEFORE encoding (for prediction dropdowns)
categorical_values_dict = {}
if categorical_cols:
    for col in categorical_cols:
        categorical_values_dict[col] = sorted(df_train[col].astype(str).unique().tolist())
        log_and_print(f"Categorical column '{col}' has {len(categorical_values_dict[col])} unique values")

if args.encoding_type.lower() == "one-hot":
    df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    logging.info("Categorical columns converted using One-Hot Encoding.")
elif args.encoding_type.lower() == "label":
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        label_encoders[col] = le
    logging.info("Categorical columns converted using Label Encoding.")
elif args.encoding_type.lower() == "target":
    from collections import defaultdict
    target_means_dict = {}  # Store for prediction use
    for col in categorical_cols:
        target_means = df_train.groupby(col)[output_column].mean().to_dict()
        target_means_dict[col] = target_means
        df_train[col] = df_train[col].map(target_means)
        log_and_print(f"Target Encoding: '{col}' mapped to {len(target_means)} mean values")
    logging.info("Categorical columns converted using Target Encoding.")
elif args.encoding_type.lower() == "none":
    skipped_categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if skipped_categorical_cols:
        logging.info(f"Skipping categorical encoding. Using only numeric features. Skipped columns: {skipped_categorical_cols}")
    else:
        logging.info("No categorical columns found. Proceeding with only numeric features.")
    df_train = df_train.select_dtypes(include=[np.number])
else:
    logging.error(f"Invalid encoding method: {args.encoding_type}. Supported: 'one-hot', 'label', 'target', 'none'")
    sys.exit(1)

logging.info("Training Data Cleaning Completed.")
print("\nData Cleaning & Exploration Done. Check logs for details.")

# ========== Critical Check: Ensure Dataset is Not Empty ==========
if df_train.shape[0] == 0:
    log_and_print("❌ ERROR: Dataset is empty after cleaning!")
    log_and_print("Possible causes:")
    log_and_print("  1. Too aggressive missing value removal (all rows had missing values)")
    log_and_print("  2. Outlier removal deleted all data")
    log_and_print("  3. Data type conversion failures")
    log_and_print("  4. All categorical values were invalid/NaN")
    log_and_print("\nSuggestions:")
    log_and_print("  - Try 'Fill with Mean/Median' instead of 'Drop Rows'")
    log_and_print("  - Use less aggressive outlier removal")
    log_and_print("  - Check your CSV file for data quality issues")
    log_and_print("  - Ensure numeric columns contain actual numbers")
    sys.exit(1)

log_and_print(f"✅ Dataset after cleaning: {df_train.shape[0]} rows, {df_train.shape[1]} columns")

if df_train.shape[0] < 10:
    log_and_print(f"⚠️ WARNING: Only {df_train.shape[0]} rows remaining. Results may be unreliable.")
    log_and_print("Consider using less aggressive data cleaning options.")






# ========== Prepare Final Training Data ==========
def get_existing_train_columns(selected_cols, df):
    result = []
    encoding_added_cols = {}
    for col in selected_cols:
        if col in df.columns:
            result.append(col)
        else:
            dummy_cols = [c for c in df.columns if c.startswith(f"{col}_")]
            if dummy_cols:
                encoding_added_cols[col] = dummy_cols
            result.extend(dummy_cols)
    return result, encoding_added_cols

existing_train_columns, encoding_added_cols = get_existing_train_columns(train_columns, df_train)
if len(existing_train_columns) < len(train_columns):
    log_and_print("Warning: Some specified train_columns were not found directly after cleaning. Using only existing columns (including dummy columns) among the selected subset.")
if not existing_train_columns:
    log_and_print("Error: No valid train columns remain after cleaning.")
    sys.exit(1)
if encoding_added_cols:
    for orig_col, new_cols in encoding_added_cols.items():
        log_and_print(f"Encoding Notice: The column '{orig_col}' was encoded into {len(new_cols)} new columns: {new_cols}\n")

X_train = df_train[existing_train_columns]
y_train = df_train[output_column]
logging.info(f"Final train_columns: {existing_train_columns}")
logging.info(f"Available columns in dataset: {df_train.columns.tolist()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Store the FULL training data before any splits (for final model training)
X_train_full = X_train_scaled.copy()
y_train_full = y_train.copy()

print("\nData Cleaning & Exploration Done. Check logs for details.")

# ====== Create Model Factory Function (for CV and Learning Curve) ======
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def create_model():
    """Factory function to create a model instance based on regularization type"""
    if regularization_type == "ridge":
        return Ridge(alpha=alpha_value)
    elif regularization_type == "lasso":
        return Lasso(alpha=alpha_value, max_iter=10000)
    elif regularization_type == "elasticnet":
        return ElasticNet(alpha=alpha_value, max_iter=10000)
    else:
        return LinearRegression()

# ====== Cross-Validation on FULL Training Data (Before Split) ======
cv_mean_r2 = None
cv_std_r2 = None
cv_mean_mse = None
cv_std_mse = None
cv_fold_models = []  # Store all CV fold models with their scores

if enable_cv:
    from sklearn.model_selection import KFold, cross_val_score
    
    log_and_print(f"\n{'='*80}")
    log_and_print(f"Performing {cv_folds}-Fold Cross-Validation on FULL Training Dataset...")
    log_and_print(f"Saving ALL fold models for user selection...")
    log_and_print(f"{'='*80}")
    
    model_type_name = {
        "ridge": f"Ridge (α={alpha_value})",
        "lasso": f"Lasso (α={alpha_value})",
        "elasticnet": f"ElasticNet (α={alpha_value})",
        "none": "Linear Regression"
    }[regularization_type]
    
    log_and_print(f"Model Type: {model_type_name}")
    
    # Manually perform CV to save each fold's model
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_scores_r2 = []
    fold_scores_mse = []
    
    log_and_print(f"\nTraining and saving individual fold models:")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        # Split data for this fold
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model on this fold
        fold_model = create_model()
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Evaluate on validation set
        y_fold_pred = fold_model.predict(X_fold_val)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_rmse = np.sqrt(fold_mse)
        
        # Calculate Adjusted R²
        n = len(y_fold_val)
        p = X_fold_val.shape[1]
        fold_adj_r2 = 1 - (1 - fold_r2) * (n - 1) / (n - p - 1) if n > p + 1 else fold_r2
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        fold_mape = np.mean(np.abs((y_fold_val - y_fold_pred) / np.where(y_fold_val != 0, y_fold_val, 1))) * 100
        
        # Calculate accuracy for binary classification
        fold_accuracy = None
        unique_fold_vals = np.unique(y_fold_val)
        if len(unique_fold_vals) == 2 and all(v in [0, 1] for v in unique_fold_vals):
            y_fold_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_fold_pred]
            fold_accuracy = accuracy_score(y_fold_val, y_fold_pred_binary)
        
        fold_scores_r2.append(fold_r2)
        fold_scores_mse.append(fold_mse)
        
        # Store model info
        cv_fold_models.append({
            'fold': fold_idx,
            'model': fold_model,
            'r2_score': fold_r2,
            'mse': fold_mse,
            'mae': fold_mae,
            'rmse': fold_rmse,
            'adj_r2': fold_adj_r2,
            'mape': fold_mape,
            'accuracy': fold_accuracy,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })
        
        accuracy_str = f", Accuracy={fold_accuracy:.4f}" if fold_accuracy is not None else ""
        log_and_print(f"  Fold {fold_idx}: R²={fold_r2:.4f}, MSE={fold_mse:.4f}{accuracy_str} (trained on {len(train_idx)} samples)")
    
    # Calculate aggregate statistics
    cv_mean_r2 = np.mean(fold_scores_r2)
    cv_std_r2 = np.std(fold_scores_r2)
    cv_mean_mse = np.mean(fold_scores_mse)
    cv_std_mse = np.std(fold_scores_mse)
    
    log_and_print(f"\nCross-Validation Summary:")
    log_and_print(f"  Mean R² Score: {cv_mean_r2:.4f} (± {cv_std_r2:.4f})")
    log_and_print(f"  Mean MSE: {cv_mean_mse:.4f} (± {cv_std_mse:.4f})")
    log_and_print(f"  Min R² Score: {min(fold_scores_r2):.4f}")
    log_and_print(f"  Max R² Score: {max(fold_scores_r2):.4f}")
    log_and_print(f"{'='*80}\n")
    log_and_print(f"✓ Saved {len(cv_fold_models)} CV fold models")
    log_and_print("Now proceeding to train final model on full dataset...\n")
else:
    log_and_print("Cross-validation disabled.\n")





# ========== TEST DATA PROCESSING ==========
if df_test_original is not None:
    logging.info("Starting data cleaning for test dataset...")
    df_test_original.columns = df_test_original.columns.str.strip()
    missing_in_test = [col for col in all_needed_columns if col not in df_test_original.columns]
    if missing_in_test:
        logging.error(f"Error: The following columns are missing in the test dataset: {missing_in_test}")
        sys.exit(1)
    df_test = df_test_original[all_needed_columns].copy()
    numeric_cols_test = df_test.select_dtypes(include=[np.number]).columns
    non_numeric_cols_test = df_test.select_dtypes(exclude=[np.number]).columns

    logging.info("Applying missing value handling for test dataset (using train set statistics)...")
    if selected_missingval_tech == "Mean Imputation":
        df_test[numeric_cols_test] = df_test[numeric_cols_test].fillna(df_train[numeric_cols].mean())
        logging.info("Test Set: Missing values in numeric columns filled using Mean Imputation (train mean).")
    elif selected_missingval_tech == "Median Imputation":
        df_test[numeric_cols_test] = df_test[numeric_cols_test].fillna(df_train[numeric_cols].median())
        logging.info("Test Set: Missing values in numeric columns filled using Median Imputation (train median).")
    elif selected_missingval_tech == "Mode Imputation":
        for col in numeric_cols_test:
            df_test[col] = df_test[col].fillna(df_train[col].mode()[0])
        for col in non_numeric_cols_test:
            df_test[col] = df_test[col].fillna(df_train[col].mode()[0])
        logging.info("Test Set: Missing values filled using Mode Imputation (train mode).")
    elif selected_missingval_tech == "Forward/Backward Fill":
        df_test.ffill(inplace=True)
        df_test.bfill(inplace=True)
        logging.info("Test Set: Missing values filled using Forward/Backward Fill.")
    elif selected_missingval_tech == "Drop Rows with Missing Values":
        before_rows = df_test.shape[0]
        df_test.dropna(inplace=True)
        after_rows = df_test.shape[0]
        logging.info(f"Test Set: Dropped {before_rows - after_rows} rows due to missing values.")
    else:
        logging.error(f"Invalid missing value handling technique: {selected_missingval_tech}")
        sys.exit(1)

    if args.remove_duplicates:
        before_rows = df_test.shape[0]
        df_test.drop_duplicates(inplace=True)
        after_rows = df_test.shape[0]
        logging.info(f"Test Set: Removed {before_rows - after_rows} duplicate rows.")

    if args.enable_outlier_detection.lower() == "true":
        numeric_cols_test = df_test.select_dtypes(include=[np.number]).columns
        logging.info(f"🛠️ Outlier removal enabled for test set using method: {args.outlier_method}")
        method = str(args.outlier_method).strip().lower().strip('"').strip("'")
        if method == "z-score":
            logging.info(f"🔹 Applying Z-score method to test set (threshold={args.z_score_threshold})...")
            if len(numeric_cols_test) == 0:
                logging.warning("⚠️ No numeric columns found in test set. Skipping outlier removal.")
            else:
                try:
                    z_scores_test = np.abs(stats.zscore(df_test[numeric_cols_test]))
                    df_test = df_test[(z_scores_test < args.z_score_threshold).all(axis=1)]
                    logging.info(f"✅ Outliers removed using Z-score method in test set. New shape: {df_test.shape}")
                except Exception as e:
                    logging.error(f"⚠️ Error in Z-score computation for test set: {e}")
        elif method == "iqr":
            logging.info("🔹 Applying IQR method to test set...")
            Q1 = df_test[numeric_cols_test].quantile(0.25)
            Q3 = df_test[numeric_cols_test].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (args.iqr_lower * IQR)
            upper_bound = Q3 + (args.iqr_upper * IQR)
            df_test = df_test[~((df_test[numeric_cols_test] < lower_bound) | (df_test[numeric_cols_test] > upper_bound)).any(axis=1)]
            logging.info(f"✅ Outliers removed using IQR method in test set. New shape: {df_test.shape}")
        elif method == "winsorization":
            logging.info("🔹 Applying Winsorization to test set...")
            from scipy.stats.mstats import winsorize
            for col in numeric_cols_test:
                df_test[col] = winsorize(df_test[col], limits=(args.winsor_lower/100, (100-args.winsor_upper)/100))
            logging.info("✅ Outliers capped using Winsorization.")
    else:
        logging.info("⚠️ Outlier removal is disabled for test set. Skipping outlier detection.")

    if feature_scaling:
        logging.info(f"✅ Applying Feature Scaling to Test Set: {feature_scaling}")
        if scaler:
            # Scale only the training features (existing_train_columns)
            X_test = df_test[existing_train_columns]
            X_test_scaled = scaler.transform(X_test)
            logging.info(f"✅ Feature Scaling applied to test data using {feature_scaling}.")
        else:
            logging.warning("⚠️ No scaler available from training. Skipping scaling on test set.")
    else:
        logging.info(" -- No feature scaling selected for test set. Skipping feature scaling. --")
    
    X_test = df_test[existing_train_columns]
    y_test = df_test[output_column]
    X_test_scaled = scaler.transform(X_test)
    logging.info("Test Data Cleaning Completed.")

    logging.info("Applying categorical encoding for test dataset...")
    if args.encoding_type.lower() == "one-hot":
        test_feature_cols = [col for col in df_test.columns if col != output_column]
        df_test = pd.get_dummies(df_test, columns=[col for col in test_feature_cols if df_test[col].dtype == 'object'], drop_first=True)
        logging.info("Test Set: Categorical columns converted to dummy variables (one-hot encoding).")
    elif args.encoding_type.lower() == "label":
        from sklearn.preprocessing import LabelEncoder
        label_encoders_test = {}
        for col in non_numeric_cols_test:
            le = LabelEncoder()
            if col in label_encoders_train:
                df_test[col] = df_test[col].map(lambda x: label_encoders_train[col].transform([x])[0] if x in label_encoders_train[col].classes_ else -1)
            else:
                df_test[col] = le.fit_transform(df_test[col])
                label_encoders_test[col] = le
        logging.info("Test Set: Categorical columns converted using Label Encoding.")
    elif args.encoding_type.lower() == "target":
        if 'target_means' not in locals():
            logging.error("Target encoding failed: Train target means not found.")
            sys.exit(1)
        for col in non_numeric_cols_test:
            if col in target_means:
                df_test[col] = df_test[col].map(target_means[col]).fillna(df_train[output_column].mean())
        logging.info("Test Set: Categorical columns converted using Target Encoding.")
    elif args.encoding_type.lower() == "none":
        skipped_categorical_cols = df_test.select_dtypes(exclude=[np.number]).columns.tolist()
        if skipped_categorical_cols:
            logging.info(f"Skipping categorical encoding. Using only numeric features. Skipped columns: {skipped_categorical_cols}")
        else:
            logging.info("No categorical columns found. Proceeding with only numeric features.")
        df_test = df_test.select_dtypes(include=[np.number])
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
    
    
else:
    # Only perform train_test_split if CV is NOT enabled
    # When CV is enabled, we keep the full dataset for training
    if not enable_cv:
        test_size = args.test_split_ratio if args.test_split_ratio else 0.2
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X_train_scaled, y_train, test_size=test_size, random_state=42
        )
        logging.info("Test Data Cleaning Completed via train_test_split.")
    else:
        # When CV is enabled without a separate test file, use the full dataset
        # We'll create a test set from a small portion for evaluation purposes
        X_test_scaled = X_train_scaled
        y_test = y_train
        logging.info("CV enabled without separate test file. Using full dataset for evaluation.")






# ====== Train Final Model ======
log_and_print(f"\n{'='*80}")
log_and_print("Training Final Model...")
log_and_print(f"{'='*80}")

# Create model using factory function
model = create_model()

# Log model type
model_type_name = {
    "ridge": f"Ridge Regression with alpha={alpha_value}",
    "lasso": f"Lasso Regression with alpha={alpha_value}",
    "elasticnet": f"ElasticNet Regression with alpha={alpha_value}",
    "none": "Standard Linear Regression (no regularization)"
}[regularization_type]
log_and_print(f"Model Type: {model_type_name}")

# If CV is enabled, train on FULL dataset (best practice after CV validation)
# If CV is disabled, train on the split training set
if enable_cv:
    log_and_print(f"Training on FULL dataset ({len(y_train_full)} samples) - CV validated")
    model.fit(X_train_full, y_train_full)
    training_data_used = "full"
else:
    log_and_print(f"Training on split training set ({len(y_train)} samples)")
    model.fit(X_train_scaled, y_train)
    training_data_used = "split"

log_and_print("✓ Model Training Completed!")

# ====== Make Predictions ======
y_pred = model.predict(X_test_scaled)

# Set X_train_full_scaled for learning curves and other plots
# When CV is enabled, X_train_scaled still contains the full dataset
# When CV is disabled, we need to use the original full dataset saved earlier
if not enable_cv:
    X_train_full_scaled = X_train_full  # Use the original full dataset from line 530
else:
    X_train_full_scaled = X_train_scaled  # Already the full dataset when CV is enabled


# Detect if classification or regression
unique_vals = np.unique(y_test)
is_classification = (len(unique_vals) == 2)

# For classification, compute accuracy
if is_classification:
    # Convert to binary using a threshold of 0.5
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)
else:
    accuracy = None

# ====== Calculate Performance Metrics ======
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate Adjusted R²
n_samples = len(y_test)
n_features = X_test_scaled.shape[1]
adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1) if n_samples > n_features + 1 else r2

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100






# ====== Plots ======

# Compute residuals
residuals = y_test - y_pred

# ----- Learning Curve (Generated First) -----
if selected_graphs is None or any(item.startswith("Learning Curve") for item in (selected_graphs if selected_graphs else [])) or selected_graphs is None:
    from sklearn.model_selection import learning_curve, KFold
    
    log_and_print("Generating Learning Curves...")
    
    # Use the appropriate training data for learning curve
    if enable_cv:
        X_learning = X_train_full_scaled
        # Convert pandas Series to numpy array to avoid indexing issues
        y_learning = y_train_full.values if hasattr(y_train_full, 'values') else y_train_full
    else:
        # For non-CV, use the split training data (after train_test_split)
        X_learning = X_train_scaled
        # Convert pandas Series to numpy array to avoid indexing issues
        y_learning = y_train.values if hasattr(y_train, 'values') else y_train
    
    log_and_print(f"Learning curve data: X shape = {X_learning.shape}, y shape = {y_learning.shape}")
    
    # Define training set sizes (from 10% to 100% in steps)
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    if enable_cv and cv_folds > 0 and (selected_graphs is None or "Learning Curve - All Folds" in selected_graphs):
        # Generate individual learning curve for each CV fold
        log_and_print(f"Generating {cv_folds} individual learning curves (one per CV fold)...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_learning), 1):
                
            X_fold = X_learning[train_idx]
            y_fold = y_learning[train_idx]  # y_learning is already a numpy array
            
            # Calculate learning curve for this fold
            train_sizes_abs, train_scores, val_scores = learning_curve(
                create_model(),
                X_fold,
                y_fold,
                train_sizes=train_sizes,
                cv=3,  # Inner CV
                scoring='r2',
                n_jobs=-1,
                shuffle=True,
                random_state=42
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create the plot for this fold
            learning_curve_path = os.path.join(output_dir, f"learning_curve_fold_{fold_idx}.png")
            plt.figure(figsize=(10, 6))
            
            # Plot training scores
            plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
            
            # Plot validation scores
            plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
            
            plt.xlabel('Training Set Size', fontsize=12)
            plt.ylabel('R² Score', fontsize=12)
            plt.title(f'Learning Curve - CV Fold {fold_idx}', fontsize=14, fontweight='bold')
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(learning_curve_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            generated_graphs.append(normalize_path(learning_curve_path))
            log_and_print(f"  Fold {fold_idx} Learning Curve saved: {learning_curve_path}")
            log_and_print(f"    Training samples: {train_sizes_abs[0]:.0f} to {train_sizes_abs[-1]:.0f}")
            log_and_print(f"    Final R² - Train: {train_mean[-1]:.4f} (± {train_std[-1]:.4f}), Val: {val_mean[-1]:.4f} (± {val_std[-1]:.4f})")
    
    # Always generate overall learning curve (aggregate or single)
    if selected_graphs is None or "Learning Curve - Overall" in selected_graphs:
        log_and_print("Generating overall learning curve...")
        
        # Calculate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            create_model(),
            X_learning,
            y_learning,
            train_sizes=train_sizes,
            cv=cv_folds if enable_cv else 3,
            scoring='r2',
            n_jobs=-1,
            shuffle=True,
            random_state=42
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create the plot
        learning_curve_path = os.path.join(output_dir, "learning_curve_overall.png")
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        title = 'Learning Curve - Overall Performance' if enable_cv else 'Learning Curve - Model Performance'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(learning_curve_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        generated_graphs.append(normalize_path(learning_curve_path))
        log_and_print(f"Overall Learning Curve saved to {learning_curve_path}")
        log_and_print(f"  Training samples range: {train_sizes_abs[0]:.0f} to {train_sizes_abs[-1]:.0f}")
        log_and_print(f"  Final training R² score: {train_mean[-1]:.4f} (± {train_std[-1]:.4f})")
        log_and_print(f"  Final validation R² score: {val_mean[-1]:.4f} (± {val_std[-1]:.4f})")


# ====== Correlation & Basic Exploration ======

# ====== Correlation Matrix ======
# Only for numeric columns in the selected subset
numeric_corr = df_train.select_dtypes(include=[np.number]).corr()
logging.info(f"\nCorrelation Matrix:\n{numeric_corr}")

# ====== Save Feature Correlation Heatmap (Selected Columns Only) ======
if selected_graphs is None or "Heatmap" in selected_graphs:
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(heatmap_path)
    plt.close()
    generated_graphs.append(normalize_path(heatmap_path))
    log_and_print(f"Feature correlation heatmap saved to {heatmap_path}")

# ====== Save Histogram of Output Distribution ======
if selected_graphs is None or "Histogram Distribution" in selected_graphs:
    histogram_path = os.path.join(output_dir, "Histogram_distribution.png")
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train[output_column], bins=3, kde=True)
    plt.title("Distribution of Output")
    plt.xlabel(output_column)
    plt.savefig(histogram_path)
    plt.close()
    generated_graphs.append(normalize_path(histogram_path))
    log_and_print(f"Output distribution histogram saved to {histogram_path}")

# ====== Save Box Plots for All Columns ======
if selected_graphs is None or "Box Plot" in selected_graphs:
    boxplot_path = os.path.join(output_dir, "Box_plots.png")
    
    # Select only numeric columns for box plot visualization
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_train[numeric_cols])
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.title("Box Plots for All Numeric Features")
        plt.savefig(boxplot_path, bbox_inches="tight")  # Save with tight layout
        plt.close()
        generated_graphs.append(normalize_path(boxplot_path))
        log_and_print(f"Box plots saved to {boxplot_path}")
    else:
        log_and_print("No numeric columns found for box plots.")



# this is for residual and reuse
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')

# Residual Plot: Predicted vs. Residuals
if selected_graphs is None or "Residual Plot" in selected_graphs:
    residual_plot_path = os.path.join(output_dir, "residual_plot.png")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(residual_plot_path)
    plt.close()
    generated_graphs.append(normalize_path(residual_plot_path))
    log_and_print(f"Residual Plot saved to {residual_plot_path}")

# Histogram of Residuals
if selected_graphs is None or "Histogram Residuals" in selected_graphs:
    hist_resid_path = os.path.join(output_dir, "histogram_residuals.png")
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residuals")
    plt.title("Histogram of Residuals")
    plt.savefig(hist_resid_path)
    plt.close()
    generated_graphs.append(normalize_path(hist_resid_path))
    log_and_print(f"Histogram of Residuals saved to {hist_resid_path}")


weight_plot_path = os.path.join(output_dir, "weight_plot.png")
plt.figure(figsize=(10, 6))

# ----- Model Coefficients Plot (Weight Plot) -----
if selected_graphs is None or "Model Coefficients" in selected_graphs:
    # ----- Weight Plot (Coefficient Plot) -----
    
    coefficients = model.coef_
    features = existing_train_columns
    weight_plot_path = os.path.join(output_dir, "model_coefficients.png")
    plt.figure(figsize=(10, 6))
    plt.bar(features, coefficients, color='steelblue')
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient Value")
    plt.title("Model Coefficients")
    plt.tight_layout()
    plt.savefig(weight_plot_path)
    plt.close()
    generated_graphs.append(normalize_path(weight_plot_path))
    log_and_print(f"Model Coefficients plot saved to {weight_plot_path}")

# ----- Effect Plot for Selected Features -----
# (Also known as a Partial Dependence or Effect Plot)
if selected_graphs is None or "Effect Plot" in selected_graphs:
    # Determine which features to plot
    features_to_plot = effect_features if effect_features else [existing_train_columns[0]]
    
    for selected_feature in features_to_plot:
        if selected_feature not in existing_train_columns:
            log_and_print(f"Feature '{selected_feature}' not in train columns. Skipping.")
            continue
            
        if np.issubdtype(df_train[selected_feature].dtype, np.number):
            effect_plot_path = os.path.join(output_dir, f"effect_plot_{selected_feature}.png")
        
            # Create a range for the selected feature
            x_vals = np.linspace(df_train[selected_feature].min(), df_train[selected_feature].max(), 100)
        
            # Create a baseline vector (mean for each feature) using the training set
            baseline = X_train.mean(axis=0)
            X_effect = np.tile(baseline, (100, 1))
        
            # Replace the column corresponding to the selected feature with the range of values
            idx = existing_train_columns.index(selected_feature)
            X_effect[:, idx] = x_vals
        
            # Scale the effect array using the same scaler
            X_effect_scaled = scaler.transform(X_effect)
            # Predict using the trained model
            y_effect = model.predict(X_effect_scaled)
        
            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_effect, color='darkorange', lw=2)
            plt.xlabel(f"{selected_feature}")
            plt.ylabel(f"Predicted {output_column}")
            plt.title(f"Effect of {selected_feature} on {output_column}")
            plt.tight_layout()
            plt.savefig(effect_plot_path)
            plt.close()
            generated_graphs.append(normalize_path(effect_plot_path))
            log_and_print(f"Effect Plot for {selected_feature} saved to {effect_plot_path}")
        else:
            log_and_print(f"Selected feature '{selected_feature}' is not numeric. Skipping Effect Plot.")



# ----- Mean Effect Plot -----
if selected_graphs is None or "Mean Effect Plot" in selected_graphs:
    features_to_plot = effect_features if effect_features else [existing_train_columns[0]]
    
    for selected_feature in features_to_plot:
        if selected_feature not in existing_train_columns:
            continue
            
        # Create bins for the selected feature (using original, unscaled values)
        feature_vals = df_train[selected_feature]
        
        bins = np.linspace(feature_vals.min(), feature_vals.max(), 20)
        bin_indices = np.digitize(feature_vals, bins)
        mean_effect = []
        bin_centers = []
        for b in np.unique(bin_indices):
            # Compute the center of the bin
            bin_center = bins[b-1] + (bins[1]-bins[0]) / 2
            bin_centers.append(bin_center)
            # Create a baseline vector from X_train (mean values)
            baseline = X_train.mean(axis=0)
            X_effect = np.tile(baseline, (1, 1))
            idx = existing_train_columns.index(selected_feature)
            X_effect[0, idx] = bin_center
            X_effect_df = pd.DataFrame(X_effect, columns=existing_train_columns)
            X_effect_scaled = scaler.transform(X_effect_df)
            y_eff = model.predict(X_effect_scaled)
            mean_effect.append(y_eff[0])

        mean_effect_plot_path = os.path.join(output_dir, f"mean_effect_plot_{sanitize_filename(selected_feature)}.png")
        plt.figure(figsize=(8,6))
        plt.plot(bin_centers, mean_effect, marker='o', linestyle='-')
        plt.xlabel(selected_feature)
        plt.ylabel(f"Mean Predicted {output_column}")
        plt.title(f"Mean Effect Plot for {selected_feature}")
        plt.tight_layout()
        plt.savefig(mean_effect_plot_path)
        plt.close()
        generated_graphs.append(normalize_path(mean_effect_plot_path))
        log_and_print(f"Mean Effect Plot for {selected_feature} saved to {mean_effect_plot_path}")





X_train_full_scaled = scaler.transform(X_train)  # X_train from df_train (full set)
y_pred_full = model.predict(X_train_full_scaled)

# ----- Individual Effect Plot -----
if selected_graphs is None or "Individual Effect Plot" in selected_graphs:
    features_to_plot = effect_features if effect_features else [existing_train_columns[0]]
    
    for selected_feature in features_to_plot:
        if selected_feature not in existing_train_columns:
            continue
            
        # Generate predictions on the full cleaned training set
        plt.figure(figsize=(8,6))
        plt.scatter(df_train[selected_feature].values, y_pred_full, alpha=0.5)
        plt.xlabel(selected_feature)
        plt.ylabel(f"Predicted {output_column}")
        plt.title(f"Individual Effect Plot for {selected_feature}")
        plt.tight_layout()
        individual_effect_plot_path = os.path.join(output_dir, f"individual_effect_plot_{sanitize_filename(selected_feature)}.png")
        plt.savefig(individual_effect_plot_path)
        plt.close()
        generated_graphs.append(normalize_path(individual_effect_plot_path))
        log_and_print(f"Individual Effect Plot for {selected_feature} saved to {individual_effect_plot_path}")




# ----- Trend Effect Plot -----
if selected_graphs is None or "Trend Effect Plot" in selected_graphs:
    features_to_plot = effect_features if effect_features else [existing_train_columns[0]]
    
    for selected_feature in features_to_plot:
        if selected_feature not in existing_train_columns:
            continue
            
        # Sort instances by the selected feature using the full training set (df_train)
        sorted_indices = np.argsort(df_train[selected_feature].values)
        sorted_feature = df_train[selected_feature].values[sorted_indices]
        # Use the full training set predictions for plotting
        sorted_predictions = model.predict(X_train_full_scaled)[sorted_indices]

        trend_effect_plot_path = os.path.join(output_dir, f"trend_effect_plot_{sanitize_filename(selected_feature)}.png")
        plt.figure(figsize=(8,6))
        plt.plot(sorted_feature, sorted_predictions, color='purple', lw=2)
        plt.xlabel(selected_feature)
        plt.ylabel(f"Predicted {output_column}")
        plt.title(f"Trend Effect Plot for {selected_feature}")
        plt.tight_layout()
        plt.savefig(trend_effect_plot_path)
        plt.close()
        generated_graphs.append(normalize_path(trend_effect_plot_path))
        log_and_print(f"Trend Effect Plot for {selected_feature} saved to {trend_effect_plot_path}")




# ----- SHAP Summary Plot for Linear Model -----
if selected_graphs is None or "Shap Summary Plot" in selected_graphs:
    # ----- SHAP Summary Plot for Linear Model -----
    try:
        import shap
    except ImportError:
        log_and_print("SHAP library not installed. Installing now...")
        run_command("python -m pip install shap")
        import shap

    if selected_graphs is None or "Shap Summary Plot" in selected_graphs:
        # Use a linear explainer without the deprecated parameter
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_train_scaled)

        shap_summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.figure()
        X_shap = pd.DataFrame(X_train_scaled, columns=existing_train_columns)
        shap.summary_plot(shap_values, X_shap, feature_names=existing_train_columns, show=False)
        plt.title("SHAP Summary Plot")
        plt.savefig(shap_summary_plot_path, bbox_inches="tight")
        plt.close()
        generated_graphs.append(normalize_path(shap_summary_plot_path))
        log_and_print(f"SHAP Summary Plot saved to {shap_summary_plot_path}")



# ====== Save Models (Final + All CV Fold Models) ======
log_and_print(f"\n{'='*80}")
log_and_print("Saving Models...")
log_and_print(f"{'='*80}")

model_path = os.path.join(output_dir, "model.pkl")
scaler_path = os.path.join(output_dir, "scaler.pkl")
preprocessing_path = os.path.join(output_dir, "preprocessing.pkl")

# Save the final model (trained on full data if CV enabled)
joblib.dump(model, model_path)
log_and_print(f"✓ Final model saved: {model_path}")

# Save all CV fold models if CV was enabled
available_models = []
if enable_cv and len(cv_fold_models) > 0:
    log_and_print(f"\nSaving {len(cv_fold_models)} CV fold models...")
    for fold_info in cv_fold_models:
        fold_model_path = os.path.join(output_dir, f"model_fold_{fold_info['fold']}.pkl")
        joblib.dump(fold_info['model'], fold_model_path)
        log_and_print(f"  ✓ Fold {fold_info['fold']} model saved (R²={fold_info['r2_score']:.4f})")
        
        # Add to available models list
        available_models.append({
            'name': f"CV Fold {fold_info['fold']}",
            'filename': f"model_fold_{fold_info['fold']}.pkl",
            'r2_score': float(fold_info['r2_score']),
            'mse': float(fold_info['mse']),
            'mae': float(fold_info['mae']),
            'rmse': float(fold_info['rmse']),
            'adj_r2': float(fold_info['adj_r2']),
            'mape': float(fold_info['mape']),
            'accuracy': float(fold_info['accuracy']) if fold_info['accuracy'] is not None else None,
            'type': 'cv_fold',
            'fold_number': fold_info['fold'],
            'train_size': fold_info['train_size'],
            'val_size': fold_info['val_size']
        })

# Add final model to available models
available_models.insert(0, {
    'name': f"Final Model ({'CV-Trained on Full Data' if enable_cv else 'Train/Test Split'})",
    'filename': 'model.pkl',
    'r2_score': float(r2),
    'mse': float(mse),
    'mae': float(mae),
    'rmse': float(rmse),
    'adj_r2': float(adj_r2),
    'mape': float(mape),
    'accuracy': float(accuracy) if accuracy is not None else None,
    'type': 'final',
    'trained_on_full': enable_cv,
    'train_size': len(y_train_full) if enable_cv else len(y_train)
})

log_and_print(f"\n✓ Total models available for prediction: {len(available_models)}")

# Detect if target is binary classification
target_unique_values = sorted(df_train[output_column].unique())
is_binary = len(target_unique_values) <= 2 and all(v in [0, 1] for v in target_unique_values)
log_and_print(f"Target column '{output_column}': {len(target_unique_values)} unique values - {'Binary Classification' if is_binary else 'Regression'}")

# Prepare preprocessing information
preprocessing_info = {
    'original_train_columns': train_columns,  # User-selected columns before encoding
    'final_feature_names': existing_train_columns,  # Actual features after encoding
    'encoding_type': args.encoding_type,
    'categorical_cols': categorical_cols,
    'categorical_values': categorical_values_dict,  # Unique values for each categorical column (extracted before encoding)
    'label_encoders': label_encoders if args.encoding_type.lower() == 'label' else None,
    'target_means': target_means_dict if args.encoding_type.lower() == 'target' else None,  # Mean mappings for Target Encoding
    'encoding_added_cols': encoding_added_cols,  # Mapping of original → encoded columns
    'is_binary_classification': is_binary,  # Whether target is binary (0/1)
    'target_name': output_column,  # Name of target column for display
    'regularization_type': regularization_type,  # Type of regularization used
    'alpha': alpha_value,  # Regularization strength
    'available_models': available_models,  # List of all available models with scores
    'cv_enabled': enable_cv,  # Whether CV was used
    'cv_folds': cv_folds if enable_cv else None  # Number of folds used
}

joblib.dump(scaler, scaler_path)
joblib.dump(preprocessing_info, preprocessing_path)
log_and_print(f"✓ Scaler saved: {scaler_path}")
log_and_print(f"✓ Preprocessing info saved: {preprocessing_path}")

# ====== Save Paths to a File ======
path_file = os.path.join(output_dir, "saved_paths.txt")
with open(path_file, "w") as f:
    f.write(f"Model Path: {model_path}\\n")
    f.write(f"CSV Path: {train_csv_path}\\n")
log_and_print(f"Saved paths in '{path_file}'")



log_and_print("\n" + "="*100)
log_and_print("MODEL PERFORMANCE SUMMARY")
log_and_print("="*100)

# Create a comprehensive results table
if enable_cv and len(cv_fold_models) > 0:
    log_and_print("\n📊 COMPREHENSIVE RESULTS TABLE (All Models)")
    log_and_print("-"*150)
    
    # Table header
    if is_classification:
        header = f"{'Model':<25} {'R²':<10} {'Adj R²':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Accuracy':<12} {'Train':<8}"
    else:
        header = f"{'Model':<25} {'R²':<10} {'Adj R²':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Train':<8}"
    log_and_print(header)
    log_and_print("-"*150)
    
    # CV Fold rows
    for fold_info in cv_fold_models:
        model_name = f"CV Fold {fold_info['fold']}"
        r2_str = f"{fold_info['r2_score']:.4f}"
        adj_r2_str = f"{fold_info['adj_r2']:.4f}"
        mse_str = f"{fold_info['mse']:.4f}"
        mae_str = f"{fold_info['mae']:.4f}"
        rmse_str = f"{fold_info['rmse']:.4f}"
        mape_str = f"{fold_info['mape']:.2f}%"
        train_size = fold_info['train_size']
        
        if is_classification and fold_info['accuracy'] is not None:
            acc_str = f"{fold_info['accuracy']:.4f}"
            row = f"{model_name:<25} {r2_str:<10} {adj_r2_str:<10} {mse_str:<12} {mae_str:<12} {rmse_str:<12} {mape_str:<10} {acc_str:<12} {train_size:<8}"
        else:
            row = f"{model_name:<25} {r2_str:<10} {adj_r2_str:<10} {mse_str:<12} {mae_str:<12} {rmse_str:<12} {mape_str:<10} {train_size:<8}"
        log_and_print(row)
    
    # Separator before final model
    log_and_print("-"*150)
    
    # Final model row (highlighted)
    model_name = "Final Model (Full)"
    r2_str = f"{r2:.4f}"
    adj_r2_str = f"{adj_r2:.4f}"
    mse_str = f"{mse:.4f}"
    mae_str = f"{mae:.4f}"
    rmse_str = f"{rmse:.4f}"
    mape_str = f"{mape:.2f}%"
    train_size = len(y_train_full)
    
    if is_classification and accuracy is not None:
        acc_str = f"{accuracy:.4f}"
        row = f"{model_name:<25} {r2_str:<10} {adj_r2_str:<10} {mse_str:<12} {mae_str:<12} {rmse_str:<12} {mape_str:<10} {acc_str:<12} {train_size:<8}"
    else:
        row = f"{model_name:<25} {r2_str:<10} {adj_r2_str:<10} {mse_str:<12} {mae_str:<12} {rmse_str:<12} {mape_str:<10} {train_size:<8}"
    log_and_print(row + " ⭐")
    
    log_and_print("="*150)
    
    # Summary statistics
    log_and_print("\n📈 CV STATISTICS:")
    log_and_print(f"  Mean R² Score: {cv_mean_r2:.4f} (± {cv_std_r2:.4f})")
    log_and_print(f"  Mean MSE: {cv_mean_mse:.4f} (± {cv_std_mse:.4f})")
    log_and_print(f"  Model Stability: {'✓ Stable' if cv_std_r2 < 0.1 else '⚠ Variable'} (Std Dev: {cv_std_r2:.4f})")
    
    log_and_print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    log_and_print(f"  R² Score: {r2:.4f}")
    log_and_print(f"  Adjusted R² Score: {adj_r2:.4f}")
    log_and_print(f"  Mean Squared Error (MSE): {mse:.4f}")
    log_and_print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    log_and_print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    log_and_print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    if is_classification and accuracy is not None:
        log_and_print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_and_print(f"  Training Samples: {len(y_train_full)} (100% of data)")
    log_and_print(f"  Test Samples: {len(y_test)}")

else:
    # No CV - just show final model results
    log_and_print("\n📊 MODEL RESULTS (Train/Test Split)")
    log_and_print("-"*100)
    
    if is_classification:
        header = f"{'Metric':<40} {'Value':<20}"
    else:
        header = f"{'Metric':<40} {'Value':<20}"
    log_and_print(header)
    log_and_print("-"*100)
    
    log_and_print(f"{'R² Score':<40} {r2:.4f}")
    log_and_print(f"{'Adjusted R² Score':<40} {adj_r2:.4f}")
    log_and_print(f"{'Mean Squared Error (MSE)':<40} {mse:.4f}")
    log_and_print(f"{'Mean Absolute Error (MAE)':<40} {mae:.4f}")
    log_and_print(f"{'Root Mean Squared Error (RMSE)':<40} {rmse:.4f}")
    log_and_print(f"{'Mean Absolute Percentage Error (MAPE)':<40} {mape:.2f}%")
    if is_classification and accuracy is not None:
        log_and_print(f"{'Accuracy':<40} {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_and_print(f"{'Training Samples':<40} {len(y_train)}")
    log_and_print(f"{'Test Samples':<40} {len(y_test)}")
    log_and_print("="*100)

# ====== Print Generated Graphs for Frontend ======
import json
print(f"__GENERATED_GRAPHS_JSON__{json.dumps(generated_graphs)}__END_GRAPHS__")
sys.stdout.flush()

log_and_print("========== FINISHED SUCCESSFULLY ==========")
