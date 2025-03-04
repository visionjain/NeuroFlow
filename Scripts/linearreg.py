import subprocess
import sys
import ast
import os
import argparse
import logging
import json



def log_and_print(message):
    """Logs and prints the message with extra blank lines and a separator."""
    print(message)
    print("-" * 200)
    sys.stdout.flush()





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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
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
    logging.info(f"Dropped {before_rows - after_rows} rows due to missing values.")
else:
    logging.error(f"Invalid missing value handling technique: {selected_missingval_tech}")
    sys.exit(1)

# ========== Remove Duplicates ==========
if args.remove_duplicates:
    before_rows = df_train.shape[0]
    df_train.drop_duplicates(inplace=True)
    after_rows = df_train.shape[0]
    logging.info(f"Removed {before_rows - after_rows} duplicate rows from the training dataset.")

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
                z_scores = np.abs(stats.zscore(df_train[numeric_cols]))
                df_train = df_train[(z_scores < args.z_score_threshold).all(axis=1)]
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
    target_means = defaultdict(float)
    for col in categorical_cols:
        target_means[col] = df_train.groupby(col)[output_column].mean()
        df_train[col] = df_train[col].map(target_means[col])
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
print("\nData Cleaning & Exploration Done. Check logs for details.")









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
    test_size = args.test_split_ratio if args.test_split_ratio else 0.2
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=test_size, random_state=42
    )
    logging.info("Test Data Cleaning Completed via train_test_split.")






# ====== Train Model ======
model = LinearRegression()
model.fit(X_train_scaled, y_train)
log_and_print("Model Training Completed!")

# ====== Make Predictions ======
y_pred = model.predict(X_test_scaled)
# After computing X_train_scaled on the full df_train:
X_train_full_scaled = X_train_scaled.copy()


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






# ====== Plots ======

# ====== Correlation & Basic Exploration ======

# Compute residuals
residuals = y_test - y_pred

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
    log_and_print(f"Model Coefficients plot saved to {weight_plot_path}")

# ----- Effect Plot for a Selected Feature -----
# (Also known as a Partial Dependence or Effect Plot)
if selected_graphs is None or "Effect Plot" in selected_graphs:
    selected_feature = existing_train_columns[0]
    if np.issubdtype(df_train[selected_feature].dtype, np.number):
        effect_plot_path = os.path.join(output_dir, "effect_plot.png")
    
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
        log_and_print(f"Effect Plot for {selected_feature} saved to {effect_plot_path}")
    else:
        log_and_print(f"Selected feature '{selected_feature}' is not numeric. Skipping Effect Plot.")



selected_feature = existing_train_columns[0]
# Create bins for the selected feature (using original, unscaled values)
feature_vals = df_train[selected_feature]

# ----- Mean Effect Plot -----
if selected_graphs is None or "Mean Effect Plot" in selected_graphs:
    
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

    mean_effect_plot_path = os.path.join(output_dir, "mean_effect_plot.png")
    plt.figure(figsize=(8,6))
    plt.plot(bin_centers, mean_effect, marker='o', linestyle='-')
    plt.xlabel(selected_feature)
    plt.ylabel(f"Mean Predicted {output_column}")
    plt.title(f"Mean Effect Plot for {selected_feature}")
    plt.tight_layout()
    plt.savefig(mean_effect_plot_path)
    plt.close()
    log_and_print(f"Mean Effect Plot saved to {mean_effect_plot_path}")





X_train_full_scaled = scaler.transform(X_train)  # X_train from df_train (full set)
y_pred_full = model.predict(X_train_full_scaled)
plt.figure(figsize=(8,6))
# ----- Individual Effect Plot -----
if selected_graphs is None or "Individual Effect Plot" in selected_graphs:
    # Generate predictions on the full cleaned training set
    plt.scatter(df_train[selected_feature].values, y_pred_full, alpha=0.5)
    plt.xlabel(selected_feature)
    plt.ylabel(f"Predicted {output_column}")
    plt.title(f"Individual Effect Plot for {selected_feature}")
    plt.tight_layout()
    individual_effect_plot_path = os.path.join(output_dir, "individual_effect_plot.png")
    plt.savefig(individual_effect_plot_path)
    plt.close()
    log_and_print(f"Individual Effect Plot saved to {individual_effect_plot_path}")




# ----- Trend Effect Plot -----
if selected_graphs is None or "Trend Effect Plot" in selected_graphs:
    selected_feature = existing_train_columns[0]
    # Sort instances by the selected feature using the full training set (df_train)
    sorted_indices = np.argsort(df_train[selected_feature].values)
    sorted_feature = df_train[selected_feature].values[sorted_indices]
    # Use the full training set predictions for plotting
    sorted_predictions = model.predict(X_train_full_scaled)[sorted_indices]

    trend_effect_plot_path = os.path.join(output_dir, "trend_effect_plot.png")
    plt.figure(figsize=(8,6))
    plt.plot(sorted_feature, sorted_predictions, color='purple', lw=2)
    plt.xlabel(selected_feature)
    plt.ylabel(f"Predicted {output_column}")
    plt.title(f"Trend Effect Plot for {selected_feature}")
    plt.tight_layout()
    plt.savefig(trend_effect_plot_path)
    plt.close()
    log_and_print(f"Trend Effect Plot saved to {trend_effect_plot_path}")




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
        log_and_print(f"SHAP Summary Plot saved to {shap_summary_plot_path}")






# ====== Save Model ======
model_path = os.path.join(output_dir, "model.pkl")
joblib.dump(model, model_path)
log_and_print(f"Model saved as '{model_path}'")

# ====== Save Paths to a File ======
path_file = os.path.join(output_dir, "saved_paths.txt")
with open(path_file, "w") as f:
    f.write(f"Model Path: {model_path}\\n")
    f.write(f"CSV Path: {train_csv_path}\\n")
log_and_print(f"Saved paths in '{path_file}'")



log_and_print("\n======= Model Performance =======")
log_and_print(f"Mean Squared Error: {mse:.4f}")
log_and_print(f"R-squared Score: {r2:.4f}")

if is_classification:
    log_and_print(f"Accuracy Score: {accuracy:.4f}")
else:
    log_and_print("Skipping accuracy score because target is continuous.")




log_and_print("========== FINISHED SUCCESSFULLY ==========")
