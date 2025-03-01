import subprocess
import sys
import ast
import os
import argparse
import logging

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

packages = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "six", "scipy"]
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
import joblib
from scipy import stats


# ====== Parse Command-Line Arguments ======
parser = argparse.ArgumentParser(description="Run Linear Regression Model")
parser.add_argument("--train_csv_path", required=True, help="Path to the training dataset CSV")
parser.add_argument("--test_csv_path", required=False, help="Path to the test dataset CSV (optional)")
parser.add_argument("--test_split_ratio", type=float, help="Test split ratio if test dataset is not provided")
parser.add_argument("--train_columns", required=True, help="Comma-separated column names for training features")
parser.add_argument("--output_column", required=True, help="Name of the target output column")

args = parser.parse_args()

train_csv_path = args.train_csv_path
test_csv_path = args.test_csv_path if args.test_csv_path and args.test_csv_path.lower() != "none" else None

# This will be refined via ast.literal_eval further below
train_columns = args.train_columns.split(",")  
output_column = args.output_column

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
df_train_original.columns = df_train_original.columns.str.strip()  # Strip whitespace from column names

if test_csv_path and os.path.exists(test_csv_path):
    df_test_original = pd.read_csv(test_csv_path)
    df_test_original.columns = df_test_original.columns.str.strip()  # Do the same for test data
else:
    df_test_original = None

log_and_print(f"Dataset Loaded Successfully! Shape: {df_train_original.shape}")


logging.info("========== Data Cleaning & Exploration ==========")
logging.info(f"\nFirst 5 rows:\n{df_train_original.head()}")
logging.info(f"\nData Summary:\n{df_train_original.describe()}")
logging.info(f"\nMissing Values:\n{df_train_original.isnull().sum()}")

# ====== Ensure train_columns is a Proper List ======
try:
    train_columns = ast.literal_eval(args.train_columns)
    if not isinstance(train_columns, list):
        raise ValueError("train_columns should be a list.")
except Exception as e:
    log_and_print(f"Error parsing train_columns: {e}")
    sys.exit(1)

# We'll subset to train_columns + output_column so we only clean relevant columns
# ====== Subset the DataFrame for cleaning ======
all_needed_columns = list(set(train_columns + [output_column]))
df_train = df_train_original[all_needed_columns].copy()

# ====== Data Cleaning on Only the Selected Columns ======
# Fill missing numeric columns with their median
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].median())

# Fill missing non-numeric columns with 'Unknown'
non_numeric_cols = df_train.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    df_train[col].fillna("Unknown", inplace=True)

# Remove duplicates among the selected columns
df_train.drop_duplicates(inplace=True)

# (Optional) Remove outliers using Z-score on numeric columns only
z_scores = np.abs(stats.zscore(df_train.select_dtypes(include=[np.number])))
df_train = df_train[(z_scores < 3).all(axis=1)]

# Convert non-numeric feature columns (excluding the output) to dummy variables
feature_cols = [col for col in df_train.columns if col != output_column]
df_train = pd.get_dummies(df_train, columns=[col for col in feature_cols if df_train[col].dtype == 'object'], drop_first=True)

logging.info("Data cleaning completed.")
print("\nData Cleaning & Exploration Done. Check logs for details.")


# ====== Correlation & Basic Exploration ======
# Only for numeric columns in the selected subset
numeric_corr = df_train.select_dtypes(include=[np.number]).corr()
logging.info(f"\nCorrelation Matrix:\n{numeric_corr}")

# ====== Save Feature Correlation Heatmap (Selected Columns Only) ======
heatmap_path = os.path.join(output_dir, "heatmap.png")
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(heatmap_path)
plt.close()
log_and_print(f"Feature correlation heatmap saved to {heatmap_path}")

# ====== Save Histogram of Output Distribution ======
histogram_path = os.path.join(output_dir, "Histogram_distribution.png")
plt.figure(figsize=(8, 5))
sns.histplot(df_train[output_column], bins=3, kde=True)
plt.title("Distribution of Output")
plt.xlabel(output_column)
plt.savefig(histogram_path)
plt.close()
log_and_print(f"Output distribution histogram saved to {histogram_path}")

# ====== Prepare Final Training Data ======
# Now that we've cleaned only the columns we need:
# If some specified train_columns got dropped due to outlier removal or didn't exist,
# we might warn the user or just proceed with what's left
existing_train_columns = [c for c in train_columns if c in df_train.columns]
if len(existing_train_columns) < len(train_columns):
    log_and_print("Warning: Some specified train_columns not found after cleaning. "
                  "Using only existing columns among the selected subset.")

# If we still have none, there's no point continuing
if not existing_train_columns:
    log_and_print("Error: No valid train columns remain after cleaning.")
    sys.exit(1)

# We'll do a final subset
X_train = df_train[existing_train_columns]
y_train = df_train[output_column]

log_and_print(f"train_columns: {existing_train_columns}")
log_and_print(f"Available columns in the selected dataset: {df_train.columns.tolist()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ====== If There's a Test Dataset ======
if df_test_original is not None:
    # Strip whitespace from test dataset column names
    df_test_original.columns = df_test_original.columns.str.strip()
    
    # Check that required columns exist in the test data
    missing_in_test = [col for col in all_needed_columns if col not in df_test_original.columns]
    if missing_in_test:
        log_and_print(f"Error: The following columns are missing in the test dataset: {missing_in_test}")
        sys.exit(1)
    
    # Subset to same columns as training
    df_test = df_test_original[all_needed_columns].copy()

    # Apply the same cleaning steps for the test subset
    df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
    for col in non_numeric_cols:
        if col in df_test.columns:
            df_test[col].fillna("Unknown", inplace=True)

    # (Optional) Apply one-hot encoding on feature columns if needed,
    # then reindex the test DataFrame to match the training DataFrame's columns
    test_feature_cols = [col for col in df_test.columns if col != output_column]
    df_test = pd.get_dummies(df_test, columns=[col for col in test_feature_cols if df_test[col].dtype == 'object'], drop_first=True)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
    
    # Now scale
    X_test = df_test[existing_train_columns]
    y_test = df_test[output_column]
    X_test_scaled = scaler.transform(X_test)
else:
    # If no test CSV is provided, do train_test_split
    test_size = args.test_split_ratio if args.test_split_ratio else 0.2
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=test_size, random_state=42
    )

log_and_print("Data Pre-processing Completed.")


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


# ====== Residual Plots ======
# Compute residuals
residuals = y_test - y_pred

# Residual Plot: Predicted vs. Residuals
residual_plot_path = os.path.join(output_dir, "residual_plot.png")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(residual_plot_path)
plt.close()
log_and_print(f"Residual Plot saved to {residual_plot_path}")

# Histogram of Residuals
hist_resid_path = os.path.join(output_dir, "histogram_residuals.png")
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Histogram of Residuals")
plt.savefig(hist_resid_path)
plt.close()
log_and_print(f"Histogram of Residuals saved to {hist_resid_path}")

# ----- Weight Plot (Coefficient Plot) -----
weight_plot_path = os.path.join(output_dir, "weight_plot.png")
plt.figure(figsize=(10, 6))


# ----- Model Coefficients Plot (Weight Plot) -----
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



# ----- Mean Effect Plot -----
selected_feature = existing_train_columns[0]
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



# ----- Individual Effect Plot -----
# Generate predictions on the full cleaned training set
X_train_full_scaled = scaler.transform(X_train)  # X_train from df_train (full set)
y_pred_full = model.predict(X_train_full_scaled)

plt.figure(figsize=(8,6))
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
try:
    import shap
except ImportError:
    log_and_print("SHAP library not installed. Installing now...")
    run_command("python -m pip install shap")
    import shap

# Use a linear explainer for your model. For linear models, the exact method is efficient.
explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_train_scaled)

shap_summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
plt.figure()
# After computing SHAP values on the scaled training data:
X_shap = pd.DataFrame(X_train_scaled, columns=existing_train_columns)
shap.summary_plot(shap_values, X_shap, feature_names=existing_train_columns, show=False)
plt.title("SHAP Summary Plot")
plt.savefig(shap_summary_plot_path, bbox_inches="tight")
plt.close()
log_and_print(f"SHAP Summary Plot saved to {shap_summary_plot_path}")






log_and_print("\n======= Model Performance =======")
log_and_print(f"Mean Squared Error: {mse:.4f}")
log_and_print(f"R-squared Score: {r2:.4f}")

if is_classification:
    log_and_print(f"Accuracy Score: {accuracy:.4f}")
else:
    log_and_print("Skipping accuracy score because target is continuous.")



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

log_and_print("========== FINISHED SUCCESSFULLY ==========")
