import subprocess
import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

# ====== Function to Create Log File in Output Directory Early ======
def get_output_dir(csv_path):
    """Creates and returns an output directory based on the CSV filename."""
    base_path = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path).split('.')[0]
    output_dir = os.path.join(base_path, f"linearregression-{csv_filename}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ====== Set Paths for Datasets (Modify these as needed) ======
train_csv_path = "D:\\web projects\\NeuroFlow\\DataSets\\Diabetes Dataset\\archive\\diabetes.csv"
test_csv_path = "D:\\web projects\\NeuroFlow\\DataSets\\Diabetes Dataset\\archive\\diabetes-test.csv"  # Set this to None if only train file is available

# ====== Initialize Logging Early to Capture Everything ======
output_dir = get_output_dir(train_csv_path)
log_file = os.path.join(output_dir, "setup_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logging.info("========== SCRIPT STARTED ==========")

# ====== Function to Run Commands and Capture Logs ======
def run_command(command):
    """Executes a shell command and logs output."""
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end="")
            logging.info(line.strip())
        for line in process.stderr:
            print(line, end="")
            logging.error(line.strip())
        process.wait()
        if process.returncode != 0:
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}\nError: {e}")
        sys.exit(1)

# ====== Check Python Installation ======
try:
    run_command("python --version")
except FileNotFoundError:
    logging.error("Python is not installed. Please install Python first.")
    sys.exit(1)

# ====== Upgrade pip ======
run_command("python -m pip install --upgrade pip")

# ====== Install Required Packages with Logging ======
def install_package(package):
    """Installs a package and logs the process."""
    try:
        logging.info(f"Installing {package}...")
        run_command(f"python -m pip install {package}")
        logging.info(f"Successfully installed {package}")
    except Exception as e:
        logging.error(f"Failed to install {package}: {e}")
        sys.exit(1)

packages = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "six"]
for package in packages:
    try:
        __import__(package)
    except ImportError:
        install_package(package)

# ====== Check If Dataset Exists ======
if not os.path.exists(train_csv_path):
    logging.error(f"Training dataset not found at {train_csv_path}. Exiting.")
    sys.exit(1)

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path) if test_csv_path and os.path.exists(test_csv_path) else None

logging.info("Loaded dataset successfully.")

# ====== Save Feature Correlation Heatmap ======
heatmap_path = os.path.join(output_dir, "heatmap.png")
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(heatmap_path)
plt.close()
logging.info(f"Feature correlation heatmap saved to {heatmap_path}")

# ====== Save Histogram of Outcome Distribution ======
histogram_path = os.path.join(output_dir, "Histogram_distribution.png")
plt.figure(figsize=(8, 5))
sns.histplot(df_train['Outcome'], bins=3, kde=True)
plt.title("Distribution of Diabetes Outcome")
plt.savefig(histogram_path)
plt.close()
logging.info(f"Outcome distribution histogram saved to {histogram_path}")

# ====== Prepare Features and Target ======
X_train = df_train.drop(columns=["Outcome"])
y_train = df_train["Outcome"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

if df_test is not None:
    X_test = df_test.drop(columns=["Outcome"])
    y_test = df_test["Outcome"]
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

logging.info("Data pre-processing completed.")

# ====== Train Model ======
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ====== Make Predictions ======
y_pred = model.predict(X_test_scaled)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# ====== Calculate Performance Metrics ======
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)

logging.info("\nModel Performance:")
logging.info(f"Mean Squared Error: {mse}")
logging.info(f"R-squared Score: {r2}")
logging.info(f"Accuracy Score: {accuracy}")

# ====== Save Model ======
model_path = os.path.join(output_dir, "model.pkl")
joblib.dump(model, model_path)
logging.info(f"Model saved as '{model_path}'")

# ====== Save Paths to a File ======
path_file = os.path.join(output_dir, "saved_paths.txt")
with open(path_file, "w") as f:
    f.write(f"Model Path: {model_path}\n")
    f.write(f"CSV Path: {train_csv_path}\n")

logging.info(f"Saved paths in '{path_file}'")
logging.info("========== SCRIPT FINISHED SUCCESSFULLY ==========")
