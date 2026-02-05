
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

# Create directories if they don't exist
for dir_path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# File paths for duplicate checking
INPUT_FILE = os.path.join(RAW_DIR, "essentia_features.csv")
EXACT_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "exact_duplicates.csv")
FUZZY_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "near_duplicates.csv")
FLAGGED_FULL_FILE = os.path.join(PROCESSED_DIR, "dataset_without_duplicates.csv")
LOG_FILE = os.path.join(PROCESSED_DIR, "duplicate_review_log.csv")

# Parameters
FUZZY_REVIEW_THRESHOLD = 40
REMIX_PENALTY = 20
VARIANCE_THRESHOLD = 1e-8
WINSORIZE_LIMITS = (0.0, 0.0)

# Analysis parameters
RANK_EXPONENT = 1.5
MIN_SONGS = 5
