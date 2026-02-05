import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import OUTPUT_DIR

INPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_AND_checked_data.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "review_log.csv")

# Thresholds for finding any problematic features
TOP_VALUE_THRESHOLD = 0.05  # single value appears in >5% of rows
EXTREME_VALUE_THRESHOLD = 0.10  # max/min appears in >10%
MIN_EFFECTIVE_SPREAD = 1e-3  # minimum std dev of middle 80%
MIN_NON_NAN = 50

# Load data
df = pd.read_csv(INPUT_FILE)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nLoaded {df.shape[0]} rows, {len(num_cols)} numeric features\n")

review_log = []
cols_to_drop = []
auto_drop = False


def categorize_feature(col_name):
    """Group features by type for better organization"""
    col_lower = col_name.lower()
    if any(x in col_lower for x in ['spectral', 'frequency', 'flux', 'rolloff', 'centroid']):
        return "spectral"
    elif any(x in col_lower for x in ['rhythm', 'bpm', 'beat', 'onset', 'tempo']):
        return "rhythm"
    elif any(x in col_lower for x in ['tonal', 'chroma', 'key', 'tuning', 'pitch']):
        return "tonal"
    elif any(x in col_lower for x in ['loudness', 'dynamic', 'level', 'rms']):
        return "dynamics"
    elif any(x in col_lower for x in ['mfcc', 'bark', 'erb', 'mel']):
        return "timbre"
    return "other"


# Check each numeric feature
for col in num_cols:
    series = df[col].dropna()
    n = len(series)

    if n < MIN_NON_NAN or series.nunique() < 2:
        continue

    # Checks for value clustering
    value_counts = series.value_counts(normalize=True)
    top_val = value_counts.index[0]
    top_freq = value_counts.iloc[0]

    max_val = series.max()
    min_val = series.min()
    max_freq = (series == max_val).mean()
    min_freq = (series == min_val).mean()

    # Calculate spread of middle 80% (excluding extremes)
    q05, q95 = series.quantile([0.05, 0.95])
    middle_data = series[(series > q05) & (series < q95)]

    if len(middle_data) > 10:
        effective_spread = middle_data.std()
        middle_iqr = middle_data.quantile(0.75) - middle_data.quantile(0.25)
    else:
        effective_spread = 0
        middle_iqr = 0

    # Flag if concentrated or piled up at extremes with low spread
    high_concentration = top_freq >= TOP_VALUE_THRESHOLD
    extreme_pileup = (max_freq >= EXTREME_VALUE_THRESHOLD or min_freq >= EXTREME_VALUE_THRESHOLD)
    low_spread = effective_spread < MIN_EFFECTIVE_SPREAD

    flagged = (high_concentration or extreme_pileup) and low_spread

    if not flagged:
        continue

    # diagnostics
    feature_type = categorize_feature(col)
    print(f"\n{'=' * 70}")
    print(f"Feature: {col} [{feature_type.upper()}]")
    print(f"Count: {n} | Unique: {series.nunique()}")
    print(f"Top value: {top_val} ({top_freq * 100:.2f}%)")
    print(f"Max: {max_val} ({max_freq * 100:.2f}%) | Min: {min_val} ({min_freq * 100:.2f}%)")
    print(f"Middle 80% - std: {effective_spread:.6f}, IQR: {middle_iqr:.6f}")

    if high_concentration:
        print(f"WARNING: High concentration at single value")
    if extreme_pileup:
        print(f"WARNING: Pile-up at boundaries")
    if low_spread:
        print(f"WARNING: Low spread in middle data")

    # Decision Time!
    if auto_drop:
        drop = True
        print("-> AUTO-DROPPED")
    else:
        decision = input("Drop? [y/N/all/quit]: ").strip().lower()
        if decision == 'all':
            auto_drop = True
            drop = True
            print("-> DROPPED (auto-drop enabled)")
        elif decision == 'quit':
            print("Exiting review")
            break
        else:
            drop = (decision == 'y')
            print(f"-> {'DROPPED' if drop else 'KEPT'}")

    # Logging
    review_log.append({
        "feature": col,
        "type": feature_type,
        "n": n,
        "unique": series.nunique(),
        "top_freq_pct": top_freq * 100,
        "max_freq_pct": max_freq * 100,
        "min_freq_pct": min_freq * 100,
        "middle_std": effective_spread,
        "dropped": drop
    })

    if drop:
        cols_to_drop.append(col)

# Apply removals
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"\n{'=' * 70}")
    print(f"DROPPED {len(cols_to_drop)} FEATURES")
    print(f"{'=' * 70}")

    # Summary by category
    by_category = {}
    for col in cols_to_drop:
        cat = categorize_feature(col)
        by_category.setdefault(cat, []).append(col)

    for cat in sorted(by_category.keys()):
        print(f"\n{cat.upper()} ({len(by_category[cat])}):")
        for col in by_category[cat]:
            print(f"  - {col}")
else:
    print("\nNo features dropped")

# Save
df.to_csv(OUTPUT_FILE, index=False)
pd.DataFrame(review_log).to_csv(LOG_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE} ({df.shape[0]} rows x {df.shape[1]} cols)")
print(f"Log: {LOG_FILE} ({len(review_log)} features reviewed)")
