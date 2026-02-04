import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import sys
import os

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import OUTPUT_DIR, VARIANCE_THRESHOLD, WINSORIZE_LIMITS

# Features where NaN means "none detected" - impute with 0
ZERO_IMPUTE = [
    'silence_rate_20dB.mean', 'silence_rate_30dB.mean', 'silence_rate_60dB.mean',
    'rhythm.onset_rate', 'tonal.chords_changes_rate', 'tonal.chords_number_rate',
]

# Domain-specific clipping ranges
CLIP_RULES = {
    'lowlevel.loudness_ebu128.integrated': (-60, 0),
    'lowlevel.average_loudness': (0, 1),
}

# Pattern-based clipping
CLIP_PATTERNS = [
    (lambda col: 'skewness' in col, (-10, 10)),
    (lambda col: 'kurtosis' in col, (-10, 50)),
    (lambda col: 'spectral_energy' in col or 'hfc' in col, (0, None)),
]


def clean_essentia_features(df, verbose=True):
    """Clean and preprocess flattened Essentia features"""
    df = df.copy()

    # Drop non-scalar feature types (covariance matrices, histograms, etc)
    drop_patterns = ['mfcc.cov', 'mfcc.icov', 'gfcc.cov', 'gfcc.icov',
                     'hpcp', 'thpcp', 'chords_histogram', 'bpm_histogram']

    cols_to_drop = [col for col in df.columns
                    if any(pattern in col for pattern in drop_patterns)]

    if cols_to_drop and verbose:
        print(f"Dropping {len(cols_to_drop)} non-scalar columns")
    df.drop(columns=cols_to_drop, inplace=True)

    # Drop columns with >50% missing data
    high_nan = df.columns[df.isna().mean() > 0.5].tolist()
    if high_nan:
        if verbose:
            print(f"Dropping {len(high_nan)} high-NaN columns: {', '.join(high_nan[:5])}...")
        df.drop(columns=high_nan, inplace=True)

        
    metadata = ['songID', 'title', 'artist', 'shazam_url', 'apple_preview_audio',
                'had_title_variant', 'had_artist_variant', 'is_exact_duplicate',
                'exact_group_size', 'filename', 'filepath', 'duration']

    feature_cols = [c for c in df.columns if (
            c.startswith(('lowlevel.', 'rhythm.', 'tonal.', 'mfcc_', 'gfcc_',
                          'hpcp_', 'barkbands_', 'melbands_', 'spectral_')) or
            c in ['computed_bpm', 'beat_interval_var', 'tempo_cv', 'beats_count_recomputed']
    ) and c not in metadata]

    numeric_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.float32, np.int64]]

    if verbose:
        print(f"\nProcessing {len(numeric_cols)} numeric features")
        
    # Handle missing values
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            if col in ZERO_IMPUTE:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Remove near-constant columns
    stds = df[numeric_cols].std()
    low_var = stds[stds < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        if verbose:
            print(f"Removing {len(low_var)} near-constant columns")
        df.drop(columns=low_var, inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in low_var]

    # Clipping outliers and winsorizing
    for col in numeric_cols:
        # Check for clipping rules
        lower, upper = None, None

        if col in CLIP_RULES:
            lower, upper = CLIP_RULES[col]
        else:
            for pattern_check, bounds in CLIP_PATTERNS:
                if pattern_check(col):
                    lower, upper = bounds
                    break

        if lower is not None or upper is not None:
            df[col] = df[col].clip(lower=lower, upper=upper)

        # Winsorize to handle remaining outliers
        if df[col].nunique() > 2:
            df[col] = winsorize(df[col].values, limits=WINSORIZE_LIMITS, nan_policy='omit')

    if verbose:
        remaining_nans = df[numeric_cols].isna().sum().sum()
        print(f"\nCleaning complete: {df.shape[0]} rows, {len(numeric_cols)} features")
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaNs still present")

    return df


if __name__ == "__main__":
    # Load data
    input_file = os.path.join(OUTPUT_DIR, "flattened_essentia_features_full.xlsx")
    df = pd.read_excel(input_file)
    print(f"Original shape: {df.shape}")

    cleaned = clean_essentia_features(df, verbose=True)

    # Quick stats
    print(f"\nFinal shape: {cleaned.shape}")
    numeric_features = cleaned.select_dtypes(include=[np.number]).columns
    print(f"NaN count: {cleaned[numeric_features].isna().sum().sum()}")
    print(f"Inf count: {np.isinf(cleaned[numeric_features]).sum().sum()}")

    output_file = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
    cleaned.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
