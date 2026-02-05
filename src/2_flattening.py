import pandas as pd
import ast
import numpy as np
import warnings
import sys
import os

warnings.simplefilter("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import PROCESSED_DIR, OUTPUT_DIR

# Columns to expand into individual features
EXPAND_CONFIG = {
    'lowlevel.mfcc.mean': 'mfcc_mean',
    'lowlevel.mfcc.var': 'mfcc_var',
    'lowlevel.mfcc.stdev': 'mfcc_stdev',
    'lowlevel.barkbands.mean': 'barkbands_mean',
    'lowlevel.barkbands.var': 'barkbands_var',
    'lowlevel.barkbands.stdev': 'barkbands_stdev',
    'lowlevel.melbands.mean': 'melbands_mean',
    'lowlevel.melbands.var': 'melbands_var',
    'lowlevel.melbands.stdev': 'melbands_stdev',
    'lowlevel.melbands128.mean': 'melbands128_mean',
    'lowlevel.melbands128.var': 'melbands128_var',
    'lowlevel.melbands128.stdev': 'melbands128_stdev',
    'tonal.hpcp.mean': 'hpcp_mean',
    'tonal.hpcp.var': 'hpcp_var',
    'tonal.hpcp.stdev': 'hpcp_stdev',
    'lowlevel.spectral_contrast_coeffs.mean': 'spectral_contrast_coeffs_mean',
    'lowlevel.spectral_contrast_coeffs.var': 'spectral_contrast_coeffs_var',
    'lowlevel.spectral_contrast_valleys.mean': 'spectral_contrast_valleys_mean',
    'lowlevel.spectral_contrast_valleys.var': 'spectral_contrast_valleys_var',
    'lowlevel.gfcc.mean': 'gfcc_mean',
    'lowlevel.gfcc.var': 'gfcc_var',
}

# Metadata columns that aren't useful for analysis
DROP_COLUMNS = [
    'metadata.audio_properties.analysis.equal_loudness',
    'metadata.audio_properties.analysis.length',
    'metadata.audio_properties.analysis.sample_rate',
    'metadata.audio_properties.analysis.start_time',
    'metadata.audio_properties.bit_rate',
    'metadata.audio_properties.length',
    'metadata.audio_properties.lossless',
    'metadata.audio_properties.number_channels',
    'metadata.audio_properties.replay_gain',
    'metadata.audio_properties.sample_rate',
    'metadata.audio_properties.analysis.downmix',
    'metadata.audio_properties.codec',
    'metadata.audio_properties.md5_encoded',
    'metadata.tags.file_name',
    'metadata.version.essentia',
    'metadata.version.essentia_git_sha',
    'metadata.version.extractor',
]


def parse_list_column(x):
    # Convert string representations of lists to actual lists
    if isinstance(x, str) and x.strip().startswith('['):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x if isinstance(x, list) else []


def expand_list_column(df, col_name, prefix):
    # Expand a column containing lists into separate columns for each element
    df[col_name] = df[col_name].apply(parse_list_column)

    valid = df[col_name][df[col_name].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    if valid.empty:
        return df

    # Make sure all lists are same length
    lengths = valid.apply(len).unique()
    if len(lengths) > 1:
        print(f"Warning: {col_name} has inconsistent lengths, skipping")
        return df

    # then expand into separate columns
    n = lengths[0]
    for i in range(n):
        df[f"{prefix}_{i + 1}"] = df[col_name].apply(
            lambda lst: lst[i] if isinstance(lst, list) and len(lst) > i else np.nan
        )

    df.drop(columns=[col_name], inplace=True)
    return df


def process_beats(df):
    # Calculate BPM and tempo variation from beat positions
    beats_col = 'rhythm.beats_position'
    df[beats_col] = df[beats_col].apply(parse_list_column)

    def calc_tempo_features(beats):
        if not isinstance(beats, list) or len(beats) < 2:
            return np.nan, np.nan, np.nan, 0

        intervals = np.diff(beats)
        avg_interval = np.mean(intervals)

        bpm = 60.0 / avg_interval if avg_interval > 0 else np.nan
        interval_var = np.var(intervals)
        tempo_cv = np.std(intervals) / avg_interval if avg_interval > 0 else np.nan

        return bpm, interval_var, tempo_cv, len(beats)

    df[['computed_bpm', 'beat_interval_var', 'tempo_cv', 'beats_count_recomputed']] = \
        df.apply(lambda row: calc_tempo_features(row[beats_col]), axis=1, result_type='expand')

    df.drop(columns=[beats_col], inplace=True)
    return df


if __name__ == "__main__":
    input_file = os.path.join(PROCESSED_DIR, "dataset_without_duplicates.csv")
    df = pd.read_csv(input_file)
    print(f"Shape: {df.shape}")

    # Expanding list columns into individual features
    for col, prefix in EXPAND_CONFIG.items():
        if col in df.columns:
            df = expand_list_column(df, col, prefix)

    # Process rhythm features
    if 'rhythm.beats_position' in df.columns:
        df = process_beats(df)

    # remove metadata
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

    # Leave out unprocessed
    list_cols = [col for col in df.columns
                 if df[col].dtype == 'object' and df[col].astype(str).str.startswith('[').any()]
    if list_cols:
        df.drop(columns=list_cols, inplace=True)

    # Converting numeric columns (but keep text fields as is)
    protected = ['songID', 'title', 'artist', 'shazam_url', 'apple_preview_audio']
    for col in df.select_dtypes(include=['object']).columns:
        if col not in protected:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Final shape: {df.shape}")

    # Save outputs
    output_csv = os.path.join(OUTPUT_DIR, "flattened_data.csv")
    output_xlsx = os.path.join(OUTPUT_DIR, "flattened_data.xlsx")
    df.to_csv(output_csv, index=False)
    df.to_excel(output_xlsx, index=False)
    print(f"Saved to: {output_csv}")
