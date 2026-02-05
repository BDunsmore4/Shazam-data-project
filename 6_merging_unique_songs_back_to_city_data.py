"""
Merge UK City song data with unique song audio features file.
Handles duplicate removal and data cleaning
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import OUTPUT_DIR, RAW_DIR, PROCESSED_DIR

UK_CITIES_FILE = os.path.join(RAW_DIR, "city_song_data.xlsx")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "post_UMAP.csv")
REMOVAL_ACTIONS_FILE = os.path.join(PROCESSED_DIR, "removal_actions.csv")

OUTPUT_CLEANED = os.path.join(OUTPUT_DIR, "city_song_data_cleaned.xlsx")
OUTPUT_MERGED = os.path.join(OUTPUT_DIR, "full_city_data.xlsx")

MERGE_KEY = 'apple_preview_audio' 
KEEP_ALL_ROWS = True 


def load_file(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)


def save_file(df, filepath):
    """Save to CSV or Excel based on extension"""
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    else:
        df.to_excel(filepath, index=False)


cities_df = load_file(UK_CITIES_FILE)
features_df = load_file(FEATURES_FILE)
actions_df = pd.read_csv(REMOVAL_ACTIONS_FILE)

print(f"Cities: {cities_df.shape}")
print(f"Features: {features_df.shape}")
print(f"Actions: {len(actions_df)}")

# Replace removed songs in cities dataset
print("\nReplacing removed songs...")

removal_map = dict(zip(actions_df['removed_songID'], actions_df['kept_songID']))

removed_in_cities = set(cities_df['songID']) & set(removal_map.keys())
print(f"Found {len(removed_in_cities)} songs to replace")

if removed_in_cities:
    cities_cleaned = cities_df.copy()

    for removed_id in removed_in_cities:
        kept_id = removal_map[removed_id]
        mask = cities_cleaned['songID'] == removed_id
        cities_cleaned.loc[mask, 'songID'] = kept_id

        # Update other fields if they exist
        action_row = actions_df[actions_df['removed_songID'] == removed_id].iloc[0]
        if 'title' in cities_cleaned.columns:
            cities_cleaned.loc[mask, 'title'] = action_row.get('kept_title')
        if 'artist' in cities_cleaned.columns:
            cities_cleaned.loc[mask, 'artist'] = action_row.get('kept_artist')
        if 'apple_preview_audio' in cities_cleaned.columns:
            cities_cleaned.loc[mask, 'apple_preview_audio'] = action_row.get('kept_apple_preview_audio')

    print(f"Replaced {mask.sum()} rows")
else:
    cities_cleaned = cities_df.copy()

# Save cleaned cities
save_file(cities_cleaned, OUTPUT_CLEANED)
print(f"Saved: {OUTPUT_CLEANED}")

# Remove duplicates from features
print(f"\nChecking for duplicates in features (using {MERGE_KEY})...")

# Drop if merge keys not there
features_clean = features_df[features_df[MERGE_KEY].notna()].copy()
print(f"Dropped {len(features_df) - len(features_clean)} null keys")

# Find and remove duplicates
n_before = len(features_clean)
features_clean = features_clean.drop_duplicates(subset=[MERGE_KEY], keep='first')
n_dupes = n_before - len(features_clean)

if n_dupes > 0:
    print(f"Removed {n_dupes} duplicate rows")
else:
    print("No duplicates found")

Check overlap

cities_keys = set(cities_cleaned[MERGE_KEY].dropna())
features_keys = set(features_clean[MERGE_KEY].dropna())

overlap = cities_keys & features_keys
cities_only = cities_keys - features_keys

print(f"Matched: {len(overlap)}/{len(cities_keys)} ({len(overlap) / len(cities_keys) * 100:.1f}%)")
if cities_only:
    print(f"Warning: {len(cities_only)} songs in cities won't have features")

# Merging time! 

merge_type = 'left' if KEEP_ALL_ROWS else 'inner'
print(f"Using {merge_type} join on '{MERGE_KEY}'")

merged_df = pd.merge(
    cities_cleaned,
    features_clean,
    on=MERGE_KEY,
    how=merge_type,
    suffixes=('', '_features')
)

print(f"Result: {merged_df.shape}")

# Check for row multiplication
if len(merged_df) > len(cities_cleaned):
    print(f"Warning: Merge created extra rows! Deduplicating...")
    merged_df = merged_df.drop_duplicates(
        subset=['date', 'city_name', MERGE_KEY, 'rank'],
        keep='first'
    )
    print(f"After dedup: {len(merged_df)} rows")

# Removing duplicate columns
duplicate_cols = [col for col in merged_df.columns if col.endswith('_features')]
if duplicate_cols:
    print(f"Removing {len(duplicate_cols)} duplicate columns")
    for col in duplicate_cols:
        base = col.replace('_features', '')
        # Keep cities version for metadata, features version for audio data
        if base in ['title', 'artist', 'songID']:
            merged_df.drop(columns=[col], inplace=True)
        else:
            merged_df.drop(columns=[base], inplace=True)
            merged_df.rename(columns={col: base}, inplace=True)

# Count feature columns
feature_cols = [c for c in merged_df.columns if any(
    c.startswith(p) for p in ['lowlevel.', 'rhythm.', 'tonal.', 'mfcc', 'melbands']
)]
print(f"Added {len(feature_cols)} feature columns")

# Check how many rows have features
has_features = merged_df[feature_cols].notna().any(axis=1).sum()
print(f"Rows with features: {has_features}/{len(merged_df)} ({has_features / len(merged_df) * 100:.1f}%)")


print("\nSaving merged dataset...")
save_file(merged_df, OUTPUT_MERGED)
print(f"Saved: {OUTPUT_MERGED}")

print(f"\nSummary:")
print(f"Cleaned {len(removed_in_cities)} duplicate songs")
print(f"Removed {n_dupes} duplicate features")
print(f"Match rate: {len(overlap) / len(cities_keys) * 100:.1f}%")
print(f"Final: {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")
print("\nDone!")
