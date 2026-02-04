import pandas as pd
import re
import unicodedata
from rapidfuzz import fuzz
from tqdm import tqdm
import sys
import os

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import INPUT_FILE, EXACT_OUTPUT_FILE, FUZZY_OUTPUT_FILE, FLAGGED_FULL_FILE, LOG_FILE, FUZZY_REVIEW_THRESHOLD, REMIX_PENALTY

TITLE_COL = "title"
ARTIST_COL = "artist"


def normalize_text(text):
    # Clean up text for matching
    if pd.isna(text):
        return "", False

    original = text
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Checking for featuring/remix variations
    had_variant = bool(
        re.search(r"\b(feat|featuring|with|ft\.?|&|remix|edit|version)\b",
                  original.lower())
    )

    # Simplifying
    text = re.sub(r"\b(feat|featuring|with|ft\.?|&|remix|edit|version)\b.*", "", text)
    text = re.sub(r"[,&+/]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text, had_variant


def detect_and_review_duplicates(df):
    df = df.copy()
    review_log = []

    # Normalise titles and artists
    title_data = df[TITLE_COL].apply(normalize_text)
    df["title_norm"] = title_data.apply(lambda x: x[0])
    df["had_title_variant"] = title_data.apply(lambda x: x[1])

    artist_data = df[ARTIST_COL].apply(normalize_text)
    df["artist_norm"] = artist_data.apply(lambda x: x[0])
    df["had_artist_variant"] = artist_data.apply(lambda x: x[1])

    df["song_key"] = df["title_norm"] + " - " + df["artist_norm"]

    # Find exact duplicates
    print("\nChecking for exact duplicates...")
    exact_counts = df["song_key"].value_counts()
    exact_dups = exact_counts[exact_counts > 1].index
    df["is_exact_duplicate"] = df["song_key"].isin(exact_dups)
    df["exact_group_size"] = df["song_key"].map(exact_counts)

    print(f"Found {len(exact_dups)} exact duplicate groups ({exact_counts[exact_dups].sum()} songs)")

    if len(exact_dups) > 0:
        exact_summary = (
            df[df["is_exact_duplicate"]]
            .groupby("song_key")
            .agg({
                TITLE_COL: 'first',
                ARTIST_COL: 'first',
                'exact_group_size': 'first'
            })
            .sort_values('exact_group_size', ascending=False)
        )
        print("\nExact duplicate groups:")
        print(exact_summary.to_string())
        exact_summary.to_csv(EXACT_OUTPUT_FILE, index=True)
        print(f"Saved to: {EXACT_OUTPUT_FILE}")

    # fuzzy matching for near-duplicates
    print("\nChecking for fuzzy matches (same title, similar artist)...")
    suspect_pairs = []

    title_groups = df.groupby("title_norm")

    for title_norm, group in tqdm(title_groups):
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                artist1_norm = group.iloc[i]["artist_norm"]
                artist2_norm = group.iloc[j]["artist_norm"]

                sim = fuzz.token_sort_ratio(artist1_norm, artist2_norm)

                # I've decided to lower the similarity rating for feat/remix as these tended to show very similar despite being acceptably different
                if group.iloc[i]["had_artist_variant"] or group.iloc[j]["had_artist_variant"]:
                    sim -= REMIX_PENALTY

                if sim >= FUZZY_REVIEW_THRESHOLD:
                    suspect_pairs.append({
                        "title_norm": title_norm,
                        "title_1": group.iloc[i][TITLE_COL],
                        "title_2": group.iloc[j][TITLE_COL],
                        "artist_1": group.iloc[i][ARTIST_COL],
                        "artist_2": group.iloc[j][ARTIST_COL],
                        "similarity": sim,
                        "row_1": group.index[i],
                        "row_2": group.index[j],
                        "had_variant_1": group.iloc[i]["had_artist_variant"],
                        "had_variant_2": group.iloc[j]["had_artist_variant"]
                    })

    suspects_df = pd.DataFrame(suspect_pairs)

    if not suspects_df.empty:
        suspects_df = suspects_df.sort_values("similarity", ascending=False)
        print(f"\nFound {len(suspects_df)} fuzzy near-duplicate pairs")
        print(suspects_df.to_string(index=False))
        suspects_df.to_csv(FUZZY_OUTPUT_FILE, index=False)
        print(f"Saved to: {FUZZY_OUTPUT_FILE}")
    else:
        print("No fuzzy near-duplicates found.")

    # I've decided to check near duplicates myself as it was a manageable number
    print("\nReviewing flagged pairs...")
    auto_drop_all = False
    cols_to_drop = []

    for idx, row in suspects_df.iterrows():
        print("\n" + "=" * 70)
        print(f"Pair {idx + 1} (similarity: {row['similarity']}%)")
        print(f"Row {row['row_1']}: {row['title_1']} by {row['artist_1']} (variant: {row['had_variant_1']})")
        print(f"Row {row['row_2']}: {row['title_2']} by {row['artist_2']} (variant: {row['had_variant_2']})")

        if auto_drop_all:
            drop = True
            print("→ AUTO-DROPPED")
        else:
            decision = input("Drop second row? [y/N/all/quit]: ").strip().lower()
            if decision == 'all':
                auto_drop_all = True
                drop = True
                print("→ AUTO-DROPPED (auto mode on)")
            elif decision == 'quit':
                print("→ Quitting")
                break
            else:
                drop = decision == 'y'

        review_log.append({
            "pair_id": idx + 1,
            "similarity": row['similarity'],
            "row_1": row['row_1'],
            "row_2": row['row_2'],
            "dropped": drop
        })

        if drop:
            cols_to_drop.append(int(row['row_2']))
            print("→ Row marked for removal")

    # removal time
    if cols_to_drop:
        df = df.drop(index=cols_to_drop)
        print(f"\nDropped {len(cols_to_drop)} rows")
    else:
        print("\nNo rows dropped")


    df.to_csv(FLAGGED_FULL_FILE, index=False)
    pd.DataFrame(review_log).to_csv(LOG_FILE, index=False)
    print(f"\nSaved flagged dataset to: {FLAGGED_FULL_FILE}")
    print(f"Review log saved to: {LOG_FILE}")

    return df, suspects_df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded dataset: {df.shape}")
    detect_and_review_duplicates(df)
