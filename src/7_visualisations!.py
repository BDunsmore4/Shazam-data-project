"""
The fun part! 
Cultural Geography Analysis: How do shazam music discoveries vary across the UK?

Analyses:
- City centroids from Shazam top 50 charts (tf-idf weighting)
- Geographic vs cultural distance correlation
- National boundary effects
- Does a North-South divide in England exist here?
- Identification of culturally similar/different city pairs

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.stats import pearsonr, mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import sys
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import OUTPUT_DIR, VIZ_DIR, RAW_DIR, RANK_EXPONENT, MIN_SONGS

DATA_FILE = os.path.join(OUTPUT_DIR, 'full_city_data.xlsx.xlsx')

# City coordinates (manually compiled from Google Maps/ONS)
CITY_COORDS = {
    'Aberdeen': (57.1497, -2.0943), 'Dundee': (56.4620, -2.9707),
    'Edinburgh': (55.9533, -3.1883), 'Glasgow': (55.8642, -4.2518),
    'Stirling': (56.1165, -3.9369), 'Inverness': (57.4778, -4.2247),
    'Belfast': (54.5973, -5.9301), 'Londonderry County Borough': (54.9966, -7.3086),
    'Derry': (54.9966, -7.3086), 'Cardiff': (51.4816, -3.1791),
    'Swansea': (51.6214, -3.9436), 'London': (51.5074, -0.1278),
    'Birmingham': (52.4862, -1.8904), 'Manchester': (53.4808, -2.2426),
    'Liverpool': (53.4084, -2.9916), 'Newcastle upon Tyne': (54.9783, -1.6178),
    'Newcastle': (54.9783, -1.6178), 'Sheffield': (53.3811, -1.4701),
    'Leeds': (53.8008, -1.5491), 'Bristol': (51.4545, -2.5879),
    'Leicester': (52.6369, -1.1398), 'Nottingham': (52.9548, -1.1581),
    'Southampton': (50.9097, -1.4044), 'Brighton': (50.8225, -0.1372),
    'Brighton and Hove': (50.8225, -0.1372), 'Portsmouth': (50.8198, -1.0880),
    'Reading': (51.4543, -0.9781), 'Coventry': (52.4068, -1.5197),
    'Bradford': (53.7960, -1.7594), 'Derby': (52.9226, -1.4746),
    'Plymouth': (50.3755, -4.1427), 'Stoke-on-Trent': (53.0027, -2.1794),
    'Wolverhampton': (52.5865, -2.1281), 'Norwich': (52.6309, 1.2974),
    'Cambridge': (52.2053, 0.1218), 'Oxford': (51.7520, -1.2577),
    'York': (53.9600, -1.0873), 'Ipswich': (52.0594, 1.1556),
    'Swindon': (51.5558, -1.7797), 'Blackpool': (53.8175, -3.0357),
    'Middlesbrough': (54.5742, -1.2349), 'Huddersfield': (53.6458, -1.7850),
    'Sunderland': (54.9069, -1.3838), 'Southend-on-Sea': (51.5460, 0.7077),
    'Bournemouth': (50.7192, -1.8808), 'Cheltenham': (51.8994, -2.0783),
    'Bath': (51.3758, -2.3599), 'Exeter': (50.7184, -3.5339),
    'Gloucester': (51.8642, -2.2382), 'Lancaster': (54.0466, -2.8007),
    'Lincoln': (53.2307, -0.5406), 'Preston': (53.7632, -2.7031),
    'Salford': (53.4876, -2.2906), 'Carlisle': (54.8951, -2.9382),
    'Durham': (54.7753, -1.5849), 'Chester': (53.1905, -2.8908),
    'Warwick': (52.2819, -1.5849), 'Canterbury': (51.2798, 1.0789),
    'Winchester': (51.0632, -1.3080), 'Peterborough': (52.5695, -0.2405),
    'Luton': (51.8787, -0.4200), 'Milton Keynes': (52.0406, -0.7594),
    'Northampton': (52.2405, -0.9027), 'Doncaster': (53.5228, -1.1285),
    'Kingston upon Hull': (53.7457, -0.3367), 'Hull': (53.7457, -0.3367),
    'Saint Peters': (50.5558, -2.4553), 'Taunton': (51.0150, -3.1061),
    'Torquay': (50.4619, -3.5253), 'Wigan': (53.5450, -2.6325),
    'Worcester': (52.1936, -2.2208),
}

COUNTRY_MAP = {
    'Aberdeen': 'Scotland', 'Dundee': 'Scotland', 'Edinburgh': 'Scotland',
    'Glasgow': 'Scotland', 'Stirling': 'Scotland', 'Inverness': 'Scotland',
    'Belfast': 'Northern Ireland', 'Londonderry County Borough': 'Northern Ireland',
    'Derry': 'Northern Ireland', 'Cardiff': 'Wales', 'Swansea': 'Wales',
}

# North-South divide roughly at 53°N (Sheffield/Lincoln latitude)
REGIONS = {
    'Newcastle upon Tyne': 'North', 'Newcastle': 'North', 'Sunderland': 'North',
    'Durham': 'North', 'Carlisle': 'North', 'Liverpool': 'North',
    'Manchester': 'North', 'Leeds': 'North', 'Sheffield': 'North',
    'Bradford': 'North', 'Kingston upon Hull': 'North', 'Hull': 'North',
    'York': 'North', 'Lancaster': 'North', 'Preston': 'North',
    'Blackpool': 'North', 'Salford': 'North', 'Wigan': 'North',
    'Huddersfield': 'North', 'Middlesbrough': 'North', 'Doncaster': 'North',
    'Birmingham': 'South', 'Nottingham': 'South', 'Leicester': 'South',
    'Coventry': 'South', 'Derby': 'South', 'Stoke-on-Trent': 'South',
    'Wolverhampton': 'South', 'Lincoln': 'South', 'Northampton': 'South',
    'Peterborough': 'South', 'Worcester': 'South', 'London': 'South',
    'Bristol': 'South', 'Southampton': 'South', 'Portsmouth': 'South',
    'Brighton': 'South', 'Brighton and Hove': 'South', 'Reading': 'South',
    'Oxford': 'South', 'Cambridge': 'South', 'Norwich': 'South',
    'Ipswich': 'South', 'Plymouth': 'South', 'Exeter': 'South',
    'Bournemouth': 'South', 'Bath': 'South', 'Cheltenham': 'South',
    'Gloucester': 'South', 'Swindon': 'South', 'Canterbury': 'South',
    'Winchester': 'South', 'Southend-on-Sea': 'South', 'Luton': 'South',
    'Milton Keynes': 'South', 'Taunton': 'South', 'Torquay': 'South',
    'Saint Peters': 'South', 'Chester': 'South', 'Warwick': 'South',
}


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points on Earth (km)."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def load_data(filepath):
    """Load and validate Shazam data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading: {filepath}")
    df = pd.read_excel(filepath)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    required = ['city_name', 'rank', 'UMAP_1', 'UMAP_2', 'UMAP_3', 'songID']
    if missing := [c for c in required if c not in df.columns]:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=['UMAP_1', 'UMAP_2', 'UMAP_3'])
    df['country'] = df['city_name'].map(COUNTRY_MAP).fillna('England')
    df['region'] = df['city_name'].map(REGIONS).fillna('Other')

    print(f"  {df['city_name'].nunique()} cities, {df['songID'].nunique()} songs")
    return df


def compute_centroids(df, weighting='tfidf'):
    """
    Calculate cultural centroid for each city in UMAP space.

    TF-IDF weighting: higher-ranked songs (TF) that appear in fewer cities (IDF)
    receive more weight, identifying distinctive local preferences.
    """
    song_counts = df.groupby('songID')['city_name'].nunique()
    n_cities = df['city_name'].nunique()

    centroids = []
    for city in df['city_name'].unique():
        city_data = df[df['city_name'] == city]

        if len(city_data) < MIN_SONGS or city not in CITY_COORDS:
            continue

        coords = city_data[['UMAP_1', 'UMAP_2', 'UMAP_3']].values

        if weighting == 'tfidf':
            tf = (city_data['rank'] / 50.0) ** RANK_EXPONENT
            idf = np.log(n_cities / city_data['songID'].map(song_counts))
            weights = (tf * idf) / (tf * idf).sum()
        else:
            weights = np.ones(len(city_data)) / len(city_data)

        centroid = (coords.T @ weights).T
        lat, lon = CITY_COORDS[city]

        centroids.append({
            'city_name': city,
            'UMAP_1': centroid[0], 'UMAP_2': centroid[1], 'UMAP_3': centroid[2],
            'n_songs': len(city_data),
            'country': COUNTRY_MAP.get(city, 'England'),
            'region': REGIONS.get(city, 'Other'),
            'latitude': lat, 'longitude': lon
        })

    print(f"\nCentroids computed: {len(centroids)} cities")
    return pd.DataFrame(centroids)


def compute_diversity(df, centroids_df):
    """Calculate within-city diversity as mean distance from centroid."""
    results = []

    for _, row in centroids_df.iterrows():
        centroid = np.array([row['UMAP_1'], row['UMAP_2'], row['UMAP_3']])
        songs = df[df['city_name'] == row['city_name']]
        coords = songs[['UMAP_1', 'UMAP_2', 'UMAP_3']].values
        distances = [euclidean(centroid, c) for c in coords]

        results.append({
            'city_name': row['city_name'],
            'mean_diversity': np.mean(distances),
            'std_diversity': np.std(distances)
        })

    diversity_df = pd.DataFrame(results)
    return centroids_df.merge(diversity_df, on='city_name')


def compute_distance_matrices(centroids_df):
    """Calculate cultural and geographic distance matrices."""
    cities = centroids_df['city_name'].tolist()

    # Cultural distance in UMAP space
    coords = centroids_df[['UMAP_1', 'UMAP_2', 'UMAP_3']].values
    cultural_dist = squareform(pdist(coords, 'euclidean'))
    cultural_dist = pd.DataFrame(cultural_dist, index=cities, columns=cities)

    # Geographic distance
    geo_coords = centroids_df[['latitude', 'longitude']].values
    geo_dist = np.zeros((len(cities), len(cities)))

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            dist = haversine(*geo_coords[i], *geo_coords[j])
            geo_dist[i, j] = geo_dist[j, i] = dist

    geo_dist = pd.DataFrame(geo_dist, index=cities, columns=cities)

    return cultural_dist, geo_dist


def mantel_test(X, Y, n_permutations=10000):
    """Mantel test for matrix correlation."""
    def correlation(A, B):
        mask = np.triu_indices_from(A, k=1)
        return pearsonr(A[mask], B[mask])[0]

    obs_corr = correlation(X, Y)
    null = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = np.random.permutation(len(X))
        Y_perm = Y[perm][:, perm]
        null[i] = correlation(X, Y_perm)

    p_value = np.mean(null >= obs_corr)
    return obs_corr, p_value


def test_england_boundary(centroids_df, cultural_dist):
    """Test if cultural distance differs within vs across England border."""
    cities = centroids_df['city_name'].tolist()
    within = []
    across = []

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city1, city2 = cities[i], cities[j]
            country1 = centroids_df[centroids_df['city_name'] == city1]['country'].values[0]
            country2 = centroids_df[centroids_df['city_name'] == city2]['country'].values[0]

            if country1 == 'England' and country2 == 'England':
                within.append(cultural_dist.loc[city1, city2])
            elif country1 == 'England' or country2 == 'England':
                across.append(cultural_dist.loc[city1, city2])

    U, p = mannwhitneyu(within, across, alternative='two-sided')
    return {
        'within_mean': np.mean(within),
        'across_mean': np.mean(across),
        'p_value': p,
        'effect_size': (np.mean(across) - np.mean(within)) / np.std(within + across)
    }


def test_north_south_divide(centroids_df, cultural_dist):
    """Test if North and South England cities differ culturally."""
    england = centroids_df[centroids_df['country'] == 'England']
    if len(england) < 10:
        return None

    cities = england['city_name'].tolist()
    within_north, within_south, across = [], [], []

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city1, city2 = cities[i], cities[j]
            region1 = england[england['city_name'] == city1]['region'].values[0]
            region2 = england[england['city_name'] == city2]['region'].values[0]

            if region1 == 'North' and region2 == 'North':
                within_north.append(cultural_dist.loc[city1, city2])
            elif region1 == 'South' and region2 == 'South':
                within_south.append(cultural_dist.loc[city1, city2])
            elif region1 in ['North', 'South'] and region2 in ['North', 'South']:
                across.append(cultural_dist.loc[city1, city2])

    U, p = mannwhitneyu(within_north + within_south, across, alternative='two-sided')
    return {
        'within_north_mean': np.mean(within_north),
        'within_south_mean': np.mean(within_south),
        'across_mean': np.mean(across),
        'p_value': p
    }


def find_interesting_pairs(centroids_df, cultural_dist, geo_dist):
    """Find city pairs that are geographically close but culturally different, and vice versa."""
    cities = centroids_df['city_name'].tolist()
    pairs = []

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city1, city2 = cities[i], cities[j]
            pairs.append({
                'city1': city1,
                'city2': city2,
                'cultural_dist': cultural_dist.loc[city1, city2],
                'geo_dist_km': geo_dist.loc[city1, city2]
            })

    df = pd.DataFrame(pairs)

    # Close but different (small geo, large cultural)
    close_threshold = df['geo_dist_km'].quantile(0.25)
    cultural_threshold = df['cultural_dist'].quantile(0.75)
    close_diff = df[(df['geo_dist_km'] < close_threshold) &
                    (df['cultural_dist'] > cultural_threshold)].sort_values('cultural_dist', ascending=False)

    # Far but similar (large geo, small cultural)
    far_threshold = df['geo_dist_km'].quantile(0.75)
    similarity_threshold = df['cultural_dist'].quantile(0.25)
    far_sim = df[(df['geo_dist_km'] > far_threshold) &
                 (df['cultural_dist'] < similarity_threshold)].sort_values('geo_dist_km', ascending=False)

    return df, close_diff, far_sim


def plot_dendrogram(centroids_df, cultural_dist, output_path):
    """Hierarchical clustering dendrogram."""
    linkage_matrix = linkage(squareform(cultural_dist.values), method='ward')

    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=centroids_df['city_name'].tolist(),
              leaf_font_size=10, leaf_rotation=90)
    plt.title('Hierarchical Clustering of UK Cities (Cultural Distance)')
    plt.xlabel('City')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmaps(cultural_dist, geo_dist, output_path):
    """Side-by-side heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    sns.heatmap(cultural_dist, cmap='viridis', ax=ax1, square=True, cbar_kws={'label': 'Distance'})
    ax1.set_title('Cultural Distance')

    sns.heatmap(geo_dist, cmap='plasma', ax=ax2, square=True, cbar_kws={'label': 'km'})
    ax2.set_title('Geographic Distance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation(centroids_df, cultural_dist, geo_dist, output_path):
    """Scatter plot of geographic vs cultural distance."""
    cities = centroids_df['city_name'].tolist()
    pairs = []

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city1, city2 = cities[i], cities[j]
            country1 = centroids_df[centroids_df['city_name'] == city1]['country'].values[0]
            country2 = centroids_df[centroids_df['city_name'] == city2]['country'].values[0]

            same_country = (country1 == country2)

            pairs.append({
                'geo': geo_dist.loc[city1, city2],
                'cultural': cultural_dist.loc[city1, city2],
                'same_country': same_country
            })

    df = pd.DataFrame(pairs)

    fig, ax = plt.subplots(figsize=(10, 8))

    for same, label, color in [(True, 'Same country', '#3498DB'),
                                (False, 'Different countries', '#E74C3C')]:
        subset = df[df['same_country'] == same]
        ax.scatter(subset['geo'], subset['cultural'], alpha=0.6,
                  label=label, s=30, color=color)

    # Regression
    z = np.polyfit(df['geo'], df['cultural'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['geo'].min(), df['geo'].max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)

    r, p_val = pearsonr(df['geo'], df['cultural'])
    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.4f}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Geographic Distance (km)', fontsize=12)
    ax.set_ylabel('Cultural Distance', fontsize=12)
    ax.set_title('Geographic vs Cultural Distance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return r, p_val


def plot_map_comparison(centroids_df, cultural_dist, output_path):
    """Geographic map colored by cultural similarity to London."""
    if 'London' not in centroids_df['city_name'].values:
        print("Warning: London not in dataset, skipping map comparison")
        return

    london_dist = cultural_dist['London'].to_dict()
    centroids_df['dist_to_london'] = centroids_df['city_name'].map(london_dist)

    fig, ax = plt.subplots(figsize=(12, 14))

    scatter = ax.scatter(centroids_df['longitude'], centroids_df['latitude'],
                        c=centroids_df['dist_to_london'], cmap='RdYlGn_r',
                        s=200, edgecolors='black', linewidth=1.5, alpha=0.8)

    for _, row in centroids_df.iterrows():
        ax.annotate(row['city_name'], (row['longitude'], row['latitude']),
                   fontsize=8, ha='center', va='bottom')

    plt.colorbar(scatter, ax=ax, label='Cultural Distance to London')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Cultural Similarity to London')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_interactive_3d(centroids_df, output_path):
    """Interactive 3D scatter in UMAP space."""
    colors = {'England': '#3498DB', 'Scotland': '#E74C3C',
              'Wales': '#2ECC71', 'Northern Ireland': '#F39C12'}

    fig = go.Figure()

    for country in centroids_df['country'].unique():
        data = centroids_df[centroids_df['country'] == country]
        fig.add_trace(go.Scatter3d(
            x=data['UMAP_1'], y=data['UMAP_2'], z=data['UMAP_3'],
            mode='markers+text', name=country,
            marker=dict(size=8, color=colors.get(country, 'gray')),
            text=data['city_name'], textposition='top center',
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

    fig.update_layout(
        title='UK Cities in Cultural Space (UMAP)',
        scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'),
        width=1000, height=800
    )

    fig.write_html(output_path)


def plot_interactive_comparison(centroids_df, cultural_dist, output_path):
    """Interactive map colored by cultural distance to selected city."""
    reference_city = 'London' if 'London' in centroids_df['city_name'].values else centroids_df.iloc[0]['city_name']
    city_dist = cultural_dist[reference_city].to_dict()
    centroids_df['distance'] = centroids_df['city_name'].map(city_dist)

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=centroids_df['longitude'],
        lat=centroids_df['latitude'],
        text=centroids_df['city_name'],
        mode='markers',
        marker=dict(
            size=12,
            color=centroids_df['distance'],
            colorscale='RdYlGn_r',
            colorbar=dict(title="Cultural Distance"),
            line=dict(width=1, color='black')
        ),
        hovertemplate='<b>%{text}</b><br>Distance: %{marker.color:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Cultural Distance to {reference_city}',
        geo=dict(scope='europe', projection_scale=6,
                center=dict(lat=54, lon=-2)),
        width=900, height=1000
    )

    fig.write_html(output_path)


def plot_interactive_correlation(centroids_df, cultural_dist, geo_dist, output_path):
    """Interactive scatter of geographic vs cultural distance."""
    cities = centroids_df['city_name'].tolist()
    pairs = []

    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city1, city2 = cities[i], cities[j]
            country1 = centroids_df[centroids_df['city_name'] == city1]['country'].values[0]
            country2 = centroids_df[centroids_df['city_name'] == city2]['country'].values[0]

            pairs.append({
                'city1': city1,
                'city2': city2,
                'geo': geo_dist.loc[city1, city2],
                'cultural': cultural_dist.loc[city1, city2],
                'type': 'Within country' if country1 == country2 else 'Cross-border'
            })

    df = pd.DataFrame(pairs)

    fig = go.Figure()

    for border_type in ['Within country', 'Cross-border']:
        subset = df[df['type'] == border_type]
        fig.add_trace(go.Scatter(
            x=subset['geo'], y=subset['cultural'],
            mode='markers', name=border_type,
            marker=dict(size=5, color='#3498DB' if border_type == 'Within country' else '#E74C3C'),
            hovertemplate='%{customdata[0]} - %{customdata[1]}<br>' +
                         'Distance: %{x:.0f} km<extra></extra>',
            customdata=subset[['city1', 'city2']].values
        ))

    # Regression
    z = np.polyfit(df['geo'], df['cultural'], 1)
    x_line = np.linspace(df['geo'].min(), df['geo'].max(), 100)

    r, p = pearsonr(df['geo'], df['cultural'])

    fig.add_trace(go.Scatter(
        x=x_line, y=np.poly1d(z)(x_line),
        mode='lines', name=f'r = {r:.3f}',
        line=dict(color='black', dash='dash')
    ))

    fig.update_layout(
        title=f'Geographic vs Cultural Distance (p = {p:.4f})',
        xaxis_title='Geographic Distance (km)',
        yaxis_title='Cultural Distance',
        width=1000, height=700
    )

    fig.write_html(output_path)
    return r, p


def plot_interactive_diversity(centroids_df, output_path):
    """Interactive map showing diversity."""
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=centroids_df['longitude'],
        lat=centroids_df['latitude'],
        text=centroids_df['city_name'],
        mode='markers',
        marker=dict(
            size=centroids_df['mean_diversity'] * 30,
            color=centroids_df['mean_diversity'],
            colorscale='Viridis',
            colorbar=dict(title="Diversity")
        ),
        hovertemplate='<b>%{text}</b><br>Diversity: %{marker.color:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Within-City Cultural Diversity',
        geo=dict(scope='europe', projection_scale=6,
                center=dict(lat=54, lon=-2)),
        width=900, height=1000
    )

    fig.write_html(output_path)


def plot_interactive_network(centroids_df, cultural_dist, output_path):
    """Network of culturally similar cities."""
    threshold = np.percentile([cultural_dist.loc[centroids_df.iloc[i]['city_name'],
                                                centroids_df.iloc[j]['city_name']]
                              for i in range(len(centroids_df))
                              for j in range(i+1, len(centroids_df))], 25)

    colors = {'England': '#3498DB', 'Scotland': '#E74C3C',
              'Wales': '#2ECC71', 'Northern Ireland': '#F39C12'}

    cities = centroids_df['city_name'].tolist()
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(cultural_dist.loc[cities, cities].values)

    centroids_df['x'] = pos[:, 0]
    centroids_df['y'] = pos[:, 1]

    fig = go.Figure()

    # Edges
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            if cultural_dist.loc[cities[i], cities[j]] < threshold:
                x0, y0 = centroids_df.iloc[i][['x', 'y']]
                x1, y1 = centroids_df.iloc[j][['x', 'y']]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines', line=dict(width=0.5, color='gray'),
                    opacity=0.3, hoverinfo='skip', showlegend=False
                ))

    # Nodes
    for country in centroids_df['country'].unique():
        data = centroids_df[centroids_df['country'] == country]
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'],
            mode='markers+text', name=country,
            marker=dict(size=15, color=colors[country]),
            text=data['city_name'], textposition='top center'
        ))

    fig.update_layout(
        title='Cultural Similarity Network',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1200, height=900
    )

    fig.write_html(output_path)


def main():
    print("="*60)
    print("CULTURAL GEOGRAPHY ANALYSIS")
    print("="*60)

    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load and process
    df = load_data(DATA_FILE)
    centroids = compute_centroids(df, weighting='tfidf')
    centroids = compute_diversity(df, centroids)

    print(f"\nMost diverse: {centroids.nlargest(5, 'mean_diversity')['city_name'].tolist()}")
    print(f"Most homogeneous: {centroids.nsmallest(5, 'mean_diversity')['city_name'].tolist()}")

    # Distance analysis
    print("\nComputing distance matrices...")
    cultural_dist, geo_dist = compute_distance_matrices(centroids)

    mantel_r, mantel_p = mantel_test(cultural_dist.values, geo_dist.values)
    england_results = test_england_boundary(centroids, cultural_dist)
    ns_results = test_north_south_divide(centroids, cultural_dist)

    # Find interesting pairs
    print("\nFinding interesting pairs...")
    all_pairs, close_diff, far_sim = find_interesting_pairs(centroids, cultural_dist, geo_dist)

    print("\nClose but different:")
    print(close_diff[['city1', 'city2', 'geo_dist_km', 'cultural_dist']].head(5))

    print("\nFar but similar:")
    print(far_sim[['city1', 'city2', 'geo_dist_km', 'cultural_dist']].head(5))

    # Visualizations
    print("\nGenerating visualizations...")

    plot_dendrogram(centroids, cultural_dist,
                   os.path.join(VIZ_DIR, 'dendrogram.png'))
    plot_heatmaps(cultural_dist, geo_dist,
                 os.path.join(VIZ_DIR, 'heatmaps.png'))
    plot_correlation(centroids, cultural_dist, geo_dist,
                    os.path.join(VIZ_DIR, 'correlation.png'))
    plot_map_comparison(centroids, cultural_dist,
                       os.path.join(VIZ_DIR, 'map_comparison.png'))

    plot_interactive_3d(centroids,
                       os.path.join(VIZ_DIR, 'interactive_3d.html'))
    plot_interactive_comparison(centroids, cultural_dist,
                               os.path.join(VIZ_DIR, 'interactive_comparison.html'))
    plot_interactive_correlation(centroids, cultural_dist, geo_dist,
                                os.path.join(VIZ_DIR, 'interactive_correlation.html'))
    plot_interactive_diversity(centroids,
                              os.path.join(VIZ_DIR, 'interactive_diversity.html'))
    plot_interactive_network(centroids, cultural_dist,
                            os.path.join(VIZ_DIR, 'interactive_network.html'))

    # Save
    centroids.to_csv(os.path.join(OUTPUT_DIR, 'centroids.csv'), index=False)
    all_pairs.to_csv(os.path.join(OUTPUT_DIR, 'all_pairs.csv'), index=False)
    close_diff.to_csv(os.path.join(OUTPUT_DIR, 'close_but_different.csv'), index=False)
    far_sim.to_csv(os.path.join(OUTPUT_DIR, 'far_but_similar.csv'), index=False)

    print(f"Cities: {len(centroids)}")
    print(f"Mantel: r = {mantel_r:.4f}, p = {mantel_p:.4f}")
    print(f"England boundary: p = {england_results['p_value']:.4f}")
    if ns_results:
        print(f"North-South divide: p = {ns_results['p_value']:.4f}")
    print(f"\nVisualizations saved to: {VIZ_DIR}")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("="*60)

    return centroids, all_pairs


if __name__ == "__main__":
    centroids, pairs = main()
