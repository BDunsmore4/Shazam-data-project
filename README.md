**Shazam Discovery Analysis: Cultural Geography in UK**

Analysing music discovery patterns across UK cities using 2151 unique songs, 50 cities, approximately 35000 city-song pairs, containing millions of shazam events. The project is very much ongoing, but I will share some preliminary work exploring whether: (1) geographic distance correlates with cultural distance in music preferences, (2) a north-sound divide exists, (3) discovery patterns are markedly different across the national boundaries of the British Isles. 

https://britishtopography.manus.space/


**Data**

The Global Music Discovery dataset used in this project was collected by Manuel Anglada-Tort and Harin Lee, collecting the top 50 most discovered songs each day in 1423 cities in 53 countries across 3 years. For my purposes here, I am focusing on UK cities (N=50) across the span of 1 year, 2021/2022. 

**Method/Pipeline**

The method used here was very data-driven, as opposed to being more hypothesis led. This is near the start of the project so I wanted to get to grips with the data.

•	First, I used Essentia to extract 600 features from the 30-sec song preview from apple music. These spanned low-level features (MFCCs, spectral features), bpm, harmonic/tonal content etc. 

•	Then, I used rapidfuzz to check for duplicates and exclude them. Quite a few tracks were rerecorded a second time with a featured singer or remixed, these were flagged but accepted. Naming discrepancies were removed. 

•	Some features were flattened or dummy-coded to make them usable further along pipeline. 

•	I cleaned the data (removing non-scalar variables, variables with too many NaNs, data that seemed artificially capped or too many repeated values, trimmed certain variables etc.)  and checked for univariate and multivariate outliers using mahalanobis distance. Having listened to songs flagged here, these were not excluded as they were genuine outliers reflecting diversity in song discovery.

•	Then, using a methodology informed by a spotify blog: 

        Standardised variables > PCA (reduce from high dim) > UMAP > HDBSCAN 


•	Once all songs are embedded in this space, I calculated the city centroids, weighting them using TF-IDF, so while higher ranked songs are weighted more in each city, the more locally distinctive discoveries also have more weight.




**Now the fun begins:**

•	Pearson correlation between geographic distance (haversine formula) and “cultural distance” (Euclidean in UMAP space)… moderate correlation detected.

<img width="2969" height="2369" alt="correlation" src="https://github.com/user-attachments/assets/3ef75514-3279-4a73-a3e7-0993a19799b4" />


<img width="5913" height="2968" alt="heatmaps" src="https://github.com/user-attachments/assets/dc31b57f-8287-46d4-8cdf-3e203b9b1a2d" />




•	Metric calculated for extent “cultural distance” changes per unit geographic distance for all pairwise comparisons. 

•	Used Mann-Whitney U test to see if there was a significant difference between within England vs cross-english-border cultural change…. There was! Although, not a big surprise given shazam reflects both bottom-up curiosity and top-down data, like radio plays which likely differ across borders. However, no significant difference between north and south England.

**Cultural Topography of Britain**

•	Tried to turn a 2D map of Great Britain (excluded NI here) into 3D, where the mountains/hills between cities correlate to the extent to which their discovery patterns diverge. Eg. If a city is totally unique…let’s say Inverness… the Cairngorms and Scottish Highlands would be very high indeed! 

•	I tried using python in blender to create a 3D model before taking this mesh and combining it with some graphics in Manus. Code quite simple and approximate: broadly, each intercity cultural distance score would rise and fall around midpoint and when there was a crossing of paths, an average of the different path heights was taken. Execution is slightly lacking but see my efforts below. 
