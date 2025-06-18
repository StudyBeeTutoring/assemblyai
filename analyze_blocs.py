# analyze_blocs.py (Corrected Version)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

print("--- Phase 2: Unsupervised Learning Analysis ---")
print("This script will analyze voting data decade-by-decade to find political blocs.")

# --- Step 1: Load Pivoted Data (with the CORRECT index) ---
print("\nStep 1: Loading the pivoted data from Phase 1...")
try:
    # --- THE FIX IS HERE ---
    # We tell pandas that the first two columns (0 and 1) together form our MultiIndex.
    df = pd.read_csv('un_voting_pivoted.csv', index_col=[0, 1])
    print(" - Pivoted data loaded successfully with correct MultiIndex.")
except FileNotFoundError:
    print("\n[ERROR] 'un_voting_pivoted.csv' not found.")
    print("Please make sure you have successfully run the Phase 1 script first.")
    exit()

# --- Step 2: Loop Through Decades and Analyze ---
print("Step 2: Analyzing each decade...")

decades = range(1940, 2030, 10)
all_results = {}

for decade in decades:
    print(f"  - Processing {decade}s...")

    # Filter the DataFrame based on the 'year' level of the MultiIndex
    decade_df = df[df.index.get_level_values('year') >= decade]
    decade_df = decade_df[decade_df.index.get_level_values('year') < decade + 10]

    if decade_df.empty:
        print(f"    - No data found for the {decade}s. Skipping.")
        continue

    X = decade_df.transpose()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 6
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    results_df = pd.DataFrame({
        'country_iso': X.index,
        'bloc': clusters,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })

    all_results[decade] = results_df
    print(f"    - Found {k} voting blocs for the {decade}s.")

# --- Step 3: Save the Complete Analysis ---
joblib.dump(all_results, 'un_analysis_results.joblib')
print("\n--- Phase 2 Complete! ---")
print("Successfully saved all analysis results to 'un_analysis_results.joblib'.")
print("You are now ready to build the final app in Phase 3.")