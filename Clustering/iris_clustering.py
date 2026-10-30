
# Chapter 17: Clustering
# Step 2: Load and Explore the Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame with the four measurements
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add species names for later comparison
df["Species"] = [
    iris.target_names[i] for i in iris.target
]

# Display the dataset
print("--- IRIS DATASET ---")
print(df.head())

# Dataset information
print("\n--- DATASET INFORMATION ---")
print("Total Flowers:", len(df))
print("Total Features:", len(iris.feature_names))

print("\nFeature Names:")
for feature in iris.feature_names:
    print("-", feature)

print("\nSpecies:")
print(df["Species"].value_counts())

print("\nIris dataset loaded successfully!")



# STEP 3: PREPARE DATA FOR UNSUPERVISED LEARNING

# Select only the four numerical features
X = df[iris.feature_names].copy()

print("\n--- UNSUPERVISED LEARNING DATA ---")
print(X.head())

# Confirm the dimensions
print("\nFeature Data Shape:", X.shape)
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

# Check whether species labels are excluded
print("\nSpecies Labels Excluded:", "Species" not in X.columns)

# Keep the real species separately for later comparison
actual_species = df["Species"]

print("\nData prepared successfully!")



# STEP 4: APPLY K-MEANS CLUSTERING

# Create K-means model with 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train the model and assign clusters
cluster_labels = kmeans.fit_predict(X)

# Add cluster assignments to the original DataFrame
df["Cluster"] = cluster_labels

print("\n--- K-MEANS CLUSTERING RESULTS ---")
print(df[["Species", "Cluster"]].head(15))

# Count flowers in each cluster
print("\n--- CLUSTER DISTRIBUTION ---")
print(df["Cluster"].value_counts().sort_index())

# Display cluster centers
print("\n--- CLUSTER CENTERS ---")
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=iris.feature_names
)
print(centers.round(2))

print("\nTotal Clusters:", kmeans.n_clusters)
print("K-means clustering completed successfully!")



# STEP 5: VISUALIZE K-MEANS CLUSTERS

from pathlib import Path

# Save graphs in the Clustering folder
folder = Path(__file__).resolve().parent

plt.figure(figsize=(9, 6))

# Plot each cluster separately
for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["petal length (cm)"],
        cluster_data["petal width (cm)"],
        label=f"Cluster {cluster}",
        s=60,
        alpha=0.75
    )

# Add labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering of Iris Flowers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save graph
plt.savefig(folder / "Iris_KMeans_Clusters.png", dpi=300)
plt.close()

print("\n--- CLUSTER VISUALIZATION ---")
print("K-means scatter plot saved successfully!")
print("File: Iris_KMeans_Clusters.png")
print("\nStep 5 completed!")



# STEP 6: ANALYZE CLUSTERS

# Compare actual species with predicted clusters
cluster_comparison = pd.crosstab(
    df["Cluster"],
    df["Species"]
)

print("\n--- CLUSTER VS ACTUAL SPECIES ---")
print(cluster_comparison)

# Calculate percentage of each species within each cluster
cluster_percentages = pd.crosstab(
    df["Cluster"],
    df["Species"],
    normalize="index"
) * 100

print("\n--- SPECIES PERCENTAGE IN EACH CLUSTER ---")
print(cluster_percentages.round(2))

# Find the most common species in each cluster
print("\n--- DOMINANT SPECIES IN EACH CLUSTER ---")

for cluster in cluster_comparison.index:
    dominant_species = cluster_comparison.loc[cluster].idxmax()
    count = cluster_comparison.loc[cluster].max()
    total = cluster_comparison.loc[cluster].sum()

    percentage = (count / total) * 100

    print(
        f"Cluster {cluster}: {dominant_species} "
        f"({count}/{total} flowers, {percentage:.2f}%)"
    )

# Calculate overall cluster purity
correctly_grouped = cluster_comparison.max(axis=1).sum()
cluster_purity = correctly_grouped / len(df) * 100

print("\n--- OVERALL CLUSTER PURITY ---")
print(f"Cluster Purity: {cluster_purity:.2f}%")

print("\nStep 6 cluster analysis completed!")



# STEP 7: IMPROVE CLUSTERING WITH FEATURE SCALING

from sklearn.preprocessing import StandardScaler

# Standardize the four flower measurements
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-means using standardized features
scaled_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

scaled_clusters = scaled_kmeans.fit_predict(X_scaled)

# Compare scaled clusters with actual species
scaled_comparison = pd.crosstab(
    scaled_clusters,
    actual_species
)

print("\n--- SCALED K-MEANS CLUSTER COMPARISON ---")
print(scaled_comparison)

# Calculate cluster purity after scaling
scaled_correct = scaled_comparison.max(axis=1).sum()
scaled_purity = scaled_correct / len(df) * 100

print("\n--- SCALING COMPARISON ---")
print(f"Original Cluster Purity: {cluster_purity:.2f}%")
print(f"Scaled Cluster Purity: {scaled_purity:.2f}%")

if scaled_purity > cluster_purity:
    print("Feature scaling improved cluster purity.")
elif scaled_purity < cluster_purity:
    print("Feature scaling decreased cluster purity.")
else:
    print("Feature scaling produced the same cluster purity.")

print("\nStep 7 completed successfully!")



# STEP 8: SAVE CLUSTER PURITY COMPARISON

plt.figure(figsize=(7, 5))

methods = ["Original K-Means", "Scaled K-Means"]
purities = [cluster_purity, scaled_purity]

bars = plt.bar(
    methods,
    purities,
    color=["steelblue", "mediumseagreen"],
    width=0.55
)

for bar, purity in zip(bars, purities):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{purity:.2f}%",
        ha="center",
        fontsize=11
    )

plt.ylabel("Cluster Purity (%)")
plt.title("K-Means Clustering: Original vs Scaled Features")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(folder / "Clustering_Purity_Comparison.png", dpi=300)
plt.close()

print("\nClustering purity comparison graph saved successfully!")
print("Chapter 17 completed!")
