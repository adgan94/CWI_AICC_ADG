import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and Explore the Dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# Summarize the feature space
df_X = pd.DataFrame(X, columns=feature_names)
print("\nFeature Space Summary (First 5 rows):")
print(df_X.head())
print("\nFeature Space Info:")
print(df_X.info())

# Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA and Analyze Variance Explained
pca = PCA().fit(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Scree plot of explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', color='blue')
plt.title('Scree Plot (Explained Variance per Component)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.savefig('scree_plot.png')
plt.close()

# Cumulative explained variance plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='red')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.axhline(0.80, color='gray', linestyle='--', label='80% Threshold')
plt.axhline(0.90, color='orange', linestyle='--', label='90% Threshold')
plt.axhline(0.95, color='green', linestyle='--', label='95% Threshold')
plt.legend()
plt.grid(True)
plt.savefig('cumulative_variance_plot.png')
plt.close()

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled) # X_scaled is the standardized data from Task 2

# Create a DataFrame for plotting
df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_pca['Target'] = y # y is the target class (0, 1, 2)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], 
                      c=df_pca['Target'], 
                      cmap='viridis', 
                      s=50, 
                      alpha=0.8)

# Add legend based on the target classes
legend1 = plt.legend(*scatter.legend_elements(), 
                    title="Wine Class",
                    loc="upper right")
plt.gca().add_artist(legend1)

plt.title('Wine Data Visualization in 2D PCA Space (PC1 vs. PC2)')
plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.grid(True)
plt.savefig('pca_2d_visualization.png')
plt.close()

print("Visualization complete. Saved as pca_2d_visualization.png")

# Fit PCA with 3 Components
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Create a DataFrame for easier plotting
df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['Target'] = y

# Calculate variance explained by the first three components
variance_explained = pca_3d.explained_variance_ratio_

# Create the 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot: PC1, PC2, PC3 colored by target class
scatter = ax.scatter(df_pca_3d['PC1'], 
                     df_pca_3d['PC2'], 
                     df_pca_3d['PC3'], 
                     c=df_pca_3d['Target'], 
                     cmap='viridis', 
                     s=50, 
                     alpha=0.8)

ax.set_title('Wine Data Visualization in 3D PCA Space')
ax.set_xlabel(f'Principal Component 1 ({variance_explained[0]*100:.1f}% Variance)')
ax.set_ylabel(f'Principal Component 2 ({variance_explained[1]*100:.1f}% Variance)')
ax.set_zlabel(f'Principal Component 3 ({variance_explained[2]*100:.1f}% Variance)')

legend1 = ax.legend(*scatter.legend_elements(), 
                    title="Wine Class", 
                    loc="upper right")
ax.add_artist(legend1)

plt.savefig('pca_3d_visualization.png')
plt.close()

# Number of components required to reach thresholds
components_80 = np.argmax(cumulative_variance >= 0.80) + 1
components_90 = np.argmax(cumulative_variance >= 0.90) + 1
components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Components required to reach 80% variance: {components_80}")
print(f"Components required to reach 90% variance: {components_90}")
print(f"Components required to reach 95% variance: {components_95}")

# Save results for later use
results = {
    'components_80': components_80,
    'components_90': components_90,
    'components_95': components_95
}
pd.Series(results).to_csv('pca_variance_results.csv', header=False)