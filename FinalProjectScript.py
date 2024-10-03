import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://query.data.world/s/z6zanxqnqxqa7sbg4h3l6kzabyuwgz?dws=00000')

# df_analyse=df
# # Ensure each column has consistent data types
# for column in df_analyse.columns:
#     # Convert all values in the column to strings
#     df_analyse[column] = df_analyse[column].astype(str)
#
# # Investigate all elements within each Feature/Column
# for column in df_analyse:
#     unique_vals = np.unique(df_analyse[column])
#     nr_values = len(unique_vals)
#     if nr_values < 10:
#         print(f"The nb of values for feature {column} : {nr_values} -- {unique_vals}")
#     else:
#         print(f"The nb of values for feature {column} : {nr_values}")
#
# # Print columns
# print(df_analyse.columns)


#   1. missing data
print(df.isna().sum())

# Analize dataset
print("Original df: ", df.shape)

# delete rows with empty cells
df=df.dropna()

#see shape of dataframe after empty cell dropped
print("After deleting empty rows: ", df.shape)



#   2. Check data type
print(df.dtypes, '\n')
print("Summarize the data types and count of columns:",'\n', df.dtypes.value_counts(), '\n')



# Create a new df x with the columns from the df
x=df[[ "GEO Region", "Activity Type Code", "Price Category Code", "Passenger Count"]]


# 3. converts categorical variables into dummy/indicator variables
x = pd.get_dummies(x, drop_first=True) #encode categoric variables

print("Current dataframe that we will work with has: ",x.shape)


# 4. K MEANS

#Take the values from the dataframe x that we created with the features that i am interrested in
X_train = x.values


# Calculate inertia for different numbers of clusters to find best K

no_of_clusters = range(2,20)
inertia = []
for k in no_of_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)
    print("The innertia for: ", k, "Clusters is: ", kmeans.inertia_)



# Creating the scree plot for Intertia - elbow method
# we can see here best k=5 based on how steep the line is

fig, (ax1) = plt.subplots(1, figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx, inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters, rotation='vertical')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia Score')
plt.title("Inertia Plot per k")
plt.show()

#---------------------------------------KMEAN ALG------------


# Apply K-means clustering with 5 clusters (optimal solution)
kmeans = KMeans(n_clusters=5, random_state=2)
kmeans.fit(X_train)

# Get cluster labels
cluster_labels = kmeans.labels_
print()
print("Cluster labels: ",cluster_labels)
print()

# "predictions" for new data
predictions = kmeans.predict(X_train)
print(predictions)

# calculating the Counts of the cluster (how many flights does each cluster have-count of flights)
unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,5)

# Creating a datagrame
countscldf = pd.DataFrame(counts, columns = ["Cluster_0","Cluster_1","Cluster_2", "Cluster_3","Cluster_4"])
print(countscldf)


# Running PCA to Visualize the data

X = X_train
y_num = predictions

target_names = ["Cluster_0", "Cluster_1", "Cluster_2", "Cluster_3", "Cluster_4"]



# Normalization can help in achieving better PCA results, especially if the features are on different scales
# Standardize the data before applying PCA
# StandardScaler: This scales each feature to have a mean of 0 and a variance of 1.

scaler = StandardScaler()
X = scaler.fit_transform(X_train)


# Check the mean and variance of the scaled data
print("Mean of each feature after scaling:", np.mean(X, axis=0))
print("Variance of each feature after scaling:", np.var(X, axis=0))

#PCA WITH 2 COMPONENTS
pca = PCA(n_components=2, random_state=453)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('Explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
# The output indicates that the first two principal components explain about 26.59% of the variance,
# which suggests that my data's variance is spread across more components


# # Plotting the data
# plt.figure()
# plt.figure(figsize=(12, 8))
# colors = ['navy', 'turquoise', 'darkorange', 'red', 'black']
# lw = 2
# for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], target_names):
#     plt.scatter(X_r[y_num == i, 0], X_r[y_num == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('PCA of 2 Items')
# plt.show()
#
# explained_variance = np.cumsum(pca.explained_variance_ratio_)
# print("EXPLAINED VARIANCE",explained_variance)


# Apply PCA with 3 components
pca = PCA(n_components=3)
components = pca.fit_transform(X)

# Explained variance
total_var = pca.explained_variance_ratio_.sum() * 100
print(f'Total Explained Variance: {total_var:.2f}%')

# Use the predictions (clusters) as colors
color = predictions  # Ensure this is a 1D array with the same length as the number of rows in X_scaled

# Plotting the 3D PCA results using Plotly
fig = px.scatter_3d(
    x=components[:, 0], y=components[:, 1], z=components[:, 2],
    color=color,
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'x': 'PC 1', 'y': 'PC 2', 'z': 'PC 3'}
)
fig.show()

# NOW TRY WITH MORE COMPONENTS FOR PCA
# Trying with Dimentionality reduction and then Kmeans

n_components = X.shape[1]
# Running PCA with all components
pca = PCA(n_components=n_components, random_state = 453)
X_r = pca.fit(X).transform(X)


# Calculating the 95% Variance
total_variance = sum(pca.explained_variance_)
print("Total Variance in our dataset is: ", total_variance)
var_95 = total_variance * 0.95
print("The 95% variance we want to have is: ", var_95)
print("")

# Creating a df with the components and explained variance
a = zip(range(0,n_components), pca.explained_variance_)
a = pd.DataFrame(a, columns=["PCA Comp", "Explained Variance"])

# Trying to hit 95%
print("Variance explain with 3 n_compononets: ", sum(a["Explained Variance"][0:3]))
print("Variance explain with 4 n_compononets: ", sum(a["Explained Variance"][0:4]))
print("Variance explain with 5 n_compononets: ", sum(a["Explained Variance"][0:5]))
print("Variance explain with 6 n_compononets: ", sum(a["Explained Variance"][0:6]))
print("Variance explain with 7 n_compononets: ", sum(a["Explained Variance"][0:7]))
print("Variance explain with 8 n_compononets: ", sum(a["Explained Variance"][0:8]))
print("Variance explain with 9 n_compononets: ", sum(a["Explained Variance"][0:9]))
print("Variance explain with 10 n_compononets: ", sum(a["Explained Variance"][0:10]))
print("Variance explain with 11 n_compononets: ", sum(a["Explained Variance"][0:11]))
print("Variance explain with 12 n_compononets: ", sum(a["Explained Variance"][0:12]))

# Plotting the Data
plt.figure(1, figsize=(14, 8))
plt.plot(pca.explained_variance_ratio_, linewidth=2, c="r")
plt.xlabel('n_components')
plt.ylabel('explained_ratio_')

# Plotting line with 95% e.v.
plt.axvline(10,linestyle=':', label='n_components - 95% explained', c ="blue")
plt.legend(prop=dict(size=12))

# adding arrow
plt.annotate('10 eigenvectors used to explain 95% variance', xy=(10, pca.explained_variance_ratio_[10]),
             xytext=(10, pca.explained_variance_ratio_[10]),
            arrowprops=dict(facecolor='blue', shrink=0.05))

plt.show()

# Running PCA again

pca = PCA(n_components=10, random_state = 453)
X_r = pca.fit(X).transform(X)

clusters= kmeans.labels_

inertia = []

#running Kmeans

for f in no_of_clusters:
    kmeans = KMeans(n_clusters=f, random_state=2)
    kmeans = kmeans.fit(X_r)
    u = kmeans.inertia_
    inertia.append(u)
    print("The innertia for :", f, "Clusters is:", u)

# Creating the scree plot for Intertia - elbow method
fig, (ax1) = plt.subplots(1, figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx, inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters, rotation='vertical')
plt.xlabel('n_components Value')
plt.ylabel('Inertia Score')
plt.title("Inertia Plot per k")
plt.show()

#Conclusion:
# We managed to improve our results by reducing inertia

# Add cluster labels to the DataFrame
df['Kmeans_Cluster'] = cluster_labels


# Create a mapping from numeric labels to descriptive names
label_mapping = {0: "Cluster_1", 1: "Cluster_2", 2: "Cluster_3", 3: "Cluster_4", 4: "Cluster_5"}

# Map the numeric labels to descriptive names
df['Kmeans_Cluster'] = df['Kmeans_Cluster'].map(label_mapping)

df.to_csv("df.csv")


# Calculate Silhouette Score -> metric used to evaluate the quality of clustering
silhouette_avg = silhouette_score(x, cluster_labels)
# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(x, cluster_labels)
# Calculate Calinski-Harabasz Index
ch_index = calinski_harabasz_score(x, cluster_labels)
print("Silhouette Score:", silhouette_avg)  #  0.44 is generally considered good  -> data point is well-clustered
print("Davies-Bouldin Index:", db_index) #  0.85 moderate separation between clusters but some overlap or similarity between clusters.
print("Calinski-Harabasz Index:", ch_index)  # 19579.24 clusters have a relatively high degree of separation and compactness -> generally desirable in clustering


kmeans_info = {
    'Silhouette Score': silhouette_avg,
    'Davies-Bouldin Index': db_index,
    'Calinski-Harabasz Index': ch_index
}



# --------------------------------HIERARCHICAL ALG

# Calculate the linkage matrix
Z = linkage(x, method='ward')

# # Plot the dendrogram   COMENTEZ ASTA CA E PREA MARE PT PROCESARE
# plt.figure(figsize=(12, 6))
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data Point Index')
# plt.ylabel('Distance')
# plt.show()

#Create sample for dendogram
# Define the size of the sample (e.g., 10% of the dataset)
sample_size = int(0.01 * len(x))

# Randomly sample the data
sampled_df = x.sample(n=sample_size, random_state=42)  # Use a fixed random_state for reproducibility

# Perform hierarchical clustering
Z = linkage(sampled_df, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram (Sampled Data)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()



# Apply hierarchical clustering with the optimal number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)  # Specify the number of clusters
agg_cluster_labels = agg_clustering.fit_predict(x)

# Calculate Silhouette Score
agg_silhouette_avg = silhouette_score(x, agg_cluster_labels)

# Calculate Davies-Bouldin Index
agg_db_index = davies_bouldin_score(x, agg_cluster_labels)

# Calculate Calinski-Harabasz Index
agg_ch_index = calinski_harabasz_score(x, agg_cluster_labels)

print("Silhouette Score (Hierarchical Clustering):", agg_silhouette_avg)

print("Davies-Bouldin Index (Hierarchical Clustering):", agg_db_index)

print("Calinski-Harabasz Index (Hierarchical Clustering):", agg_ch_index)


hierarchical_info = {
    'Silhouette Score': agg_silhouette_avg,
    'Davies-Bouldin Index': agg_db_index,
    'Calinski-Harabasz Index': agg_ch_index
}


# Create DataFrames for K-means and Hierarchical Clustering metrics
kmeans_df = pd.DataFrame(kmeans_info, index=['KMeans'])

hierarchical_df = pd.DataFrame(hierarchical_info, index=['Hierarchical'])

# Concatenate the K-means and Hierarchical Clustering DataFrames vertically
combined_df = pd.concat([kmeans_df, hierarchical_df])

# Transpose the DataFrame so that the columns represent the index names
combined_df_transposed = combined_df.T

# Save the DataFrame to a CSV file
combined_df_transposed.to_csv('METRICS.csv')