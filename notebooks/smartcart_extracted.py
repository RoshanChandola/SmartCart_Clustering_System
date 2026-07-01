import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('E:/smartcart_clustering_system/data/smartcart_customers.csv')  # Load your dataset here

# --- CELL ---

df.head()

# --- CELL ---

df.shape

# --- CELL ---

df.isnull().sum()

# --- CELL ---

df['Income'] = df['Income'].fillna(df['Income'].median())# why we can not fill income with mean value because the mean is sensitive to outliers, which can skew the data. The median is a better measure of central tendency for skewed distributions, as it is less affected by extreme values.

# --- CELL ---

df.head()

# --- CELL ---

# getting age

df["Age"] = 2026 - df["Year_Birth"]
df.head()

# --- CELL ---

# Customer joing date
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

refrence_date = df["Dt_Customer"].max()

df["Customer_Tenure_Days"] = (refrence_date - df["Dt_Customer"]).dt.days

# --- CELL ---

df.head()  # Display the first few rows of the dataset

# --- CELL ---

df.columns

# --- CELL ---

# Spending  new feature =  total _spending

df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

# --- CELL ---

# children

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

# --- CELL ---

# education

print(df["Education"].value_counts())

# UnderGraduate, Graduate, PostGraduate

df["Education"] = df["Education"].replace({"Basic": "UnderGraduate", "2n Cycle": "UnderGraduate", "Master": "PostGraduate", "PhD": "PostGraduate"})

# --- CELL ---

df.head()

# --- CELL ---

# handling martial status

df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone",
    "Widow": "Alone"
})

# --- CELL ---

df["Living_With"].value_counts()

# --- CELL ---

df.shape
df.head()

# --- CELL ---

cols = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status","Kidhome", "Teenhome"]

spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

cols_to_drop = cols + spending_cols

df_cleaned =df.drop(columns=cols_to_drop)

# --- CELL ---

df_cleaned.shape
df_cleaned.head()

# --- CELL ---

cols = ["Income","Recency","Response","Age","Total_Spending","Total_Children"]

# relative plots of some features = pair plots

sns.pairplot(df_cleaned[cols]) # the pair plot is a great way to visualize the relationships between multiple variables in a dataset. It creates scatter plots for each pair of variables and histograms for each individual variable, allowing you to see potential correlations, distributions, and patterns in the data. 



# --- CELL ---

# remove outliers

print("data size with outliers:", len(df_cleaned))

df_cleaned = df_cleaned[(df_cleaned["Age"] < 90)]
df_cleaned = df_cleaned[(df_cleaned["Income"] < 600_000)]

print("data size without outliers:", len(df_cleaned))

# --- CELL ---

corr_mat = df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_mat, 
    annot=True,
    annot_kws={"size": 6}, # reduces the size of the annotation text in the heatmap for better readability
    cmap="coolwarm"
    )

# --- CELL ---

df_cleaned.shape
df_cleaned.head()

# --- CELL ---

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- CELL ---

ohe = OneHotEncoder()

cat_cols = ["Education", "Living_With"]

enc_cols = ohe.fit_transform(df_cleaned[cat_cols]).toarray() 

# --- CELL ---

enc_df = pd.DataFrame(enc_cols,columns = ohe.get_feature_names_out(cat_cols),index=df_cleaned.index)  

enc_df.head()

# --- CELL ---

df_cleaned = df_cleaned.drop(columns=cat_cols)

df_encoded = pd.concat([df_cleaned, enc_df], axis=1)

# --- CELL ---

df_encoded.shape
df_encoded.head()

# --- CELL ---

X = df_encoded

# --- CELL ---

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --- CELL ---

X_scaled.shape

# --- CELL ---

#2d
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)


# --- CELL ---

pca.explained_variance_ratio_

# --- CELL ---

#plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Scatter Plot')

# --- CELL ---

from sklearn.cluster import KMeans
from kneed import KneeLocator

wcss= []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k,  random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)

# --- CELL ---

knee = KneeLocator(range(1,11),wcss,curve = "convex", direction = "decreasing")

optimal_k = knee.elbow

print("best k ", optimal_k)

# --- CELL ---

# plot

plt.plot(range(1,11), wcss, marker = 'o')
plt.xlabel("k")
plt.ylabel("wcss")

# --- CELL ---

from sklearn.metrics import silhouette_score
scores = []

for k in range(2,11):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca,labels)
    scores.append(score)

# --- CELL ---

#plot

plt.plot(range(2,11),scores, marker ='o')
plt.xlabel("K")
plt.ylabel("Silhouette")

# --- CELL ---

# combined plot for both

k_range = range(2,11)

fig,ax1 = plt.subplots(figsize= (8,6))

ax1.plot(k_range, wcss[:len(k_range)],marker = 'o', color = "blue")
ax1.set_xlabel("K")
ax1.set_ylabel("wcss")

ax2 = ax1.twinx()

ax2.plot(k_range, scores[:len(k_range)], marker="x", color="red", linestyle="--")

ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette")

# --- CELL ---

# kmeans

kmeans = KMeans(
    n_clusters = 4,
    random_state=42
)

labels_mean = kmeans.fit_predict(X_pca)

# --- CELL ---

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(111, projection='3d') # 111 means 1 row, 1 column, 1st subplot other values we have 221 means 2 rows, 2 columns, 1st subplot and so on

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_mean)

# --- CELL ---

# agglomerative clustering

from sklearn.cluster import AgglomerativeClustering

agg_cls = AgglomerativeClustering(
    n_clusters=4,
    # affinity='euclidean', # distance metric to use for clustering
    linkage='ward' # linkage method to minimize the variance of merged clusters. ward means that it minimizes the total within-cluster variance. At each step, the pair of clusters with the smallest increase in total within-cluster variance after merging is combined
)

labels_agg = agg_cls.fit_predict(X_pca)

# --- CELL ---

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_agg)


# --- CELL ---

 # using db scan for clustering
 
from sklearn.cluster import DBSCAN
 
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

# plotting the dbscan clusters

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_dbscan)
 

# --- CELL ---

X["clusters"] = labels_agg


# --- CELL ---

X.head()


# --- CELL ---

pal = ["red", "blue", "yellow", "green"]

sns.countplot(x=X["clusters"], palette=pal, hue = X["clusters"])

# --- CELL ---

# income & spending patterns

sns.scatterplot(x=X["Total_Spending"], y=X["Income"], hue=X["clusters"], palette=pal)

# --- CELL ---

cluster_summary = X.groupby("clusters").mean()

print(cluster_summary)

# --- CELL ---

