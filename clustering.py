# HOMEWORK 8 - Clustering
# Tsakiris Giorgos

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')

k = 2
name = ['KMeans', 'SpectralClustering']
pca = False

breastCancer = load_breast_cancer()
X = breastCancer.data
y = breastCancer.target

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if pca:
    print('PCA for 2 components')
    pca_model = PCA(n_components=2)
    pca_data = pca_model.fit_transform(X)
    data = pca_data
else:
    data = X

for k in range(2, 11):
    for n in name:
        # KMeans
        if n == 'KMeans':
            model = KMeans(n_clusters=k, random_state=0).fit(data)
        elif n == 'SpectralClustering':
            model = SpectralClustering(n_clusters=k, random_state=0).fit(data)

        labels = model.labels_
        if pca:
            clustering = np.vstack((data.T, labels)).T
            clustering = pd.DataFrame(data=clustering, columns=("Dim_1", "Dim_2", "label"))
            sn.FacetGrid(clustering, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
            plt.title(n)
            plt.savefig(n+'_clusters'+str(k)+'.png')
            plt.show()

        sc = silhouette_score(data, labels)
        print('Silhouette Score for %d clusters (%s): %.5f' %(k, n, sc))
