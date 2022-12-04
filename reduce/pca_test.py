from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def test_pca(data, k):
    scaled_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=k)
    pca.fit(scaled_data)
    plt.plot(pca.explained_variance_ratio_)
    return pca, pca.transform(scaled_data)
