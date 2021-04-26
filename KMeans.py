import pandas as pd
from sklearn.cluster import KMeans

from ClusteringTools import analyzeCluster
import matplotlib.pyplot as plt


def apply_KMeans(X_train, y_train, random_state):
    n_init = 10
    max_iter = 300
    tol = 0.0001

    kMeans_inertia = pd.DataFrame(data=[], index=range(2, 21), columns=['inertia'])
    overallAccuracy_kMeansDF = pd.DataFrame(data=[], index=range(2, 21), columns=['overallAccuracy'])

    for n_clusters in range(2, 21):
        print("Testing with " + str(n_clusters) + " clusters")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=max_iter, tol=tol, random_state=random_state)
        kmeans.fit(X_train)
        kMeans_inertia.loc[n_clusters] = kmeans.inertia_

        X_train_kmeansClustered = kmeans.predict(X_train)
        X_train_kmeansClustered = pd.DataFrame(data=X_train_kmeansClustered, index=X_train.index, columns=['cluster'])

        print(X_train_kmeansClustered)

        countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel = analyzeCluster(
            X_train_kmeansClustered, y_train)

        overallAccuracy_kMeansDF.loc[n_clusters] = overallAccuracy

    overallAccuracy_kMeansDF.plot()
    print(overallAccuracy_kMeansDF)
    kMeans_inertia.plot()
    plt.show()

