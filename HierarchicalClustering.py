import fastcluster
import numpy as np
from scipy.cluster.hierarchy import fcluster
import pandas as pd
from sklearn.model_selection import train_test_split

import utilPerformanceComputation
from DataframePreprocessor import preprocessDataframe
from DisplayTools import showClassificationPerformances
from SupervisedLearning import applySupervisedModel


def find_distance_thres(n_clusters, Z, X):
    distance_threshold = 0
    while True:
        print("Trying distance threshold " + str(distance_threshold))
        clusters = fcluster(Z, distance_threshold, criterion='distance')
        X_train_hierClustered = pd.DataFrame(data=clusters, index=X.index, columns=['cluster'])
        if len(X_train_hierClustered['cluster'].unique()) <= n_clusters:
            break
        distance_threshold = distance_threshold + 1

    return distance_threshold

def applyHierarchicalClustering(X, n_clusters):
    Z = fastcluster.linkage_vector(X, method='ward',
                                   metric='euclidean')

    Z_dataFrame = pd.DataFrame(data=Z, columns=['clusterOne',
                                                'clusterTwo', 'distance', 'newClusterSize'])


    distance = find_distance_thres(n_clusters, Z, X)

    clusters = fcluster(Z, distance, criterion='distance')
    clusters = pd.DataFrame(data=clusters, index=X.index, columns=['cluster'])
    print("Number of distinct clusters: ", len(clusters['cluster'].unique()))

    # cluster number from int to string
    clusters['cluster'] = clusters['cluster'].apply(str)

    return clusters

def trainAndTestHierModel(data, random_state, n_clusters):
    # Do all preprocessing operations on dataframe and return the features (X) and the labels (y)
    X, y = preprocessDataframe(data)

    clusters = applyHierarchicalClustering(X, n_clusters)

    # Split data with 70% of train data and 30% of test data, the labels are the clusters in which the instances were grouped by the clustering algorithm
    X_train, X_test, y_train, y_test = train_test_split(X, clusters, test_size=0.3, random_state=random_state)

    # Train a supervised model on the data and try to predict the test instances, the function returns the result of the classification of the test instances
    result = applySupervisedModel(X_train, y_train, X_test)

    result['genre'] = y

    # Prints a confusion matrix and an average mean of the global performance, returns a dataframe that contains the genre that's most present in each cluster and the number of instances in that cluster
    best_by_cluster, average_precision = showClassificationPerformances(result)

    # We add a column to the results that represents the predicted value for each test instances
    cluster_to_genre_dict = pd.Series(best_by_cluster.genre.values, index=best_by_cluster.index).to_dict()
    result['predicted'] = result.cluster
    result = result.replace({"predicted": cluster_to_genre_dict})

    class_names = np.unique(y)

    # Prints different performance metrics for a multiclass model
    performances = utilPerformanceComputation.Performances()
    cm = utilPerformanceComputation.compute_performances_for_multiclass(result['genre'], result['predicted'],
                                                                   class_names, performances)
    return average_precision, cm.cohen_kappa_score, cm.weighted_precision, cm.weighted_recall, cm.weighted_f1_score, cm.matthews_corrcoef


