import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

import utilPerformanceComputation
from DataframePreprocessor import preprocessDataframe
from DisplayTools import showClassificationPerformances
from SupervisedLearning import applySupervisedModel


def trainAndTestDBSCANModel(data, random_state, eps, min_samples):
    # Do all preprocessing operations on dataframe and return the features (X) and the labels (y)
    X, y = preprocessDataframe(data)

    clusters = applyDBSCAN(X, eps, min_samples)

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


def applyDBSCAN(X, eps, min_samples):
    leaf_size = 30
    n_jobs = -1

    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)

    clusters = db.fit_predict(X)
    clusters = pd.DataFrame(data=clusters, index=X.index, columns=['cluster'])

    # cluster number from int to string
    clusters['cluster'] = clusters['cluster'].apply(str)

    return clusters
