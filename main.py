import pandas as pd
import numpy as np

import utilPerformanceComputation
from DataframeAnalyser import AnalyseDataframe
from DataframePreprocessor import preprocessDataframe
from sklearn.model_selection import train_test_split
from DisplayTools import showClassificationPerformances
from SupervisedLearning import applySupervisedModel
from HierarchicalClustering import applyHierarchicalClustering, trainAndTestHierModel
from DBSCAN import trainAndTestDBSCANModel

from KMeans import apply_KMeans, trainAndTestKmeansModel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('genres_v2.csv')

# Print information about the dataframe, the last parameters dictates if the function should print graphs
AnalyseDataframe(data, 'genre')  # , True)

# USING KMEANS
'''
n_clusters = 20
n_init = 10
max_iter = 300
tol = 0.0001
kmeansResults = pd.DataFrame(columns=['average_precision', 'kappa_score', 'weighted_precision', 'weighted_recall', 'weighted_f1_score', 'matthews_correlation_coefficient'])

for random_state in range(2020, 2023):
    for i in range(0, 3):
        kmeansResults.loc[len(kmeansResults)] = trainAndTestKmeansModel(data, random_state, n_clusters, n_init, max_iter, tol)


kmeansResultsMeanValues = kmeansResults.mean()
kmeansResultsCSV = pd.read_csv('kmeansResults.csv')

kmeansResultsCSV.loc[len(kmeansResultsCSV)] = [n_clusters, n_init, max_iter, tol, kmeansResultsMeanValues.average_precision,
                                               kmeansResultsMeanValues.kappa_score, kmeansResultsMeanValues.weighted_precision,
                                               kmeansResultsMeanValues.weighted_recall, kmeansResultsMeanValues.weighted_f1_score,
                                               kmeansResultsMeanValues.matthews_correlation_coefficient]

kmeansResultsCSV.to_csv('kmeansResults.csv', index=False)

print(kmeansResultsCSV)'''

# USING HIERARCHICAL CLUSTERING

n_clusters = 20
HierResults = pd.DataFrame(columns=['average_precision', 'kappa_score', 'weighted_precision', 'weighted_recall', 'weighted_f1_score', 'matthews_correlation_coefficient'])

for random_state in range(2020, 2023):
    for i in range(0, 3):
        HierResults.loc[len(HierResults)] = trainAndTestHierModel(data, random_state, n_clusters)


hierResultsMeanValues = HierResults.mean()
hierResultsCSV = pd.read_csv('hierResults.csv')


hierResultsCSV.loc[len(hierResultsCSV)] = [n_clusters, hierResultsMeanValues.average_precision,
                                           hierResultsMeanValues.kappa_score, hierResultsMeanValues.weighted_precision,
                                           hierResultsMeanValues.weighted_recall, hierResultsMeanValues.weighted_f1_score,
                                           hierResultsMeanValues.matthews_correlation_coefficient]

hierResultsCSV.to_csv('hierResults.csv', index=False)

print(hierResultsCSV)


# USING DBSCAN
'''
eps = 1.43
min_samples = 5
DBSCANResults = pd.DataFrame(columns=['average_precision', 'kappa_score', 'weighted_precision', 'weighted_recall', 'weighted_f1_score', 'matthews_correlation_coefficient'])

for random_state in range(2020, 2023):
    for i in range(0, 3):
        DBSCANResults.loc[len(DBSCANResults)] = trainAndTestDBSCANModel(data, random_state, eps, min_samples)


DBSCANResultsMeanValues = DBSCANResults.mean()
DBSCANResultsCSV = pd.read_csv('DBSCANResults.csv')


DBSCANResultsCSV.loc[len(DBSCANResultsCSV)] = [eps, min_samples, DBSCANResultsMeanValues.average_precision,
                                               DBSCANResultsMeanValues.kappa_score, DBSCANResultsMeanValues.weighted_precision,
                                               DBSCANResultsMeanValues.weighted_recall, DBSCANResultsMeanValues.weighted_f1_score,
                                               DBSCANResultsMeanValues.matthews_correlation_coefficient]

DBSCANResultsCSV.to_csv('DBSCANResults.csv', index=False)

print(DBSCANResultsCSV)
'''