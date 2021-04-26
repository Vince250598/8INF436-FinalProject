import fastcluster
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import numpy as np

import utilDecisionTreeClassification
import utilPerformanceComputation
from DataframeAnalyser import AnalyseDataframe
from DataframePreprocessor import preprocessDataframe
from ClusteringTools import analyzeCluster

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from KMeans import apply_KMeans

random_state = 2021

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('genres_v2.csv')

AnalyseDataframe(data, 'genre') #, True)

X, y = preprocessDataframe(data)

print(X.head())
print(y.head())

# 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
original_test = X_test
original_labels_test = y_test
# K-MEANS

n_init = 10
max_iter = 300
tol = 0.0001
n_clusters = 20


kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
kmeans.fit(X_train)


X_train_cluster = kmeans.predict(X_train)
X_train_cluster = pd.DataFrame(data=X_train_cluster, index=X_train.index, columns=['cluster'])

print(X_train_cluster)
print(X_train)

# X_train['cluster'] = X_train_cluster.cluster

'''
RDC = RandomForestClassifier(n_estimators=100)

RDC.fit(X_train, X_train_cluster.values.ravel())

y_pred = RDC.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
'''

decision_tree_parameters = utilDecisionTreeClassification.DecisionTreeParameters()
decision_tree_parameters.criterion = 'entropy'
decision_tree_parameters.splitter = 'best'
decision_tree_parameters.max_depth = None
decision_tree_parameters.min_samples_split = 2
decision_tree_parameters.min_samples_leaf = 1
decision_tree_parameters.max_feature = 'sqrt'
decision_tree_parameters.max_leaf_nodes = None

# cluster number from int to string
X_train_cluster['cluster'] = X_train_cluster['cluster'].apply(str)

#X_train, X_test, y_train, y_test = train_test_split(X_train, X_train_cluster, test_size=0.3, random_state=random_state)

print("     The training dataset has : " + str(len(X_train_cluster)) + " instances\n")  # It work if it is a list
print("     The testing dataset has : " + str(len(X_test)) + " instances\n")  # Same
print("     All the instances are splitting into " + str(len(np.unique(y_test))) + " classes, which are : \n")  # Same

'''
# Get the classes (unique labels) of the problem from the list "y_train"
class_names = np.unique(y_train)

# Display the list of classes of the problem
for i in range(0, len(class_names), 1):
    print("         - " + str(class_names[i]))

# Convert the list of class names into an array to display results
class_names = np.array(class_names)  # It was useful in a previous version of this code
'''
# Create a class object that define the performances container of the decision tree classifier
performances = utilPerformanceComputation.Performances()

print("The decision tree algorithm is executing. Please wait ...")

# Create and train the model of the decision tree
decision_tree_classifier, training_running_time = \
    utilDecisionTreeClassification.train_decision_tree_classifier(X_train, X_train_cluster, decision_tree_parameters)

print("The training process of the model of the decision tree took : %.8f second" % training_running_time)

y_test_predicted = decision_tree_classifier.predict(X_test)

print(X_test)
print(y_test_predicted)
X_test['cluster'] = y_test_predicted
X_test['genre'] = original_labels_test
print(X_test)


confusion_matrix = pd.pivot_table(X_test, values='key_0', index=['cluster'], columns=['genre'], aggfunc='count')
confusion_matrix = confusion_matrix.fillna(0)
confusion_matrix.loc[:, 'total'] = confusion_matrix.sum(axis=1)
print(confusion_matrix)
confusion_matrix = (confusion_matrix.loc[:, 'Dark Trap':'trap'].div(confusion_matrix['total'], axis=0)) * 100
print(confusion_matrix)
best_by_cluster = pd.DataFrame()
best_by_cluster['percentage'] = confusion_matrix.max(axis=1)
best_by_cluster['genre'] = confusion_matrix.idxmax(axis=1)
print(best_by_cluster)
average_precision = best_by_cluster.percentage.mean()
print(average_precision)

'''
# Test the trained model of the decision tree
y_test_predicted, testing_running_time = \
    utilDecisionTreeClassification.test_decision_tree_classifier(X_test, decision_tree_classifier)

print("The testing process of decision tree took : %.8f second" % testing_running_time)

# Compute the performances of the decision tree classifier
cm = utilPerformanceComputation.compute_performances_for_multiclass(y_test, y_test_predicted, class_names,performances)

# Display the results
utilPerformanceComputation.display_confusion_matrix(performances, class_names)
utilPerformanceComputation.display_features_and_classification_for_dt_classifier(X_test, y_test, class_names,
                                                                                 decision_tree_classifier)
                                                                                 '''




'''
# Clustering hierarchique
Z = fastcluster.linkage_vector(X_train, method='ward',
                               metric='euclidean')

Z_dataFrame = pd.DataFrame(data=Z, columns=['clusterOne',
                                  'clusterTwo', 'distance', 'newClusterSize'])

def find_distance_thres(n_clusters, Z, X_train):
    distance_threshold = 0
    while True:
        print("Trying distance threshold " + str(distance_threshold))
        clusters = fcluster(Z, distance_threshold, criterion='distance')
        X_train_hierClustered = pd.DataFrame(data=clusters, index=X_train.index, columns=['cluster'])
        if len(X_train_hierClustered['cluster'].unique()) <= n_clusters:
            break
        distance_threshold = distance_threshold + 1

    return distance_threshold


distance = find_distance_thres(32, Z, X_train)
print(distance)

clusters = fcluster(Z, distance, criterion='distance')
X_train_hierClustered = pd.DataFrame(data=clusters, index=X_train.index,
                 columns=['cluster'])
print("Number of distinct clusters: ",
      len(X_train_hierClustered['cluster'].unique()))

countByCluster_hierClust, countByLabel_hierClust, \
countMostFreq_hierClust, accuracyDF_hierClust, \
overallAccuracy_hierClust, accuracyByLabel_hierClust = \
    analyzeCluster(X_train_hierClustered, y_train)
print("Overall accuracy from hierarchical clustering: ",
      overallAccuracy_hierClust)
'''
'''
eps = 3
min_samples = 5
leaf_size = 30
n_jobs = -1

db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)

X_train_dbscanClustered = db.fit_predict(X_train)
X_train_dbscanClustered = pd.DataFrame(data=X_train_dbscanClustered, index=X_train.index, columns=['cluster'])

countByCluster_dbscan, countByLabel_dbscan, countMostFreq_dbscan, accuracyDF_dbscan, overallAccuracy_dbscan, accuracyByLabel_dbscan = analyzeCluster(X_train_dbscanClustered, y_train)

print(overallAccuracy_dbscan)
'''

