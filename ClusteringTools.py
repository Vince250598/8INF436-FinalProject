import pandas as pd
import matplotlib.pyplot as plt


def showClusterDistribution(X_train, trainedModelLabels, clusterNumber, *attributes):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X_train.index.values
    cluster_map['cluster'] = trainedModelLabels
    chosenClusterindexes = cluster_map[cluster_map.cluster == clusterNumber]
    chosenCluster = X_train.loc[chosenClusterindexes.data_index]
    for attribute in attributes:
        chosenCluster[attribute].plot(kind='hist', title='Distribution of ' + str(attribute) + ' in cluster ' + str(clusterNumber))
        #chosenCluster.hist(column=attribute)

    plt.show()

