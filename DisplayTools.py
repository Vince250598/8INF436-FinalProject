import pandas as pd


def showClassificationPerformances(X_test):
    confusion_matrix = pd.pivot_table(X_test, values='key_0', index=['cluster'], columns=['genre'], aggfunc='count')
    confusion_matrix = confusion_matrix.fillna(0)
    confusion_matrix.loc[:, 'total'] = confusion_matrix.sum(axis=1)
    confusion_matrix = confusion_matrix.sort_values(['total'], ascending=False)
    print(confusion_matrix)

    confusion_matrix_percentages = (confusion_matrix.loc[:, 'Dark Trap':'trap'].div(confusion_matrix['total'],
                                                                                    axis=0)) * 100
    print(confusion_matrix_percentages)

    best_by_cluster = pd.DataFrame()
    best_by_cluster['percentage'] = confusion_matrix_percentages.max(axis=1)
    best_by_cluster['genre'] = confusion_matrix_percentages.idxmax(axis=1)
    best_by_cluster['Number of instances'] = confusion_matrix['total']
    print(best_by_cluster)

    average_precision = (best_by_cluster.percentage * best_by_cluster['Number of instances']).sum() / best_by_cluster[
        'Number of instances'].sum()
    print(average_precision)

    return best_by_cluster, average_precision
