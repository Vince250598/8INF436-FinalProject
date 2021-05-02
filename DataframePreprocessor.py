from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocessDataframe(data, balanceDataClasses=True):
    data = removeUselessColumns(data)

    print("Values per class: \n", data['genre'].value_counts())
    print("Number of instances: ", len(data.index))

    if balanceDataClasses:
        data = balanceDataframe(data)
        print("Values per class (after oversampling): \n", data['genre'].value_counts())
        print("Number of instances (after oversampling): ", len(data.index))

    # Séparation des données et des labels
    X = data.loc[:, data.columns != 'genre']
    labels = data.genre
    y = pd.Series(data=labels, name="genre")

    X = encodeCategoricalAttributes(X)

    X = scaleNumericalAttributes(X)

    return X, y


def removeUselessColumns(data):
    data = data.drop(['id'], axis=1)
    data = data.drop(['uri'], axis=1)
    data = data.drop(['track_href'], axis=1)
    data = data.drop(['analysis_url'], axis=1)
    data = data.drop(['song_name'], axis=1)
    data = data.drop(['Unnamed: 0'], axis=1)
    data = data.drop(['title'], axis=1)
    data = data.drop(['type'], axis=1)

    return data


def encodeCategoricalAttributes(X):
    X = pd.get_dummies(X, columns=["key", "time_signature"])

    return X


# Oversampling
def balanceDataframe(X):
    max_size = X['genre'].value_counts().max()

    lst = [X]
    for class_index, group in X.groupby('genre'):
        lst.append(group.sample(max_size - len(group), replace=True))
    X_new = pd.concat(lst)

    X_new = X_new.reset_index(drop=True)

    return X_new


def scaleNumericalAttributes(X):
    # Mise a l'echelle des attributs
    scaled_features = X.copy()

    attributesToScale = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                         'liveness', 'valence', 'tempo', 'duration_ms']
    features = scaled_features[attributesToScale]

    scaler = StandardScaler()
    features = scaler.fit_transform(features.values)

    scaled_features[attributesToScale] = features

    return scaled_features
