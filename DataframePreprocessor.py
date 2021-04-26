from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


# La somme de train_percentage, test_percentage et validation_percentage doit être égale à 1
def train_test_validation_split(X, y, test_percentage, validation_percentage, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=random_state,
                                                        stratify=y)

    train_percentage = 1 - (validation_percentage + test_percentage)

    validation_size = validation_percentage / (validation_percentage + train_percentage)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                      random_state=random_state, stratify=y_train)

    return X_train, X_test, X_val, y_train, y_test, y_val


def preprocessDataframe(data):
    data = removeUselessColumns(data)

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

def scaleNumericalAttributes(X):
    # Mise a l'echelle des attributs
    scaled_features = X.copy()

    attributesToScale = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    features = scaled_features[attributesToScale]

    scaler = StandardScaler()
    features = scaler.fit_transform(features.values)

    scaled_features[attributesToScale] = features

    return scaled_features
