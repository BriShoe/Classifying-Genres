from sklearn.neighbors import KNeighborsClassifier

def KNN(dataset, k):
    model = KNeighborsClassifier(n_neighbors=k)
    X = dataset[dataset.columns.difference(["genre", "key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0"])]
    y = dataset["genre"]
    model.fit(X, y)
    return model




