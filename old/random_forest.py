from sklearn.ensemble import RandomForestClassifier


def random_forest(dataset, trees):
    model = RandomForestClassifier(n_estimators=trees)
    X = dataset[dataset.columns.difference(
        ["key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0", "genre"])]
    y = dataset["genre"]
    model.fit(X, y)
    return model
