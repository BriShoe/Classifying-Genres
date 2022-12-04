from sklearn.linear_model import LogisticRegression


def logreg(dataset):
    model = LogisticRegression()
    X = dataset[dataset.columns.difference(
        ["key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0", "genre"])]
    y = dataset["genre"]
    model.fit(X, y)
    return model
