from sklearn import svm as supportvectormachine

def svm(dataset):
    model = supportvectormachine.SVC()
    X = dataset[dataset.columns.difference(["genre", "key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0"])]
    y = dataset["genre"]
    model.fit(X, y)
    return model
