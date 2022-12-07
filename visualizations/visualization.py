from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def pcavisualization():
    pca = PCA()
    df = pd.read_csv("../data/rock1edited.csv")
    df2 = pd.read_csv("../data/rock2edited.csv")
    df = pd.concat([df, df2])
    columns = []
    for column in df.columns.values:
        if column.startswith("tonal") or column.startswith(
                "rhythm") or column.startswith("lowlevel"):
            columns.append(column)
    X = df[columns].to_numpy()
    transformedX = pca.fit_transform(X)
    plot = plt.scatter(transformedX[:, 0], transformedX[:, 1], c="b")
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.title("PCA Visualization")
    plt.savefig("../visualizations/pcavisualization.png")


def plotGenres():
    genres = dict({
        'rock---alternative': 1580,
        'rock---indierock': 574,
        'rock---singersongwriter': 648,
        'rock---classicrock': 1804,
        'rock---poprock': 130,
        'rock---progressiverock': 744,
        'rock---rockabilly': 214,
        'rock---rocknroll': 284,
        'soul': 46,
        'soul---rnb': 22,
        'rock---hardcorepunk': 68,
        'rock---punk': 1006,
        'rock---newwave': 460,
        'rock---postpunk': 322,
        'rock---alternativerock': 956,
        'rock---indie': 1540,
        'rock---hardrock': 858,
        'rock---hairmetal': 54,
        'rock---artrock': 72,
        'rock---bluesrock': 212,
        'rock---alternativepunk': 30,
        'rock---latinrock': 18,
        'rock---powerpop': 70,
        'rock---indiepop': 204,
        'soul---funk': 24,
        'rock---psychobilly': 86,
        'rock---stonerrock': 110,
        'rock---glamrock': 150,
        'rock---aor': 26,
        'rock---psychedelicrock': 314,
        'rock---britpop': 176,
        'rock---newromantic': 42,
        'rock---emo': 102,
        'rock---softrock': 148,
        'rock---grunge': 200,
        'rock---pianorock': 18,
        'rock---american': 98,
        'rock---rockabillysoul': 12,
        'rock---krautrock': 44,
        'rock---noisepop': 22,
        'rock---stoner': 20,
        'rock---garagerock': 90,
        'rock---lofi': 64,
        'rock---spacerock': 108,
        'rock---indiefolk': 24,
        'rock---alternativemetal': 34,
        'rock---guitarvirtuoso': 48,
        'rock---powerballad': 10,
        'rock---symphonicrock': 32,
        'rock---rockballad': 20,
        'rock---arenarock': 4,
        'rock---protopunk': 14,
        'rock---numetal': 44,
        'rock---rapcore': 26,
        'rock---funkrock': 16,
        'soundtrack': 36,
        'soundtrack---moviesoundtrack': 4,
        'soundtrack---musical': 2,
        'rock---folkpunk': 18,
        'rock---surfrock': 26,
        'rock---anarchopunk': 16,
        'rock---stonermetal': 14,
        'rock---southernrock': 92,
        'soul---motown': 6,
        'rock---poppunk': 88,
        'rock---jamband': 28,
        'rock---funkmetal': 12,
        'rock---madchester': 18,
        'rock---britishinvasion': 6,
        'rock---chamberpop': 6,
        'rock---russianrock': 58,
        'rock---experimentalrock': 28,
        'rock---melodicrock': 34,
        'rock---postgrunge': 12,
        'rock---horrorpunk': 14,
        'rock---streetpunk': 18,
        'rock---jazzrock': 26,
        'soundtrack---score': 2,
        'rock---symphonicprog': 10,
        'rock---glam': 14,
        'rock---acousticrock': 8,
        'rock---psychedelicpop': 4,
        'world': 2,
        'soul---neosoul': 2
    })
    sortedGenres = np.array(
        sorted(genres.items(), key=lambda x: x[1], reverse=True))
    genres = sortedGenres[:12]
    print(genres)
    #topgenres, values = list(genres[:, 0]), list(genres[:, 1])
    #topgenres = [str[7:] for str in topgenres]
    #values = [int(num) for num in values]
    #plt.bar(topgenres, values, color='maroon', width=0.4, bottom=0)
    #plt.ylim(0, 2000)
    #plt.xlabel("Subgenres")
    #plt.ylabel("Number of Examples")
    #plt.xticks(rotation=45, ha="right")
    #plt.title("Top 10 Subgenres by Number of Examples Present in Dataset")
    #plt.savefig("../visualizations/top10examples.png", bbox_inches="tight")


def valueHistogram():
    genres = dict({
        'rock---alternative': 1580,
        'rock---indierock': 574,
        'rock---singersongwriter': 648,
        'rock---classicrock': 1804,
        'rock---poprock': 130,
        'rock---progressiverock': 744,
        'rock---rockabilly': 214,
        'rock---rocknroll': 284,
        'rock---hardcorepunk': 68,
        'rock---punk': 1006,
        'rock---newwave': 460,
        'rock---postpunk': 322,
        'rock---alternativerock': 956,
        'rock---indie': 1540,
        'rock---hardrock': 858,
        'rock---hairmetal': 54,
        'rock---artrock': 72,
        'rock---bluesrock': 212,
        'rock---alternativepunk': 30,
        'rock---latinrock': 18,
        'rock---powerpop': 70,
        'rock---indiepop': 204,
        'rock---psychobilly': 86,
        'rock---stonerrock': 110,
        'rock---glamrock': 150,
        'rock---aor': 26,
        'rock---psychedelicrock': 314,
        'rock---britpop': 176,
        'rock---newromantic': 42,
        'rock---emo': 102,
        'rock---softrock': 148,
        'rock---grunge': 200,
        'rock---pianorock': 18,
        'rock---american': 98,
        'rock---rockabillysoul': 12,
        'rock---krautrock': 44,
        'rock---noisepop': 22,
        'rock---stoner': 20,
        'rock---garagerock': 90,
        'rock---lofi': 64,
        'rock---spacerock': 108,
        'rock---indiefolk': 24,
        'rock---alternativemetal': 34,
        'rock---guitarvirtuoso': 48,
        'rock---powerballad': 10,
        'rock---symphonicrock': 32,
        'rock---rockballad': 20,
        'rock---arenarock': 4,
        'rock---protopunk': 14,
        'rock---numetal': 44,
        'rock---rapcore': 26,
        'rock---funkrock': 16,
        'rock---folkpunk': 18,
        'rock---surfrock': 26,
        'rock---anarchopunk': 16,
        'rock---stonermetal': 14,
        'rock---southernrock': 92,
        'rock---poppunk': 88,
        'rock---jamband': 28,
        'rock---funkmetal': 12,
        'rock---madchester': 18,
        'rock---britishinvasion': 6,
        'rock---chamberpop': 6,
        'rock---russianrock': 58,
        'rock---experimentalrock': 28,
        'rock---melodicrock': 34,
        'rock---postgrunge': 12,
        'rock---horrorpunk': 14,
        'rock---streetpunk': 18,
        'rock---jazzrock': 26,
        'rock---symphonicprog': 10,
        'rock---glam': 14,
        'rock---acousticrock': 8,
        'rock---psychedelicpop': 4
    })
    #sortedGenres = np.array(sorted(genres.items(), key=lambda x: x[1], reverse=True))
    #genres = sortedGenres[:10]
    #topgenres, values = list(genres[:, 0]), list(genres[:, 1])
    #topgenres = [str[7:] for str in topgenres]
    #values = [int(num) for num in values]
    count = 0
    for key, value in genres.items():
        if value > 100:
            count += 1
    print(count)
    genrevalues = genres.values()
    genrevalues = [int(value) for value in genrevalues]
    hist = np.histogram(genrevalues, range=(0, 2000))
    plt.hist(genrevalues, bins="auto", color="g")
    #plt.ylim(0, 2000)
    plt.xlabel("Number of Examples")
    plt.ylabel("Number of Subgenres")
    #plt.xticks(rotation=45, ha="right")
    plt.title("Histogram of Examples per Subgenre")
    plt.savefig("../visualizations/histogram.png", bbox_inches="tight")


listOfGenres = sorted([
    'rock---alternative', 'rock---alternativerock', 'rock---bluesrock',
    'rock---britpop', 'rock---classicrock', 'rock---garagerock',
    'rock---glamrock', 'rock---grunge', 'rock---hardrock', 'rock---indie',
    'rock---indiepop', 'rock---indierock', 'rock---newwave', 'rock---poprock',
    'rock---postpunk', 'rock---progressiverock', 'rock---psychedelicrock',
    'rock---punk', 'rock---rockabilly', 'rock---rocknroll',
    'rock---singersongwriter', 'rock---softrock', 'rock---spacerock',
    'rock---stonerrock'
])


def labelCorrelation():
    data_p1 = pd.read_csv('data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)

    num_genres = 24
    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(Y[listOfGenres].corr())
    xticks_labels = [
        'alternative', 'alternativerock', 'bluesrock', 'britpop',
        'classicrock', 'garagerock', 'glamrock', 'grunge', 'hardrock', 'indie',
        'indiepop', 'indierock', 'newwave', 'poprock', 'postpunk',
        'progressiverock', 'psychedelicrock', 'punk', 'rockabilly',
        'rocknroll', 'singersongwriter', 'softrock', 'spacerock', 'stonerrock'
    ]
    plt.xticks(np.arange(24) + .5, labels=xticks_labels)
    plt.yticks(np.arange(24) + .5, labels=xticks_labels)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    labelCorrelation()
