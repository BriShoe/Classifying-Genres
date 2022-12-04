import pandas as pd
import os
import ast
import numpy as np
from collections import defaultdict


def createGenreList(df):
    genres = df[["genre1", "genre2", "genre3", "genre4", "genre5", "genre6", "genre7", "genre8", "genre9", "genre10"]]
    genrelist = []
    for index, row in genres.iterrows():
        trackgenres = []
        for i in range(1, 11):
            if not pd.isna(row[f"genre{i}"]):
                trackgenres.append(row[f"genre{i}"])
            else:
                break
        genrelist.append(trackgenres)
    df["genres"] = genrelist
    df.drop(["Unnamed: 0", "genre1", "genre2", "genre3", "genre4", "genre5", "genre6", "genre7", "genre8",
              "genre9", "genre10", "rhythm_bpm_histogram_second_peak_weight_dvar"], axis=1, inplace=True)


def normalize(series):
    return (series - series.mean()) / series.std()


def genreToColumn(df):
    listOfGenres = ['rock', 'rock---alternative', 'rock---indierock', 'rock---singersongwriter',
                    'rock---classicrock', 'rock---poprock', 'rock---progressiverock', 'rock---rockabilly', 'rock---rocknroll',
                    'rock---hardcorepunk', 'rock---punk', 'rock---newwave', 'rock---postpunk', 'rock---alternativerock',
                    'rock---indie', 'rock---hardrock', 'rock---hairmetal', 'rock---artrock', 'rock---bluesrock','rock---alternativepunk',
                    'rock---latinrock', 'rock---powerpop', 'rock---indiepop', 'rock---psychobilly',
                    'rock---stonerrock', 'rock---glamrock', 'rock---aor', 'rock---psychedelicrock', 'rock---britpop', 'rock---newromantic',
                    'rock---emo', 'rock---softrock', 'rock---grunge', 'rock---pianorock', 'rock---american', 'rock---rockabillysoul',
                    'rock---krautrock', 'rock---noisepop', 'rock---stoner', 'rock---garagerock', 'rock---lofi', 'rock---spacerock',
                    'rock---indiefolk', 'rock---alternativemetal', 'rock---guitarvirtuoso', 'rock---powerballad', 'rock---symphonicrock',
                    'rock---rockballad', 'rock---arenarock', 'rock---protopunk', 'rock---numetal', 'rock---rapcore', 'rock---funkrock',
                    'rock---folkpunk', 'rock---surfrock',
                    'rock---anarchopunk', 'rock---stonermetal', 'rock---southernrock', 'rock---poppunk', 'rock---jamband',
                    'rock---funkmetal', 'rock---madchester', 'rock---britishinvasion', 'rock---chamberpop', 'rock---russianrock',
                    'rock---experimentalrock', 'rock---melodicrock', 'rock---postgrunge', 'rock---horrorpunk', 'rock---streetpunk',
                    'rock---jazzrock', 'rock---symphonicprog', 'rock---glam', 'rock---acousticrock',
                    'rock---psychedelicpop']
    genreDict = dict()
    for i in range(len(listOfGenres)):
        genreDict[listOfGenres[i]] = i
    genres = df["genres"]
    genreColumns = [[0 for j in range(genres.shape[0])] for i in range(len(listOfGenres))]
    for i in range(genres.shape[0]):
        list1 = ast.literal_eval(genres.values[i])
        for j in list1:
            if j in genreDict.keys():
                genreColumns[genreDict[j]][i] = 1
    for i in range(len(listOfGenres)):
        df[listOfGenres[i]] = genreColumns[i]
    df.drop("genres", axis=1, inplace=True)



if __name__ == "__main__":
    root = os.getcwd()
    filename = "rock2edited"
    df = pd.read_csv(f"{root}/data/{filename}.csv")
    genreToColumn(df)
    df.to_csv(f"{root}/data/rock2edited.csv")
