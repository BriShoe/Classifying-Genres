import pandas as pd
import os
from ast import literal_eval
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
    listOfGenres = ['rock---alternative', 'rock---indierock', 'rock---singersongwriter',
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
                    'rock---psychedelicpop'].sort()
    genreDict = dict()
    for i in range(len(listOfGenres)):
        genreDict[listOfGenres[i]] = i
    genres = df["genres"]
    genreColumns = [[0 for j in range(genres.shape[0])] for i in range(len(listOfGenres))]
    for i in range(genres.shape[0]):
        list1 = literal_eval(genres.values[i])[1:]
        for j in list1:
            if j in genreDict.keys():
                genreColumns[genreDict[j]][i] = 1
    for i in range(len(listOfGenres)):
        df[listOfGenres[i]] = genreColumns[i]
    df.drop("genres", axis=1, inplace=True)


def trylist(str):
    try:
        return literal_eval(str)
    except Exception as e:
        print(e)
        return "not list-able"
def unlistify(df):
    newcolumns = []
    newvalues = []
    listcolumns = []
    for column in df.columns.values:
        listified = trylist(df[column].iloc[0])
        if type(listified) is list:
            listcolumns.append(column)
    #        s = df[column]
    #        mincolumnlen = s.apply(lambda x: len(x)).min()
    #        listColumns.append((column, mincolumnlen))
    #        for i in range(mincolumnlen):
    #            newcolumns.append(f"{column}_{i}")
    #            newvalues.append([])
    #    else:
    #        continue
    #for index, row in df.iterrows():
    #    iter = 0
    #    for column in listColumns:
    #        columnlist, numentries = trylist(row[column])
    #        for i in range(numentries):
    #            newvalues[iter].append(columnlist[i])
    #            iter += 1
    df.drop(listcolumns, axis=1, inplace=True)


def bringrocktoback(df):
    for column in df.columns.values:
        if column.startswith("rock"):
            popped = df.pop(column)
            df[column] = popped


if __name__ == "__main__":
    root = os.getcwd()
    df = pd.read_csv(f"{root}/data/rock1edited.csv")
    df2 = pd.read_csv(f"{root}/data/rock2edited.csv")
    df3 = pd.read_csv(f"{root}/data/rockvalidedited.csv")
    sharedcolumns = list(set(df.columns.values) & set(df2.columns.values) & set(df3.columns.values))
    #rock1
    df = df[sharedcolumns]
    unlistify(df)
    df.apply(pd.to_numeric, errors="ignore")
    df.drop(["recordingmbid", "releasegroupmbid", "rock", "Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
    df = df.sort_index(axis=1)
    bringrocktoback(df)
    #rock2
    df2 = df2[sharedcolumns]
    unlistify(df2)
    df2.apply(pd.to_numeric, errors="ignore")
    df2.drop(["recordingmbid", "releasegroupmbid", "rock", "Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
    df2 = df2.sort_index(axis=1)
    bringrocktoback(df2)
    #rockvalid
    df3 = df3[sharedcolumns]
    unlistify(df3)
    df3.apply(pd.to_numeric, errors="ignore")
    df3.drop(["recordingmbid", "releasegroupmbid", "rock", "Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
    df3 = df3.sort_index(axis=1)
    bringrocktoback(df3)

    df.to_csv(f"{root}/data/rock1edited2.csv")
    df2.to_csv(f"{root}/data/rock2edited2.csv")
    df3.to_csv(f"{root}/data/rockvalidedited2.csv")
