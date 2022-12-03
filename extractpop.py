import pandas as pd
import os
import numpy as np
import json

def createDf(row, columns, dictionary, root, top):
    addData(row, columns, dictionary, root, top)
    newdf = pd.DataFrame(columns=columns)
    newdf.loc[len(newdf.index)] = row
    return newdf


def addData(row, columns, dictionary, root, top):
    for key, value in dictionary.items():
        if top:
            itemRef = f'{key}'
        else:
            itemRef = f'{root}_{key}'
        if type(value) is dict:
            addData(row, columns, value, itemRef, False)
        else:
            columns.append(f'{itemRef}')
            row.append(value)


def augmentDataframe(df):
    root = os.getcwd()
    first = True
    numtracks = 1
    for track in df["recordingmbid"].values:
        print(f"Finished Track {numtracks}")
        newcolumns = ["recordingmbid"]
        row = [track]
        folder = track[:2]
        attributes = open(f"{root}/data/acousticbrainz-mediaeval-validation/{folder}/{track}.json", )
        data = json.load(attributes)
        if first:
            newdf = createDf(row, newcolumns, data, "", True)
            first = False
        else:
            newdf2 = createDf(row, newcolumns, data, "", True)
            sharedcolumns = list(set(newdf.columns.values.tolist()) & set(newdf2.columns.values.tolist()))
            newdf = newdf[sharedcolumns].append(newdf2[sharedcolumns])
        numtracks += 1
    return newdf


if __name__ == "__main__":
    root = os.getcwd()
    filename = "acousticbrainz-mediaeval-discogs-validation"
    df = pd.read_csv(f"{root}/data/{filename}.tsv", sep="\t")
    df = df.loc[df['genre1'] == "pop"]
    df = df.loc[df['genre2'].notna()]
    recordingmbid, genre2 = df["recordingmbid"].values, df["genre2"].values
    boolean = [a.startswith(('0', '1', '2', '3', '4', '5', '6', '7')) and b.startswith("pop") for a, b in zip(recordingmbid, genre2)]
    validlabels = df.loc[boolean]
    #boolean = [(a.startswith("2") or a.startswith("3")) and b.startswith("pop") for a, b in zip(recordingmbid, genre2)]
    #pop23labels = df.loc[boolean]
    validdata = augmentDataframe(validlabels)
    #pop23data = augmentDataframe(pop23labels)
    validdf = pd.merge(validlabels, validdata, on="recordingmbid", how="right")
    #pop23df = pd.merge(pop23labels, pop23data, on="recordingmbid", how="right")
    validdf.to_csv(f"{root}/data/valid.csv")
    #pop23df.to_csv(f"{root}/data/pop23.csv")