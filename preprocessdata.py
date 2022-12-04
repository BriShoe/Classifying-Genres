import pandas as pd
import os
import numpy as np
from collections import defaultdict


def createGenreList(df, outputname):
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
              "genre9", "genre10"], axis=1, inplace=True)
    df.to_csv(f"{root}/data/{outputname}.csv")



if __name__ == "__main__":
    root = os.getcwd()
    filename = "rock2"
    df = pd.read_csv(f"{root}/data/{filename}.csv")
    createGenreList(df, "rock2edited")
