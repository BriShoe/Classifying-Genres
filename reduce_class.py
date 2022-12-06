import os

import pandas as pd


def filter_df(df: pd.DataFrame, labels, labels_to_keep, n_labels, n_columns):
    columns_to_drop = labels.columns.values.tolist() ^ labels_to_keep
    df.drop(columns_to_drop, axis=1, inplace=True)
    for i, row in df.iterrows():
        to_keep = False
        for label in labels_to_keep:
            if row[label] == 1:
                to_keep = True
                break

        if not to_keep:
            df.drop(i, inplace=True)

    return df


if __name__ == "__main__":
    root = os.getcwd()
    df = pd.read_csv(f"{root}/data/rock1edited.csv", index_col=0)
    df2 = pd.read_csv(f"{root}/data/rock2edited.csv", index_col=0)
    df3 = pd.read_csv(f"{root}/data/rockvalidedited.csv", index_col=0)
    print(df.shape)
    print(df2.shape)
    print(df3.shape)

    full_set = df.append(df2)

    n_labels = 74
    threshold = 100
    l, w = full_set.shape

    labels = full_set.iloc[:, w - n_labels:]
    sums = [labels[i].sum() for i in labels.columns]
    label_counts = dict(zip(labels.columns, sums))
    filtered_counts = {}
    for k, v in label_counts.items():
        if v > 100:
            filtered_counts[k] = v

    filtered_labels = filtered_counts.keys()
    df = filter_df(df, labels, filtered_labels, n_labels, w)
    df.to_csv(f"{root}/data/rock1edited_filtered.csv")
    df2 = filter_df(df2, labels, filtered_labels, n_labels, w)
    df2.to_csv(f"{root}/data/rock2edited_filtered.csv")
    df3 = filter_df(df3, labels, filtered_labels, n_labels, w)
    df3.to_csv(f"{root}/data/rockvalidedited_filtered.csv")
    print(df.shape)
    print(df2.shape)
    print(df3.shape)





