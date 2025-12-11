import sys

import pandas as pd
import numpy as np

def creat_head_dict_from_csv():
    meta = pd.read_csv("survey-key-question.csv") # first row of survey with codes and question
    meta = meta.columns.to_series().reset_index(drop=True)
    meta = meta.to_frame(name='raw')

    meta[["id", "text"]] = meta.raw.str.split(".", n=1, expand=True)
    meta = meta.drop(columns="raw")

    meta["id"] = meta["id"].str.strip()
    meta["text"] = meta["text"].str.strip()
    meta.to_csv("survey-key-question.csv", index=False)

def get_full_question(df: pd.DataFrame):
    meta = pd.read_csv("survey-key-question.csv").set_index("id")["text"]
    mapping = meta.to_dict()
    df = df.rename(columns=mapping, errors="raise")
    return df

if __name__ == '__main__':
    df_young = pd.read_csv("results-survey779776.csv")
    df_young = df_young.assign(young_group=1)

    df_old = pd.read_csv("results-survey374736.csv")
    df_old = df_old.assign(young_group=0)



    df = pd.concat([df_young, df_old], ignore_index=True)
    # print(df.iloc[1]) # with question key
    # print(get_full_question(df).iloc[1]) # with full questions
