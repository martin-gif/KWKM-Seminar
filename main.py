from math import inf

import numpy as np
import pandas as pd

from src.correlation_matrix import (
    calc_correlation,
    calc_correlation_motivation_skilling,
)
from src.survey_analysis import SurveyAnalyzer
from src.survey_statistics import SurveyStatistics
from src.ttest import do_ttest
from src.group import group_data
from src.descriptives import descriptives_by_group
from src.linear_regression import linear_regression


def creat_head_dict_from_csv():
    meta = pd.read_csv(
        "data/survey-key-question.csv"
    )  # first row of survey with codes and question
    meta = meta.columns.to_series().reset_index(drop=True)
    meta = meta.to_frame(name="raw")

    meta[["id", "text"]] = meta.raw.str.split(".", n=1, expand=True)
    meta = meta.drop(columns="raw")

    meta["id"] = meta["id"].str.strip()
    meta["text"] = meta["text"].str.strip()
    meta.to_csv("survey-key-question.csv", index=False)


def get_full_question(df: pd.DataFrame):
    meta = pd.read_csv("data/survey-key-question.csv").set_index("id")["text"]
    mapping = meta.to_dict()
    df = df.rename(columns=mapping, errors="raise")
    return df


def check_age(df, low_bound, upper_bound):
    return df[(df["G02Q04"] >= low_bound) & (df["G02Q04"] <= upper_bound)]


if __name__ == "__main__":
    df_young = pd.read_csv("data/results-survey779776.csv")
    df_young = check_age(df_young, 18, 35)
    df_young = df_young.assign(young_group=1)

    df_old = pd.read_csv("data/results-survey374736.csv")
    df_old = check_age(df_old, 35, np.inf)
    df_old = df_old.assign(young_group=0)

    df = pd.concat([df_young, df_old], ignore_index=True)
    # print(df.iloc[1]) # with question key
    # print(get_full_question(df).iloc[1]) # with full questions

    column_answer_percentage = 0.8
    min_count = int(column_answer_percentage * len(df))

    # delete rows where user didnt finished
    df = df[df["submitdate"].notna()]
    # drop columns with to litte partisans
    df = df.dropna(axis="columns", thresh=min_count)
    # df = df.dropna()
    # print(df[df["young_group"] == 1].shape)
    # print(df[df["young_group"] == 0].shape)

    df_grouped = group_data(df, print_cronbach=False)

    vars_usefulness = ["usefulness_work", "usefulness_learning"]
    vars_motivation = ["controlled_motivation", "autonomous_motivation"]

    desc = descriptives_by_group(
        df=df_grouped,
        group_col="young_group",
        vars_usefulness=vars_usefulness,
        vars_motivation=vars_motivation,
        confidence=0.95,
    )

    print(desc)

    if False:
        analyzer = SurveyAnalyzer(
            young_csv="data/results-survey779776.csv",
            old_csv="data/results-survey374736.csv",
            key_csv="data/survey-key-question.csv",
            group_col="young_group",
            young_value=1,
            old_value=0,
        )

        df_clean = analyzer.prepare_clean_dataset()

        analyzer.run_ttest_autonomous_by_group(df_clean)
        analyzer.run_ancova(df_clean)

        df_clean.to_csv("data/analysis.csv", index=False)

        analyzer.plot_group_box_and_points(df_clean)
        analyzer.plot_histograms(df_clean)
        analyzer.plot_scatter_autonomous_vs_reskill(df_clean)

        # Survey statistics using data from main
        survey_stats = SurveyStatistics(df=df)
        survey_stats.print_summary()

    calc_correlation(
        df_grouped[["upskilling", "reskilling", "usage", "age", "young_group"]],
        save_fig=True,
        fig_title="Correlation Matrix",
    )
    corr_matrix = calc_correlation_motivation_skilling(
        df_grouped[
            [
                "upskilling",
                "reskilling",
                "controlled_motivation",
                "autonomous_motivation",
            ]
        ]
    )
    print(corr_matrix)

    if False:
        for y in ["autonomous_motivation", "controlled_motivation"]:
            linear_regression(
                df_X=df_grouped[
                    [
                        "upskilling",
                        "reskilling",
                        "age",
                        "usage",
                    ]
                ],
                df_Y=df_grouped[y],
                print_summary=True,
            )

        do_ttest(
            df_grouped[
                [
                    "autonomous_motivation",
                    "controlled_motivation",
                    "usefulness_work",
                    "usefulness_learning",
                    "young_group",
                ]
            ],
            print_results=True,
        )
