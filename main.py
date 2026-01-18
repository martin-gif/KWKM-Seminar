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

# ===========================
# CONFIGURATION
# ===========================
PRINT_OUTPUT = False      # Print analysis results to console
GENERATE_FILES = False    # Save plots and files

# Data configuration
YOUNG_CSV = "data/results-survey779776.csv"
OLD_CSV = "data/results-survey374736.csv"
KEY_CSV = "data/survey-key-question.csv"
GROUP_COL = "young_group"
YOUNG_VALUE = 1
OLD_VALUE = 0

# Data cleaning configuration
COLUMN_ANSWER_PERCENTAGE = 0.8


def creat_head_dict_from_csv():
    meta = pd.read_csv(
        KEY_CSV
    )  # first row of survey with codes and question
    meta = meta.columns.to_series().reset_index(drop=True)
    meta = meta.to_frame(name="raw")

    meta[["id", "text"]] = meta.raw.str.split(".", n=1, expand=True)
    meta = meta.drop(columns="raw")

    meta["id"] = meta["id"].str.strip()
    meta["text"] = meta["text"].str.strip()
    meta.to_csv(KEY_CSV, index=False)


def get_full_question(df: pd.DataFrame):
    meta = pd.read_csv(KEY_CSV).set_index("id")["text"]
    mapping = meta.to_dict()
    df = df.rename(columns=mapping, errors="raise")
    return df


def check_age(df, low_bound, upper_bound):
    return df[(df["G02Q04"] >= low_bound) & (df["G02Q04"] <= upper_bound)]


if __name__ == "__main__":
    # Load and combine datasets
    df_young = pd.read_csv(YOUNG_CSV)
    df_young = df_young.assign(young_group=YOUNG_VALUE)

    df_old = pd.read_csv(OLD_CSV)
    df_old = df_old.assign(young_group=OLD_VALUE)

    df = pd.concat([df_young, df_old], ignore_index=True)
    # print(df.iloc[1]) # with question key
    # print(get_full_question(df).iloc[1]) # with full questions

    # Clean dataset
    min_count = int(COLUMN_ANSWER_PERCENTAGE * len(df))

    # delete rows where user didnt finished
    df = df[df["submitdate"].notna()]
    # drop columns with to litte partisans
    df = df.dropna(axis="columns", thresh=min_count)
    # df = df.dropna()
    # print(df[df["young_group"] == 1].shape)
    # print(df[df["young_group"] == 0].shape)

    # Create grouped dataset with calculated variables
    # Create grouped dataset with calculated variables
    df_grouped = group_data(df, print_cronbach=False)

    # Define variables for analysis
    vars_usefulness = ["usefulness_work", "usefulness_learning"]
    vars_motivation = ["controlled_motivation", "autonomous_motivation"]

    # Calculate descriptives by group
    desc = descriptives_by_group(
        df=df_grouped,
        group_col=GROUP_COL,
        vars_usefulness=vars_usefulness,
        vars_motivation=vars_motivation,
        confidence=0.95,
    )
    print(desc)

    # Initialize analyzer and prepare its specific dataset
    # Note: SurveyAnalyzer uses different variables than df_grouped
    analyzer = SurveyAnalyzer(
        young_csv=YOUNG_CSV,
        old_csv=OLD_CSV,
        key_csv=KEY_CSV,
        group_col=GROUP_COL,
        young_value=YOUNG_VALUE,
        old_value=OLD_VALUE,
    )
    
    # Prepare clean dataset for SurveyAnalyzer methods
    # (uses autonomous_use, upskill_orientation, reskill_orientation)
    df_clean = analyzer.prepare_clean_dataset()

    # Run SurveyAnalyzer analyses
    analyzer.run_ttest_autonomous_by_group(
        df_clean,
        print_output=PRINT_OUTPUT,
        generate_files=GENERATE_FILES
    )
    analyzer.run_ancova(
        df_clean,
        print_output=PRINT_OUTPUT,
        generate_files=GENERATE_FILES
    )

    df_clean.to_csv("data/analysis.csv", index=False)

    analyzer.plot_group_box_and_points(
        df_clean,
        print_output=PRINT_OUTPUT,
        generate_files=GENERATE_FILES
    )
    analyzer.plot_histograms(
        df_clean,
        print_output=PRINT_OUTPUT,
        generate_files=GENERATE_FILES
    )
    analyzer.plot_scatter_autonomous_vs_reskill(
        df_clean,
        print_output=PRINT_OUTPUT,
        generate_files=GENERATE_FILES
    )

    # Survey statistics 
    survey_stats = SurveyStatistics(df=df)
    survey_stats.print_summary(
        print_output=True,
        generate_files=GENERATE_FILES
    )

    # Additional analyses
    calc_correlation(df_grouped, save_fig=False)
    do_ttest(df)
