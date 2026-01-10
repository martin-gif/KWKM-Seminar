import pandas as pd

from src.survey_analysis import SurveyAnalyzer
from src.ttest import do_ttest


def creat_head_dict_from_csv():
    meta = pd.read_csv("data/survey-key-question.csv")  # first row of survey with codes and question
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


if __name__ == "__main__":
    df_young = pd.read_csv("data/results-survey779776.csv")
    df_young = df_young.assign(young_group=1)

    df_old = pd.read_csv("data/results-survey374736.csv")
    df_old = df_old.assign(young_group=0)

    df = pd.concat([df_young, df_old], ignore_index=True)
    # print(df.iloc[1]) # with question key
    # print(get_full_question(df).iloc[1]) # with full questions

    column_answer_percentage = 0.8
    min_count = int(column_answer_percentage * len(df))

    # delete rows where user didnt finished
    df = df[df['submitdate'].notna()]
    # drop columns with to litte partisans
    df = df.dropna(axis="columns", thresh=min_count)
    # df = df.dropna()

    if True:
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

    do_ttest(df)
