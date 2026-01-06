from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


class SurveyAnalyzer:
    def __init__(
        self,
        young_csv: str = "data/results-survey779776.csv",
        old_csv: str = "data/results-survey374736.csv",
        key_csv: str = "data/survey-key-question.csv",
        group_col: str = "young_group",
        young_value: int = 1,
        old_value: int = 0,
    ):
        self.young_csv = young_csv
        self.old_csv = old_csv
        self.key_csv = key_csv
        self.group_col = group_col
        self.young_value = young_value
        self.old_value = old_value

    # -----------------------------
    # Mapping (id -> full question)
    # -----------------------------
    def creat_head_dict_from_csv(self) -> None:
        meta = pd.read_csv(self.key_csv)  # reads with first row as header
        meta = meta.columns.to_series().reset_index(drop=True).to_frame(name="raw")

        meta[["id", "text"]] = meta["raw"].str.split(".", n=1, expand=True)
        meta = meta.drop(columns="raw")

        meta["id"] = meta["id"].astype(str).str.strip()
        meta["text"] = meta["text"].astype(str).str.strip()
        meta.to_csv(self.key_csv, index=False)

    def get_full_question(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = pd.read_csv(self.key_csv).set_index("id")["text"]
        mapping = meta.to_dict()
        return df.rename(columns=mapping, errors="raise")

    # -----------------------------
    # Likert + scales
    # -----------------------------
    @staticmethod
    def to_numeric_likert(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        nums = pd.to_numeric(s.str.extract(r"^(\d+)", expand=False), errors="coerce")

        word_map = {
            "not at all": 1,
            "very little": 2,
            "a little": 3,
            "moderately": 4,
            "strongly": 5,
            "very strongly": 6,
            "completely": 7,
        }
        words = s.str.lower().map(word_map).astype("Float64")

        out = nums.astype("Float64")
        out = out.where(out.notna(), words)
        return out

    def compute_scale(self, df: pd.DataFrame, prefix: str, new_name: str) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            raise ValueError(f"No columns found for prefix: {prefix}")

        df[new_name] = df[cols].apply(self.to_numeric_likert).mean(axis=1)
        return df

    # -----------------------------
    # Load + prepare dataset
    # -----------------------------
    def load_two_groups(self) -> pd.DataFrame:
        df_young = pd.read_csv(self.young_csv).assign(**{self.group_col: self.young_value})
        df_old = pd.read_csv(self.old_csv).assign(**{self.group_col: self.old_value})
        return pd.concat([df_young, df_old], ignore_index=True)

    def prepare_clean_dataset(self) -> pd.DataFrame:
        df = self.load_two_groups()

        # G05Q18 -> autonomous_use, then reuse as upskill_orientation (your current definition)
        df = self.compute_scale(df, prefix="G05Q18[", new_name="autonomous_use")
        df["upskill_orientation"] = df["autonomous_use"]

        # G05Q19 -> reskill_orientation
        df = self.compute_scale(df, prefix="G05Q19[", new_name="reskill_orientation")

        df_clean = df[
            [self.group_col, "autonomous_use", "upskill_orientation", "reskill_orientation"]
        ].dropna()

        return df_clean

    # -----------------------------
    # Stats
    # -----------------------------
    def run_ttest_autonomous_by_group(self, df_clean: pd.DataFrame) -> None:
        g1 = df_clean.loc[df_clean[self.group_col] == self.young_value, "autonomous_use"]
        g0 = df_clean.loc[df_clean[self.group_col] == self.old_value, "autonomous_use"]

        lev_stat, lev_p = stats.levene(g1, g0)
        equal_var = lev_p > 0.05
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=equal_var)

        print("=== T-TEST autonomous_use by group ===")
        print("Levene p =", lev_p)
        print("t-stat   =", t_stat)
        print("p-value  =", p_val)
        print("mean young =", g1.mean())
        print("mean old   =", g0.mean())
        print()

    def run_ancova(self, df_clean: pd.DataFrame):
        model = smf.ols(
            f"autonomous_use ~ C({self.group_col}) + upskill_orientation + reskill_orientation",
            data=df_clean,
        ).fit()

        print("=== ANCOVA (Type II SS) ===")
        print(sm.stats.anova_lm(model, typ=2))
        print()
        print("=== Regression Summary ===")
        print(model.summary())
        print()

        return model

    # -----------------------------
    # Plots
    # -----------------------------
    def plot_group_box_and_points(self, df_clean: pd.DataFrame, out_png: str = "figures/plot_box_autonomous_use.png") -> None:
        g0 = df_clean.loc[df_clean[self.group_col] == self.old_value, "autonomous_use"].to_numpy()
        g1 = df_clean.loc[df_clean[self.group_col] == self.young_value, "autonomous_use"].to_numpy()

        plt.figure()
        plt.boxplot([g1, g0], labels=["young (1)", "old (0)"], showmeans=True)

        rng = np.random.default_rng(42)
        x1 = 1 + rng.uniform(-0.06, 0.06, size=len(g1))
        x0 = 2 + rng.uniform(-0.06, 0.06, size=len(g0))
        plt.scatter(x1, g1, alpha=0.8)
        plt.scatter(x0, g0, alpha=0.8)

        plt.ylabel("autonomous_use (mean of G05Q18[1..5])")
        plt.title("Autonomous LLM use by group")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        # plt.show()

    def plot_histograms(self, df_clean: pd.DataFrame, out_png: str = "figures/plot_hist_autonomous_use.png") -> None:
        g0 = df_clean.loc[df_clean[self.group_col] == self.old_value, "autonomous_use"]
        g1 = df_clean.loc[df_clean[self.group_col] == self.young_value, "autonomous_use"]

        plt.figure()
        plt.hist(g1, bins=10, alpha=0.6, label="young (1)")
        plt.hist(g0, bins=10, alpha=0.6, label="old (0)")
        plt.xlabel("autonomous_use")
        plt.ylabel("count")
        plt.title("Distribution of autonomous_use by group")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        # plt.show()

    def plot_scatter_autonomous_vs_reskill(
        self,
        df_clean: pd.DataFrame,
        out_png: str = "figures/plot_scatter_autonomous_vs_reskill.png",
    ) -> None:
        plt.figure()
        for grp, label in [(self.young_value, "young (1)"), (self.old_value, "old (0)")]:
            sub = df_clean[df_clean[self.group_col] == grp]
            plt.scatter(sub["reskill_orientation"], sub["autonomous_use"], alpha=0.8, label=label)

        plt.xlabel("reskill_orientation (mean of G05Q19[1..6])")
        plt.ylabel("autonomous_use")
        plt.title("Autonomous use vs reskill orientation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        # plt.show()
