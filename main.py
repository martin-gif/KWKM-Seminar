import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# 1) build mapping (id → text)
def creat_head_dict_from_csv():
    meta = pd.read_csv("survey-key-question.csv")  # first row of survey with codes and question
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
    return df.rename(columns=mapping, errors="raise")


# 2) LIKERT SCALES
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
    words = s.map(word_map).astype("Float64")

    out = nums.astype("Float64")
    out = out.where(out.notna(), words)
    return out


def compute_scale(df: pd.DataFrame, prefix: str, new_name: str):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found for prefix {prefix}")
    df[new_name] = df[cols].apply(to_numeric_likert).mean(axis=1)
    return df


# 3) VISUALS
def plot_group_box_and_points(df_clean: pd.DataFrame, out_png: str = "plot_box_autonomous_use.png"):
    g0 = df_clean.loc[df_clean["age_group"] == 0, "autonomous_use"].to_numpy()
    g1 = df_clean.loc[df_clean["age_group"] == 1, "autonomous_use"].to_numpy()

    plt.figure()
    plt.boxplot([g1, g0], labels=["young (1)", "old (0)"], showmeans=True)

    # jitter points
    rng = np.random.default_rng(42)
    x1 = 1 + rng.uniform(-0.06, 0.06, size=len(g1))
    x0 = 2 + rng.uniform(-0.06, 0.06, size=len(g0))
    plt.scatter(x1, g1, alpha=0.8)
    plt.scatter(x0, g0, alpha=0.8)

    plt.ylabel("autonomous_use (mean of G05Q18[1..5])")
    plt.title("Autonomous LLM use by age group")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def plot_histograms(df_clean: pd.DataFrame, out_png: str = "plot_hist_autonomous_use.png"):
    g0 = df_clean.loc[df_clean["age_group"] == 0, "autonomous_use"]
    g1 = df_clean.loc[df_clean["age_group"] == 1, "autonomous_use"]

    plt.figure()
    plt.hist(g1, bins=10, alpha=0.6, label="young (1)")
    plt.hist(g0, bins=10, alpha=0.6, label="old (0)")
    plt.xlabel("autonomous_use")
    plt.ylabel("count")
    plt.title("Distribution of autonomous_use by age group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()


def plot_scatter_autonomous_vs_reskill(df_clean: pd.DataFrame, out_png: str = "plot_scatter_autonomous_vs_reskill.png"):
    plt.figure()
    for grp, label in [(1, "young (1)"), (0, "old (0)")]:
        sub = df_clean[df_clean["age_group"] == grp]
        plt.scatter(sub["reskill_orientation"], sub["autonomous_use"], alpha=0.8, label=label)

    plt.xlabel("reskill_orientation (mean of G05Q19[1..6])")
    plt.ylabel("autonomous_use")
    plt.title("Autonomous use vs reskill orientation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

if __name__ == '__main__':

    # Load two groups
    df_young = pd.read_csv("results-survey779776.csv").assign(age_group=1)
    df_old   = pd.read_csv("results-survey374736.csv").assign(age_group=0)
    df = pd.concat([df_young, df_old], ignore_index=True)

    # Compute scales (G05Q18 for upskill/autonomous)
    df = compute_scale(df, prefix="G05Q18[", new_name="autonomous_use")
    df["upskill_orientation"] = df["autonomous_use"]

    # Compute reskill scale (G05Q19)
    df = compute_scale(df, prefix="G05Q19[", new_name="reskill_orientation")

    # Clean
    df_clean = df[["age_group", "autonomous_use",
                   "upskill_orientation", "reskill_orientation"]].dropna()

    # T-TEST
    young = df_clean.loc[df_clean["age_group"] == 1, "autonomous_use"]
    old   = df_clean.loc[df_clean["age_group"] == 0, "autonomous_use"]

    lev_stat, lev_p = stats.levene(young, old)
    equal_var = lev_p > 0.05
    t_stat, p_val = stats.ttest_ind(young, old, equal_var=equal_var)

    print("=== T-TEST autonomous_use by age group ===")
    print("Levene p =", lev_p)
    print("t-stat   =", t_stat)
    print("p-value  =", p_val)
    print("mean young =", young.mean())
    print("mean old   =", old.mean())
    print()

    # ANCOVA
    model = smf.ols(
        "autonomous_use ~ C(age_group) + upskill_orientation + reskill_orientation",
        data=df_clean
    ).fit()

    print("=== ANCOVA (Type II SS) ===")
    print(sm.stats.anova_lm(model, typ=2))
    print()
    print("=== Regression Summary ===")
    print(model.summary())
    print()

    # EXPORT
    df_clean.to_csv("analysis.csv", index=False)

    # VISUALS
    plot_group_box_and_points(df_clean)
    plot_histograms(df_clean)
    plot_scatter_autonomous_vs_reskill(df_clean)
