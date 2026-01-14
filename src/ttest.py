from numpy.ma.core import equal
from scipy.stats import ttest_ind, levene
import pandas as pd
import warnings


def do_ttest(df: pd.DataFrame, print_results: bool = False):
    results = []
    column_list = df.columns[df.columns != "young_group"]

    for column in column_list:
        g0 = df.loc[df["young_group"] == 0, column].dropna()
        g1 = df.loc[df["young_group"] == 1, column].dropna()
        levene_stat, p_value = levene_test(g0, g1)
        equal_var = p_value < 0.05
        # print(equal_var)

        if g0.dtype != "float64" or g1.dtype != "float64":
            warnings.warn(f"{column} is not numeric")
            continue

        t, p = ttest_ind(g0, g1, equal_var=equal_var, nan_policy="omit")

        results.append(
            {
                "scale": column,
                "mean_old": g0.mean(),
                "mean_young": g1.mean(),
                "t": t,
                "p": p,
            }
        )
        if print_results:
            print(f"{column} has a levene p value of {round(p_value, 3)}")
            if equal_var:
                print(f"t-test results \t\t\t t:{round(t, 3)} \t p:{round(p, 3)}")
            else:
                print(f"Welchs t-test results \t t:{round(t, 3)} \t p:{round(p, 3)}")

    results_df = pd.DataFrame(results)
    if print_results:
        print(results_df)
    return results_df


def levene_test(df_1: pd.DataFrame, df_2: pd.DataFrame):
    levene_stat, p_value = levene(df_1, df_2)
    return levene_stat, p_value
