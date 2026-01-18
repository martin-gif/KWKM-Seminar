from scipy.stats import ttest_ind
import pandas as pd
import re

def do_ttest(df: pd.DataFrame):
    results = []
    r = re.compile(r"G\d{2}Q\d{2}")

    scales = df.columns


    for scale in list(filter(r.match,scales)):
        g0 = df.loc[df["young_group"] == 0, scale].dropna()
        g1 = df.loc[df["young_group"] == 1, scale].dropna()

        if g0.dtype != 'float64' or g1.dtype != 'float64':
            continue

        t, p = ttest_ind(g0, g1, equal_var=False, nan_policy="omit")

        results.append({
            "scale": scale,
            "mean_old": g0.mean(),
            "mean_young": g1.mean(),
            "t": t,
            "p": p
        })

    results_df = pd.DataFrame(results)
    print(results_df.loc[results_df["p"] <= 0.05])
    return results_df
