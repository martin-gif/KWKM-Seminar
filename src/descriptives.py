#descriptives.py

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)

    m = float(x.mean())
    se = float(stats.sem(x))  # Standardfehler
    tcrit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    half = tcrit * se
    return (m - half, m + half)


def descriptives_by_group(
    df: pd.DataFrame,
    group_col: str,
    vars_usefulness: list[str],
    vars_motivation: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
 
    targets = vars_usefulness + vars_motivation
    for c in targets:
        if c not in df.columns:
            raise KeyError(f"Spalte fehlt im DataFrame: {c}")

    def summarize(sub: pd.DataFrame, group_label: str) -> pd.DataFrame:
        rows = []
        for var in targets:
            s = pd.to_numeric(sub[var], errors="coerce")
            n = int(s.notna().sum())
            mean = float(s.mean()) if n > 0 else np.nan
            median = float(s.median()) if n > 0 else np.nan
            std = float(s.std(ddof=1)) if n > 1 else np.nan

            ci_low, ci_high = mean_ci(s, confidence=confidence)

            rows.append(
                {
                    "group": group_label,
                    "variable": var,
                    "n": n,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        return pd.DataFrame(rows)

    out = []
    #for all groups
    out.append(summarize(df, "overall"))

    #per group
    for g in sorted(df[group_col].dropna().unique()):
        out.append(summarize(df[df[group_col] == g], f"{group_col}={int(g)}"))

    return pd.concat(out, ignore_index=True)
