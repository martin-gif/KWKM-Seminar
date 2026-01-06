#correlation.py

# =========================
# IFBL + LLM Survey Analysis
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

# -------------------------
# 1) Dateien laden
# -------------------------
DATA_FILE = "results-survey374736.csv"   # eure Antworten (25 x 74)
LABEL_FILE = "results-full.csv"          # eure "Spaltennamen detaillierter" (optional)

df = pd.read_csv(DATA_FILE)

# Optional: Labels laden (nur falls du später was nachschauen willst)
try:
    df_labels = pd.read_csv(LABEL_FILE)
except Exception:
    df_labels = None


# -------------------------
# 2) Kleine Helferfunktionen
# -------------------------
def require_columns(data: pd.DataFrame, cols: list, label: str) -> None:
    """Bricht mit verständlicher Fehlermeldung ab, wenn Spalten fehlen."""
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"[{label}] Fehlende Spalten: {missing}\n"
            f"Vorhandene Spalten (Auszug): {list(data.columns[:15])} ..."
        )

def row_mean(data: pd.DataFrame, cols: list, label: str) -> pd.Series:
    """Bildet den Zeilen-Mittelwert über mehrere Items."""
    require_columns(data, cols, label)
    # pd.to_numeric: falls mal Strings drin sind
    tmp = data[cols].apply(pd.to_numeric, errors="coerce")
    return tmp.mean(axis=1, skipna=True)

def cronbach_alpha(data: pd.DataFrame, cols: list, label: str) -> float:
    """
    Cronbachs Alpha (Reliabilität) für eine Skala.
    Funktioniert nur, wenn mind. 2 Items und genug nicht-missing Werte existieren.
    """
    require_columns(data, cols, label)
    X = data[cols].apply(pd.to_numeric, errors="coerce")

    # zu wenige Items
    if X.shape[1] < 2:
        return np.nan

    # Drop rows with all NaN
    X = X.dropna(how="all")
    if len(X) < 3:
        return np.nan

    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)

    if total_var == 0 or np.isnan(total_var):
        return np.nan

    k = X.shape[1]
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)

from typing import List, Tuple

def corr_pvalues(df_in, cols):
    """
    Gibt (corr_matrix, p_matrix) zurück für Pearson-Korrelation.
    """
    sub = df_in[cols].apply(pd.to_numeric, errors="coerce")

    corr = sub.corr(method="pearson")

    p = pd.DataFrame(np.ones_like(corr), columns=cols, index=cols, dtype=float)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                p.iloc[i, j] = np.nan
                continue
            x = sub.iloc[:, i]
            y = sub.iloc[:, j]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                p.iloc[i, j] = np.nan
            else:
                _, pv = pearsonr(x[valid], y[valid])
                p.iloc[i, j] = pv
    return corr, p


# -------------------------
# 3) Konstrukte definieren (echte Spalten aus euren CSVs)
# -------------------------

# Motivation: G04Q16[1..20]
# - Attention check: [9] (raus)
# - Amotivation-like items: [4], [11], [17] (nicht in autonomous/controlled reinrechnen)

autonomous_items = [
    "G04Q16[1]",   # personally important (identified)
    "G04Q16[3]",   # exciting (intrinsic)
    "G04Q16[7]",   # personal significance (identified)
    "G04Q16[8]",   # fun (intrinsic)
    "G04Q16[14]",  # interesting (intrinsic)
    "G04Q16[15]",  # personal values (identified)
]

controlled_items = [
    "G04Q16[2]",   # criticized by others (external)
    "G04Q16[5]",   # proud of myself (introjected)
    "G04Q16[6]",   # risk losing job (external)
    "G04Q16[10]",  # get others’ approval (external/introjected)
    "G04Q16[12]",  # prove to myself (introjected)
    "G04Q16[13]",  # show others I put effort (external)
    "G04Q16[16]",  # others respect me more (external/introjected)
    "G04Q16[18]",  # otherwise feel bad (introjected)
    "G04Q16[19]",  # (sehr ähnlich wie 13) -> vorsichtshalber drin lassen
    "G04Q16[20]",  # ashamed (introjected)
]

# Upskilling: G05Q18[1..5]
upskilling_items = [f"G05Q18[{i}]" for i in range(1, 6)]

# Reskilling: G05Q19[1..6]
reskilling_items = [f"G05Q19[{i}]" for i in range(1, 7)]

# Perceived Usefulness for IFBL: G03Q14[1..10]
# (Wenn ihr lieber "useful for job performance" wollt: G03Q13, aber da ist ein Attention Check [7].)
usefulness_items = [f"G03Q14[{i}]" for i in range(1, 11)]

# Kontrollvariablen:
age_col = "G02Q04"      # How old are you?
freq_work_col = "G03Q12"  # LLM frequency in work-related context


# -------------------------
# 4) Konstrukte berechnen
# -------------------------
df["autonomous_motivation"] = row_mean(df, autonomous_items, "Autonomous Motivation")
df["controlled_motivation"] = row_mean(df, controlled_items, "Controlled Motivation")
df["upskilling"] = row_mean(df, upskilling_items, "Upskilling")
df["reskilling"] = row_mean(df, reskilling_items, "Reskilling")
df["perceived_usefulness"] = row_mean(df, usefulness_items, "Perceived Usefulness (IFBL)")

# Alter und Frequenz numerisch machen
require_columns(df, [age_col, freq_work_col], "Controls")
df["age"] = pd.to_numeric(df[age_col], errors="coerce")
df["llm_use_freq_work"] = pd.to_numeric(df[freq_work_col], errors="coerce")

# Altersgruppen für RQ1 (18–35 vs 45+)
df["age_group_rq1"] = np.where(
    df["age"].between(18, 35, inclusive="both"), "younger_18_35",
    np.where(df["age"] >= 45, "older_45plus", np.nan)
)

# -------------------------
# 5) Mini-Check: Reliabilität (optional, aber sehr gut im Seminar)
# -------------------------
print("\n--- Cronbach Alpha (optional) ---")
print("Autonomous:", cronbach_alpha(df, autonomous_items, "Autonomous"))
print("Controlled:", cronbach_alpha(df, controlled_items, "Controlled"))
print("Upskilling:", cronbach_alpha(df, upskilling_items, "Upskilling"))
print("Reskilling:", cronbach_alpha(df, reskilling_items, "Reskilling"))
print("Usefulness:", cronbach_alpha(df, usefulness_items, "Usefulness"))

# -------------------------
# 6) Korrelationsanalyse (RQ2-Kern)
# -------------------------
vars_for_corr = [
    "upskilling",
    "reskilling",
    "autonomous_motivation",
    "controlled_motivation",
    "age",
    "llm_use_freq_work"
]

corr, pvals = corr_pvalues(df, vars_for_corr)

print("\n--- Pearson Correlations (r) ---")
print(corr.round(3))

print("\n--- p-values ---")
print(pvals.round(4))

# Heatmap (ohne seaborn, nur matplotlib)
plt.figure()
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(vars_for_corr)), vars_for_corr, rotation=45, ha="right")
plt.yticks(range(len(vars_for_corr)), vars_for_corr)
plt.colorbar(label="Pearson r")
plt.title("Correlation Matrix (Pearson r)")
plt.tight_layout()
plt.show()


# -------------------------
# 7) OPTIONAL: Regression (RQ2) - Kontrolle für Alter & Nutzungshäufigkeit
# -------------------------
# Beispiel: autonomous_motivation ~ upskilling + reskilling + age + llm_use_freq_work
reg_vars = ["upskilling", "reskilling", "age", "llm_use_freq_work"]

reg_df = df[["autonomous_motivation"] + reg_vars].dropna()
if len(reg_df) >= 8:  # grobe Mindestgröße, sonst wird's sehr instabil
    X = sm.add_constant(reg_df[reg_vars])
    y = reg_df["autonomous_motivation"]
    model = sm.OLS(y, X).fit()
    print("\n--- Regression: Autonomous Motivation ---")
    print(model.summary())
else:
    print("\n[Regression Autonomous] Zu wenige vollständige Fälle (nach dropna).")

# Beispiel: controlled_motivation ~ upskilling + reskilling + age + llm_use_freq_work
reg_df2 = df[["controlled_motivation"] + reg_vars].dropna()
if len(reg_df2) >= 8:
    X2 = sm.add_constant(reg_df2[reg_vars])
    y2 = reg_df2["controlled_motivation"]
    model2 = sm.OLS(y2, X2).fit()
    print("\n--- Regression: Controlled Motivation ---")
    print(model2.summary())
else:
    print("\n[Regression Controlled] Zu wenige vollständige Fälle (nach dropna).")


# -------------------------
# 8) OPTIONAL: RQ1 Unterschied zwischen Altersgruppen (nur als Extra)
# -------------------------
# Das ist KEINE Korrelation, sondern Gruppendifferenz. (Hilft aber fürs Seminar.)
rq1_vars = ["autonomous_motivation", "controlled_motivation", "perceived_usefulness"]

rq1_df = df[df["age_group_rq1"].isin(["younger_18_35", "older_45plus"])].copy()
print("\n--- RQ1 Deskriptiv nach Altersgruppe ---")
print(rq1_df.groupby("age_group_rq1")[rq1_vars].mean().round(3))
print("\nN pro Gruppe:")
print(rq1_df["age_group_rq1"].value_counts())
