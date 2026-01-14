import statsmodels.api as sm
import pandas as pd


def linear_regression(df_X: pd.DataFrame, df_Y: pd.DataFrame, print_summary=False):
    X = sm.add_constant(df_X)
    model = sm.OLS(df_Y, X).fit()
    summary = model.summary()
    if print_summary:
        print(summary)
