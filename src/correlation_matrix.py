import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def _fig(df, title: str = None) -> plt.Figure:
    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax = sns.heatmap(data=df, annot=True, cmap="crest", square=True, vmin=0, vmax=1)
    plt.xticks(rotation=45)
    fig.tight_layout()

    plt.show()


def calc_correlation(df: pd.DataFrame, save_fig=False):
    corr_matrix_young = df.loc[
        df["young_group"] == 1, df.columns != "young_group"
    ].corr(method="pearson")

    corr_matrix_old = df.loc[df["young_group"] == 0, df.columns != "young_group"].corr(
        method="pearson"
    )

    if save_fig:
        _fig(corr_matrix_young, title="Young correlation")
        _fig(corr_matrix_old, title="Old correlation")

    return corr_matrix_young, corr_matrix_old
