import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def _fig(df, file_name: str, title: str = None):
    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax = sns.heatmap(data=df, annot=True, cmap="crest", square=True, vmin=0, vmax=1)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    fig.tight_layout()

    # plt.show()
    plt.savefig(file_name)


def calc_correlation_motivation_skilling(
    df: pd.DataFrame, save_fig=False, fig_title: str = None
):
    if save_fig and fig_title is None:
        raise ValueError("fig_title is required for Figure")

    corr_matrix = df.corr(method="pearson")
    corr_matrix = corr_matrix.drop(
        columns=["controlled_motivation", "autonomous_motivation"], axis="1"
    )
    corr_matrix = corr_matrix.drop(index=["upskilling", "reskilling"], axis="0")
    if save_fig:
        _fig(
            corr_matrix,
            title=fig_title,
            file_name=f"figures/{fig_title}.png",
        )

    return corr_matrix


def calc_correlation(df: pd.DataFrame, save_fig=False, fig_title: str = None):
    if save_fig and fig_title is None:
        raise ValueError("fig_title is required for Figure")

    corr_matrix = df.corr(method="pearson")
    if save_fig:
        _fig(
            corr_matrix,
            title=fig_title,
            file_name=f"figures/{fig_title}.png",
        )

    return corr_matrix
