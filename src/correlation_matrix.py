import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def _fig(df, file_name: str, title: str = None, vmin: float = 0, vmax: float = 1):
    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    df = df.astype(float)

    sns.heatmap(
        data=df, annot=True, cmap="crest", square=True, vmin=vmin, vmax=vmax, ax=ax
    )
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

    cols = pd.DataFrame(columns=df.columns).sort_index(axis=1)
    r_matrix = cols.transpose().join(cols, how="outer")
    p_matrix = cols.transpose().join(cols, how="outer")

    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            r_val, p_val = pearsonr(tmp[r], tmp[c])
            r_matrix.loc[r, c] = round(r_val, 4)
            p_matrix.loc[r, c] = p_val

    # For correlation Matrix
    r_matrix = r_matrix.drop(
        columns=["controlled_motivation", "autonomous_motivation"], axis="1"
    )
    r_matrix = r_matrix.drop(index=["upskilling", "reskilling"], axis="0")

    # For P-Value Matrix
    p_matrix = p_matrix.drop(
        columns=["controlled_motivation", "autonomous_motivation"], axis="1"
    )
    p_matrix = p_matrix.drop(index=["upskilling", "reskilling"], axis="0")

    if save_fig:
        _fig(
            r_matrix,
            title=fig_title,
            file_name=f"figures/{fig_title}_corrMatrix.png",
        )
        _fig(
            p_matrix,
            title=fig_title,
            file_name=f"figures/{fig_title}_pMatrix.png",
            vmax=None,
            vmin=None,
        )

    return r_matrix


def calc_correlation(
    df: pd.DataFrame,
    save_fig=False,
    fig_title_correlation: str = None,
    fig_title_pValue: str = None,
):
    if save_fig and fig_title_correlation is None:
        raise ValueError("fig_title is required for Figure")

    cols = pd.DataFrame(columns=df.columns).sort_index(axis=1)
    r_matrix = cols.transpose().join(cols, how="outer")
    p_matrix = cols.transpose().join(cols, how="outer")

    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            r_val, p_val = pearsonr(tmp[r], tmp[c])
            r_matrix.loc[r, c] = round(r_val, 4)
            p_matrix.loc[r, c] = round(p_val, 4)

    if fig_title_correlation is not None and fig_title_pValue is None:
        fig_title_pValue = fig_title_correlation

    if save_fig:
        _fig(
            r_matrix,
            title=fig_title_correlation,
            file_name=f"figures/{fig_title_correlation}_corrMatrix.png",
        )
        _fig(
            p_matrix,
            title=fig_title_pValue,
            file_name=f"figures/{fig_title_pValue}_pMatrix.png",
        )

    return r_matrix
