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


def calc_correlation(df: pd.DataFrame, save_fig=False, split_age_groups=False):
    conditions = []
    if split_age_groups:
        conditions.append(
            df["young_group"] == 1
        )  # split into young group and older group
        conditions.append(df["young_group"] == 0)
    else:
        conditions.append(df["young_group"] <= 2)  # take all

    return_matrices = []

    for index, condition in enumerate(conditions):
        corr_matrix = df.loc[condition, df.columns != "young_group"].corr(
            method="pearson"
        )
        return_matrices.append(corr_matrix)

        if save_fig:
            _fig(
                corr_matrix,
                title=f"correlation group {index}",
                file_name=f"figures/corr_matrix_group_{index}.png",
            )

    return return_matrices
