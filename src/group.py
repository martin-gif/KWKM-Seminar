import pandas as pd
import pingouin as pg

SCALES = {
    "usefulness_work": [
        "G03Q13[1]",
        "G03Q13[2]",
        "G03Q13[3]",
        "G03Q13[4]",
        "G03Q13[5]",
        "G03Q13[6]",
        "G03Q13[8]",
        "G03Q13[9]",
        "G03Q13[10]",
        "G03Q13[11]",
    ],
    "usefulness_learning": [
        "G03Q14[1]",
        "G03Q14[2]",
        "G03Q14[3]",
        "G03Q14[4]",
        "G03Q14[5]",
        "G03Q14[6]",
        "G03Q14[7]",
        "G03Q14[8]",
        "G03Q14[9]",
        "G03Q14[10]",
    ],
    "mot_intrinsic": ["G04Q16[3]", "G04Q16[8]", "G04Q16[14]"],
    "mot_ext_social": ["G04Q16[2]", "G04Q16[10]", "G04Q16[16]"],
    "upskilling": ["G05Q18[1]", "G05Q18[2]", "G05Q18[3]", "G05Q18[4]", "G05Q18[5]"],
    "reskilling": [
        "G05Q19[1]",
        "G05Q19[2]",
        "G05Q19[3]",
        "G05Q19[4]",
        "G05Q19[5]",
        "G05Q19[6]",
    ],
    "young_group": [
        "young_group",
    ],
}


def group_data(input_df, print_cronbach=False) -> pd.DataFrame:
    all_items = [item for items in SCALES.values() for item in items]
    input_df[all_items] = input_df[all_items].apply(pd.to_numeric, errors="coerce")

    df = pd.DataFrame()
    for key, column_list in SCALES.items():
        if print_cronbach:
            cronbach = pg.cronbach_alpha(data=input_df[column_list])
            print(f"Cronbach's Alpha für group {key} = {round(cronbach[0], 3)}")
        df[key] = input_df[column_list].mean(axis=1)

    # print(df)
    return df
