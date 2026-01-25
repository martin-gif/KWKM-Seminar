import pandas as pd
import pingouin as pg

SCALES = {
    "usage": [
        "G03Q12",
    ],
    "age": ["G02Q04"],
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
    "external_regulation_material": ["G04Q16[13]", "G04Q16[6]", "G04Q16[19]"],
    "external_regulation_social": ["G04Q16[10]", "G04Q16[16]", "G04Q16[2]"],
    "introjected_regulation": ["G04Q16[12]", "G04Q16[5]", "G04Q16[18]", "G04Q16[20]"],
    "intrinsic_regulation": ["G04Q16[8]", "G04Q16[3]", "G04Q16[14]"],
    "identified_regulation": ["G04Q16[1]", "G04Q16[15]", "G04Q16[7]"],
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
    unique_items = list(dict.fromkeys(all_items))
    # cols = [c for c in unique_items if c in input_df.columns]
    # input_df[all_items] = input_df[all_items].apply(pd.to_numeric, errors="coerce")
    input_df.loc[:, all_items] = input_df.loc[:, all_items].apply(
        pd.to_numeric, errors="coerce"
    )

    df = pd.DataFrame()
    for key, column_list in SCALES.items():
        if print_cronbach:
            if len(column_list) > 1:
                cronbach = pg.cronbach_alpha(data=input_df[column_list])
                print(
                    f"Cronbach's Alpha für group {key} = {round(cronbach[0], 3)}, mit der grenze {cronbach[1]}"
                )
        df[key] = input_df[column_list].mean(axis=1)

    ext_mat = "external_regulation_material"
    ext_soc = "external_regulation_social"
    introj = "introjected_regulation"
    intrin = "intrinsic_regulation"
    ident = "identified_regulation"

    df["controlled_motivation"] = (((df[ext_mat] + df[ext_soc]) / 2) + df[introj]) / 2

    df["autonomous_motivation"] = (df[intrin] + df[ident]) / 2

    # print(df)
    return df
