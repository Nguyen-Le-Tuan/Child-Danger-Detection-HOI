# eda/data_loader.py

import glob
import pandas as pd

def load_all_danger_data(root_dir=".", pattern="danger*/merged_danger*.csv"):
    paths = glob.glob(f"{root_dir}/{pattern}")
    dfs = []

    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = p
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    return data
