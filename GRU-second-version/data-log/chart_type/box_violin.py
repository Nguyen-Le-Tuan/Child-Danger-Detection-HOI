# eda/chart_type/box_violin.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_box_violin(df, feature, label_col):
    plt.figure(figsize=(6,5))
    sns.violinplot(
        x=label_col,
        y=feature,
        data=df,
        inner="box"
    )

    plt.xticks([0,1], ["Safe", "Danger"])
    plt.title(f"{feature}: Safe vs Danger")
    plt.grid(True)
