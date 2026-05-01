# eda/plots/scatter.py

import matplotlib.pyplot as plt

def plot_distance_velocity(df, label_col):
    safe = df[df[label_col] == 0]
    danger = df[df[label_col] == 1]

    plt.figure(figsize=(7,6))
    plt.scatter(
        safe["distance"], safe["velocity"],
        alpha=0.4, label="Safe"
    )
    plt.scatter(
        danger["distance"], danger["velocity"],
        alpha=0.4, label="Danger"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Distance–Velocity distribution")
    plt.legend()
    plt.grid(True)
