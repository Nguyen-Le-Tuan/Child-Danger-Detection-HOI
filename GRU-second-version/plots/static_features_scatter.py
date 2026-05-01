import matplotlib.pyplot as plt

def distance_velocity_status(
    df,
    distance_col='distance',
    velocity_col='velocity',
    label_col='status_label'
):
    """
    Vẽ scatter plot phân bố Distance - Velocity,
    phân biệt trạng thái Safe / Danger.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu
    distance_col : str
        Tên cột distance
    velocity_col : str
        Tên cột velocity
    label_col : str
        Tên cột label (0 = safe, 1 = danger)
    """

    # Tách dữ liệu theo nhãn
    safe_df = df[df[label_col] == 0]
    danger_df = df[df[label_col] == 1]

    plt.figure(figsize=(8, 6))

    # Safe points
    plt.scatter(
        safe_df[distance_col],
        safe_df[velocity_col],
        alpha=0.4,
        label='Safe',
        marker='o'
    )

    # Danger points
    plt.scatter(
        danger_df[distance_col],
        danger_df[velocity_col],
        alpha=0.6,
        label='Danger',
        marker='x'
    )

    # Labels & title
    plt.xlabel('Distance')
    plt.ylabel('Velocity')
    plt.title('Distribution of Distance and Velocity\n(Safe vs Danger)')

    # Legend & grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
