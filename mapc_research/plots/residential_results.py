from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mapc_research.plots.config import AGENT_COLORS

DISTANCE_MAP ={
    10: 0,
    20: 1,
    30: 2
}

LABELS_MAP = {
    "DCF + Ideal MCS": "DCF",
    "SR -72 dBM": "SR",
    "MAB [F] + ideal MCS": "MAB",
    "MAB [H] + Ideal MCS": "H-MAB",
    "Upper bound [min]": "F-Optimal",
    "Upper bound [sum]": "T-Optimal"
}

X_TICKS_LABELS = ["2x2", "2x3", "3x3", "3x4", "4x4"]


def clean_data(df: pd.DataFrame):

    # Drop columns that are not needed
    old_columns = df.columns
    new_columns = ["DCF + Ideal MCS", "SR -72 dBM", "MAB [F] + ideal MCS", "MAB [H] + Ideal MCS", "Upper bound [min]", "Upper bound [sum]"]
    df = df.drop(columns=[c for c in old_columns if c not in new_columns], axis=1)
    
    # Rename the columns
    df = df.rename(columns=LABELS_MAP)

    # The values are in the form of strings, so we need to convert them to floats.
    # But the floating point is represented by a comma, so we need to replace it with a dot.
    # Also, there are some values marked by `x` that are not valid, so we need to replace them with `NaN`.
    df = df.replace('x', '-1000,0')
    df = df.replace(',', '.', regex=True)
    df = df.astype(float)

    return df


def plot_for_distance(distance: float, df_mean: pd.DataFrame, results_path: str):

    # Filter the dataframes by the distance
    modulo = DISTANCE_MAP[distance]
    df_mean_iter = df_mean.iloc[lambda x: x.index % 3 == modulo].reset_index(drop=True)

    # Setups for the plot
    fig, ax = plt.subplots()

    # Plot the data for each transmission mode
    xs = np.arange(5)
    barwidth = 0.12
    for i, column in enumerate(df_mean_iter.columns):
        if column == "T-Optimal":
            ax.bar(xs, df_mean_iter[column], color="gray", width=5*barwidth, label=column, alpha=0.5)

        else:
            ax.bar(xs + (i-2) * barwidth, df_mean_iter[column], color=AGENT_COLORS[column], width=barwidth, label=column)
    for i, column in enumerate(df_mean_iter.columns):
        if column == "T-Optimal":
            pass
        else:
            ax.bar(xs + (i-2) * barwidth, df_mean_iter[column], color=AGENT_COLORS[column], width=barwidth)

    # Add the 0 line
    ax.axhline(0, color='black', linewidth=0.5)

    # Add a table with convergance times
    columns = X_TICKS_LABELS
    rows = ["MAB", "H-MAB"]
    convergence_data = [
        ["2.96 s", "7.24 s", "N/A", "N/A", "N/A"],
        ["3.62 s", "9.87 s", "78.97 s", "658.08 s", "inf"]
    ]
    the_table = ax.table(
        cellText=convergence_data,
        rowLabels=rows,
        rowColours=[AGENT_COLORS["MAB"], AGENT_COLORS["H-MAB"]],
        colLabels=columns,
        loc='bottom',
        cellLoc='center',
        colLoc='center',
    )

    # - Set linewidth for table borders
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(0.5)
    the_table.set_fontsize(9)

    # Set up the plot layout
    ax.set_xticks([])
    ax.set_ylim(-30, ax.get_ylim()[1])
    ax.set_ylabel('Effective data rate [Mb/s]')
    ax.legend(loc='upper left', fontsize=6, ncols=2)

    # Reorder the legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=6, ncol=2)

    # Save the plot
    save_path = results_path.replace(".csv", f"_{distance}.pdf")
    plt.grid(axis='y')
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-r', '--results_path', type=str, required=True)
    args = args.parse_args()

    # Load the data
    df_mean = clean_data(pd.read_csv(args.results_path))

    # Plot the data for each distance
    for distance in DISTANCE_MAP.keys():
        plot_for_distance(distance, df_mean, args.results_path)