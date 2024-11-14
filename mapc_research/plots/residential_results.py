from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mapc_research.plots.config import get_cmap

DISTANCE_MAP ={
    10: 0,
    20: 1,
    30: 2
}

LABELS_MAP = {
    "IdealDCF": "DCF",
    "IdealSR": "SR",
    "IdealFMAB": "MAB",
    "IdealHMAB": "H-MAB",
    "MinUB": "F-Optimal",
    "SumUB": "T-Optimal"
}

X_TICKS_LABELS = ["2x2", "2x3", "3x3", "3x4", "4x4"]


def clean_data(df: pd.DataFrame):

    # Drop columns that are not needed
    df.columns = ['Scenario', 'CSTX', 'DCF', 'IdealDCF', 'IdealSR', 'FMAB', 'IdealFMAB', 'HMAB', 'IdealHMAB', 'MinUB', 'SumUB']
    df = df.drop(['Scenario', 'CSTX', 'DCF', 'FMAB', 'HMAB',], axis=1)

    # The values are in the form of strings, so we need to convert them to floats.
    # But the floating point is represented by a comma, so we need to replace it with a dot.
    # Also, there are some values marked by `x` that are not valid, so we need to replace them with `NaN`.
    df = df.replace('x', np.nan)
    df = df.replace(',', '.', regex=True)
    df = df.astype(float)

    return df


def plot_for_distance(distance: float, df_mean: pd.DataFrame, df_low: pd.DataFrame, df_high: pd.DataFrame, results_dir: str):

    # Filter the dataframes by the distance
    modulo = DISTANCE_MAP[distance]
    df_mean_iter = df_mean.iloc[lambda x: x.index % 3 == modulo].reset_index(drop=True)
    df_low_iter = df_low.iloc[lambda x: x.index % 3 == modulo].reset_index(drop=True)
    df_high_iter = df_high.iloc[lambda x: x.index % 3 == modulo].reset_index(drop=True)

    # Setups for the plot
    colors = get_cmap(len(df_mean_iter.columns))
    fig, ax = plt.subplots()

    # Plot the data for each transmission mode
    xs = np.arange(5)
    barwidth = 0.12
    for i, (color, column) in enumerate(zip(colors, df_mean_iter.columns)):
        if LABELS_MAP[column] == "T-Optimal":
            ax.bar(xs, df_mean_iter[column], color="gray", width=5*barwidth, label=LABELS_MAP[column], alpha=0.5)

        else:
            ax.bar(xs + (i-2) * barwidth, df_mean_iter[column], color=color, width=barwidth, label=LABELS_MAP[column])
    for i, (color, column) in enumerate(zip(colors, df_mean_iter.columns)):
        if LABELS_MAP[column] == "T-Optimal":
            pass
        else:
            ax.bar(xs + (i-2) * barwidth, df_mean_iter[column], color=color, width=barwidth)

    # Set up the plot layout
    ax.set_xticks(range(5))
    ax.set_xticklabels(X_TICKS_LABELS)
    ax.set_xlabel("AP Grid Size")
    ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
    ax.legend(loc='upper left', fontsize=6, ncols=2)

    # REorder the legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=6, ncol=2)

    # Save the plot
    plt.savefig(os.path.join(results_dir, f"results_residential_d{distance}.pdf"), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--results_dir', type=str, default=f'results')
    args = args.parse_args()

    results_dir = args.results_dir

    # Load the data
    df_mean = clean_data(pd.read_csv(os.path.join(results_dir, "residential_mean.csv")))
    df_low = clean_data(pd.read_csv(os.path.join(results_dir, "residential_low.csv")))
    df_high = clean_data(pd.read_csv(os.path.join(results_dir, "residential_high.csv")))

    # Plot the data for each distance
    for distance in DISTANCE_MAP.keys():
        plot_for_distance(distance, df_mean, df_low, df_high, results_dir)