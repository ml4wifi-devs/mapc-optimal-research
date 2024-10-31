from dataclasses import dataclass
from chex import Array, Scalar
from typing import Dict, List

import os
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, TheilSenRegressor

from mapc_research.plots.config import set_style, get_cmap
from mapc_research.plots.utils import confidence_interval

RESULTS_PATH = "mapc_research/scalability/results"
DB_COLUMNS = ["start_timestamp", "n_sta_per_ap", "x_aps", "y_aps", "repetition", "seed", "time"]
CMAP = get_cmap(4)
COLOR_MAP = {
    "CBC": CMAP[1],
    "CPLEX": CMAP[3],
}
QUANTILE = 0.75
CONFIDENCE_INTERVAL = 0.95


def create_db(name: str) -> str:

    path = os.path.join(os.getcwd(), "mapc_research", "scalability", "database", name)
    path = path if path.endswith(".csv") else path + ".csv"

    if os.path.exists(path):
        return path
    
    db = open(path, "a")
    db.write(",".join(DB_COLUMNS) + "\n")
    db.close()

    return path

def plot_results(df: pd.DataFrame, n_aps_threshold: int, save_path: str, uncertainty: str = "quantile", with_outliers: bool = False):

    # Unpack results
    aps = df["n_aps"].unique()
    times_median = df.groupby('n_aps').median().reset_index()["time"].values
    times_count = df.groupby('n_aps').count().reset_index()["time"].values
    mask = times_count > 4

    # Calculate uncertainty
    if uncertainty == "quantile":
        # Calculate quantiles
        times_low = df.groupby('n_aps').apply(lambda x: x.quantile(1 - QUANTILE))["time"].values
        times_high = df.groupby('n_aps').apply(lambda x: x.quantile(QUANTILE))["time"].values
    elif uncertainty == "std":
        # Alternative: Use standard deviation
        times_low = df.groupby('n_aps').apply(lambda x: x.mean(axis=0) - x.std(axis=0))["time"].values
        times_high = df.groupby('n_aps').apply(lambda x: x.mean(axis=0) + x.std(axis=0))["time"].values
    elif uncertainty == "ci":
        # Alternative: Use confidence intervals
        times_low = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=CONFIDENCE_INTERVAL)[1])["time"].values
        times_high = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=CONFIDENCE_INTERVAL)[2])["time"].values
    else:
        raise ValueError("Invalid uncertainty type, choose from 'quantile', 'std', 'ci'")
    
    # Find IQR, whiskers and outliers
    iqr = times_high - times_low
    whiskers_low = times_low - 1.5*iqr
    whiskers_high = times_high + 1.5*iqr
    outliers = ([], [])
    for n_aps, low, high in zip(aps[mask], whiskers_low[mask], whiskers_high[mask]):
        outliers_mask = (df["n_aps"] == n_aps) & ((df["time"] < low) | (df["time"] > high))
        outliers[0].extend(df[outliers_mask]["n_aps"].values)
        outliers[1].extend(df[outliers_mask]["time"].values)

    # Fit curve
    model = TheilSenRegressor()
    df_fitted = df[df["n_aps"] >= n_aps_threshold]
    aps_fitted = df_fitted["n_aps"].values
    times_fitted = df_fitted["time"].values
    model.fit(aps_fitted.reshape(-1, 1), jnp.log(times_fitted))
    a, b = model.coef_[0], model.intercept_
    scale, exponent = jnp.exp(b), jnp.exp(a)

    # Define resolution for fitted curve
    xs = jnp.linspace(0, aps[-1], 100)

    # Plot results
    label = "Data"
        # Median
    plt.scatter(aps, times_median, marker="_", label=label, color="C0", s=20, linewidths=0.5)
        # Quantiles
    plt.fill_between(aps[mask], times_low[mask], times_high[mask], alpha=0.3, color="C0")
        # Whiskers
    # plt.scatter(aps[mask], whiskers_low[mask], color=COLOR_MAP[label], marker="_")
    # plt.scatter(aps[mask], whiskers_high[mask], color=COLOR_MAP[label], marker="_")
        # Outliers
    plt.scatter(outliers[0], outliers[1], color="C0", marker=".") if with_outliers else None
    plt.plot(
        xs, scale*jnp.power(exponent, xs),
        color="tab:grey", linestyle="--", linewidth=0.5, label=f"Fit"
    )
    plt.yscale("log")
    plt.xticks(list(range(0, 17, 2)))
    plt.xlabel("Number of access points")
    plt.ylabel("Execution time [s]")
    plt.legend()
    plt.title(f"Scale = {scale:.5f}, Exponent = {exponent:.5f}")
    plt.tight_layout()
    plt.savefig(save_path)


def plot_combined(dfs: List[pd.DataFrame], labels: List[str], n_aps_thresholds: List[int], save_path: str, uncertainty: str = "quantile", with_outliers: bool = False):

    # Setup figure
    set_style()
    plt.figure(figsize=(4,3))

    for i, (df, label, n_aps_threshold) in enumerate(zip(dfs, labels, n_aps_thresholds)):

        # Unpack results
        aps = df["n_aps"].unique()
        times_median = df.groupby('n_aps').median().reset_index()["time"].values
        times_count = df.groupby('n_aps').count().reset_index()["time"].values
        mask = times_count > 4

            # Calculate uncertainty
        if uncertainty == "quantile":
            # Calculate quantiles
            times_low = df.groupby('n_aps').apply(lambda x: x.quantile(1 - QUANTILE))["time"].values
            times_high = df.groupby('n_aps').apply(lambda x: x.quantile(QUANTILE))["time"].values
        elif uncertainty == "std":
            # Alternative: Use standard deviation
            times_low = df.groupby('n_aps').apply(lambda x: x.mean(axis=0) - x.std(axis=0))["time"].values
            times_high = df.groupby('n_aps').apply(lambda x: x.mean(axis=0) + x.std(axis=0))["time"].values
        elif uncertainty == "ci":
            # Alternative: Use confidence intervals
            times_low = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=CONFIDENCE_INTERVAL)[1])["time"].values
            times_high = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=CONFIDENCE_INTERVAL)[2])["time"].values
        else:
            raise ValueError("Invalid uncertainty type, choose from 'quantile', 'std', 'ci'")
        
        # Find IQR, whiskers and outliers
        iqr = times_high - times_low
        whiskers_low = times_low - 1.5*iqr
        whiskers_high = times_high + 1.5*iqr
        outliers = ([], [])
        for n_aps, low, high in zip(aps[mask], whiskers_low[mask], whiskers_high[mask]):
            outliers_mask = (df["n_aps"] == n_aps) & ((df["time"] < low) | (df["time"] > high))
            outliers[0].extend(df[outliers_mask]["n_aps"].values)
            outliers[1].extend(df[outliers_mask]["time"].values)

        # Fit curve
        model = TheilSenRegressor()
        df_fitted = df[df["n_aps"] >= n_aps_threshold]
        aps_fitted = df_fitted["n_aps"].values
        times_fitted = df_fitted["time"].values
        model.fit(aps_fitted.reshape(-1, 1), jnp.log(times_fitted))
        a, b = model.coef_[0], model.intercept_
        scale, exponent = jnp.exp(b), jnp.exp(a)

        # Define resolution for fitted curve
        xs = jnp.linspace(0, aps[-1], 100)

        # Plot results
            # Just for the legend
        latex_label = f"{label}:\n" + r"$$\alpha =  {:.2f}, \beta = {:.2f}$$".format(scale, exponent)
        plt.plot([], [], color=COLOR_MAP[label], linewidth=1., label=latex_label)
            # Median
        # plt.scatter(aps, times_median, marker="_", color=COLOR_MAP[label], s=20, linewidths=0.5)
            # Quantiles
        # plt.fill_between(aps[mask], times_low[mask], times_high[mask], alpha=0.3, color=COLOR_MAP[label])
            # Whiskers
        # plt.scatter(aps[mask], whiskers_low[mask], color=COLOR_MAP[label], marker="_")
        # plt.scatter(aps[mask], whiskers_high[mask], color=COLOR_MAP[label], marker="_")
            # Outliers
        # plt.scatter(outliers[0], outliers[1], color=COLOR_MAP[label], marker=".") if with_outliers else None

        sns.boxplot(
            data=df, x='n_aps', y='time', order=jnp.arange(len(df['n_aps'])),
            color=COLOR_MAP[label], showfliers=True, linewidth=0.5,
            flierprops=dict(markeredgecolor=COLOR_MAP[label], markersize=1)
        )
        plt.plot(
            xs.tolist(), scale*jnp.power(exponent, xs),
            color="tab:grey", linestyle="--", linewidth=0.5, label=f"Theil-Sen regression" if label == labels[-1] else None
        )
    
    plt.yscale("log")
    plt.xticks(list(range(0, 17, 2)))
    plt.xlabel("Number of access points")
    plt.ylabel("Execution time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


@dataclass
class ExperimentResult:
    config: Dict
    aps: Array
    times_mean: Array
    times_std_low: Array
    times_std_high: Array
    total_time: Scalar
    scale: Scalar
    exponent: Scalar