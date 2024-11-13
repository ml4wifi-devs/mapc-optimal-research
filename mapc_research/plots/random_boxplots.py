import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mapc_research.plots.config import get_cmap, PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HEIGHT


if __name__ == "__main__":
    PLOT_PARAMS['figure.figsize'] = (1.1 * COLUMN_WIDTH, 1.3 * COLUMN_HEIGHT)
    PLOT_PARAMS['figure.dpi'] = 300
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv("random_results.csv")
    dcf = df.pop('DCF')
    df = df.div(dcf / 100, axis=0)

    plt.axhline(100, color='gray', linestyle='--', label='DCF', linewidth=0.5)
    sns.boxplot(
        data=pd.melt(df),
        x='variable', y='value', hue='variable',
        palette=get_cmap(6).tolist(),
        width=0.5,
        boxprops=dict(linewidth=0.5),
        whiskerprops=dict(linewidth=0.5),
        medianprops=dict(linewidth=0.5, color='k'),
        capprops=dict(linewidth=0.5),
        flierprops=dict(marker='o', markersize=2, markeredgecolor='k', markerfacecolor='k'),
    )

    plt.xlabel('')
    plt.xticks(rotation=30)
    plt.ylabel(r'Relative improvement over DCF [\%]')
    plt.ylim(0, 350)

    plt.grid(axis='y', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('random_boxplot.pdf', bbox_inches='tight')
    plt.show()
