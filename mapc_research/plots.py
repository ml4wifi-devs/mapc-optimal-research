import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array
from scipy.stats import t


COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + 5 ** 0.5)

PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 5.5,
    'lines.linewidth': 1.,
    'lines.markersize': 2,
    'patch.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


def set_style() -> None:
    plt.rcParams.update(PLOT_PARAMS)


def get_cmap(n: int) -> plt.cm:
    return plt.cm.viridis(jnp.linspace(0., 0.75, n))


def confidence_interval(data: Array, ci: float = 0.99) -> tuple:
    measurements = data.shape[0]
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    alpha = 1 - ci
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / jnp.sqrt(measurements)
    ci_high = mean + z * std / jnp.sqrt(measurements)

    return mean, ci_low, ci_high
