import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from mapc_sim.constants import DATA_RATES, TAU

from mapc_research.plots.config import COLUMN_WIDTH, COLUMN_HEIGHT, get_cmap
from mapc_research.plots.utils import confidence_interval


plt.rcParams.update({
    'figure.figsize': (3 * COLUMN_WIDTH, COLUMN_HEIGHT + 0.7),
    'legend.fontsize': 9
})

AGENT_NAMES = ["C-Single TX"] + 2 * [r"$\varepsilon$-greedy"] + 2 * ["Softmax"] + 2 * ["UCB"] + 2 * ["TS"]
HIERARCHICAL = [False] + 4 * [True, False]

N_STEPS = [600, 600, 3000]
AGGREGATE_STEPS = [20, 20, 100]
SWITCH_STEPS = [None, 300, 1500]
TITLES = [r"(a) $d=10$ m", r"(b) $d=20$ m", r"(c) $d=30$ m"]
CLASSIC_MAB = [4, 4, 4]


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-f', '--file', type=str, default=f'../mab/small_office_results.json')
    args = args.parse_args()

    with open(args.file, 'r') as file:
        results = json.load(file)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.)
    colors = get_cmap(4)

    for i, (ax, scenario) in enumerate(zip(axes, results)):
        scenario_results = [
            [np.array(run).reshape((-1, AGGREGATE_STEPS[i])).mean(axis=-1) for run in agent_results['runs']]
            for agent_results in scenario
        ]

        xs = np.linspace(0, N_STEPS[i], N_STEPS[i] // AGGREGATE_STEPS[i]) * TAU

        if SWITCH_STEPS[i] is not None:
            ax.axvline(SWITCH_STEPS[i] * TAU, linestyle='--', color='red')

        for j, data in enumerate(scenario_results):
            mean, ci_low, ci_high = confidence_interval(np.asarray(data))

            if j == 0:
                ax.plot(xs, mean, linestyle='--', c='gray')
            elif HIERARCHICAL[j]:
                ax.plot(xs, mean, c=colors[(j - 1) // 2], marker='o')
                ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=colors[(j - 1) // 2], linewidth=0.0)
            elif j == CLASSIC_MAB[i]:
                ax.plot(xs, mean, linestyle='--', marker='^', c='gray', markersize=2)
                ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color='gray', linewidth=0.0)

        ax.set_title(TITLES[i], y=-0.45, fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=550)
        ax.set_yticks(range(0, 501, 100))
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.grid()

        if i == 0:
            ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
            ax.plot([], [], 'o', linestyle='-', c=colors[0], label=r'$\varepsilon$-greedy')
            ax.plot([], [], 'o', linestyle='-', c=colors[1], label='Softmax')
            ax.plot([], [], 'o', linestyle='-', c=colors[2], label='UCB')
            ax.plot([], [], 'o', linestyle='-', c=colors[3], label='TS')
            ax.legend(loc='upper left', title='Hierarchical MABs')

            ax2 = ax.twinx()
            ax2.axis('off')
            ax2.plot([], [], linestyle='--', c="gray", label='C-Single TX')
            ax2.plot([], [], '^', linestyle='--', c='gray', label='Best flat MAB')
            ax2.legend(loc='upper right', title='Baselines')

    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.show()
