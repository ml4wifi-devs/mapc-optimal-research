import json
from collections import defaultdict
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mapc_research.envs.test_scenarios import INDOOR_SCENARIOS
from mapc_research.plots.config import AGENT_COLORS

DISTANCE_MAP = {
    10: 0,
    20: 1,
    30: 2
}
X_TICKS_LABELS = ["2x2", "2x3", "3x3", "3x4", "4x4"]


def plot_for_distance(distance, fairness_index):
    mean_iter_list = [fairness_index[i] for i in range(DISTANCE_MAP[distance], len(fairness_index), 3)]
    mean_iter = mean_iter_list[0]

    for i in range(1, len(mean_iter_list)):
        for key in mean_iter_list[i].keys():
            mean_iter[key] = np.hstack((mean_iter[key], mean_iter_list[i][key]))

    for k in mean_iter.keys():
        mean_iter[k][np.where(mean_iter[k] == None)] = -0.05

    fig, ax = plt.subplots()
    xs = np.arange(5)
    barwidth = 0.12

    for i, column in enumerate(mean_iter.keys()):
        if column == "F-Optimal":
            ax.bar(xs, mean_iter[column], color="gray", width=5 * barwidth, label=column, alpha=0.5, zorder=0)
        elif column == "T-Optimal":
            ax.bar(xs + (i - 3) * barwidth, mean_iter[column], color=AGENT_COLORS[column], width=barwidth, label=column)
        else:
            ax.bar(xs + (i - 2) * barwidth, mean_iter[column], color=AGENT_COLORS[column], width=barwidth, label=column)

    ax.set_xticks(range(5))
    ax.set_xticklabels(X_TICKS_LABELS)
    ax.set_xlabel("AP Grid Size")
    ax.set_ylabel('Jain\'s Fairness Index')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', ncols=3, bbox_to_anchor=(0.06, 0.975))
    ax.grid(axis='y', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'fairness_residential_d{d}.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()


def jains_fairness_index(data_rate):
    if (n := len(data_rate)) == 0:
        return None
    return (np.sum(data_rate) ** 2) / (n * np.sum(data_rate ** 2))


if __name__ == '__main__':
    with open('../mab/node_thr_h_mab.json') as f:
        mab_h = json.load(f)

    with open('../mab/node_thr_f_mab.json') as f:
        mab_f = json.load(f)

    with open('../upper_bound/all_results.json') as f:
        optimal = json.load(f)

    all_results = []

    for i, scenario in tqdm(enumerate(INDOOR_SCENARIOS)):
        stas = np.asarray(list(chain.from_iterable(scenario.associations.values())))

        mab_h_thr = np.asarray(mab_h[i]).mean(axis=(0, 1))[stas]
        mab_f_thr = np.asarray(mab_f[i]).mean(axis=(0, 1))[stas] if len(mab_f[i]) > 0 else np.asarray([])
        t_optimal_thr = defaultdict(float)
        f_optimal_thr = defaultdict(float)

        for conf, weight in optimal[i][0]['shares'][0].items():
            for link, rate in optimal[i][0]['link_rates'][0][conf].items():
                t_optimal_thr[link] += rate * weight
        t_optimal_thr = np.asarray(list(t_optimal_thr.values()))

        for conf, weight in optimal[i][1]['shares'][0].items():
            for link, rate in optimal[i][1]['link_rates'][0][conf].items():
                f_optimal_thr[link] += rate * weight
        f_optimal_thr = np.asarray(list(f_optimal_thr.values()))

        all_results.append({
            'MAB': jains_fairness_index(mab_f_thr),
            'H-MAB': jains_fairness_index(mab_h_thr),
            'T-Optimal': jains_fairness_index(t_optimal_thr),
            'F-Optimal': jains_fairness_index(f_optimal_thr),
        })

    with open('fairness_index.json', 'w') as f:
        json.dump(all_results, f)

    for d in DISTANCE_MAP.keys():
        plot_for_distance(d, all_results)
