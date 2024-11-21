import json

import numpy as np
import matplotlib.pyplot as plt
from mapc_sim.constants import DATA_RATES, TAU

from mapc_research.plots.config import COLUMN_WIDTH, COLUMN_HEIGHT, AGENT_COLORS
from mapc_research.plots.utils import confidence_interval


plt.rcParams.update({
    'figure.figsize': (3 * COLUMN_WIDTH, COLUMN_HEIGHT + 0.7),
    'legend.fontsize': 9
})

N_STEPS = [600, 1000, 3000]
AGGREGATE_STEPS = [20, 25, 100]
SWITCH_STEPS = [None, 500, 1500]
TITLES = [r"(a) $d=10$ m", r"(b) $d=20$ m", r"(c) $d=30$ m"]
H_MAB_IDX = 5
MAB_IDX = 4


if __name__ == '__main__':
    with open('../mab/small_office_mab_results.json') as file:
        mab_results = json.load(file)

    dcf_results = []

    with open('oracle/small_office/static_small_office_10.0_2.0.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('oracle/small_office/dynamic_static_small_office_20.0_2.0_a.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('oracle/small_office/dynamic_static_small_office_20.0_2.0_b.json') as file:
        dcf_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('oracle/small_office/dynamic_static_small_office_30.0_2.0_a.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('oracle/small_office/dynamic_static_small_office_30.0_2.0_b.json') as file:
        dcf_results[-1].append(json.load(file)['DataRate']['Mean'])

    sr_results = []

    with open('sr/small_office/static_small_office_10.0_2.0.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/small_office/dynamic_static_small_office_20.0_2.0_a.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/small_office/dynamic_static_small_office_20.0_2.0_b.json') as file:
        sr_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('sr/small_office/dynamic_static_small_office_30.0_2.0_a.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/small_office/dynamic_static_small_office_30.0_2.0_b.json') as file:
        sr_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('../mab/mean_optimal_results.json') as file:
        optimal_results = json.load(file)

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.)

    for i, (ax, scenario) in enumerate(zip(axes, mab_results)):
        scenario_results = [
            [np.array(run).reshape((-1, AGGREGATE_STEPS[i])).mean(axis=-1) for run in agent_results['runs']]
            for agent_results in scenario
        ]

        xs = np.linspace(0, N_STEPS[i], N_STEPS[i] // AGGREGATE_STEPS[i]) * TAU

        if SWITCH_STEPS[i] is not None:
            ax.axvline(SWITCH_STEPS[i] * TAU, linestyle='--', color='gray')

            xs_first, xs_sec = xs[:SWITCH_STEPS[i] // AGGREGATE_STEPS[i]], xs[SWITCH_STEPS[i] // AGGREGATE_STEPS[i]:]
            x_mid = (xs_first[-1] + xs_sec[0]) / 2
            xs_first = np.concatenate((xs_first, [x_mid]))
            xs_sec = np.concatenate(([x_mid], xs_sec))

            ax.plot(xs_first, len(xs_first) * [dcf_results[i][0]], c=AGENT_COLORS['DCF'])
            ax.plot(xs_sec, len(xs_sec) * [dcf_results[i][1]], c=AGENT_COLORS['DCF'])
            ax.plot(xs_first, len(xs_first) * [sr_results[i][0]], c=AGENT_COLORS['SR'])
            ax.plot(xs_sec, len(xs_sec) * [sr_results[i][1]], c=AGENT_COLORS['SR'])
            ax.plot(xs_first, len(xs_first) * [optimal_results[i][0]['runs'][0]], c=AGENT_COLORS['T-Optimal'])
            ax.plot(xs_sec, len(xs_sec) * [optimal_results[i][0]['runs'][1]], c=AGENT_COLORS['T-Optimal'])
        else:
            ax.plot(xs, len(xs) * [dcf_results[i][0]], c=AGENT_COLORS['DCF'])
            ax.plot(xs, len(xs) * [sr_results[i][0]], c=AGENT_COLORS['SR'])
            ax.plot(xs, len(xs) * [optimal_results[i][0]['runs'][0]], c=AGENT_COLORS['T-Optimal'])

        for j, data in enumerate(scenario_results):
            mean, ci_low, ci_high = confidence_interval(np.asarray(data))

            if j == H_MAB_IDX:
                ax.plot(xs, mean, c=AGENT_COLORS['H-MAB'], marker='o')
                ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=AGENT_COLORS['H-MAB'], linewidth=0.0)
            elif j == MAB_IDX:
                ax.plot(xs, mean, c=AGENT_COLORS['MAB'], marker='o')
                ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=AGENT_COLORS['MAB'], linewidth=0.0)

        ax.set_title(TITLES[i], y=-0.45, fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=600)
        ax.set_yticks(range(0, 601, 100))
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.grid()

        if i == 0:
            ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
            ax.plot([], [], marker='o', c=AGENT_COLORS['MAB'], label='MAB')
            ax.plot([], [], marker='o', c=AGENT_COLORS['H-MAB'], label='H-MAB')
            ax.legend(loc='upper left')

            ax2 = ax.twinx()
            ax2.axis('off')
            ax2.plot([], [], c=AGENT_COLORS['DCF'], label='DCF (average)')
            ax2.plot([], [], c=AGENT_COLORS['SR'], label='SR (average)')
            ax2.plot([], [], c=AGENT_COLORS['T-Optimal'], label='T-Optimal')
            ax2.legend(loc='upper right', title='Baselines')

    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.show()
