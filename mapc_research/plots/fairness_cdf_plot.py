import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from collections import defaultdict
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mapc_sim.constants import TAU

from mapc_research.envs.scenario_impl import residential_scenario
from mapc_research.plots.config import AGENT_COLORS


N_RUNS = 10
N_STEPS = 600
SIM_TIME = N_STEPS * TAU


if __name__ == '__main__':
    scenario_idx = 4
    scenario = residential_scenario(seed=104, n_steps=3000, x_apartments=2, y_apartments=3, n_sta_per_ap=4, size=20.0)
    stas = np.asarray(list(chain.from_iterable(scenario.associations.values())))

    with open('../mab/node_thr_h_mab.json') as f:
        mab_h = json.load(f)

    with open('../mab/node_thr_f_mab.json') as f:
        mab_f = json.load(f)

    with open('../upper_bound/all_results.json') as f:
        optimal = json.load(f)

    dcf = pd.read_csv('dcf/residential/static_residential_104_2_3_4_20.0.csv')
    sr = pd.read_csv('sr/residential/static_residential_104_2_3_4_20.0.csv')

    dcf_thr = np.asarray(dcf.groupby("Dst")["AMPDUSize"].sum() * 1e-6 / SIM_TIME / N_RUNS)
    sr_thr = np.asarray(sr.groupby("Dst")["AMPDUSize"].sum() * 1e-6 / SIM_TIME / N_RUNS)
    mab_h_thr = np.asarray(mab_h[scenario_idx]).mean(axis=(0, 1))[stas]
    mab_f_thr = np.asarray(mab_f[scenario_idx]).mean(axis=(0, 1))[stas]
    t_optimal_thr = defaultdict(float)
    f_optimal_thr = defaultdict(float)

    for conf, weight in optimal[scenario_idx][0]['shares'][0].items():
        for link, rate in optimal[scenario_idx][0]['link_rates'][0][conf].items():
            t_optimal_thr[link] += rate * weight
    t_optimal_thr = np.asarray(list(t_optimal_thr.values()))

    for conf, weight in optimal[scenario_idx][1]['shares'][0].items():
        for link, rate in optimal[scenario_idx][1]['link_rates'][0][conf].items():
            f_optimal_thr[link] += rate * weight
    f_optimal_thr = np.asarray(list(f_optimal_thr.values()))

    dcf_cdf = np.sort(dcf_thr)
    sr_cdf = np.sort(sr_thr)
    mab_f_cdf = np.sort(mab_f_thr)
    mab_h_cdf = np.sort(mab_h_thr)
    f_optimal_cdf = np.sort(f_optimal_thr)
    t_optimal_cdf = np.sort(t_optimal_thr)
    ys = np.arange(len(stas)) / (len(stas) - 1)

    plt.plot(dcf_cdf, ys, label='DCF', color=AGENT_COLORS['DCF'])
    plt.plot(sr_cdf, ys, label='SR', color=AGENT_COLORS['SR'])
    plt.plot(mab_f_cdf, ys, label='MAB', color=AGENT_COLORS['MAB'])
    plt.plot(mab_h_cdf, ys, label='H-MAB', color=AGENT_COLORS['H-MAB'])
    plt.plot(f_optimal_cdf, ys, label='F-Optimal', color=AGENT_COLORS['F-Optimal'])
    plt.plot(t_optimal_cdf, ys, label='T-Optimal', color=AGENT_COLORS['T-Optimal'])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', ncol=2)

    plt.xlabel('Effective data rate [Mb/s]')
    plt.ylabel('CDF')
    plt.xlim(0, 120)
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.savefig('residential_cdf_2_3_4_d20.pdf', bbox_inches='tight')
    plt.show()
