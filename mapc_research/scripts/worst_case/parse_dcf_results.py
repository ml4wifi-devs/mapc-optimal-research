import os

from argparse import ArgumentParser
import pandas as pd

from mapc_sim.constants import TAU

N_RUNS = 10
N_STEPS = 600
SIM_TIME = N_STEPS * TAU


def parse_agent_results(agent_dir: str) -> dict:

    # Read results
    results_files = os.listdir(agent_dir)
    results_files = [f for f in results_files if f.endswith(".csv")]
    results_files = sorted(results_files)
    results_dfs = {int(f.split('_')[2]) - 100: pd.read_csv(os.path.join(agent_dir, f)) for f in results_files}

    # Calculate total throughput
    get_total_thr = lambda df: (df["AMPDUSize"].sum() * 1e-6 / SIM_TIME / N_RUNS)
    total_thr = {f: get_total_thr(df) for f, df in results_dfs.items()}

    # Calculate worst case throughput
    get_worst_case_thr = lambda df: (df.groupby("Dst")["AMPDUSize"].sum() * 1e-6 / SIM_TIME / N_RUNS).min()
    worst_case_thr = {f: get_worst_case_thr(df) for f, df in results_dfs.items()}

    return total_thr, worst_case_thr


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-r', '--results_dir', type=str, required=True)
    args.add_argument('-o', '--out_path', type=str, required=True)
    args = args.parse_args()

    agents = [
        "dcf",
        # "oracle",
        "sr"
    ]

    data = []
    for agent in agents:
        agent_dir = os.path.join(args.results_dir, agent, "residential")
        total_thr, worst_case_thr = parse_agent_results(agent_dir)

        for scenario_id in total_thr.keys():
            data.append(
                {
                    "ScenarioID": scenario_id,
                    "Agent": agent,
                    "TotalThr": total_thr[scenario_id],
                    "WorstCaseThr": worst_case_thr[scenario_id],
                }
            )

    results_df = pd.DataFrame(data)
    results_df.to_csv(args.out_path, index=False)
