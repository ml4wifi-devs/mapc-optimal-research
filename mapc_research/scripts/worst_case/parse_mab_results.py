import json

from argparse import ArgumentParser
import pandas as pd
import numpy as np


def parse_json_results(results: list, agent: str) -> pd.DataFrame:

    data = []
    n_scenarios = sum([1 for res in results if len(res) > 0])
    for scenario_id in range(n_scenarios):

        link_thr = np.asarray(results_json[scenario_id]).mean(axis=(0, 1))
        n_nodes = link_thr.shape[0]
        indeces = set(range(n_nodes))
        indeces = indeces - set([5 * i for i in range(n_nodes // 5)])
        link_thr = link_thr[np.asarray(list(indeces))]

        total_thr = np.sum(link_thr)
        min_thr = np.min(link_thr)

        data.append(
            {
                "ScenarioID": scenario_id,
                "Agent": agent,
                "TotalThr": total_thr,
                "WorstCaseThr": min_thr,
            }
        )

    return pd.DataFrame(data)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-r', '--results_path', type=str, required=True)
    args.add_argument('-a', '--agent', type=str, required=True)
    args = args.parse_args()

    # Load results
    with open(args.results_path, "r") as f:
        results_json = json.load(f)

    # Parse results
    results_df = parse_json_results(results_json, args.agent)

    # Save results
    results_df.to_csv(args.results_path.replace(".json", ".csv"), index=False)
    