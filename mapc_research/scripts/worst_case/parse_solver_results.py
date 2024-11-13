import json

from argparse import ArgumentParser
import pandas as pd
import numpy as np

def parse_json_results(results: dict) -> pd.DataFrame:

    data = []
    for scenario_id, scenario_results in enumerate(results):
        for agent_results in scenario_results:
            agent = agent_results["agent"]
            total_thr = agent_results["runs"][0]
            shares = agent_results["shares"][0]
            link_rates = agent_results["link_rates"][0]

            sta_rates = {}
            for tau, share in shares.items():
                configuration = link_rates[tau]
                for link_str, rate in configuration.items():
                    sta_str = link_str.split("'")[3]
                    if sta_str not in sta_rates:
                        sta_rates[sta_str] = 0
                    sta_rates[sta_str] += rate * share

            min_thr = np.mean(np.array(list(sta_rates.values())))

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
    args = args.parse_args()

    # Load results
    with open(args.results_path, "r") as f:
        results_json = json.load(f)

    # Parse results
    results_df = parse_json_results(results_json)

    # Save results
    results_df.to_csv(args.results_path.replace(".json", ".csv"), index=False)
    