import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from itertools import chain

import pulp as plp
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss
from tqdm import tqdm

from mapc_research.envs.static_scenario import StaticScenario
from mapc_research.envs.test_scenarios import ALL_SCENARIOS


def convert_to_string(data: list) -> str:
    new_data = []
    for item in data:
        new_item = {}
        for key, inner_dict in item.items():
            new_inner_dict = {}
            for k, v in inner_dict.items():
                new_inner_dict[str(k)] = v  # Convert tuple keys to strings
            new_item[key] = new_inner_dict
        new_data.append(new_item)
    return new_data


def run_solver(scenario: StaticScenario, opt_type: OptimizationType, baseline: dict) -> tuple:
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))

    positions = scenario.pos
    walls = scenario.walls
    path_loss = positions_to_path_loss(positions, walls)

    solver = Solver(stations, access_points, opt_type=opt_type, solver=plp.CPLEX_CMD(msg=False, threads=None))
    return solver(path_loss, associations, baseline, return_objectives=True)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-b', '--baseline', type=str, default='dcf_baselines.json')
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    with open(args.baseline, 'r') as file:
        baselines = json.load(file)

    all_results = []

    for scenario, scenario_baseline in tqdm(zip(ALL_SCENARIOS, baselines), total=len(ALL_SCENARIOS), desc='Scenarios'):
        scenario_results = []

        for opt_type, opt_name in zip(
                [OptimizationType.SUM, OptimizationType.MAX_MIN, OptimizationType.MAX_MIN_BASELINE, OptimizationType.PROPORTIONAL],
                ['sum', 'max_min', 'max_min_baseline', 'proportional']
        ):
            split_rate = []
            split_shares = []
            split_links_rates = []

            for (static, _), baseline in zip(scenario.split_scenario(), scenario_baseline):
                configuration, solver_rate, obj = run_solver(static, opt_type, baseline)
                split_rate.append(solver_rate)
                split_shares.append(configuration["shares"])
                split_links_rates.append(configuration["link_rates"])

            scenario_results.append({
                'agent': opt_name,
                'runs': split_rate,
                'shares': split_shares,
                'link_rates': convert_to_string(split_links_rates)
            })

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(all_results, file, indent=4)
