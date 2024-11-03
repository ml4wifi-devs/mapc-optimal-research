import json
from argparse import ArgumentParser
from itertools import chain

import pulp as plp
from mapc_optimal import Solver, positions_to_path_loss
from tqdm import tqdm

from mapc_research.envs.static_scenario import StaticScenario
from mapc_research.envs.test_scenarios import ALL_SCENARIOS


def run_solver(scenario: StaticScenario, solver_kwargs: dict) -> tuple:
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))

    positions = scenario.pos
    walls = scenario.walls
    path_loss = positions_to_path_loss(positions, walls)

    solver = Solver(stations, access_points, **solver_kwargs, solver=plp.CPLEX_CMD(msg=False))
    return solver(path_loss, associations)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    all_results = []

    for scenario in tqdm(ALL_SCENARIOS, desc='Scenarios'):
        scenario_results = []

        for opt_sum, opt_name in zip([True, False], ['opt_sum', 'opt_min']):
            split_rate = []

            for static, _ in scenario.split_scenario():
                configuration, solver_rate = run_solver(static, {'opt_sum': opt_sum})
                split_rate.append(solver_rate)

            scenario_results.append({
                'agent': opt_name,
                'runs': split_rate
            })

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
