import json
from argparse import ArgumentParser
from itertools import chain

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

    solver = Solver(stations, access_points, **solver_kwargs)
    return solver(path_loss)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    all_results = []

    for scenario in tqdm(ALL_SCENARIOS, desc='Scenarios'):
        for opt_sum, opt_name in zip([True, False], ['sum', 'min']):
            split_conf, split_rate = [], []

            for static, _ in scenario.split_scenario():
                configuration, solver_rate, _ = run_solver(static, {'opt_sum': opt_sum})
                split_conf.append(configuration)
                split_rate.append(solver_rate)

            all_results.append({
                'solver': opt_name,
                'runs': split_rate,
                'actions': split_conf
            })

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
