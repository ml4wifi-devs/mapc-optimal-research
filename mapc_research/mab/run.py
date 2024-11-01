import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

import json
from argparse import ArgumentParser

import jax
from mapc_mab import MapcAgentFactory
from reinforced_lib.agents.mab import *
from tqdm import tqdm

from mapc_research.envs.scenario import Scenario
from mapc_research.envs.test_scenarios import ALL_SCENARIOS


def run_scenario(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> tuple[list, list]:
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    for _ in range(n_reps):
        agent = agent_factory.create_mapc_agent()
        scenario.reset()
        runs.append([])
        actions.append([])
        reward = 0.

        for _ in range(n_steps):
            key, scenario_key = jax.random.split(key)
            tx_matrix, tx_power = agent.sample(reward)
            data_rate, reward = scenario(scenario_key, tx_matrix, tx_power)
            runs[-1].append(data_rate.item())
            actions[-1].append(scenario.tx_matrix_to_action(tx_matrix))

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config', type=str, default='default_config.json')
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    all_results = []

    for scenario in tqdm(ALL_SCENARIOS, desc='Scenarios'):
        scenario_results = []

        for agent_config in tqdm(config['agents'], desc='Agents', leave=False):
            agent_factory = MapcAgentFactory(
                associations=scenario.associations,
                agent_type=globals()[agent_config['name']],
                agent_params_lvl1=agent_config['params1'],
                agent_params_lvl2=agent_config['params2'] if agent_config['hierarchical'] else None,
                agent_params_lvl3=agent_config['params3'] if agent_config['hierarchical'] else None,
                hierarchical=agent_config['hierarchical'],
                seed=config['seed']
            )

            runs, actions = run_scenario(agent_factory, scenario, config['n_reps'], scenario.n_steps, config['seed'])
            scenario_results.append({
                'agent': {
                    'name': agent_config['name'],
                    'params1': agent_config['params1'],
                    'params2': agent_config['params2'] if agent_config['hierarchical'] else None,
                    'params3': agent_config['params3'] if agent_config['hierarchical'] else None,
                    'hierarchical': agent_config['hierarchical']
                },
                'runs': runs,
                'actions': actions
            })

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
