import os
import unittest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mapc_research.envs.scenario_impl import toy_scenario_1, toy_scenario_2, random_scenario


class ScenarioClassTestCase(unittest.TestCase):
    def test_simple_plotting(self):
        scenario = toy_scenario_2()
        scenario.plot("test_simple_scenario.pdf")
        assert os.path.exists("test_simple_scenario.pdf")

    def test_random_plotting(self):
        scenario = random_scenario(seed=88)
        scenario.plot("test_random_scenario.png")
        assert os.path.exists("test_random_scenario.png")

    def test_simple_sim(self):
        # Define test-case key and scenario
        key = jax.random.PRNGKey(42)
        scenario = toy_scenario_1(d=10., mcs=7)

        # Transmission matrices indicating which node is transmitting to which node:
        # - in this example, AP A is transmitting to STA 1
        tx1 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, AP A is transmitting to STA 2 and AP B is transmitting to STA 3
        tx2 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, AP A is transmitting to STA 1 and AP B is transmitting to STA 4
        tx3 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])

        # Simulate the network for 150 steps
        data_rate_1, data_rate_2, data_rate_3 = [], [], []

        for _ in range(150):
            key, k1, k2, k3 = jax.random.split(key, 4)
            data_rate_1.append(scenario(k1, tx1))
            data_rate_2.append(scenario(k2, tx2))
            data_rate_3.append(scenario(k3, tx3))

        # Plot effective data rate
        xs = jnp.arange(150)
        plt.plot(xs, data_rate_1, label='AP A -> STA 1')
        plt.plot(xs, data_rate_2, label='AP A -> STA 2 and AP B -> STA 3')
        plt.plot(xs, data_rate_3, label='AP A -> STA 1 and AP B -> STA 4')
        plt.xlim(0, 150)
        plt.ylim(0, 175)
        plt.xlabel('Timestep')
        plt.ylabel('Effective data rate [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_3_rate.pdf', bbox_inches='tight')
        plt.clf()

    def test_cca_threshold(self):
        assert toy_scenario_1(d=25.).is_cca_single_tx()
        assert toy_scenario_2(d_ap=50.).is_cca_single_tx()
        assert not toy_scenario_1(d=100.).is_cca_single_tx()
        assert not toy_scenario_2(d_ap=100.).is_cca_single_tx()
