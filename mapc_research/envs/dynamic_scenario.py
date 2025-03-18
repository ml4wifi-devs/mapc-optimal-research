from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from chex import Array, Scalar, PRNGKey
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA, DATA_RATES, TAU
from mapc_sim.sim import network_data_rate
from mapc_sim.utils import default_path_loss

from mapc_research.envs.scenario import Scenario
from mapc_research.envs.static_scenario import StaticScenario


class DynamicScenario(Scenario):
    """
    Dynamic scenario with possibility to change the configuration of the scenario at runtime.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    associations: dict
        Dictionary of associations between access points and stations.
    n_steps: int
        Number of steps in the simulation.
    tx_power: Scalar
        Default transmission power of the nodes.
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    channel_width: int
        Channel width in MHz.
    pos_sec: Optional[Array]
        Array of node positions after the change.
    tx_power_sec: Optional[Scalar]
        Default transmission power of the nodes after the change.
    sigma_sec: Optional[Scalar]
        Standard deviation of the noise after the change.
    walls_sec: Optional[Array]
        Adjacency matrix of walls after the change.
    walls_pos_sec: Optional[Array]
        Array of wall positions after the change.
    channel_width_sec: int
        Channel width after the change.
    switch_steps: Optional[list]
        List of steps at which the scenario should change.
    tx_power_delta: Scalar
        Difference in transmission power between the tx power levels.
    path_loss_fn: Callable
        A function that calculates the path loss between two nodes. The function signature should be
        `path_loss_fn(distance: Array, walls: Array) -> Array`, where `distance` is the matrix of distances
        between nodes and `walls` is the adjacency matrix of walls. By default, the simulator uses the
        residential TGax path loss model.
    str_repr: str
        String representation of the scenario.
    """

    def __init__(
            self,
            pos: Array,
            associations: dict,
            n_steps: int,
            tx_power: Scalar = DEFAULT_TX_POWER,
            sigma: Scalar = DEFAULT_SIGMA,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            channel_width: int = None,
            pos_sec: Optional[Array] = None,
            tx_power_sec: Optional[Scalar] = None,
            sigma_sec: Optional[Scalar] = None,
            walls_sec: Optional[Array] = None,
            walls_pos_sec: Optional[Array] = None,
            channel_width_sec: int = None,
            switch_steps: Optional[list] = None,
            tx_power_delta: Scalar = 3.0,
            path_loss_fn: Callable = default_path_loss,
            str_repr: str = ""
    ) -> None:
        self.str_repr = "dynamic_" + str_repr if str_repr else "dynamic"
        super().__init__(associations, pos, walls, walls_pos, channel_width, path_loss_fn, self.str_repr)

        if walls is None:
            walls = jnp.zeros((pos.shape[0], pos.shape[0]))
        if switch_steps is None:
            switch_steps = []

        self.tx_power_first = jnp.full(pos.shape[0], tx_power)
        self.scenario_first = StaticScenario(
            pos, associations, n_steps, tx_power, sigma, walls, walls_pos, channel_width, tx_power_delta
        )
        self.data_rate_fn_first = jax.jit(partial(
            network_data_rate,
            pos=pos,
            mcs=None,
            sigma=sigma,
            walls=walls,
            path_loss_fn=path_loss_fn,
            channel_width=self.scenario_first.channel_width
        ))
        self.normalize_reward_first = DATA_RATES[self.scenario_first.channel_width,][-1]

        if pos_sec is None:
            pos_sec = pos.copy()
        if tx_power_sec is None:
            tx_power_sec = tx_power
        if sigma_sec is None:
            sigma_sec = sigma
        if walls_sec is None:
            walls_sec = walls.copy()

        self.tx_power_sec = jnp.full(pos_sec.shape[0], tx_power_sec)
        self.scenario_sec = StaticScenario(
            pos_sec, associations, n_steps, tx_power_sec, sigma_sec, walls_sec, walls_pos_sec, channel_width_sec, tx_power_delta
        )
        self.data_rate_fn_sec = jax.jit(partial(
            network_data_rate,
            pos=pos_sec,
            mcs=None,
            sigma=sigma_sec,
            walls=walls_sec,
            path_loss_fn=path_loss_fn,
            channel_width=self.scenario_sec.channel_width
        ))
        self.normalize_reward_sec = DATA_RATES[self.scenario_sec.channel_width][-1]

        self.data_rate_fn = self.data_rate_fn_first
        self.normalize_reward = self.normalize_reward_first
        self.tx_power = self.tx_power_first
        self.n_steps = n_steps
        self.switch_steps = switch_steps
        self.step = 0
        self.tx_power_delta = tx_power_delta

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Array) -> tuple[Scalar, Scalar]:
        if tx_power is None:
            tx_power = jnp.zeros(self.pos.shape[0])

        if self.step in self.switch_steps:
            self.switch()

        self.step += 1

        thr = self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)
        reward = thr / self.normalize_reward
        return thr, reward

    def split_scenario(self) -> list[tuple[StaticScenario, float]]:
        is_first = True
        switch_steps = [0] + self.switch_steps + [self.n_steps]
        scenarios = []

        for start, end in zip(switch_steps[:-1], switch_steps[1:]):
            if is_first:
                scenarios.append((self.scenario_first, (end - start) * TAU))
            else:
                scenarios.append((self.scenario_sec, (end - start) * TAU))

            is_first = not is_first

        return scenarios

    def reset(self) -> None:
        self.data_rate_fn = self.data_rate_fn_first
        self.step = 0

    def switch(self) -> None:
        if self.data_rate_fn is self.data_rate_fn_first:
            self.data_rate_fn = self.data_rate_fn_sec
            self.tx_power = self.tx_power_sec
            self.normalize_reward = self.normalize_reward_sec
        else:
            self.data_rate_fn = self.data_rate_fn_first
            self.tx_power = self.tx_power_first
            self.normalize_reward = self.normalize_reward_first

    @staticmethod
    def from_static_params(
            scenario: StaticScenario,
            n_steps: int = float('inf'),
            pos_sec: Optional[Array] = None,
            tx_power_sec: Optional[Scalar] = None,
            sigma_sec: Optional[Scalar] = None,
            walls_sec: Optional[Array] = None,
            walls_pos_sec: Optional[Array] = None,
            channel_width_sec: int = None,
            switch_steps: Optional[list] = None
    ) -> 'DynamicScenario':
        return DynamicScenario(
            scenario.pos,
            scenario.associations,
            n_steps,
            scenario.tx_power,
            scenario.sigma,
            scenario.walls,
            scenario.walls_pos,
            scenario.channel_width,
            pos_sec,
            tx_power_sec,
            sigma_sec,
            walls_sec,
            walls_pos_sec,
            channel_width_sec,
            switch_steps,
            str_repr=scenario.str_repr
        )

    @staticmethod
    def from_static_scenarios(
            scenario: StaticScenario,
            scenario_sec: StaticScenario,
            switch_steps: list,
            n_steps: int = float('inf')
    ) -> 'DynamicScenario':
        return DynamicScenario(
            scenario.pos,
            scenario.associations,
            n_steps,
            scenario.tx_power,
            scenario.sigma,
            scenario.walls,
            scenario.walls_pos,
            scenario.channel_width,
            scenario_sec.pos,
            scenario_sec.tx_power,
            scenario_sec.sigma,
            scenario_sec.walls,
            scenario_sec.walls_pos,
            scenario_sec.channel_width,
            switch_steps,
            str_repr=scenario.str_repr
        )
