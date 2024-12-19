import os
os.environ['JAX_ENABLE_X64'] = 'True'

from mapc_research.envs.dynamic_scenario import DynamicScenario
from mapc_research.envs.scenario_impl import *


SMALL_OFFICE_SCENARIOS = [
]

EXEMPLARY_SCENARIOS = [
    residential_scenario(seed=100, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=10.0),
    DynamicScenario.from_static_scenarios(
        residential_scenario(seed=101, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
        residential_scenario(seed=102, n_steps=600, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
        switch_steps=[600], n_steps=1200
    ),
    random_scenario(seed=3, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1200)
]

INDOOR_SCENARIOS = [
]

FREE_SPACE_SCENARIOS = [
]

ALL_SCENARIOS = EXEMPLARY_SCENARIOS + INDOOR_SCENARIOS + FREE_SPACE_SCENARIOS
