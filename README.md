# Resources for the article "Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks"

This repository is the main workspace for the article. We decided **not to create a monolithic repository** (a mono-repository), but to keep the code and the data separated accross multiple repositories (see below). This approach allows other researchers to cherypick and focus on the single tool developed by the authors of the article without needing to search for specific functionality in a larger and potentially more complex codebase.

Among the installed dependencies, you will find the open-source tools developed alongside the article. The tools are:

- [mapc_optimal](https://github.com/ml4wifi-devs/mapc-optimal) - a theoretical model of C-SR, which finds the best possible transmission schedule using mixed-integer linear programming.
- [mapc_dcf](https://github.com/ml4wifi-devs/mapc-dcf) - a discrete event simulator (built using SimPy), in which devices use either legacy IEEE 802.11 channel access (DCF) or 802.11ax spatial reuse (SR).
- [mapc_sim](https://github.com/ml4wifi-devs/mapc-sim) - a Monte Carlo simulator of consecutive C-SR transmission opportunities.
- [mapc_mab](https://github.com/ml4wifi-devs/mapc-mab) - our hierarchical multi-armed bandit framework to determine C-SR scheduling.

## Installation

To reproduce the research environment with all dependencies and developed tools, you need to first clone this repository and then install it using `pip`.

```bash
# Clone the repository
git clone https://github.com/ml4wifi-devs/mapc-optimal-research.git

# Install the package
pip install -e ./mapc-optimal-research
```

The `-e` flag installs the package in editable mode, so you can change the code and test it without reinstalling the package.


## Repository structure

The repository contains code to schedule simulation experiments, parse and analyze the results, and plot the figures presented in the article. The repository is structured as follows:

- `mapc_optimal_research/`: the main package with all the code,
  - `brute_force`: The brute force algorithm to find the optimal schedule for the C-SR problem.
  - `dcf`: Scripts to manage the DCF simulation experiments.
  - `envs`: Definition and implementation of the IEEE 802.11 C-SR scenarios.
  - `fairness`: Scripts to study the fairness of our approaches.
  - `mab`: Scripts to manage the MAB simulation experiments.
  - `plots`: Scripts to plot the figures presented in the article.
  - `scalability`: Scripts to study the scalability of the upper bound model.
  - `simulator_validation`: Contains a script to validate if the network throughput calculated by the simulator is consistent with the upper bound model.
  - `upper_bounds`: Scripts to manage the upper bound simulation experiments.

## How to reference `mapc-optimal-research`?

```
@article{wojnar2025coordinated,
  author={Wojnar, Maksymilian and Ciężobka, Wojciech and Tomaszewski, Artur and Chołda, Piotr and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Haxhibeqiri, Jetmir and Hoebeke, Jeroen and Bellalta, Boris and Zubow, Anatolij and Dressler, Falko and Szott, Szymon},
  title={{Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks}}, 
  year={2025},
}
```