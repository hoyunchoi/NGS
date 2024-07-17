# Neural Graph Simulator for Complex Systems
Fully reproducible code and data for paper 'Neural Graph Simulator for Complex Systems'.

In order to fully reproduce the contents of the paper, the following scripts must be executed in order. Alternatively, you can use the simulation results or the trained model we provide right away.

Please refer to the paper for an explanation of each term.


# Requirements
- Python 3.11 or higher
- PyTorch 2.2 or higher
- PyTorch Geometric 2.5 or higher

Or, one might use the provided `environment.yaml` file to create conda environment that was used for this project.


# 1. Simulation with numerical solvers
For the three systems presented in the paper: heat, coupled Rössler, and Kuramoto, codes and dataset are provided at the corresponding directories that simulates the systems using numerical solvers and stores the results in the `data` directory.
For example, to create a train dataset for the heat system, run the following command:
```
python heat/simulation.py heat_train --num_nodes 100 200 --mean_degree 2.0 4.0 --max_time 1.0 -1.0 --num_steps 20 -1 --dt_delta 0.8 --num_samples=1000 --hot_ratio 0.2 0.8 --dissipation 0.1 2.0 --seed=0
```
This will create a `data/heat_train.pkl` file, which is a pickle file containing the simulation results of the heat system.
Commands to create all of the datasets used in the paper are provided in `simulation.txt`.
Alternatively, one might use the datasets already provided in the `data` directory.

# 2. Training the NGS

### Provided models
All the weights of the trained NGS models that are used to present the results in the paper are provided in the `result` directory.

For the heat and coupled Rössler systems, the weights are stored in directories named of each system and the degradation level of the train dataset. For example, the `heat_p0.1_s0.001` directory contains the weights of the NGS trained on the heat system, where the train dataset is degraded with missing fraction $p=0.1$ and standard deviation of Gaussian noise $\sigma=0.001$.

For the Kuramoto system, the name of the directory indicates the $\pi / \theta_\text{th}$. i.e., the `kuramoto_th6` indicates the model is trained under condition of $\theta_\text{th}=\pi/6$.

### Training by hand
If you want to train the model yourself, `train.py` or `train.ipynb` in each system's directory is available. (These are the same files)
For example, to train the NGS on the heat system, run the following command:
```
python heat/train.py
```
Hyperparmeters can be modified in the corresponding train files.

# 3. Evaluating the NGS
The code for evaluation of the trained model are provided in the `evaluate.py` file in each system's directory.
These codes load a trained NGS model, run simulations for each dataset, and store the predictions in the directory where the model weights are stored.
Again, the simulation results of the trained NGS model are already available in the `result` directory.
Because the cuda operations included in NGS are not deterministic, running them yourself may result in different simulation results as those already provided, but the difference should be negligible.

# 4. Reproducing figures and table
The codes for the figures and table presented in the paper are provided in the `figures` directory.
