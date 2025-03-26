# Quantum Federated Learning Experiments for QCE25

This repository contains the code for experiments presented in our QCE25 paper on Quantum Federated Learning (QFL). The codebase implements several QFL training methods using a variational quantum circuit, leveraging PennyLane for quantum simulation and TensorFlow for dataset handling. We evaluate different client data distributions, including extreme label skew, label pair skew, IID, and quantity skew, on MNIST and Fashion MNIST datasets.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Distributions](#data-distributions)
- [Training Methods](#training-methods)
- [Experiment Configuration](#experiment-configuration)
- [Results and Checkpoints](#results-and-checkpoints)
- [Citing This Work](#citing-this-work)
- [License](#license)

## Overview

This project explores quantum federated learning for image classification tasks. Our experiments compare the following QFL training methods:
- **Centralized**: Standard training using the full dataset.
- **FedAvg**: Federated averaging of locally trained models.
- **FedAdam**: Federated optimization with Adam-style updates.
- **Loss Aggregation (SPSA-based)**: Aggregating SPSA updates computed on each client.
- **Loss Aggregation with Adam**: A variant combining loss aggregation and Adam updates.

Additionally, we consider various data distributions across clients:
- **C1 (Extreme Label Skew)**: Each client receives data for one class.
- **C2 (Label Pair Skew)**: Each client receives data from two consecutive classes.
- **IID**: Data is randomly distributed across clients.
- **Quantity Skew ("quantity")**: Clients receive varying numbers of samples determined by a Dirichlet distribution.

## Directory Structure

```
├── main.py                 # Entry point for running experiments.
├── training.py             # Contains the training loop, optimization, logging, and checkpointing.
├── data_utils.py           # Dataset loading and preprocessing (resizing and normalization).
├── distribution.py         # Functions for partitioning data among clients (C1, C2, IID, Quantity Skew).
├── quantum_model.py        # Quantum circuit, QNode, cost/accuracy calculation, and weight initialization.
└── README.md               # This file.
```

## Installation

The code requires Python 3.7 or higher. Install the necessary dependencies using `pip`:

```bash
pip install pennylane tensorflow numpy scikit-learn matplotlib tqdm
```

Make sure that your environment supports GPU acceleration for TensorFlow if needed. We also recommend using a virtual environment.

## Usage

Run the experiments using the concise `main.py` entry point. The script accepts several command-line arguments:

- **method**: QFL training method to use. Choices: `centralized`, `fedavg`, `fedadam`, `loss_agg`, `loss_agg_adam`
- **-d / --distribution**: Data distribution type. Choices: `c1`, `c2`, `iid`, `quantity` (default: `c2`)
- **-c / --classes**: Number of classes to use (8 for classes 0-7 or 10 for all classes; default: 10)
- **-ds / --dataset**: Dataset to use: `mnist` or `fmnist` (default: `mnist`)
- **-l / --load**: (Optional) Load from an existing run directory (e.g., `run_1`)

### Example

To run federated averaging on MNIST with a 2-class per client skew (C2):

```bash
python main.py fedavg -d c2 -c 10 -ds mnist
```

To run a quantity skew experiment using the new `"quantity"` distribution:

```bash
python main.py fedadam -d quantity -c 10 -ds fmnist
```

## Data Distributions

The data distribution is controlled via the `distribution.py` module:

- **C1 Distribution**: Each client receives data for a single class.
- **C2 Distribution**: Each client receives half of one class and half of the next class.
- **IID Distribution**: Randomly assigns data equally across clients.
- **Quantity Skew ("quantity")**: Uses a Dirichlet distribution (beta=0.5) to determine a random proportion of the dataset assigned to each client. This creates variation in the number of training samples per client.

## Training Methods

The training methods are implemented in `training.py` using a variational quantum circuit defined in `quantum_model.py`. Local optimization is performed with SPSA or Adam optimizers depending on the selected method. The training loop orchestrates federated averaging, FedAdam updates, and loss aggregation as appropriate.

## Experiment Configuration

The experiment settings (e.g., number of epochs, batch size, number of steps, number of layers, and SPSA iterations) are hardcoded in `training.py`. The configuration (including dataset, number of classes, distribution method, and model parameters) is saved in a configuration file in the output directory for reproducibility.

## Results and Checkpoints

- **Output Directory**: The results for each experiment are saved in a dedicated output directory named based on the method, distribution, class count, and dataset (e.g., `fedavg_c2_10class_mnist_output`).
- **Logging**: Training progress is logged to a CSV file and a text file within the run directory.
- **Checkpoints**: Model weights are saved periodically (every 20 steps) as `latest_ckpt.npy`, allowing you to resume experiments using the `-l` flag.
- **Final Weights**: After training, the final weights are saved as `<method>_final_weights.npy`.

## Citing This Work

If you find this code useful in your research, please consider citing our paper:

```
@inproceedings{YourPaper2025,
  title={Quantum Federated Learning: A New Paradigm for Distributed Quantum Machine Learning},
  author={Your Name and Collaborators},
  booktitle={Proceedings of QCE25},
  year={2025},
  publisher={Your Publisher}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please contact [Your Email] or open an issue on this repository.

Happy experimenting!
