# LossAgg-QFL: A Communication-Efficient Quantum Federated Learning Framework for Non-IID Data


## Overview

This repository contains the code for our paper, **LossAgg-QFL: A Communication-Efficient Quantum Federated Learning Framework for Non-IID Data**. Our work introduces a novel quantum federated learning (QFL) framework that improves communication efficiency and performance under non-IID data distributions. The repository implements several QFL training methods using a variational quantum circuit with PennyLane, along with TensorFlow for dataset handling. We evaluate different client data distributions—including extreme label skew, label pair skew, IID, and a novel quantity skew—using the MNIST and Fashion MNIST datasets.

## Directory Structure

```
├── main.py                 # Entry point for running experiments.
├── training.py             # Training loop, optimization, logging, and checkpointing.
├── data_utils.py           # Dataset loading and preprocessing (resizing/normalization).
├── distribution.py         # Data partitioning functions (C1, C2, IID, and Quantity Skew).
├── quantum_model.py        # Quantum circuit, QNode, cost/accuracy calculation, and weight initialization.
└── README.md               # This file.
```

## Installation

The project requires Python 3.7 or higher. Install the necessary dependencies with:

```bash
pip install pennylane tensorflow numpy scikit-learn matplotlib tqdm
```

We recommend creating and activating a virtual environment for managing dependencies.

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

To run a quantity skew experiment:

```bash
python main.py fedadam -d quantity -c 10 -ds fmnist
```

## Data Distributions

Data partitioning is handled in `distribution.py`:

- **C1 (Extreme Label Skew)**: Each client receives data for a single class.
- **C2 (Label Pair Skew)**: Each client receives data from two consecutive classes.
- **IID**: Data is randomly distributed equally among clients.
- **Quantity Skew ("quantity")**: Uses a Dirichlet distribution (beta = 0.5) to assign varying numbers of samples to each client. This creates non-uniform sample sizes, adding a realistic quantity skew to the client data.

## Training Methods

Training methods are implemented in `training.py` and leverage a variational quantum circuit defined in `quantum_model.py`. The methods include:
- **Centralized** training using the full dataset.
- **FedAvg**: Federated averaging of locally trained models.
- **FedAdam**: Federated optimization using Adam-style updates.
- **Loss Aggregation (SPSA-based)**: Aggregating losses from clients to server for SPSA optimization.
- **Loss Aggregation with Adam**: Combines loss aggregation with Adam updates.

## Experiment Configuration

The experiment settings (e.g., number of epochs, batch size, steps, layers, and SPSA iterations) are defined in `training.py`. The configuration is saved in a configuration file within the output directory for reproducibility. Output directories are named based on method, distribution type, number of classes, and dataset (e.g., `fedavg_c2_10class_mnist_output`).

## Results and Checkpoints

- **Output Directory**: Each experiment’s results are stored in a dedicated output directory.
- **Logging**: Training progress is logged to CSV and text files.
- **Checkpoints**: Model weights are periodically saved as `latest_ckpt.npy` for resuming experiments.
- **Final Weights**: The final trained model weights are saved as `<method>_final_weights.npy`.

## Citing This Work

If you use this code in your research, please cite our work as follows:

```
@inproceedings{TsaiCheng2025,
  title={LossAgg-QFL: A Communication-Efficient Quantum Federated Learning Framework for Non-IID Data},
  author={Cheng-En Tsai and Hao-Chung Cheng},
  booktitle={Proceedings of QCE25},
  year={2025},
  publisher={Your Publisher}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or issues, please contact johnnytsai920315@gmail.com or open an issue in this repository.

Happy experimenting!