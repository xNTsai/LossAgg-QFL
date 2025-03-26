import os
import csv
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pennylane as qml

from data_utils import load_dataset, preprocess_data
from distribution import prepare_client_data
from quantum_model import create_qnode, cost_and_accuracy, weights_init, weights_load

# Hardcoded hyperparameters
epochs = 4
batch_size = 16
steps = 100
local_epochs = 10
n_val_samples = 1024
num_layers = 30
n_spsa = 40  # SPSA iterations for loss aggregation

# Federated learning / FedAdam update
def fedadam_update(weights, delta, v, m, beta1, beta2, learning_rate, tau):
    m = beta1 * m + (1 - beta1) * delta
    v = beta2 * v + (1 - beta2) * delta**2
    updated_weights = weights + learning_rate * m / (np.sqrt(v) + tau)
    return updated_weights, v, m

# Loss aggregation helper function (modified to accept qnode)
def qfl_loss_aggregation(qnode, weights, client_data, batch_size, step):
    total_loss = 0
    total_samples = 0
    for client_X, client_y in client_data:
        data_X = np.concatenate([
            client_X[step*(batch_size//2):(step+1)*(batch_size//2)],
            client_X[len(client_X)-(step+1)*(batch_size//2):len(client_X)-step*(batch_size//2)]
        ])
        data_y = np.concatenate([
            client_y[step*(batch_size//2):(step+1)*(batch_size//2)],
            client_y[len(client_y)-(step+1)*(batch_size//2):len(client_y)-step*(batch_size//2)]
        ])
        loss, _ = cost_and_accuracy(qnode, weights, data_X, data_y)
        total_loss += loss * len(data_X)
        total_samples += len(data_X)
    return total_loss / total_samples

def run_optimization(run_dir, args, X_train, y_train, X_val, y_val, dataset_name, qnode, initial_weights, num_clients):
    # Set optimizer hyperparameters
    spsa_learning_rate = 0.02
    loss_agg_spsa_learning_rate = 0.1
    if args.distribution == 'iid' and args.classes == 10:
        spsa_learning_rate = 1.0

    opt_centralized = qml.AdamOptimizer(stepsize=0.01)
    opt_federated = qml.SPSAOptimizer(maxiter=local_epochs, alpha=0, a=spsa_learning_rate)
    opt_loss_agg = qml.SPSAOptimizer(maxiter=1, alpha=0, a=loss_agg_spsa_learning_rate)

    # FedAdam parameters
    tau = 1e-8
    Adam_learning_rate = 5e-3
    beta1 = 0.9
    beta2 = 0.999

    method = args.method
    if method in ['centralized', 'fedavg', 'loss_agg']:
        weights = initial_weights.copy()
    elif method in ['fedadam', 'loss_agg_adam']:
        weights = initial_weights.copy()
        v = np.ones_like(initial_weights) * tau**2
        m = np.zeros_like(initial_weights)
    else:
        raise ValueError("Invalid method")

    monitoring_interval = 2
    ckpt_interval = 20
    val_losses, val_accs = [], []
    steps_list = []

    # CSV and progress logging
    csv_path = os.path.join(run_dir, f'{method}_results.csv')
    progress_path = os.path.join(run_dir, f'{method}_training_progress.txt')
    file_mode = 'a' if args.load else 'w'
    with open(csv_path, file_mode, newline='') as f:
        writer = csv.writer(f)
        if not args.load:
            writer.writerow(['Epoch', 'Step', 'Val Loss', 'Val Accuracy'])

    with open(progress_path, file_mode) as progress_file:
        progress_file.write(f"Starting {method.upper()} training on {dataset_name}\n")
        for epoch in range(epochs):
            # Shuffle data each epoch
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            client_data = prepare_client_data(X_train, y_train, args.distribution, args.classes, num_clients)

            for step in tqdm(range(steps), desc=f"EPOCH {epoch+1}/{epochs}"):
                if method == 'centralized':
                    weights = opt_centralized.step(
                        lambda w: cost_and_accuracy(qnode, w,
                            X_train[step*batch_size:(step+1)*batch_size],
                            y_train[step*batch_size:(step+1)*batch_size]
                        )[0], weights
                    )
                elif method == 'fedavg':
                    local_weights = []
                    for client_X, client_y in client_data:
                        client_weights = weights.copy()
                        data_X = np.concatenate([
                            client_X[step*(batch_size//2):(step+1)*(batch_size//2)],
                            client_X[len(client_X)-(step+1)*(batch_size//2):len(client_X)-step*(batch_size//2)]
                        ])
                        data_y = np.concatenate([
                            client_y[step*(batch_size//2):(step+1)*(batch_size//2)],
                            client_y[len(client_y)-(step+1)*(batch_size//2):len(client_y)-step*(batch_size//2)]
                        ])
                        for _ in range(local_epochs):
                            client_weights = opt_federated.step(
                                lambda w: cost_and_accuracy(qnode, w, data_X, data_y)[0], client_weights
                            )
                        local_weights.append(client_weights)
                    weights = np.mean(local_weights, axis=0)
                elif method == 'fedadam':
                    local_weights = []
                    for client_X, client_y in client_data:
                        client_weights = weights.copy()
                        data_X = np.concatenate([
                            client_X[step*(batch_size//2):(step+1)*(batch_size//2)],
                            client_X[len(client_X)-(step+1)*(batch_size//2):len(client_X)-step*(batch_size//2)]
                        ])
                        data_y = np.concatenate([
                            client_y[step*(batch_size//2):(step+1)*(batch_size//2)],
                            client_y[len(client_y)-(step+1)*(batch_size//2):len(client_y)-step*(batch_size//2)]
                        ])
                        for _ in range(local_epochs):
                            client_weights = opt_federated.step(
                                lambda w: cost_and_accuracy(qnode, w, data_X, data_y)[0], client_weights
                            )
                        local_weights.append(client_weights)
                    delta = np.mean([w - weights for w in local_weights], axis=0)
                    weights, v, m = fedadam_update(weights, delta, v, m, beta1, beta2, Adam_learning_rate, tau)
                elif method == 'loss_agg':
                    aggregated_updates = []
                    for _ in range(n_spsa):
                        updated_weights = opt_loss_agg.step(
                            lambda w: qfl_loss_aggregation(qnode, w, client_data, batch_size, step), weights
                        )
                        aggregated_updates.append(updated_weights)
                    weights = np.mean(aggregated_updates, axis=0)
                elif method == 'loss_agg_adam':
                    aggregated_updates = []
                    for _ in range(n_spsa):
                        updated_weights = opt_loss_agg.step(
                            lambda w: qfl_loss_aggregation(qnode, w, client_data, batch_size, step), weights
                        )
                        aggregated_updates.append(updated_weights)
                    delta = np.mean([w - weights for w in aggregated_updates], axis=0)
                    weights, v, m = fedadam_update(weights, delta, v, m, beta1, beta2, Adam_learning_rate, tau)

                # Monitor validation every few steps
                if (step + 1) % monitoring_interval == 0:
                    val_loss, val_acc = cost_and_accuracy(qnode, weights, X_val, y_val)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    steps_list.append((epoch, step + 1))
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, step + 1, val_loss, val_acc])
                    progress_text = f"\nEpoch {epoch+1}, Step {step+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}\n"
                    tqdm.write(progress_text)
                    progress_file.write(progress_text)
                    progress_file.flush()

                # Save checkpoint periodically
                if (step + 1) % ckpt_interval == 0:
                    ckpt_path = os.path.join(run_dir, 'latest_ckpt.npy')
                    np.save(ckpt_path, weights)
                    
        return weights, val_losses, val_accs, steps_list

def save_results(weights, val_losses, val_accs, steps_list, run_dir, method_name):
    np.save(os.path.join(run_dir, f'{method_name}_final_weights.npy'), weights)
    print(f"Final results saved for {method_name}")

def run_training(args):
    # Determine number of clients and qubits from number of classes
    if args.classes == 8:
        num_clients = 8
        n_q = 8
    else:
        num_clients = 10
        n_q = 10

    # Setup output directory
    output_dir = f'{args.method}_{args.distribution}_{args.classes}class_{args.dataset}_output'
    os.makedirs(output_dir, exist_ok=True)
    if args.load:
        run_dir = os.path.join(output_dir, args.load)
        if not os.path.exists(run_dir):
            print(f"Error: Directory {run_dir} does not exist")
            exit(1)
        print(f"Loading from existing directory: {run_dir}")
    else:
        existing_ids = [int(d.split('_')[-1]) for d in os.listdir(output_dir) if d.startswith('run_')]
        max_id = max(existing_ids) if existing_ids else 0
        run_id = max_id + 1
        run_dir = os.path.join(output_dir, f'run_{run_id}')
        os.makedirs(run_dir)

    # Save configuration
    config_path = os.path.join(run_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of classes: {args.classes}\n")
        f.write(f"Distribution method: {args.distribution}\n")
        f.write(f"QFL method: {args.method}\n")
        f.write(f"Number of qubits: {n_q}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"SPSA steps for loss aggregation: {n_spsa}\n")

    # Load and preprocess dataset
    (x_train, y_train), (x_val, y_val) = load_dataset(args.dataset)
    dataset_name = "MNIST digits" if args.dataset == 'mnist' else "Fashion MNIST"
    X_train, X_val = preprocess_data(x_train, x_val, args.classes)

    if args.classes == 8:
        mask = y_train < 8
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = y_val < 8
        X_val = X_val[mask]
        y_val = y_val[mask]
        print("Filtered to 8 classes (0-7)")
    X_val, y_val = X_val[:n_val_samples], y_val[:n_val_samples]
    print(f"Image data shape: {X_train.shape}")
    print(f"Using {args.classes} classes with {num_clients} clients")

    # Create the QNode for the quantum circuit
    qnode = create_qnode(n_q)

    # Initialize or load weights
    init_weights_path = os.path.join(run_dir, 'latest_ckpt.npy')
    initial_weights = weights_load(init_weights_path) if args.load else weights_init(n_components=n_q, num_layers=num_layers)
    print(f"Number of parameters: {np.prod(initial_weights.shape)}")

    # Run the optimization loop
    weights, val_losses, val_accs, steps_list = run_optimization(
        run_dir, args, X_train, y_train, X_val, y_val, dataset_name, qnode, initial_weights, num_clients
    )
    # Save final results
    save_results(weights, val_losses, val_accs, steps_list, run_dir, args.method)
    print(f"Results for {args.method} saved successfully in {run_dir}")

# Define n_val_samples for internal use
n_val_samples = 1024
