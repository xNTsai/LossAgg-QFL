import pennylane as qml
from pennylane import numpy as np
import os

def create_qnode(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        for layer in weights:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.RX(layer[i][0], wires=i)
                qml.RZ(layer[i][1], wires=i)
                qml.RX(layer[i][2], wires=i)
        return [qml.expval(10.0 * qml.PauliZ(i)) for i in range(n_qubits)]
    return qnode

def cost_and_accuracy(qnode, weights, X, y):
    predictions = np.array([qnode(x, weights) for x in X])
    probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y.astype(int)]))
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == y)
    return loss, accuracy

def weights_init(n_components, num_layers):
    initial_weights = np.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, n_components, 3))
    return initial_weights

def weights_load(weights_file):
    if os.path.exists(weights_file):
        initial_weights = np.load(weights_file)
        print("Loaded initial weights")
    else:
        print("Weights file not found")
        return None
    return initial_weights
