import numpy as np

def prepare_C1_client_data(X_train, y_train, num_classes):
    X_train_by_class = []
    y_train_by_class = []
    client_data = []
    for class_id in range(num_classes):
        class_indices = np.where(y_train == class_id)[0]
        X_train_by_class.append(X_train[class_indices])
        y_train_by_class.append(y_train[class_indices])
    for i in range(num_classes):
        client_data.append((X_train_by_class[i], y_train_by_class[i]))
    print("Created C1 data distribution (1 class per client)")
    return client_data

def prepare_C2_client_data(X_train, y_train, num_classes):
    X_train_by_class = []
    y_train_by_class = []
    client_data = []
    for class_id in range(num_classes):
        class_indices = np.where(y_train == class_id)[0]
        X_train_by_class.append(X_train[class_indices])
        y_train_by_class.append(y_train[class_indices])
    for i in range(num_classes):
        next_class = (i + 1) % num_classes
        half_size_i = len(X_train_by_class[i]) // 2
        X_current = X_train_by_class[i][:half_size_i]
        y_current = y_train_by_class[i][:half_size_i]
        half_size_next = len(X_train_by_class[next_class]) // 2
        X_next = X_train_by_class[next_class][:half_size_next]
        y_next = y_train_by_class[next_class][:half_size_next]
        X_combined = np.concatenate([X_current, X_next])
        y_combined = np.concatenate([y_current, y_next])
        client_data.append((X_combined, y_combined))
    print("Created C2 data distribution (2 classes per client)")
    return client_data

def prepare_IID_client_data(X_train, y_train, num_clients):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    samples_per_client = len(X_train) // num_clients
    client_indices = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_indices.append(indices[start_idx:end_idx])
    print("\nIID Client Data Distribution:")
    for i in range(num_clients):
        unique, counts = np.unique(y_train[client_indices[i]], return_counts=True)
        print(f"Client {i} distribution: {dict(zip(unique, counts))}")
    return [(X_train[inds], y_train[inds]) for inds in client_indices]

import numpy as np

def prepare_QuantitySkew_client_data(X_train, y_train, num_clients):
    """
    Quantity skew: randomly assigns a different number of training samples to each client
    based on a Dirichlet distribution with beta=0.5.
    """
    beta = 0.5
    min_proportion = 0.005  # 0.5%
    
    # Ensure each client gets at least a minimal proportion
    while True:
        proportions = np.random.dirichlet([beta] * num_clients)
        if np.all(proportions >= min_proportion):
            break
    
    # Normalize proportions
    proportions = proportions / np.sum(proportions)
    
    total_samples = len(X_train)
    samples_per_client = (proportions * total_samples).astype(int)
    # Adjust last client to include any rounding error
    samples_per_client[-1] = total_samples - np.sum(samples_per_client[:-1])
    
    # Shuffle the dataset indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    client_data_indices = []
    start_idx = 0
    for n_samples in samples_per_client:
        end_idx = start_idx + n_samples
        client_indices = indices[start_idx:end_idx]
        client_data_indices.append(client_indices)
        start_idx = end_idx
    
    # Log the distribution per client
    for i in range(num_clients):
        unique, counts = np.unique(y_train[client_data_indices[i]], return_counts=True)
        print(f"Client {i} samples: {len(client_data_indices[i])} ({proportions[i]*100:.1f}%)")
        print(f"Client {i} class distribution: {dict(zip(unique, counts))}")
    
    # Return list of (X_client, y_client) tuples
    return [(X_train[client_indices], y_train[client_indices]) for client_indices in client_data_indices]


def prepare_client_data(X_train, y_train, distribution_type, num_classes, num_clients):
    if distribution_type == 'LabelSkew_c1':
        return prepare_C1_client_data(X_train, y_train, num_classes)
    elif distribution_type == 'LabelSkew_c2':
        return prepare_C2_client_data(X_train, y_train, num_classes)
    elif distribution_type == 'IID':
        return prepare_IID_client_data(X_train, y_train, num_clients)
    elif distribution_type == 'QuantitySkew':
        return prepare_QuantitySkew_client_data(X_train, y_train, num_clients)
    else:
        raise ValueError(f"Invalid distribution type: {distribution_type}")
