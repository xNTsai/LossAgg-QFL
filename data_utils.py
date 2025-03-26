import tensorflow as tf
import numpy as np

def load_dataset(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
        print("MNIST dataset loaded")
    else:
        (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
        print("Fashion MNIST dataset loaded")
    return (x_train, y_train), (x_val, y_val)

def expand_to_32x32(images):
    expanded_images = np.zeros((images.shape[0], 32, 32))
    expanded_images[:, 2:30, 2:30] = images
    expanded_images = expanded_images / 255.0
    n = 10
    expanded_images = tf.image.resize(expanded_images[..., tf.newaxis],
                                      (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
    expanded_images = expanded_images / np.sqrt(np.sum(expanded_images**2, axis=-1, keepdims=True))
    return expanded_images

def resize_to_16x16(images):
    images = images / 255.0
    n = 8
    images = tf.image.resize(images[..., tf.newaxis],
                             (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
    images = images / np.sqrt(np.sum(images**2, axis=-1, keepdims=True))
    return images

def preprocess_data(x_train, x_val, num_classes):
    if num_classes == 8:
        X_train = resize_to_16x16(x_train)
        X_val = resize_to_16x16(x_val)
    else:
        X_train = expand_to_32x32(x_train)
        X_val = expand_to_32x32(x_val)
    return X_train, X_val
