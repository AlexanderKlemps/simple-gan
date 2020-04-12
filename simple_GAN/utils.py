import numpy as np


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


def tanh(x, derivative=False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2


def sigmoid(x, derivative=False):
    if not derivative:
        return _sigmoid(x)
    else:
        return _sigmoid(x) * (1 - _sigmoid(x))


def relu(x, derivative=False):
    if not derivative:
        return x * (x > 0)
    else:
        return (x > 0).astype(x.dtype)


def sample_data(num_samples, num_classes=1):
    mus = np.random.uniform(-1, 1, num_classes)
    sigmas = np.random.uniform(0.1, 2.0, num_classes)
    data = []
    for num in range(num_classes):
        mu = mus[num]
        sigma = sigmas[num]
        data.extend(np.random.normal(mu, sigma, (num_samples, 2)))
    labels = np.kron(range(num_classes), [1.0] * num_samples)

    return np.array(data), labels


class DataChunk:
    def __init__(self, data, label):
        self.__data = data
        self.__label = label
        self.__index_in_epoch = 0
        self.perm = np.arange(self.__data.shape[0])

    def iter_batches(self, batch_size):
        batch_count = self.__data.shape[0] // batch_size
        np.random.shuffle(self.perm)
        self.__index_in_epoch = batch_size

        for idx in range(batch_count):
            start = self.__index_in_epoch
            self.__index_in_epoch += batch_size
            indices = self.perm[start:self.__index_in_epoch]
            if len(indices) < batch_size:
                break
            yield self.__data[indices], self.__label[indices]

    def get_data(self):
        return self.__data

    def get_label(self):
        return self.__label
