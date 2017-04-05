import numpy as np
from scipy.optimize import *


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    x = np.clip(x, -500, 500)
    return np.multiply(x, (1.0 - x))


class NeuralNetwork:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.hidden_layer = np.random.randn(input_size, hidden_layer_size)
        self.output_layer = np.random.randn(hidden_layer_size, output_size)
        self.number_of_outputs = output_size
        self.j = []

    def reshape(self, x):
        l1_size = self.hidden_layer.shape[0] * self.hidden_layer.shape[1]
        l2_size = self.output_layer.shape[0] * self.output_layer.shape[1]
        l1 = x[0:l1_size]
        l2 = x[l1_size:(l1_size + l2_size)]
        l1 = l1.reshape(self.hidden_layer.shape)
        l2 = l2.reshape(self.output_layer.shape)
        return l1, l2

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        a1 = input_data.transpose()
        z1 = np.dot(a1, self.hidden_layer)
        a2 = sigmoid(z1)
        z2 = np.dot(a2, self.output_layer)
        a3 = sigmoid(z2)
        return np.argmax(a3)

    def cost_fn_for_minimalization(self, x, *args):
        l1, l2 = self.reshape(x)
        learning_rate, input_data = args
        cost =  self.cost_fn(l1, l2, learning_rate, input_data)["cost"]
        self.j.append(cost)
        return cost

    def gradients_for_minimalization(self, x, *args):
        l1, l2 = self.reshape(x)
        learning_rate, input_data = args
        return self.cost_fn(l1, l2, learning_rate, input_data)["gradients"]

    def cost_fn(self, l1, l2, learning_rate, input_data):
        J = 0
        m = len(input_data)
        l1_delta = 0
        l2_delta = 0
        for i in range(m):
            # Feed forward algoritmus
            # první vrstva
            a1 = input_data[i]["input"]
            z1 = np.dot(a1, l1)
            a2 = sigmoid(z1)
            z2 = np.dot(a2, l2)
            a3 = sigmoid(z2)
            y = np.matrix([int(x) for x in np.arange(self.number_of_outputs) == input_data[i]["output"]])
            prediction = a3.T
            J += np.dot(y, np.log(prediction)) - np.dot((1 - y), np.log(1 - prediction))
            # Výpočet gradientů
            delta3 = prediction.T - y
            delta2_part = np.dot(delta3, l2.T)
            delta2 = np.multiply(delta2_part, sigmoid_derivation(z1))
            l1_delta += np.dot(a1.T, delta2)
            l2_delta += np.dot(a2.T, delta3)

        # Cost funkce bez regularizace
        J *= -(1/m)

        # Regularizace
        regularization = sum(sum(l1 ** 2)) + sum(sum(l2 ** 2))
        regularization = (learning_rate / (2 * m)) * regularization
        J += regularization
        l1_delta = l1_delta * (learning_rate / m)
        l2_delta = l2_delta * (learning_rate / m)
        l1_delta = np.squeeze(np.asarray(l1_delta.flatten()))
        l2_delta = np.squeeze(np.asarray(l2_delta.flatten()))
        J = J[0, 0]
        return {
            "cost": J,
            "gradients": np.append(l1_delta.flatten(),
                                   l2_delta.flatten())
        }

    def train(self, input_data, learning_rate):
        import matplotlib.pyplot as plt
        theta = np.append(self.hidden_layer.flatten(), self.output_layer.flatten())
        result = fmin_ncg(f=self.cost_fn_for_minimalization, fprime=self.gradients_for_minimalization, x0=theta,
                         args=(learning_rate, input_data), full_output=True)
        self.hidden_layer, self.output_layer = self.reshape(result[0])
        plt.plot(self.j)
        plt.xlabel("iterace")
        plt.ylabel("$J(\\theta)$")
        plt.show()


def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict

import random
if __name__ == '__main__':
    neural_net = NeuralNetwork(32 * 32 * 3, 25, 10)

    training_set = unpickle("cifar-10-batches-py/data_batch_1")
    input_data = []
    for i in range(int(len(training_set["labels"]) * 0.1)):
        input_data.append({
            "input": np.matrix(training_set["data"][i]),
            "output": training_set["labels"][i]
        })

    neural_net.train(input_data, 0.00001)
    correctly_classified = 0
    classified = 0
    for i in range(int(len(input_data) * 0.3)):
        classified += 1
        indx = random.randint(0, len(training_set["labels"]))
        cmp = training_set["labels"][indx] == neural_net.feed_forward(training_set["data"][indx])
        if cmp:
            correctly_classified += 1

    print("Úspěšnost %s " % str((correctly_classified / classified) * 100))
