import numpy as np
from operator import xor
import random
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    return x * (1 + np.exp(-x))


class NeuralNetwork:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, layer1_size, layer2_size, layer3_size):
        # Vygeneruji pole náhodně inicializovaných skrytých vrstev + bias
        self.l1 = np.random.randn(layer1_size, layer2_size)
        self.l2 = np.random.randn(layer2_size, layer1_size)
        self.l3 = np.random.randn(layer3_size, layer2_size)

    def reshape(self, x):
        l1_size = self.l1.shape[0] * self.l1.shape[1]
        l2_size = self.l2.shape[0] * self.l2.shape[1]
        l3_size = self.l3.shape[0] * self.l3.shape[1]
        l1 = x[0:l1_size]
        l2 = x[l1_size:(l1_size + l2_size)]
        l3 = x[(l1_size + l2_size):]
        l1 = l1.reshape(self.l1.shape)
        l2 = l2.reshape(self.l2.shape)
        l3 = l3.reshape(self.l3.shape)
        return l1, l2, l3

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        a1 = input_data
        z1 = self.l1.dot(a1)
        a2 = sigmoid(z1)
        z2 = self.l2.dot(a2)
        a3 = sigmoid(z2)
        z4 = self.l3.dot(a3)
        a4 = sigmoid(z4)
        return a4

    def cost_fn_for_minimalization(self, x, *args):
        l1, l2, l3 = self.reshape(x)
        learning_rate, input_data = args
        return self.cost_fn(l1, l2, l3, learning_rate, input_data)["cost"]

    def gradients_for_minimalization(self, x, *args):
        l1, l2, l3 = self.reshape(x)
        learning_rate, input_data = args
        return self.cost_fn(l1, l2, l3, learning_rate, input_data)["gradients"]

    def cost_fn(self, l1, l2, l3, learning_rate, input_data):
        result = 0
        m = len(input_data)
        l1_delta = 0
        l2_delta = 0
        l3_delta = 0
        for i in range(m):
            for k in range(l3.shape[0]):
                # Feed forward algoritmus
                # první vrstva
                a1 = input_data[i]["input"]
                z1 = l1.dot(a1)
                # druhá vrstva
                a2 = sigmoid(z1)
                z2 = l2.dot(a2)
                # Třetí vrstva
                a3 = sigmoid(z2)
                z3 = l3.dot(a3)
                a4 = sigmoid(z3)
                prediction = a4 # h(x_i)
                y = input_data[i]["output"]
                result += y * np.log(prediction) - np.dot((1 - y), np.log(1 - prediction))
                # chyba výstupu vrstvy
                delta4 = prediction - y
                delta3 = np.dot(l3.transpose(), delta4) * sigmoid_derivation(z3)
                delta2 = np.dot(l2.transpose(), delta3) * sigmoid_derivation(z2)
                l3_delta += learning_rate * np.dot(a4, delta4)
                l2_delta += learning_rate * np.dot(a3, delta3)
                l1_delta += learning_rate * np.dot(a2, delta2)
        # Cost funkce bez regularizace
        result *= -(1/m)

        # Regularizace
        regularization = sum(sum(l1 ** 2)) + sum(sum(l2 ** 2)) + sum(sum(l3 ** 2))
        regularization = (learning_rate / (2 * m)) * regularization
        result = result + regularization
        l1_delta += (learning_rate / m) * l1
        l2_delta += (learning_rate / m) * l2
        l3_delta += (learning_rate / m) * l3
        return {
            "cost": result[0],
            "gradients": np.append(np.append(l1_delta.flatten(),
                                             l2_delta.flatten()),
                                             l3_delta.flatten())

        }

    def train(self, input_data, learning_rate, epochs):
        theta = np.append(np.append(self.l1.flatten(), self.l2.flatten()), self.l3.flatten())
        random.shuffle(input_data)
        result = fmin_cg(f=self.cost_fn_for_minimalization, fprime=self.gradients_for_minimalization, x0=theta,
                         args=(learning_rate, input_data), maxiter=epochs)
        self.l1, self.l2, self.l3 = self.reshape(result)


if __name__ == '__main__':
    neural_net = NeuralNetwork(2, 2, 1)
    data = [np.array([x, y]).transpose() for x in range(0, 2) for y in range(0, 2)]
    input_data = [{
        "input": x,
        "output": int(xor(x[0], x[1]))
    } for x in data]
    print(input_data)
    neural_net.train(input_data, 0.00001, 90000)
    for data_point in data:
        print("nn(%s) = %s" % (data_point, neural_net.feed_forward(data_point)))
