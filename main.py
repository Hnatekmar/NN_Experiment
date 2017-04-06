import numpy as np
from scipy.optimize import *
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    return np.multiply(x, (1.0 - x))


class NeuralNetwork:
    """
    Třívrstvý vícevrsvý perceptron
    """
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.hidden_layer = np.random.randn(hidden_layer_size, input_size + 1).astype(np.float64) * 0.24 - 0.12
        self.output_layer = np.random.randn(output_size, hidden_layer_size + 1).astype(np.float64) * 0.24 - 0.12
        self.number_of_outputs = output_size
        self.memoize = {}
        self.j = []

    def reshape(self, x):
        l1_size = self.hidden_layer.shape[0] * self.hidden_layer.shape[1]
        l1 = x[0:l1_size]
        l2 = x[l1_size:]
        l1 = l1.reshape(self.hidden_layer.shape)
        l2 = l2.reshape(self.output_layer.shape)
        return l1, l2

    def feed_forward(self, input_data):
        """
        Provede predikci s pomocí feed forward algoritmu
        :param input_data:  vstupní data o velikosti
        :return: predikce
        """
        x = input_data
        x = np.insert(x, 0, 1)
        a1 = x.T
        z2 = np.dot(self.hidden_layer, a1)
        a2 = sigmoid(z2)
        bias = np.ones((1, 1))
        a2 = np.vstack([bias, a2])
        z3 = np.dot(self.output_layer, a2)
        a3 = sigmoid(z3)
        result = a3
        return np.argmax(result)

    def cost_fn_for_minimalization(self, x, *args):
        try:
            return self.memoize[str(x)]["cost"]
        except KeyError:
            l1, l2 = self.reshape(x)
            learning_rate, input_data = args
            cost = self.cost_fn(l1, l2, learning_rate, input_data)
            self.j.append(cost["cost"])
            self.memoize[str(x)] = cost
            return cost["cost"]

    def gradients_for_minimalization(self, x, *args):
        try:
            return self.memoize[str(x)]["gradients"]
        except KeyError:
            l1, l2 = self.reshape(x)
            learning_rate, input_data = args
            gradients = self.cost_fn(l1, l2, learning_rate, input_data)
            self.memoize[str(x)] = gradients
            return self.memoize[str(x)]["gradients"]

    def cost_fn(self, theta1, theta2, learning_rate, input_data):
        J = 0
        m = len(input_data)
        l1_delta = 0
        l2_delta = 0
        for i in range(m):
            # Feed forward algoritmus
            # první vrstva
            x = input_data[i]["input"]
            x = np.insert(x, 0, 1)
            a1 = x.T
            z2 = np.dot(theta1, a1)
            a2 = sigmoid(z2)
            bias = np.ones((1, 1))
            a2 = np.vstack([bias, a2])
            z3 = np.dot(theta2, a2)
            a3 = sigmoid(z3)
            y = np.matrix([int(x) for x in np.arange(self.number_of_outputs) == input_data[i]["output"]])
            prediction = a3
            J += (np.dot(-y, np.log(prediction)) - np.dot((1 - y), np.log(1 - prediction)))
            # Výpočet gradientů
            delta3 = prediction - y.T
            delta2_part = np.dot(theta2.T, delta3)
            delta2 = np.multiply(delta2_part, np.multiply(a2, (1 - a2)))
            l1_delta += np.dot(delta2[1:, :], a1.T)
            l2_delta += np.dot(delta3, a2.T)

        # Cost funkce bez regularizace
        J = J/(2 * m)
        l1_delta = l1_delta / m
        l2_delta = l2_delta / m

        th1 = theta1[:, 1:]
        th2 = theta2[:, 1:]
        # Regularizace
        regularization = sum(sum(th1 ** 2)) + sum(sum(th2 ** 2))
        regularization *= (learning_rate / (2 * m))
        J += regularization

        l1_delta[:, 1:] += np.multiply(theta1[:, 1:], (learning_rate / m))
        l2_delta[:, 1:] += np.multiply(theta2[:, 1:], (learning_rate / m))
        l1_delta = np.squeeze(np.asarray(l1_delta.flatten()))
        l2_delta = np.squeeze(np.asarray(l2_delta.flatten()))
        J = J[0, 0]
        return {
            "cost": J,
            "gradients": np.append(l2_delta,
                                   l1_delta)
        }

    def train(self, input_data, learning_rate):
        theta = np.append(self.hidden_layer.flatten(), self.output_layer.flatten())
        result = fmin_ncg(self.cost_fn_for_minimalization, theta, fprime=self.gradients_for_minimalization,
                           args=(learning_rate, input_data), retall=True)

        self.hidden_layer, self.output_layer = self.reshape(result[0])


def image_to_vector(path):
    img = Image.open(path).convert('LA')
    data = np.matrix([x[0] for x in img.getdata()], dtype=np.float64)
    data /= data.max()
    return data


if __name__ == '__main__':
    input_data = []
    labels = []
    class_number = 0
    for folder in os.listdir("./alphabet/lowercase"):
        labels.append(folder)
        for sample in os.listdir("./alphabet/lowercase/%s" % folder):
            input_data.append({
                "input": image_to_vector("./alphabet/lowercase/%s/%s" % (folder, sample)),
                "output": class_number,
                "filename": sample
            })
            break
        class_number += 1
        if class_number >= 2:
            break

    neural_net = NeuralNetwork(100 * 100, 55, len(labels))

    random.shuffle(input_data)
    training_set = input_data[:int(len(input_data) * 1)]
    neural_net.train(input_data, 0.1)

    classified = 0
    correctly_classified = 0
    classification_data = input_data[int(len(input_data) * 0.7):]
    for sample in classification_data:
        classified += 1
        if neural_net.feed_forward(sample["input"]) == sample["output"]:
            correctly_classified += 1

    print("Úspěšnost %s " % str((correctly_classified / classified) * 100))

    for i in range(5):
        indx = random.randint(0, len(input_data) - 1)
        fig = plt.imshow(input_data[indx]["input"].reshape((100, 100)))
        plt.show()
        j, k = (neural_net.feed_forward(input_data[indx]["input"]), input_data[indx]["output"])
        print("Neuronka si myslí, že vzorek je %s skutečnost je %s" % (labels[j], labels[k]))
    """
    import operator
    xor = operator.xor
    neural_net = NeuralNetwork(2, 5, 1)
    data = [np.matrix([x, y]) for x in range(0, 2) for y in range(0, 2)]
    input_data = []
    for x in data:
        input_data.append({
            "input": x,
            "output": int(xor(x[0, 0], x[0, 1]))
        })
    random.shuffle(input_data)
    neural_net.train(input_data, 0.000000000001)
    for data_point in data:
        print("nn(%s) = %s" % (data_point, neural_net.feed_forward(data_point)))
    """
