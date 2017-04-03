import numpy as np
from operator import xor


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
        self.l1[:, 0] = 1
        self.l2[:, 0] = 1
        self.l3[:, 0] = 1

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

    def learn(self, l1, l2, l3, input_data, learning_rate):
        m = len(input_data)
        for i in range(m):
            # Feed forward algoritmus
            # první vrstva
            a1 = input_data[i]["input"]
            z1 = l1.dot(a1)
            # druhá vrstva
            a2 = sigmoid(z1)
            a2 = np.vstack([a2, np.ones(a2.shape[0])])
            z2 = l2.dot(a2)
            # Třetí vrstva
            a3 = sigmoid(z2)
            z3 = l3.dot(a3)
            a4 = sigmoid(z3)
            prediction = a4 # h(x_i)
            y = input_data[i]["output"]
            # chyba výstupu vrstvy
            delta4 = prediction - y
            delta3 = np.dot(l3.transpose(), delta4) * sigmoid_derivation(z3)
            delta2 = np.dot(l2.transpose(), delta3) * sigmoid_derivation(z2)
            l3 += learning_rate * np.dot(delta4, a3)
            l2 += learning_rate * np.dot(delta3, a2)
            l1 += learning_rate * np.dot(delta2, a1)
        return l1, l2, l3

    def cost_fn(self, l1, l2, l3, learning_rate, input_data):
        result = 0
        m = len(input_data)
        for i in range(m):
            for k in range(len(l3.shape[0])):
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
        # Cost funkce bez regularizace
        result *= -(1/m)

        # Regularizace
        regularization = sum(sum(l1 ** 2)) + sum(sum(l2 ** 2)) + sum(sum(l3 ** 2))
        regularization = (learning_rate / (2 * m)) * regularization
        result = result + regularization
        return result

    def train(self, input_data, learning_rate, epochs):
        for i in range(epochs):
            self.l1, self.l2, self.l3 = self.learn(self.l1, self.l2, self.l3, input_data, learning_rate)


if __name__ == '__main__':
    neural_net = NeuralNetwork(2, 2, 1)
    data = [np.array([x, y]).transpose() for x in range(0, 2) for y in range(0, 2)]
    input_data = [{
        "input": x,
        "output": xor(x[0], x[1])
    } for x in data]
    neural_net.train(input_data, 1, 1000)
    for data_point in data:
        print("nn(%s) = %s" % (data_point, neural_net.feed_forward(data_point)))
