import math
import json
from random import random, randint

def inverse_tangent(x):
    if x >= -128 and x <= 127:
        return 2 / (1 + math.exp(-x)) - 1
    elif x >= 127:
        return 1
    elif x <= -128:
        return -1

class Neuron:
    def __init__(self, weight_count):
        self.weights = []
        self.bias = 2 * random() - 1

        for i in range(weight_count):
            self.weights.append(2 * random() - 1)

    def run(self, inputs):
        output = self.bias

        for i in range(len(self.weights)):
            output += self.weights[i] * inputs[i]

        return inverse_tangent(output)

class Layer:
    def __init__(self, neuron_count, weight_count):
        self.neurons = []

        for i in range(neuron_count):
            self.neurons.append(Neuron(weight_count))

    def run(self, inputs):
        outputs = []

        for i in range(len(self.neurons)):
            outputs.append(self.neurons[i].run(inputs[i]))

        return outputs

class Network:
    def __init__(self, input_count, hidden_layer_count, neurons_per_hidden_layer, output_count):
        self.layers = []

        self.layers.append(Layer(input_count, 1))

        for i in range(1, hidden_layer_count + 1):
            self.layers.append(Layer(neurons_per_hidden_layer, input_count if i == 1 else neurons_per_hidden_layer))

        self.layers.append(Layer(output_count, neurons_per_hidden_layer))

    def cost(self, inputs, expected_outputs):
        cost = 0

        for i in range(len(inputs)):
            outputs = self.run(inputs[i])

            for j in range(len(outputs)):
                err = outputs[j] - expected_outputs[i][j]
                cost += err * err

        return cost / len(inputs)

    def train(self, data, options):
        h = 0.5 # It's supposed to be like 0.0001 but it doesn't fuckin' work at that value.

        options = options or {}

        learning_rate = options.get("learning_rate", 0.3)
        iterations = options.get("iterations", 2000)
        batch_size = options.get("batch_size", 10)
        log = options.get("log", False)
        log_interval = options.get("log_interval", 200)
        error_threshold = options.get("error_threshold", 0.03)

        for epoch in range(iterations):
            temp_inputs = []
            temp_outputs = []

            if len(data) <= batch_size:
                for i in range(len(data)):
                    temp_inputs.append(data[i][0])
                    temp_outputs.append(data[i][1])
            else:
                for i in range(batch_size):
                    index = randint(0, len(data) - 1)

                    temp_inputs.append(data[index][0])
                    temp_outputs.append(data[index][1])

            orginal_cost = self.cost(temp_inputs, temp_outputs)

            if orginal_cost <= error_threshold:
                if log and epoch % log_interval == 0:
                    print(f"Epoch: {epoch}, Error: {orginal_cost}")

                continue

            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].neurons)):
                    for k in range(len(self.layers[i].neurons[j].weights)):
                        self.layers[i].neurons[j].weights[k] += h
                        cost_difference = self.cost(temp_inputs, temp_outputs) - orginal_cost
                        self.layers[i].neurons[j].weights[k] -= h + (cost_difference / h) * learning_rate

                    self.layers[i].neurons[j].bias += h
                    cost_difference = self.cost(temp_inputs, temp_outputs) - orginal_cost
                    self.layers[i].neurons[j].bias -= h + (cost_difference / h) * learning_rate

            if log and epoch % log_interval == 0:
                print(f"Epoch: {epoch}, Error: {self.cost(temp_inputs, temp_outputs)}")

    def as_JSON(self):
        brain = {"layers": []}

        for i in range(len(self.layers)):
            brain["layers"].append({
                "neurons": []
            })

            for j in range(len(self.layers[i].neurons)):
                brain["layers"][i]["neurons"].append({
                    "weights": self.layers[i].neurons[j].weights,
                    "bias": self.layers[i].neurons[j].bias
                })

        return brain

    def save(self, filename):
        with open(filename, "w") as file:
            file.write(json.dump(self.as_JSON()))

    @staticmethod
    def load(filename):
        brain = {}
        with open(filename, "r") as file:
            brain = json.loads(file.read())

        network = Network(
            len(brain["layers"][0]["neurons"]),
            len(brain["layers"]) - 2,
            len(brain["layers"][1]["neurons"]),
            len(brain["layers"][len(brain["layers"]) - 1]["neurons"])
        )

        for i in range(len(brain["layers"])):
            for j in range(len(brain["layers"][i]["neurons"])):
                network.layers[i].neurons[j].weights = brain["layers"][i]["neurons"][j]["weights"]
                network.layers[i].neurons[j].bias = brain["layers"][i]["neurons"][j]["bias"]

        return network

    def run(self, inputs):
        buffer = []

        for i in range(1, len(self.layers)):
            buffer.append([])

            for j in range(len(self.layers[i].neurons)):
                buffer[i - 1].append([])

        for i in range(len(self.layers[1].neurons)):
            buffer[0][i] = inputs

        for i in range(1, len(self.layers)):
            temp = self.layers[i].run(buffer[i - 1])

            if i >= len(self.layers) - 1:
                return temp

            for j in range(len(buffer[i])):
                buffer[i][j] = temp
