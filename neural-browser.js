function inverseTangent(x) {
    return 2 / (1 + Math.exp(-x)) - 1;
}

class Neuron {
    constructor(weightCount) {
        this.weights = [];
        this.bias = 2 * Math.random() - 1;

        for(let i = 0; i < weightCount; i++) {
            this.weights.push(2 * Math.random() - 1);
        }
    }

    run(inputs) {
        let output = this.bias;

        for(let i = 0; i < this.weights.length; i++) {
            output += this.weights[i] * inputs[i];
        }

        return inverseTangent(output);
    }
}

class Layer {
    constructor(neuronCount, weightCount) {
        this.neurons = [];

        for(let i = 0; i < neuronCount; i++) {
            this.neurons.push(new Neuron(weightCount));
        }
    }

    run(inputs) {
        let outputs = [];

        for(let i = 0; i < this.neurons.length; i++) {
            outputs.push(this.neurons[i].run(inputs[i]));
        }

        return outputs;
    }
}

class Network {
    constructor(inputCount, hiddenLayerCount, neuronsPerHiddenLayer, outputCount) {
        this.layers = [];

        this.layers.push(new Layer(inputCount, 1));

        for(let i = 1; i < hiddenLayerCount + 1; i++) {
            this.layers.push(new Layer(neuronsPerHiddenLayer, i == 1 ? inputCount : neuronsPerHiddenLayer));
        }

        this.layers.push(new Layer(outputCount, neuronsPerHiddenLayer));
    }

    cost(inputs, expectedOutputs) {
        let cost = 0;

        for(let i = 0; i < inputs.length; i++) {
            const outputs = this.run(inputs[i]);

            for(let j = 0; j < outputs.length; j++) {
                const err = outputs[j] - expectedOutputs[i][j];

                cost += err * err;
            }
        }

        return cost / inputs.length;
    }

    train(data, options) {
        const h = 0.5; // It's supposed to be like 0.0001 but it doesn't fuckin' work at that value.

        options = options || {};

        const learningRate = options.learningRate || 0.3;
        const iterations = options.iterations || 2000;
        const batchSize = options.batchSize || 10;
        const log = options.log || false;
        const logInterval = options.logInterval || 200;
        const errorThreshold = options.errorThreshold || 0.03;

        for(let epoch = 0; epoch < iterations; epoch++) {
            let tempInputs = [];
            let tempOutputs = [];

            if(data.length <= batchSize) {
                for(let i = 0; i < data.length; i++) {
                    tempInputs.push(data[i][0]);
                    tempOutputs.push(data[i][1]);
                }
            } else {
                for(let i = 0; i < batchSize; i++) {
                    const index = Math.floor(Math.random() * data.length);

                    tempInputs.push(data[index][0]);
                    tempOutputs.push(data[index][1]);
                }
            }

            const orginalCost = this.cost(tempInputs, tempOutputs);

            if(orginalCost <= errorThreshold) {
                if(log && epoch % logInterval == 0) {
                    console.log(`Epoch: ${epoch}, Error: ${orginalCost}`);
                }

                continue;
            }

            for(let i = 0; i < this.layers.length; i++) {
                for(let j = 0; j < this.layers[i].neurons.length; j++) {
                    for(let k = 0; k < this.layers[i].neurons[j].weights.length; k++) {
                        this.layers[i].neurons[j].weights[k] += h;
                        const costDifference = this.cost(tempInputs, tempOutputs) - orginalCost;
                        this.layers[i].neurons[j].weights[k] -= h + (costDifference / h) * learningRate;
                    }

                    this.layers[i].neurons[j].bias += h;
                    const costDifference = this.cost(tempInputs, tempOutputs) - orginalCost;
                    this.layers[i].neurons[j].bias -= h + (costDifference / h) * learningRate;

                }
            }

            if(log && epoch % logInterval == 0) {
                console.log(`Epoch: ${epoch}, Error: ${this.cost(tempInputs, tempOutputs)}`);
            }
        }
    }

    asJSON() {
        const brain = {layers: []};

        for(let i = 0; i < this.layers.length; i++) {
            brain.layers.push({
                neurons: []
            });

            for(let j = 0; j < this.layers[i].neurons.length; j++) {
                brain.layers[i].neurons.push({
                    weights: this.layers[i].neurons[j].weights,
                    bias: this.layers[i].neurons[j].bias
                });
            }
        }

        return brain;
    }

    static load(brain) {
        const network = new Network(
            brain.layers[0].neurons.length,
            brain.layers.length - 2,
            brain.layers[1].neurons.length,
            brain.layers[brain.layers.length - 1].neurons.length
        );

        for(let i = 0; i < brain.layers.length; i++) {
            for(let j = 0; j < brain.layers[i].neurons.length; j++) {
                network.layers[i].neurons[j].weights = brain.layers[i].neurons[j].weights;
                network.layers[i].neurons[j].bias = brain.layers[i].neurons[j].bias;
            }
        }

        return network;
    }

    run(inputs) {
        let buffer = [];

        for(let i = 1; i < this.layers.length; i++) {
            buffer.push([]);

            for(let j = 0; j < this.layers[i].neurons.length; j++) {
                buffer[i - 1].push([]);
            }
        }

        for(let i = 0; i < this.layers[1].neurons.length; i++) {
            buffer[0][i] = inputs;
        }

        for(let i = 1; i < this.layers.length; i++) {
            let temp = this.layers[i].run(buffer[i - 1]);

            if(i >= this.layers.length - 1) {
                return temp;
            }

            for(let j = 0; j < buffer[i].length; j++) {
                buffer[i][j] = temp;
            }
        }

    }
}
