# Neural

Example of an AI trained for a three input XOR gate.

NodeJS:
```JS
const {Network} = require("neural.js");
const net = new Network(3, 3, 5, 1); // A network with 3 inputs, 3 hidden layers with 5 neurons each, and 1 output.

net.train([
    [[0, 0, 0], [0]],
    [[0, 0, 1], [1]],
    [[0, 1, 0], [1]],
    [[0, 1, 1], [0]],
    [[1, 0, 0], [1]],
    [[1, 0, 1], [0]],
    [[1, 1, 0], [0]],
    [[1, 1, 1], [1]]
], {
    iterations: 10000,
    log: true,
    logInterval: 500,
    learningRate: 0.02,
    errorThreshold: 0.03
});

console.log(net.run([0, 1, 1])); // Expected: [0]
```

Python:
```PY
from neural import Network
net = Network(3, 3, 5, 1) # A network with 3 inputs, 3 hidden layers with 5 neurons each, and 1 output.

net.train([
    [[0, 0, 0], [0]],
    [[0, 0, 1], [1]],
    [[0, 1, 0], [1]],
    [[0, 1, 1], [0]],
    [[1, 0, 0], [1]],
    [[1, 0, 1], [0]],
    [[1, 1, 0], [0]],
    [[1, 1, 1], [1]]
], {
    "iterations": 10000,
    "log": True,
    "logInterval": 500,
    "learningRate": 0.02,
    "errorThreshold": 0.03
})

print(net.run([0, 1, 1])) # Expected: [0]
```

HTML / JavaScript:
```HTML
<script src = "https://raw.githubusercontent.com/ImaEntity/Neural/2cb5eebe2d4b668def0e2bf9dc2919d24e275cf5/neural-browser.js"></script>
<script>
    const net = new Network(3, 3, 5, 1); // A network with 3 inputs, 3 hidden layers with 5 neurons each, and 1 output.

    net.train([
        [[0, 0, 0], [0]],
        [[0, 0, 1], [1]],
        [[0, 1, 0], [1]],
        [[0, 1, 1], [0]],
        [[1, 0, 0], [1]],
        [[1, 0, 1], [0]],
        [[1, 1, 0], [0]],
        [[1, 1, 1], [1]]
    ], {
        iterations: 10000,
        log: true,
        logInterval: 500,
        learningRate: 0.02,
        errorThreshold: 0.03
    });

    console.log(net.run([0, 1, 1])); // Expected: [0]
</script>
```

##Nsdasd
