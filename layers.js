const model = tf.sequential()

const hidden = tf.layers.dense({
    units: 4,
    inputShape: 2,
    activation: 'sigmoid'
})

const output = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
})

model.add(hidden)
model.add(output)

const sgdOpt = tf.train.sgd(0.1)

model.compile({
    optimizer: sgdOpt,
    loss: 'meanSquaredError'
})

const inputs = tf.tensor2d([
    [0.25, 0.92],
    [0.12, 0.3],
    [0.4, 0.74],
    [0.1, 0.22]
])

let outputs = model.predict(inputs)
outputs.print()