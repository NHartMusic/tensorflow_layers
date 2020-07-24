const model = tf.sequential()

const hidden = tf.layers.dense({
    units: 4,
    inputShape: 2,
    activation: 'sigmoid'
})

const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
})

model.add(hidden)
model.add(output)

const sgdOpt = tf.train.sgd(0.1)

model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
})

const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1]
])

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
])


// const config = {
//     verbose: true,
//     epochs: 100
// }

train().then(() => {
    console.log('training is complete')
    let outputs = model.predict(xs)
    outputs.print()
})

async function train() {
    for (let i = 0; i < 100; i++) {
        const config = {
            shuffle: true,
            epochs: 10
        }
        const response = await model.fit(xs, ys, config)
        console.log(response.history.loss[0])
    }
}



// const xs = tf.tensor2d([
//     [0.25, 0.92],
//     [0.12, 0.3],
//     [0.4, 0.74],
//     [0.1, 0.22]
// ])

