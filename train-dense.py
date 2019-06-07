# %% Imports
import random
import tensorflow as tf
import numpy as np

SET_SIZE = 100_000
TRAIN_FRACTION = 0.75

print("A test network with a test generated data set. \
    Doesn't matter if it doesn't train well, all that \
        matters is it trains at all", end='\n\n')

print("Is GPU available: ", tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
), end='\n\n')

print("Generating data set for multiplying numbers by 2")
x = list(range(SET_SIZE))
y = list(map(lambda n: n * 2, x))
pairs = list(zip(x, y))

print("Splitting into train/test x/y sets deterministically")
random.Random(123).shuffle(pairs)

train_size = int(SET_SIZE * TRAIN_FRACTION)
test_size = int(SET_SIZE - train_size)

train = pairs[:train_size-1]
test = pairs[train_size:train_size+test_size-1]

train_x = np.array(list(map(lambda xy: [xy[0] / SET_SIZE], train)))
train_y = np.array(list(map(lambda xy: xy[1], train)))
test_x = np.array(list(map(lambda xy: [xy[0] / SET_SIZE], test)))
test_y = np.array(list(map(lambda xy: xy[1], test)))

print("Data set created", end='\n\n')

# %% Building model

print("Building & training model")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, 'relu'),
    tf.keras.layers.Dense(4, 'relu'),
    tf.keras.layers.Dense(2, 'relu'),
    tf.keras.layers.Dense(1, 'relu'),
])

model.compile(
    tf.keras.optimizers.Adamax(),
    loss=tf.keras.losses.mae,
)


model.fit(
    x=train_x, y=train_y,
    epochs=10, batch_size=20,
    validation_data=(test_x, test_y)
)

print("Evaluating model")
model.evaluate(test_x, test_y)
