#with high level api

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = tf.reshape(x_train, (-1 , 28 * 28))
x_test = tf.reshape(x_test, (-1 , 28 * 28))

print(x_train.shape)
print(x_test.shape)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1024 , activation='relu'),
        tf.keras.layers.Dense(512 , activation='relu'),
        tf.keras.layers.Dense(128 , activation='relu'),
        tf.keras.layers.Dense(10)
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size = 32, epochs = 5 , verbose = 2)

model.evaluate(x_test , y_test , batch_size = 32 , verbose = 2)
