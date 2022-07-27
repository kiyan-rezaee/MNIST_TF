import tensorflow as tf
import tensorflow_datasets

def normalizer(image, label):
    return tf.cast(image, tf.float32) / 256, label

(train, test), information = tensorflow_datasets.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
train = train.map(normalizer)
train = train.cache()
train = train.shuffle(information.splits['train'].num_examples)
train = train.batch(300)
train = train.prefetch(tf.data.experimental.AUTOTUNE)
test = test.map(normalizer)
test = test.batch(256)
test = test.cache()
test = test.prefetch(tf.data.experimental.AUTOTUNE)

# Create model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train, epochs=10, validation_data=test)


