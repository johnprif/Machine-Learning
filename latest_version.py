#https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
import numpy as np
#neurons
K=500

averange_acc=0
averange_f1=0

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()

def preprocess(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)

  return x, y

def create_dataset(xs, ys, n_classes=10):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    keras.layers.Dense(units=K, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='sgd',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_dataset.repeat(),
    epochs=2,
    steps_per_epoch=1875,
    validation_data=val_dataset.repeat(),
    validation_steps=2)

predictions = model.predict(val_dataset)

averange_acc+=accuracy_score(train_dataset, predictions)
averange_f1+=f1_score(train_dataset, predictions)

print("The accurace is", averange_acc)
print("The f1 score is", averange_f1)

np.argmax(predictions[0])

