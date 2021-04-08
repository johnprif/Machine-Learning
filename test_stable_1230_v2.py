# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.densenet import layers

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
test_images.shape

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(1, activation='sigmoid'),
   tf.keras.layers.Dense(500)
])

#model.add(layers.Dense(50,input_dim=2,activation='sigmoid'))
#model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))


# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

np.argmax(predictions_single[0])




