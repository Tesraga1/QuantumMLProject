import tensorflow as tf
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train=x_train / 255.0
x_test=x_test/255.0


y_train = tf.one_hot(y_train, 
                     depth=y_train.max() + 1, 
                     dtype=tf.float64) 
y_test = tf.one_hot(y_test, 
                   depth=y_test.max() + 1, 
                   dtype=tf.float64) 
  
y_train = tf.squeeze(y_train) 
y_test = tf.squeeze(y_test)
print(y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(126, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

batch_size = 64
num_classes = 100
epochs = 50

model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')