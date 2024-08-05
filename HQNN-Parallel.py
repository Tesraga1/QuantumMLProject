import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt

cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


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
#print(y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(126, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
])

def conv_circuit(parameters):
    length = len(parameters)
    target = QuantumCircuit(length)
    for x in range(length):
        target.rx(parameters[x], x)
    for x in range(length):
        target.rz(parameters[x],x)
    for x in range(length):
        target.ry(parameters[x],x)
    for x in range(length):
        target.rz(parameters[x], x)
    for x in range(length):
        if x == length-1:
            target.cx(x, 0)
        else:
            target.cx(x, x+1)
    target.measure_all()
    return target

params = ParameterVector("θ", length=5)
circuit = conv_circuit(params)
circuit.draw("mpl", style="clifford")
plt.show()

#batch_size = 64
#num_classes = 100
#epochs = 50

#model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['acc'])
#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)