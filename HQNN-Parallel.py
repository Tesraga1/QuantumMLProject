import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit, assemble, transpile
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
import pennylane as qml
import pennylane_qiskit as qmlq
#import matplotlib.pyplot as plt

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

cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

n_qubits = 5
    
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





params = ParameterVector("θ", length=5)
#circuit = conv_circuit(params)
#circuit.draw("mpl", style="clifford")
#plt.show()
qnode = qmlq.load(conv_circuit(params))
n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}
#qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

# Define the quantum device
dev = qml.device("default.qubit", wires=5)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(5):
        qml.RX(weights[i], wires=i)
        qml.RZ(weights[i], wires=i)
        qml.RY(weights[i], wires=i)
    for i in range(4):  # Entangle all qubits in a ring
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[4, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]

# Create the Keras layer from the QNode
qlayer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(126, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    qlayer,
    tf.keras.layers.Dense(100, activation='softmax')
])

batch_size = 5
num_classes = 100
epochs = 50

model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)