import tensorflow as tf
import numpy as np
from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.circuit import ParameterVector
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

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.circuit, self.params = self.create_quantum_circuit()
        self.backend = Aer.get_backend('aer_simulator')

    def create_quantum_circuit(self):
        params = ParameterVector('θ', self.n_qubits)
        circuit = conv_circuit(params)
        return circuit, params

    def build(self, input_shape):
        self.weights = self.add_weight(
            name='weights',
            shape=(input_shape[-1], self.n_qubits),
            initializer='uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = []

        for i in range(batch_size):
            parameter_values = {self.params[j]: tf.tensordot(inputs[i], self.weights[:, j], axes=1) for j in range(self.n_qubits)}
            circuit = self.circuit.bind_parameters(parameter_values)
            qobj = assemble(transpile(circuit, self.backend))
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            counts_array = np.array([counts.get(bin(k)[2:].zfill(self.n_qubits), 0) for k in range(2**self.n_qubits)])
            outputs.append(counts_array / sum(counts_array))
        
        return tf.convert_to_tensor(outputs)
    
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
    QuantumLayer(n_qubits),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
])



#params = ParameterVector("θ", length=5)
#circuit = conv_circuit(params)
#circuit.draw("mpl", style="clifford")
#plt.show()

batch_size = 64
num_classes = 100
epochs = 50

model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)