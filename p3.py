# # Simulating one neuron with four unique inputs
# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2

# # formula: (inputs x weights) + bias
# output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias
# print(output)

# # ================================================= #

# Simulating 3 neurons with four inputs
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# 3 neurons means we have 3 unique output for each neuron
output = [
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3,
]
print(output)

# ================================================= #

# Simplifing p2.py into a more dynamic code
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layers_output = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    neuron_output += neuron_bias
    layers_output.append(neuron_output)

print(layers_output)

# ================================================= #

# Dot product with numpy
import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]  # using only one set of weights
bias = 2

output = np.dot(weights, inputs) + bias  # always use weight first in dot product
print(output)

# ================================================= #

# Dot product for the 3 neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]  # using only one set of weights
bias = 2

output = np.dot(weights, inputs) + bias  # always use weights first in dot product
# weights is used first as it is a interpreted as a matrix (3D Array).
# if input is used first, an error is thrown and the program will not run.
# the error occurs because the matrix needs to always be in the first position.
# both columns needs to be 4 in size as each input needs to have a weight associated to it.

# the shape of weights is 3 x 4 (3 rows 4 columns) or shape(3,4)
# print(np.array(weights).shape)

# the shape of inputs is 1 x 4 (1 row 4 columns) or shape(4,)
# shape(4,) - can be understood as: shape(1,4) however this is not the syntax
# print(np.array(inputs).shape)
print(output)