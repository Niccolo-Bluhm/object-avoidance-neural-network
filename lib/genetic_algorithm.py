from copy import deepcopy
import numpy as np


def mutate(mlp):
    """We are going to combine all the weights into one 1D array.
    After chaning the weights, we need to reshape them back into their original form."""
    # The Neural network for this ship.
    mlp = deepcopy(mlp)

    all_weights, all_biases = deconstruct_mlp(mlp)

    # Mutate anywhere from 5% to 20%.
    num_weights = int(np.round((np.random.rand() * 0.15 + 0.05) * len(all_weights)))
    num_biases = int(np.round((np.random.rand() * 0.15 + 0.05) * len(all_biases)) * np.round(np.random.rand()))

    # Array of indices to mutate.
    weight_indices = np.random.choice(range(0, len(all_weights)), size=num_weights, replace=False)
    bias_indices = np.random.choice(range(0, len(all_biases)), size=num_biases, replace=False)

    selector = np.random.rand()
    if selector > 0.5:
        # Scale the weights and biases by a random number between -1 and 3
        mutate_factor = np.random.uniform(-1, 3)
        all_weights[weight_indices] *= mutate_factor
        all_biases[bias_indices] *= mutate_factor
    else:
        # Replace selected weights and biases with random number between -1 and 1
        all_weights[weight_indices] = np.random.rand(len(weight_indices)) * 2 - 1
        all_biases[bias_indices] = np.random.rand(len(bias_indices)) * 2 - 1

    # Reconstruct
    mlp = construct_mlp(mlp, all_weights, all_biases)

    return mlp


def crossover(mlp1, mlp2):
    """We are going to combine all the weights into one 1D array.
    After changing the weights, we need to reshape them back into their original form."""

    mlp1 = deepcopy(mlp1)
    weights1, biases1 = deconstruct_mlp(mlp1)

    mlp2 = deepcopy(mlp2)
    weights2, biases2 = deconstruct_mlp(mlp2)

    # The number of biases to crossover, anywhere from 5 to 25%.
    num_biases = int( np.round(len(biases1) * np.random.uniform(0.05, 0.25)) )
    num_weights = int( np.round(len(weights1) * np.random.uniform(0.05, 0.25)) )

    weight_indices = np.random.choice(range(0, len(weights1)), size=num_weights, replace=False)
    bias_indices = np.random.choice(range(0, len(biases1)), size=num_biases, replace=False)

    # Perform crossover.
    weights1[weight_indices] = weights2[weight_indices]
    biases1[bias_indices] = biases2[bias_indices]

    # Reconstruct
    mlp1 = construct_mlp(mlp1, weights1, biases1)

    return mlp1


def deconstruct_mlp(mlp):
    """ Takes all biases and combines into a 1D array. Takes all weights and combines into a 1D array.
    :param mlp: sci-kit learn mlp
    :return: (weights, biases)
    """

    biases = np.concatenate(mlp.intercepts_)
    weights = np.array(mlp.coefs_[0].flatten())
    for idx in range(1, len(mlp.coefs_)):
        weights = np.concatenate((weights, mlp.coefs_[idx].flatten()))

    return weights, biases


def construct_mlp(mlp, weights, biases):
    """ This function takes the given weight array and bias array and integrates them back into the mlp class.
    :param mlp: sci-kit learn mlp
    :param weights: 1D array of all weights
    :param biases:  1D array of all biases
    :return: mlp
    """

    pos = 0
    for idx, bias_array in enumerate(mlp.intercepts_):
        mlp.intercepts_[idx] = biases[pos:pos + len(bias_array)]
        pos += len(bias_array)

    pos = 0
    for idx, weight_matrix in enumerate(mlp.coefs_):
        num_elements = mlp.coefs_[idx].size
        matrix_shape = mlp.coefs_[idx].shape
        # Get the 1D array of weights.
        coefs_1d = weights[pos:pos + num_elements]
        # Reshape back to original dimensions
        mlp.coefs_[idx] = coefs_1d.reshape(matrix_shape)
        pos += num_elements

    return mlp
