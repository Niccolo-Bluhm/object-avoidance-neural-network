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
        mutate_factor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
        for idx in weight_indices:
            all_weights[idx] = all_weights[idx] * mutate_factor

        mutate_factor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
        for idx in bias_indices:
            all_biases[idx] = all_weights[idx] * mutate_factor
    else:
        for idx in weight_indices:
            all_weights[idx] = np.random.rand() * 2 - 1

        for idx in bias_indices:
            all_biases[idx] = np.random.rand() * 2 - 1

    # Reconstruct
    mlp = construct_mlp(mlp, all_weights, all_biases)

    return mlp


def crossover(mlp1, mlp2):
    """We are going to combine all the weights into one 1D array.
    After chaning the weights, we need to reshape them back into their original form."""
    # The MLP Neural network for this ship.
    mlp1 = deepcopy(mlp1)
    mlp2 = deepcopy(mlp2)

    # Store shape information for reconstruction.
    s0 = len(mlp1.intercepts_[0])
    s1 = len(mlp1.intercepts_[1])
    s2 = mlp1.coefs_[0].shape
    s3 = mlp1.coefs_[1].shape

    intercepts = np.concatenate( (mlp1.intercepts_[0], mlp1.intercepts_[1]))
    weights1 = mlp1.coefs_[0].flatten()
    weights2 = mlp1.coefs_[1].flatten()
    allWeights = np.concatenate((weights1, weights2))

    intercepts2= np.concatenate( (mlp2.intercepts_[0], mlp2.intercepts_[1]))
    weights12 = mlp2.coefs_[0].flatten()
    weights22 = mlp2.coefs_[1].flatten()
    allWeights2 = np.concatenate((weights12, weights22))

    # Crossover anywhere from 20% to 60%.
    num_m_intercepts = int(np.round((np.random.rand()*0.15+0.05)  * len(intercepts))) * int(np.round(np.random.rand()+0.3))
    num_m_weights = int(np.round((np.random.rand() * 0.15 + 0.05) * len(allWeights)))

    m_inds_w = np.random.choice(range(0,len(allWeights)), size = num_m_weights, replace = False)
    m_inds_i = np.random.choice(range(0,len(intercepts)), size = num_m_intercepts, replace = False)

    for ii in range(len(m_inds_w)):
        allWeights[m_inds_w[ii]] = allWeights2[m_inds_w[ii]]

    if(num_m_intercepts !=0):
        for ii in range(len(m_inds_i)):
            intercepts[m_inds_i[ii]] = intercepts2[m_inds_i[ii]]

    #Reconstruct
    intercepts_0 = intercepts[range(s0)]
    intercepts_1 = intercepts[range(s0,s1+s0)]
    coefs_0 = allWeights[range(len(weights1))].reshape(s2)
    coefs_1 = allWeights[range(len(weights1),len(weights2)+len(weights1))].reshape(s3)

    #Add the new weights back into the neural network
    mlp1.intercepts_[0] = intercepts_0
    mlp1.intercepts_[1] = intercepts_1
    mlp1.coefs_[0] = coefs_0
    mlp1.coefs_[1] = coefs_1

    return mlp1


def deconstruct_mlp(mlp):
    """ Takes all biases and combines into one array. Takes all weights and combines into one array.
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
