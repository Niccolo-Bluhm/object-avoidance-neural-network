from copy import deepcopy
import numpy as np


def mutate(mlp):
    """We are going to combine all the weights into one 1D array.
    After chaning the weights, we need to reshape them back into their original form."""
    # The Neural network for this ship
    mlp = deepcopy(mlp)

    # Store shape information for reconstruction
    s0 = len(mlp.intercepts_[0])
    s1 = len(mlp.intercepts_[1])
    s2 = mlp.coefs_[0].shape
    s3 = mlp.coefs_[1].shape

    # Combine all weights into one array
    intercepts = np.concatenate((mlp.intercepts_[0], mlp.intercepts_[1]))
    weights1 = mlp.coefs_[0].flatten()
    weights2 = mlp.coefs_[1].flatten()
    all_weights = np.concatenate((weights1, weights2))

    # Mutate anywhere from 5% to %20
    num_m_weights = int(np.round((np.random.rand() * 0.15 + 0.05) * len(all_weights)))
    num_m_intercepts = int(np.round((np.random.rand() * 0.15 + 0.05) * len(intercepts))) * int(np.round(
        np.random.rand()))

    # Array of indices to mutate
    m_inds_w = np.random.choice(range(0, len(all_weights)), size=num_m_weights, replace=False)
    m_inds_i = np.random.choice(range(0, len(intercepts)), size=num_m_intercepts, replace=False)

    selector = np.random.rand()
    if selector > 0.5:

        mutate_factor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
        for ii in range(len(m_inds_w)):
            all_weights[m_inds_w[ii]] = all_weights[m_inds_w[ii]] * mutate_factor

        mutate_factor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
        if num_m_intercepts != 0:
            for ii in range(len(m_inds_i)):
                intercepts[m_inds_i[ii]] = all_weights[m_inds_w[ii]] * mutate_factor
    else:

        for ii in range(len(m_inds_w)):
            all_weights[m_inds_w[ii]] = np.random.rand() * 2 - 1

        if num_m_intercepts != 0:
            for ii in range(len(m_inds_i)):
                intercepts[m_inds_i[ii]] = np.random.rand() * 2 - 1

    # Reconstruct
    intercepts_0 = intercepts[range(s0)]
    intercepts_1 = intercepts[range(s0, s1 + s0)]
    coefs_0 = all_weights[range(len(weights1))].reshape(s2)
    coefs_1 = all_weights[range(len(weights1), len(weights2) + len(weights1))].reshape(s3)

    # Add the new weights back into the neural network
    mlp.intercepts_[0] = intercepts_0
    mlp.intercepts_[1] = intercepts_1
    mlp.coefs_[0] = coefs_0
    mlp.coefs_[1] = coefs_1

    return mlp
