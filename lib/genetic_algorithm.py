from copy import deepcopy
import numpy as np


def select_and_evolve(space_ships):
    """Given a list of space ships, this function selects the most fit of the population, crossbreeds and mutates them,
    and passes them to the next generation.

    :param space_ships: list of SpaceShip objects.
    :return: List of SpaceShip objects.
    """

    # Remove the pygame screen because it cannot be deepcopied.
    screen_backup = space_ships[0].screen
    for ship in space_ships:
        ship.screen = None

    # The lower the distance, the better the ship did (It got closer to the white planet)
    fitness_list = np.array([ship.fitness for ship in space_ships])

    # Sort ships in order of fitness, with the ship at index 0 having the highest fitness.
    indices = np.argsort(fitness_list)[::-1]
    fitness = fitness_list[indices]
    sorted_ships = [space_ships[idx] for idx in indices]

    # Assign a selection probability to each ship based on fitness.
    scores_sum = np.sum(fitness)
    probabilities = fitness / scores_sum

    # Take best performing ships (top 20%) and introduce directly to next round.
    new_ships = []
    num_top_ships = int(len(sorted_ships) * 0.2)
    for idx in range(num_top_ships):
        new_ships.append(deepcopy(sorted_ships[idx]))

    # Take a weighted selection of 10% of ships, mutate them, and introduce to next round (Skip crossover)
    num_ten_percent = int(len(sorted_ships) * 0.1)
    ships_to_mutate = np.random.choice(sorted_ships, size=num_ten_percent, replace=False, p=probabilities)
    for ship in ships_to_mutate:
        new_ships.append(mutate(ship))

    # Whatever ships we have left mutate + crossbreed
    for idx in range(len(sorted_ships) - len(new_ships)):
        # Select two parents
        parents = np.random.choice(sorted_ships, size=2, replace=False, p=probabilities)
        child_ship = crossover(parents[0], parents[1])
        new_ships.append(mutate(child_ship))

    # Add screen property back in.
    for ship in new_ships:
        ship.screen = screen_backup

    return new_ships


def mutate(ship):
    ship = deepcopy(ship)

    # The Neural network for this ship.
    mlp = ship.mlp

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

    ship.mlp = mlp

    return ship


def crossover(ship1, ship2):
    ship1 = deepcopy(ship1)
    ship2 = deepcopy(ship2)

    mlp1 = ship1.mlp
    weights1, biases1 = deconstruct_mlp(mlp1)

    mlp2 = ship2.mlp
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

    output_ship = ship1
    output_ship.mlp = mlp1

    return output_ship


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
