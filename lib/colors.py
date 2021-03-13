class Colors:
    white = (255,) * 3
    red = (255, 0, 0)
    green = (0, 128, 0)
    blue = (0, 142, 204)
    black = (0, 0, 0)


def generate_ship_colors(num_ships):
    """ Generates a unique color for each ship, so they can be tracked during the game.
    :param num_ships: The number of ships being rendered
    :param idx: The index of the current ship
    :return: ship_colors: list of RGB colors, e.g. [(124, 255, 0), (0, 255, 0),...]
    """

    one_third = num_ships // 3
    red_indices = range(one_third)
    green_indices = range(one_third)
    blue_indices = range(num_ships - (len(red_indices) + len(green_indices)))

    ship_colors = []
    for idx in red_indices:
        red_amount = idx / len(red_indices) * 255
        ship_colors.append((red_amount, 255, 255))

    for idx in green_indices:
        green_amount = idx / len(green_indices) * 255
        ship_colors.append((255, green_amount, 255))

    for idx in blue_indices:
        blue_amount = idx / len(blue_indices) * 255
        ship_colors.append((255, 255, blue_amount))

    return ship_colors