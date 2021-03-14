from copy import deepcopy
from sklearn.neural_network import MLPClassifier
import pygame
import numpy as np
from lib.genetic_algorithm import mutate, crossover, deconstruct_mlp, construct_mlp
from lib.colors import Colors
vector = pygame.math.Vector2
MAX_ANGLE = 180


class SpaceShip:
    """The space ship class"""
    def __init__(self, screen, level, game_settings):
        self.game_settings = game_settings

        # Neural Network Structure. Hidden layer size is loaded from the game_settings file.
        n_inputs = 15
        n_hidden = self.game_settings['hidden_layer_sizes']
        n_output = 1

        # These are used for initializing the scikitlearn Neural Networks.
        X = np.zeros(n_inputs)
        X_train = np.array([X, X])
        y_train = np.array(range(n_output + 1))
        self.mlp = MLPClassifier(hidden_layer_sizes=n_hidden, max_iter=1, activation="tanh")
        self.mlp.fit(X_train, y_train)

        # Initialize the MLP with random weights and biases between -1 and 1
        weights, biases = deconstruct_mlp(self.mlp)
        weights = np.random.rand(len(weights)) * 2 - 1
        biases = np.random.rand(len(biases)) * 2 - 1
        self.mlp = construct_mlp(self.mlp, weights, biases)

        self.level = level
        self.pos = self.level['ship']['starting_pos']
        self.angle = self.level['ship']['starting_angle']
        self.screen = screen
        self.velocity = vector(0, 0)
        self.crashed = False
        self.cumulative_distance_from_goal = 0
        self.inputs = np.zeros(n_inputs)
        self.fitness2 = 0
        self.fitnessDebug = 0
        self.sawTheGoodPlanet = False
        self.donezo = False
        self.debug = False
        self.max_distance = vector(self.game_settings['width'], self.game_settings['height']).length()
        self.ship_won = False
        self.distance_from_goal = float('inf')

    def valid_ship_position(self):
        """This method checks that the ship is within the game window.

        :return: True if ship is within the window, False otherwise
        """
        # Ensure the entire ship is within the window. Check all points on the ship's triangle.
        # All x coordinates.
        ship_x_vals = np.array([self.tip[0], self.left[0], self.right[0]])
        # All y coordinates.
        ship_y_vals = np.array([self.tip[1], self.left[1], self.right[1]])

        # TODO: Verify width corresponds to x and height to y
        if np.any(ship_x_vals < 0) or np.any(ship_x_vals > self.game_settings['width']):
            return False
        if np.any(ship_y_vals < 0) or np.any(ship_y_vals > self.game_settings['height']):
            return False

        return True

    def reset_location(self):
        self.pos = self.level['ship']['starting_pos']
        self.angle = self.level["ship"]['starting_angle']
        self.velocity = vector(0, 0)
        self.crashed = False
        self.fitness2 = 0
        self.cumulative_distance_from_goal = 0
        self.sawTheGoodPlanet = False
        self.donezo = False
        self.ship_won = False
        self.distance_from_goal = float('inf')

    def update_fitness(self):
        """Updates the ships fitness value.

        :param fitness_data: Dictionary with keys 'distances' and 'classifications'. Distances is a 5 element array with
                             the distance to each object. Classification is a 5 element array with the object
                             classification, 0 for a bad planet and 1 for a good planet.
        :return: None
        """

        # When a ship penetrates a planet, its distance from that planet will be negative.
        # Ensure that it is always a positive number so the algorithm doesn't break farther downstream.
        if self.distance_from_goal < 0:
            distance = 1
        else:
            distance = self.distance_from_goal

        self.cumulative_distance_from_goal += distance

    def predict(self):
        direction = "none"
        distances, angles, classifications = self.calculate_mlp_inputs()

        # Check if the ship has crashed into anything.
        if np.any(distances <= 0):
            self.crashed = True

        # Store the distance from the white planet (Used in fitness calculation).
        good_planet_idx = np.where(classifications == 1)[0][0]
        self.distance_from_goal = distances[good_planet_idx]

        # Sort distances and keep the five closest objects.
        indices = np.argsort(distances)[0:5]

        # Normalize inputs so range is 0 to 1.
        distances_norm = distances[indices] / self.max_distance
        angles_norm = angles[indices] / MAX_ANGLE
        mlp_inputs = np.concatenate((distances_norm, angles_norm, classifications[indices]))

        # Get data for fitness calculation.
        fitness_data = {'distances': distances, 'classifications': classifications}

        # Make prediction based on inputs.
        output = self.mlp.predict(mlp_inputs.reshape(1, -1))[0]
        if output == 0:
            direction = "left"
        elif output == 1:
            direction = "right"

        return direction, fitness_data

    def calculate_mlp_inputs(self):
        """This function calculates the neural network inputs. It checks the ships distance from all walls and planets
        in the game, and returns the 5 closest objects and their distance + angle.

        :param red_planets:
        :return:
        """

        # Object classifications. 0 is for a bad object (object to avoid, i.e. a wall or red planet), 1 is for good.
        # Left wall, right wall, top wall, bottom wall.
        classifications = [0, 0, 0, 0]

        # Distance from each wall.
        distance_from_left   = self.tip[0]
        distance_from_right  = self.game_settings['width'] - self.tip[0]
        distance_from_bottom = self.game_settings['height'] - self.tip[1]
        distance_from_top    = self.tip[1]

        # Angle to closest point on each wall. Coords of closest point on wall - ship coordinates.
        left_vector   = vector([0, self.tip[1]]) - self.tip
        right_vector  = vector([self.game_settings['width'], self.tip[1]]) - self.tip
        bottom_vector = vector([self.tip[0], self.game_settings['height']]) - self.tip
        top_vector = vector([self.tip[0], 0]) - self.tip

        # Calculate the ship vector.
        back_point = (self.right + self.left) / 2
        ship_vector = self.tip - back_point

        angles = [ship_vector.angle_to(left_vector), ship_vector.angle_to(right_vector),
                  ship_vector.angle_to(bottom_vector), ship_vector.angle_to(top_vector)]

        distances = [distance_from_left, distance_from_right, distance_from_bottom, distance_from_top]

        # Calculate the distance between the ship and each red planet.
        for idx, radius in enumerate(self.level["radii_red"]):
            ship_to_planet_vector = (self.tip - vector(self.level["centers_red"][idx]))
            distance = ship_to_planet_vector.length() - radius
            distances.append(distance)
            classifications.append(0)
            angles.append(ship_vector.angle_to(ship_to_planet_vector))

        # Calculate the distance between the ship and the white planet.
        ship_to_planet_vector = (self.tip - vector(self.level["center_white"]))
        distance = ship_to_planet_vector.length() - self.level["radius_white"]
        distances.append(distance)
        classifications.append(1)
        angles.append(ship_vector.angle_to(ship_to_planet_vector))

        # Convert everything to numpy arrays.
        distances = np.array(distances)
        angles = np.array(angles)
        classifications = np.array(classifications)

        return distances, angles, classifications

    def render(self, color):
        tip = vector(10, 0)
        left = vector(-5, -5)
        right = vector(-5, 5)

        for point in (tip, right, left):
            point.rotate_ip(self.angle)
            point += self.pos
        pygame.draw.polygon(self.screen, color, (tip, left, right))

        self.back = (left + right) / 2
        self.tip, self.left, self.right = tip, left, right

    def calculate_position(self, delta_angle=0.0, stop=False, color=Colors.blue):
        if not stop:
            self.velocity = self.game_settings['speed_multiplier'] * vector(1, 0).rotate(self.angle)
            dt = self.game_settings["dt"]
            self.pos = self.pos + self.velocity * dt
            self.angle += delta_angle

        self.render(color)

