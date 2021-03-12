from copy import deepcopy
from sklearn.neural_network import MLPClassifier
import pygame
import numpy as np
from lib.genetic_algorithm import mutate, crossover, deconstruct_mlp, construct_mlp
from lib.colors import Colors
vector = pygame.math.Vector2


class SpaceShip:
    """The space ship class"""
    def __init__(self, screen, level, game_settings):
        self.game_settings = game_settings

        # Neural Network Structure. Hidden layer size is loaded from the game_settings file.
        n_inputs = 10
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
        self.fitness = 0
        self.inputs = np.zeros(n_inputs)
        self.fitness2 = 0
        self.fitnessDebug = 0
        self.sawTheGoodPlanet = False
        self.donezo = False
        self.debug = False
        self.max_distance = vector(self.game_settings['width'], self.game_settings['height']).length()

    def validShipPos(self):
        # Ensure the entire ship is within the window. Check all points on the ship's triangle.
        # All x coordinates.
        ship_x_vals = np.array([self.tip[0], self.left[0], self.right[0]])
        # All y coordinates.
        ship_y_vals = np.array([self.tip[1], self.left[1], self.right[1]])

        #TODO: Verify width corresponds to x and height to y
        if np.any(ship_x_vals < 0) or np.any(ship_x_vals > self.game_settings['width']):
            return False
        if np.any(ship_y_vals < 0) or np.any(ship_x_vals > self.game_settings['height']):
            return False

        return True

    def updateFitness(self):
        """ Update the ships fitness value.

        :return:
        """
        if not self.validShipPos():
            # If the ship has crashed, only update its fitness a tiny amount
            # TODO: Consider removing this, and having fitness not update at all.
            self.fitness2 = self.fitness2 + 1
            return

        # First five inputs of the neural network contain the ships distance in each direction.
        distances = self.inputs[range(5)]
        # Inputs 6 - 10 contain object classification in each direction (Whether its an object to avoid or go towards)
        # "Good" object = 0, "Bad" object (avoid) = 1
        classification = self.inputs[range(5, 10)]

        bad_object_idx = np.where(classification == 1)[0]
        bad_distances = distances[bad_object_idx]
        bad_distances = np.min(bad_distances)
        good_inds = np.where(classification == 0)[0]
        if len(good_inds) != 0:
            good_distances = distances[good_inds]
            good_distances = np.min(good_distances)
            good_distances = 1 / good_distances
            good_distances = good_distances * self.max_distance

            if good_distances > 50:
                good_distances = 50

            self.fitnessDebug = self.fitnessDebug + good_distances
            self.fitness2 = self.fitness2 + bad_distances + good_distances
            self.sawTheGoodPlanet = True
        else:
            self.fitness2 = self.fitness2 + bad_distances

    def predict(self, red_planets):
        string_output = "none"
        X = self.calcInputs(red_planets)

        # Normalize inputs so range is 0 to 1.
        self.inputs = deepcopy(X)
        X[range(5)] = X[range(5)] / self.max_distance

        # Make prediction based on inputs.
        output = self.mlp.predict(X.reshape(1, -1))[0]
        if output == 0:
            string_output = "left"
        elif output == 1:
            string_output = "right"
        return string_output

    def calcInputs(self, red_planets):
        avoidObject = np.zeros(5)
        objectDistances = np.zeros(5)
        # For each direction
        for i in range(5):
            # For each planet (+1 is for the good planet)
            allObjDistances = []
            for j in range(len(red_planets) + 1):
                distFromEdge = self.wallIntercept(i)
                # avoidObject[i] = 1
                if (j != len(red_planets)):  # If we're not equal to the last planet (that's the good one)
                    # red_planets[j][0] is the planet center
                    dist = self.circleIntercept(i, red_planets[j][0], red_planets[j][1])
                    if (dist == -1):
                        dist = distFromEdge
                    allObjDistances.append(dist)
                else:
                    # Make the last planet the good one
                    center = np.array(*[self.level['center_white']])
                    dist = self.circleIntercept(i, center, self.level['radius_white'])
                    if (dist == -1):
                        dist = 99999
                    # if we set the white planet radius
                    # to 0 we ignore the white planet
                    # in the fitness function.
                    if self.level['radius_white'] != 0:
                        allObjDistances.append(dist)
                    else:
                        allObjDistances.append(99999)
            objectDistances[i] = min(allObjDistances)
            ind = allObjDistances.index(objectDistances[i])
            if (ind != len(red_planets)):
                avoidObject[i] = 1

        return np.concatenate((objectDistances, avoidObject))

    def wallIntercept(self, direction):
        # m is the slope of the line. Used to describe line in direction of ship
        # direction = 4

        if (direction == 0):
            # straight
            m = self.tip - self.back
            x, y = self.tip[0], self.tip[1]
        if (direction == 1):
            # left
            m = self.left - self.right
            x, y = self.left[0], self.left[1]
        if (direction == 2):
            # right
            m = self.right - self.left
            x, y = self.right[0], self.right[1]
        if (direction == 3):
            # left-staight
            m = (self.left + self.tip) / 2 - self.right
            x, y = ((self.left + self.tip) / 2)[0], ((self.left + self.tip) / 2)[1]
        if (direction == 4):
            # right-straight
            m = (self.right + self.tip) / 2 - self.left
            x, y = ((self.right + self.tip) / 2)[0], ((self.right + self.tip) / 2)[1]
        # Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if (m[0] == 0):
            m = 999999
        else:
            m = m[1] / m[0]

        """ m_lw = the slope of the line that describes the left wall of the game world  """
        m_lw = 999999
        m_rw = 999999
        m_bw = 0
        m_tw = 0

        """rw = rightWall, lw = leftWall, bw = bottomWall, tw = topWall"""
        x_lw, y_lw = 0, 0
        x_rw, y_rw = 1000, 0
        x_bw, y_bw = 1000, 800
        x_tw, y_tw = 1000, 0

        m_walls = [m_lw, m_rw, m_bw, m_tw]
        x_w = [x_lw, x_rw, x_bw, x_tw]
        y_w = [y_lw, y_rw, y_bw, y_tw]

        lDistances = []
        for ii in range(4):
            if (m - m_walls[ii] == 0):
                x_i = 999999
            else:
                x_i = (m * x - y - m_walls[ii] * x_w[ii] + y_w[ii]) / (m - m_walls[ii])

            y_i = m * (x_i - x) + y

            dist = (vector(x_i, y_i) - vector(x, y)).length()

            if (dist != -1):
                if (direction == 0):
                    # straight
                    if (self.back - vector(x_i, y_i)).length() < (self.tip - vector(x_i, y_i)).length():
                        dist = -1
                if (direction == 1):
                    # left
                    if (self.right - vector(x_i, y_i)).length() < (self.left - vector(x_i, y_i)).length():
                        dist = -1
                if (direction == 2):
                    # right
                    if (self.left - vector(x_i, y_i)).length() < (self.right - vector(x_i, y_i)).length():
                        dist = -1
                if (direction == 3):
                    # left-staight
                    if (self.right - vector(x_i, y_i)).length() < ((self.left + self.tip) / 2 - vector(x_i, y_i)).length():
                        dist = -1
                if (direction == 4):
                    # right-straight
                    if (self.left - vector(x_i, y_i)).length() < ((self.right + self.tip) / 2 - vector(x_i, y_i)).length():
                        dist = -1
            if (dist != -1):
                lDistances.append(dist)

        # For some reason, it didn't get any distances once. This will prevent the game from crashing if that happens
        if len(lDistances) == 0:
            lDistances.append(1)
            # print("Bug!")
        return np.min(lDistances)

    def circleIntercept(self, direction, planetCenter, planetRadius):
        """https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle"""

        # m is the slope of the line. c is the y intercept. used to describe line in direction of ship
        # direction = 4

        if (direction == 0):
            # straight
            m = self.tip - self.back
            lineStart = self.tip
        if (direction == 1):
            # left
            m = self.left - self.right
            lineStart = self.left
        if (direction == 2):
            # right
            m = self.right - self.left
            lineStart = self.right
        if (direction == 3):
            # left-staight
            m = (self.left + self.tip) / 2 - self.right
            lineStart = (self.left + self.tip) / 2
        if (direction == 4):
            # right-straight
            m = (self.right + self.tip) / 2 - self.left
            lineStart = (self.right + self.tip) / 2
        # Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if (m[0] == 0):
            m = 999999
        else:
            m = m[1] / m[0]

        # We want left and right 'seeing directions' to be at the back of the ship
        c = lineStart[1] - m * lineStart[0]

        p = planetCenter[0]  # config['planet_center'][0]
        q = planetCenter[1]  # config['planet_center'][1]
        r = planetRadius  # config['planet_radius']

        A = m ** 2 + 1
        B = 2 * (m * c - m * q - p)
        C = q ** 2 - r ** 2 + p ** 2 - 2 * c * q + c ** 2

        # If B^2−4AC<0 then the line misses the circle
        # If B^2−4AC=0 then the line is tangent to the circle.
        # If B^2−4AC>0 then the line meets the circle in two distinct points.
        if (B ** 2 - 4 * A * C < 0):
            x = -1
            y = -1
            dist = -1
        elif (B ** 2 - 4 * A * C == 0):
            x = -B / (2 * A)
            y = m * x + c
            dist = (vector(x, y) - vector(lineStart[0], lineStart[1])).length()
        else:
            x1 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            x2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            y1 = m * x1 + c
            y2 = m * x2 + c

            l1 = (vector(x1, y1) - vector(lineStart[0], lineStart[1])).length()
            l2 = (vector(x2, y2) - vector(lineStart[0], lineStart[1])).length()

            # Pick the point on the circle that is closest to the ship
            if (l1 < l2):
                x = x1
                y = y1
                dist = l1
            else:
                x = x2
                y = y2
                dist = l2

        # Check to make sure the line intercepts the circle on the front side of the ship
        if (dist != -1):
            if (direction == 0):
                # straight
                if (self.back - vector(x, y)).length() < (self.tip - vector(x, y)).length():
                    dist = -1
            if (direction == 1):
                # left
                if (self.right - vector(x, y)).length() < (self.left - vector(x, y)).length():
                    dist = -1
            if (direction == 2):
                # right
                if (self.left - vector(x, y)).length() < (self.right - vector(x, y)).length():
                    dist = -1
            if (direction == 3):
                # left-staight
                if (self.right - vector(x, y)).length() < ((self.left + self.tip) / 2 - vector(x, y)).length():
                    dist = -1
            if (direction == 4):
                # right-straight
                if (self.left - vector(x, y)).length() < ((self.right + self.tip) / 2 - vector(x, y)).length():
                    dist = -1
        return dist

    def render(self, color):

        tip = vector(10, 0)
        left = vector(-5, -5)
        right = vector(-5, 5)

        for pt in (tip, right, left):
            pt.rotate_ip(self.angle)
            pt += self.pos
        pygame.draw.polygon(self.screen, color, (tip, left, right))

        self.back = (left + right) / 2
        self.tip, self.left, self.right = tip, left, right

    def physics(self, thrust=0.0, delta_angle=0.0, stop=False, color=Colors.blue):
        ppos = self.level['center_white']

        # gravity = config["gravity"]*(self.pos-ppos).normalize()
        dt = self.game_settings["dt"]
        if not stop:
            thrust_vector = vector(1, 0).rotate(self.angle) * thrust
            # self.velocity = self.velocity + (gravity+thrust_vector)*dt
            self.velocity = self.game_settings['speed_multiplier'] * vector(1, 0).rotate(self.angle)

            self.pos = self.pos + self.velocity * dt
            self.angle += delta_angle

        '''
        if thrust == 0:
            color = colors.green
        else:
            color = colors.blue
        '''

        self.render(color)

    # Begin methods to check win conditions
    def check_orientation(self):
        pangle = ((self.left - self.right).angle_to(
            self.pos - self.game_settings["planet_center"]))

        if pangle > -90 - self.game_settings["land_angle"] \
                and pangle < -90 + self.game_settings["land_angle"]:

            return True
        else:
            return False

    def check_red_planets(self, rps):
        for (ppos, rad) in rps:
            if (self.tip - ppos).length() < rad:
                return False

        return True

    def check_speed(self):
        if self.velocity.length() < self.game_settings["land_speed"]:
            return True
        else:
            return False

    def check_pos_screen(self):
        if (self.pos[0] > 0 and self.pos[0] < 1000 and self.pos[1] > 0 and self.pos[1] < 800):
            return True
        else:
            return False
        # print(self.pos)

    def check_on_planet(self):
        # if any part of the ship is touching the planet
        # we have landed
        for pt in (self.tip, self.left, self.right):

            if (pt - vector(self.level["center_white"])).length() \
                    < self.level["radius_white"]:
                return True
        return False