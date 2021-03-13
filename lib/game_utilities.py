from copy import deepcopy
from sklearn.neural_network import MLPClassifier
import os
import pygame
import numpy as np
import math
import time
import pickle
import datetime
import json
import sys
import pathlib
import builtins
from lib.genetic_algorithm import mutate, crossover, deconstruct_mlp, construct_mlp
from lib.spaceship import SpaceShip
from lib.colors import Colors, generate_ship_colors
vector = pygame.math.Vector2


class PygView(object):
    def __init__(self, settings_file, level_dir):
        """Initialize pygame, window, background, font,...
        """

        # Load the game settings.
        with open(settings_file) as json_file:
            self.game_settings = json.load(json_file)

        # Load the levels
        level_files = os.listdir(level_dir)
        self.levels = []
        for file in level_files:
            with open(os.path.join(level_dir, file)) as json_file:
                self.levels.append(json.load(json_file))

        pygame.init()
        pygame.display.set_caption("Press ESC to quit")

        self.level = self.levels[0]
        self.width = self.game_settings['width']
        self.height = self.game_settings['height']
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        self.planetFinished = False
        self.planets = []
        self.ship = SpaceShip(self.screen, self.level, self.game_settings)
        self.ships = []
        for i in range(self.game_settings['num_ships']):
            self.ships.append(SpaceShip(self.screen, self.level, self.game_settings))
        self.Newships = []
        self.game_over = False
        self.stop_printing = False
        self.generation = 0
        self.bestScore = 0
        self.prevShips = []
        self.prevFitness = []
        self.logLst = []
        self.nfitnesses = np.zeros(self.game_settings['num_ships'])

        self.maxes = np.zeros(len(self.levels))

        if self.game_settings['ship_file'] is not None:
            self.loadShips()

    def loadShips(self):
        with open(self.game_settings['ship_file'], 'rb') as f:
            lShipData = pickle.load(f)

        for i in range(self.game_settings['num_ships']):
            self.ships[i].mlp.intercepts_[0] = lShipData[-1]['intercepts1']
            self.ships[i].mlp.intercepts_[1] = lShipData[-1]['intercepts2']
            self.ships[i].mlp.coefs_[0] = lShipData[-1]['weights1']
            self.ships[i].mlp.coefs_[1] = lShipData[-1]['weights2']

    def reset(self):
        self.ship = SpaceShip(self.screen, self.level, self.game_settings)
        self.game_over = False

    def run(self):
        """

        :return:
        """
        # Game Loop.
        game_running = True
        ship_colors = generate_ship_colors(len(self.ships))
        while game_running:
            delta_angle = 0
            for idx, level in enumerate(self.levels):
                for ship in self.ships:
                    ship.level = level
                    ship.calculate_position(delta_angle=delta_angle, stop=ship.crashed)
                    ship.reset_location()

                start_time = time.time()
                all_crashed = False
                ships_alive = 10
                loop_count = 0
                while not all_crashed:
                    self.draw_text("Generation:{}".format(self.generation))
                    self.draw_text_top(("Level: {} of {} Ships Alive: {}".format(idx + 1, len(self.levels), ships_alive)))

                    # Render the planet
                    self.render_planets(level)

                    # Check if user clicked the exit button.
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                self.reset()

                    # Check if we've exceeded the time limit.
                    for j in range(self.game_settings['num_ships']):
                        if time.time() - start_time > self.game_settings['time_limit']:
                            self.ships[j].crashed = True

                        # TODO: Have neural network scale delta angle based on output
                        turn_direction = self.ships[j].predict(self.planets)
                        if turn_direction == "left":
                            delta_angle = -self.game_settings["delta_angle"]
                        elif turn_direction == "right":
                            delta_angle = self.game_settings["delta_angle"]
                        else:
                            delta_angle = 0

                        # Calculate the updated ship position.
                        self.ships[j].calculate_position(delta_angle=delta_angle, stop=self.ships[j].crashed, color=ship_colors[j])

                        # Did we land?
                        if self.ships[j].check_on_planet() or self.ships[j].check_pos_screen() == False:
                            self.ships[j].crashed = True

                        self.ships[j].updateFitness()

                        if self.ships[j].check_red_planets(self.planets) == False:
                            self.ships[j].crashed = True

                    pygame.display.flip()
                    self.screen.blit(self.background, (0, 0))

                    # Limit the framerate so game doesn't run too fast.
                    self.clock.tick(self.game_settings['fps'])

                    ships_alive = 0
                    all_crashed = True
                    for j in range(self.game_settings['num_ships']):
                        if self.ships[j].crashed == False:
                            all_crashed = False
                            ships_alive = ships_alive + 1

                    loop_count = loop_count + 1

                # Normalize fitness by the number of loops
                fitnesses = []
                for p in range(self.game_settings['num_ships']):
                    fitnesses.append(deepcopy(self.ships[p].fitness2))
                theMax = max(fitnesses)

                if (theMax > self.maxes[idx]):
                    self.maxes[idx] = theMax
                else:
                    theMax = self.maxes[idx]

                if (self.game_settings['normalize_fitness'] == True):
                    fitnesses = np.array(fitnesses) / loop_count

                self.nfitnesses = self.nfitnesses + np.array(fitnesses)

            self.updateWeights()

        pygame.quit()

    def updateWeights(self):
        newShips = []

        scores = np.zeros(self.game_settings['num_ships'])
        for i in range(self.game_settings['num_ships']):
            scores[i] = deepcopy(self.nfitnesses[i])  # deepcopy(self.ships[i].fitness2)

        scores_sort = np.sort(scores)[::-1]
        # Invert. Make highest scores to Lowest
        # scores_sort = 1/scores_sort

        reject = True
        if (scores_sort[0] >= self.bestScore):
            self.bestScore = scores_sort[0]
            reject = False
        scores_sort_ind = scores.argsort()[::-1]  # Descending order (highest to lowest)

        ##### PRINT STUFF #####
        print("")
        print("Generation: ", self.generation)
        for i in range(self.game_settings['num_ships']):
            print("Ship Score:", scores[scores_sort_ind[i]], self.ships[scores_sort_ind[i]].fitnessDebug, "Weight:")

        # If we did worse than before, reject this generation

        reject = False
        if reject:
            scores = np.zeros(self.game_settings['num_ships'])
            for i in range(self.game_settings['num_ships']):
                scores[i] = deepcopy(self.prevFitness[i])
                self.ships[i].mlp = deepcopy(self.prevShips[i])

                # self.ships[i].fitness2 = deepcopy(self.prevFitness[i])

            scores_sort = np.sort(scores)[::-1]
            # Invert. Make highest scores to Lowest
            # scores_sort = 1/scores_sort
            print("Generation Rejected")

        self.generation = self.generation + 1
        for i in range(self.game_settings['num_ships']):
            # Get Weight value of best ship
            NN1 = deepcopy(self.ships[scores_sort_ind[i]].mlp)
            intercepts = np.concatenate((NN1.intercepts_[0], NN1.intercepts_[1]))
            weights1 = NN1.coefs_[0].flatten()
            weights2 = NN1.coefs_[1].flatten()
            allWeights = np.concatenate((intercepts, weights1, weights2))
            weightSum = deepcopy(np.sum(allWeights))

            # pickle info of best ship
            if scores_sort_ind[i] == 0 and not reject:
                logdict = {
                    'ship_num': i,
                    'weights1': NN1.coefs_[0],
                    'weights2': NN1.coefs_[1],
                    'intercepts1': NN1.intercepts_[0],
                    'intercepts2': NN1.intercepts_[1],
                    'Generation': self.generation,
                    'timestamp': datetime.datetime.now(),
                    'score': scores[scores_sort_ind[i]]
                }
                self.logLst.append(logdict)
                fname = "best.pkl"
                # fname = "{}_best.pkl".format(datetime.datetime.now().isoformat().replace(':','-'))

                with open(fname, 'wb') as pfd:
                    pickle.dump(self.logLst, pfd)

        # print(self.bestScore)
        #########################

        # Sort the scores from low value to high values
        # Low values indicate a better score (Closer to landing zone)
        scores_sort_ind = scores.argsort()[::-1]
        sortedShips = []
        for i in range(self.game_settings['num_ships']):
            sortedShips.append(deepcopy(self.ships[scores_sort_ind[i]].mlp))

        # Normalize the fitness scores
        scores_sum = np.sum(scores_sort)
        scores_sort = scores_sort / scores_sum
        probabilities = scores_sort

        # Take best performing ships(Top 20%) and introduce directly to next round
        num_bestShips = int(np.floor(self.game_settings['num_ships'] * 0.2))
        for i in range(num_bestShips):
            newShips.append(deepcopy(self.ships[scores_sort_ind[i]].mlp))

        # Take two parents, mutate them, and introduce to next round (Skip crossover)
        for i in range(2):
            parents1 = np.random.choice(range(self.game_settings['num_ships']), size=2, replace=False, p=probabilities)
            theNewMlp1 = mutate(sortedShips[parents1[0]])
            newShips.append(deepcopy(theNewMlp1))

        # Whatever ships we have left mutate + crossbreed
        for i in range(int(self.game_settings['num_ships'] - len(newShips))):
            # Select two parents
            parents = np.random.choice(range(self.game_settings['num_ships']), size=2, replace=False, p=probabilities)

            NN = crossover(sortedShips[parents[0]], sortedShips[parents[1]])
            theNewMlp = mutate(NN)
            # theNewMlp = self.mutate(sortedShips[parents[0]])

            newShips.append(deepcopy(theNewMlp))

        # Save the previous ships incase all the new ships are worse
        # We don't currently need this because we're always carrying the
        # best ships to the next round
        if (reject == False):
            self.prevShips = []
            self.prevFitness = []
            for i in range(len(self.ships)):
                self.prevShips.append(deepcopy(self.ships[i].mlp))
                self.prevFitness = deepcopy(self.nfitnesses)

        for i in range(len(self.ships)):
            self.ships[i].mlp = deepcopy(newShips[i])
            self.ships[i].fitnessDebug = 0

        self.nfitnesses = np.zeros(self.game_settings['num_ships'])

    def draw_text(self, text):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render(text, True, (0, 255, 0))
        # // makes integer division in python3
        self.screen.blit(
            surface, ((self.width - fw), (self.height - fh)))

    def draw_text_top(self, text):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render(text, True, (0, 255, 0))
        # // makes integer division in python3
        self.screen.blit(
            surface, ((self.width - fw), (20 - fh)))

    def render_planets(self, level):
        """Draw the planet including the gaussian noise
        to simulate erruptions"""

        # Draw the white circle
        pygame.draw.circle(self.screen, Colors.white, np.int64(level["center_white"]), level["radius_white"])

        red_radii = level['radii_red']
        red_centers = level['centers_red']
        for idx, radius in enumerate(red_radii):
            pygame.draw.circle(self.screen, Colors.red, red_centers[idx], radius)

        self.planets = []
        for i in range(len(red_centers)):
            np_center = vector(red_centers[i])
            rp = red_planet(self.screen, np_center, red_radii[i])
            self.planets.append(rp)


class red_planet:
    def __init__(self, screen, center, radius):
        self.screen = screen
        self.radius = radius
        self.center = center
        self.idx = 0

    def __getitem__(self, key):
        if key == 0:
            return self.center

        elif key == 1:
            return self.radius

        raise KeyError("red planet index must be 0 or 1 not ", key)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        try:
            item = (self.center, self.radius)[self.idx]
        except IndexError:
            raise StopIteration("Iter error")
        self.idx += 1
        return item



def open(path, *args, **kwargs):
    wpath = pathlib.PureWindowsPath(path)
    return builtins.open(str(pathlib.Path(wpath)), *args, **kwargs)