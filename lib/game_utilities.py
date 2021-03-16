import os
import pygame
import numpy as np
import time
import pickle
import json
import sys
from lib.genetic_algorithm import select_and_evolve, mutate, crossover, deconstruct_mlp, construct_mlp
from lib.spaceship import SpaceShip
from lib.colors import Colors, generate_ship_colors
from game_settings import game_settings
vector = pygame.math.Vector2
MARGIN = 5


class PygView(object):
    def __init__(self, settings_file, level_dir):
        """Initialize pygame, window, background, font,...
        """
        self.game_settings = game_settings

        # Load the levels
        level_files = os.listdir(level_dir)
        self.levels = []
        for file in level_files:
            if not file.endswith(".txt"):
                continue
            with open(os.path.join(level_dir, file)) as json_file:
                self.levels.append(json.load(json_file))

        pygame.init()
        pygame.display.set_caption("Neural Network Evolution")
        self.level = self.levels[0]
        self.width = self.game_settings['width']
        self.height = self.game_settings['height']
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.font30 = pygame.font.SysFont('mono', 30, bold=True)
        self.font22 = pygame.font.SysFont('mono', 22, bold=True)
        self.ships = []

        if self.game_settings['ship_file'] is None:
            # Initialize ships from scratch
            for i in range(self.game_settings['num_ships']):
                self.ships.append(SpaceShip(self.screen, self.level, self.game_settings))
        else:
            # Load pre-existing ships.
            self.load_ships(self.game_settings['ship_file'])

        self.game_over = False
        self.generation = 0

    def run(self):
        generation = 0
        ship_colors = generate_ship_colors(len(self.ships))
        for idx, level in enumerate(self.levels):
            level_running = True
            max_fitness = 0
            while level_running:
                for i, ship in enumerate(self.ships):
                    ship.level = level
                    ship.color = ship_colors[i]
                    ship.reset_location()

                start_time = time.time()
                all_crashed = False
                while not all_crashed:
                    self.draw_text_top("Level: {} of {}".format(idx + 1, len(self.levels)),
                                       "Max Fitness:{:0.0f}".format(max_fitness))
                    self.draw_text_bottom("Generation:{}".format(generation))

                    # Render the planet
                    self.render_planets(level)

                    # Check if user clicked the exit button.
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            all_crashed = True
                            level_running = False
                            pygame.quit()
                            sys.exit()

                    # Check if we've exceeded the time limit.
                    for ship in self.ships:
                        if time.time() - start_time > self.game_settings['time_limit']:
                            ship.crashed = True

                        # Predict whether to turn left or right using the neural network.
                        delta_angle = ship.predict()

                        # Calculate the updated ship position.
                        ship.calculate_position(delta_angle=delta_angle)
                        ship.update_fitness()

                    pygame.display.flip()
                    self.screen.blit(self.background, (0, 0))

                    # Limit the framerate so game doesn't run too fast.
                    self.clock.tick(self.game_settings['fps'])

                    if np.all([ship.crashed for ship in self.ships]):
                        all_crashed = True
                        max_fitness = np.max([ship.fitness for ship in self.ships])

                    # If a ship reached the white planet, we won the level. Advance to the next one.
                    if np.any([ship.ship_won for ship in self.ships]):
                        level_running = False

                self.ships = select_and_evolve(self.ships)
                self.save_ships()
                generation += 1

        pygame.quit()

    def draw_text_bottom(self, text):
        fw, fh = self.font30.size(text)  # fw: font width,  fh: font height
        surface = self.font30.render(text, True, Colors.green, Colors.black)
        self.screen.blit(surface, ((self.width - fw - MARGIN), (self.height - fh - MARGIN)))

    def draw_text_top(self, text1, text2):
        fw, fh = self.font22.size(text1)  # fw: font width,  fh: font height
        surface = self.font22.render(text1, True, Colors.blue, Colors.black)
        self.screen.blit(surface, ((self.width - fw - MARGIN), (MARGIN)))

        fw, fh = self.font22.size(text2)
        surface = self.font22.render(text2, True, Colors.blue, Colors.black)
        self.screen.blit(surface, ((self.width - fw - MARGIN), ( fh + MARGIN )))

    def render_planets(self, level):
        # Draw the white circle
        pygame.draw.circle(self.screen, Colors.white, np.int64(level["center_white"]), level["radius_white"])

        red_radii = level['radii_red']
        red_centers = level['centers_red']
        for idx, radius in enumerate(red_radii):
            pygame.draw.circle(self.screen, Colors.red, red_centers[idx], radius)

    def save_ships(self):
        for ship in self.ships:
            ship.screen = None
        pickle.dump(self.ships, open("trained_models/saved_ships.p", "wb"))
        for ship in self.ships:
            ship.screen = self.screen

    def load_ships(self, filepath):
        self.ships = pickle.load( open( filepath, "rb" ) )
        for ship in self.ships:
            ship.screen = self.screen
