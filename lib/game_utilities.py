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
from lib.genetic_algorithm import select_and_evolve, mutate, crossover, deconstruct_mlp, construct_mlp
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
            if not file.endswith(".txt"):
                continue
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
        self.ships = []
        for i in range(self.game_settings['num_ships']):
            self.ships.append(SpaceShip(self.screen, self.level, self.game_settings))
        self.game_over = False
        self.stop_printing = False
        self.generation = 0

        if self.game_settings['ship_file'] is not None:
            self.loadShips()

    def reset(self):
        self.ship = SpaceShip(self.screen, self.level, self.game_settings)
        self.game_over = False

    def run(self):
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
                        turn_direction, fitness_data = self.ships[j].predict()
                        if turn_direction == "left":
                            delta_angle = -self.game_settings["delta_angle"]
                        elif turn_direction == "right":
                            delta_angle = self.game_settings["delta_angle"]
                        else:
                            delta_angle = 0

                        # Calculate the updated ship position.
                        self.ships[j].calculate_position(delta_angle=delta_angle, stop=self.ships[j].crashed, color=ship_colors[j])

                    pygame.display.flip()
                    self.screen.blit(self.background, (0, 0))

                    # Limit the framerate so game doesn't run too fast.
                    self.clock.tick(self.game_settings['fps'])

                    ships_alive = 0
                    all_crashed = True
                    for j in range(self.game_settings['num_ships']):
                        if not self.ships[j].crashed:
                            all_crashed = False
                            ships_alive = ships_alive + 1

                    loop_count = loop_count + 1

                # Update fitness values.
                for ship in self.ships:
                    ship.update_fitness()

            self.ships = select_and_evolve(self.ships)

        pygame.quit()

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
        self.screen.blit(surface, ((self.width - fw), (20 - fh)))

    def render_planets(self, level):
        """Draw the planet including the gaussian noise
        to simulate erruptions"""

        # Draw the white circle
        pygame.draw.circle(self.screen, Colors.white, np.int64(level["center_white"]), level["radius_white"])

        red_radii = level['radii_red']
        red_centers = level['centers_red']
        for idx, radius in enumerate(red_radii):
            pygame.draw.circle(self.screen, Colors.red, red_centers[idx], radius)
