game_settings = { # Game window size in pixels.
                  "width": 1000,
                  "height": 800,

                  # Frames per second game is limited to. To train as fast as possible, set to a really high number.
                  "fps": 60,

                  "delta_angle": 2,
                  "thrust": 0.01,
                  # Time step.
                  "dt": 5,
                  "num_ships": 20,
                  "speed_multiplier": 1.35,
                  # Timeout before the game resets the level. Useful if ships fly around in an infinite loop.
                  "time_limit": 10,
                  # Pickled ship data to load. If set to None, ship will be trained from scratch.
                  "ship_file": None,
                  # Each element represents the number of nodes in each hidden layer.
                  "hidden_layer_sizes": [10, 5, 3]}
