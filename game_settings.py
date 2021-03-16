game_settings = { # Game window size in pixels.
                  "width": 1000,
                  "height": 800,

                  # Frames per second game is limited to. To train as fast as possible, set to a really high number.
                  "fps": 100,

                  # Total number of ships.
                  "num_ships": 20,

                  # Ship manuevering settings.
                  "delta_angle": 5,
                  "thrust": 0.01,
                  "speed_multiplier": 1.35,

                  # Time step.
                  "dt": 5,

                  # Timeout before the game resets the level. Useful if ships fly around in an infinite loop.
                  "time_limit": 10,

                  # Pickled ship data to load. If set to None, ship will be trained from scratch.
                  "ship_file": None, #'trained_models/saved_ships.p'

                  # Each element represents the number of nodes in each hidden layer.
                  "hidden_layer_sizes": [10, 5, 3]}
