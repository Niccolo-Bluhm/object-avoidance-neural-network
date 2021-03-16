from lib.game_utilities import PygView

if __name__ == '__main__':
    level_dir = 'testing_levels'
    game = PygView(level_dir)
    # Load in a trained model.
    game.load_ships('trained_models/saved_ships.p')
    game.run()
