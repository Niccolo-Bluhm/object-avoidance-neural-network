#!/usr/bin/env python
import os
import json
from lib.game_utilities import PygView



# End win condition methods.
if __name__ == '__main__':
    settings_file = 'game_settings.py'
    level_dir = 'training_levels'
    PygView(settings_file, level_dir).run()

