# Evolving Neural Networks
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Python](https://img.shields.io/badge/scikitlearn-v0.24.1-blue.svg)
![Python](https://img.shields.io/badge/pygame-v2.0+-blue.svg)

<p align="center"><img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/demo.gif" width=90%></p>

## Basic Overview
A simple simulation visualized using PyGame to demonstrate evolving neural networks using a genetic algorithm. The goal 
of the game is for the spaceship to avoid the red planets, and navigate to the white planet. Each ship is controlled by
a simple multi-layered neural network.


## Installation
Download and install python 3.8. CD to the directory where requirements.txt is located,
then run:

    pip install -r requirements.txt

## Running the Simulation
Run <b>train.py</b> to kickoff the neural network training. The ships will train on each level in the <b>training_levels</b>
folder until they beat it. The ships are saved in the <b>trained_models</b> folder,
and can be reloaded later. 

The file <b>game_settings.py</b> contains various parameters that can be tweaked to change the simulation, such as the 
number of ships, the number of hidden layers in the neural network, and the game FPS.

<b>test.py</b> can be run to test your trained model on a level it has never encountered before. Unfortunately, trained 
spaceships don't tend to generalize well to new levels and require additional training on that level to perform well.


## Background

