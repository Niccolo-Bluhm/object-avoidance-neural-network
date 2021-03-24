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
Download the source code for the project and install python 3.8. CD to the directory where requirements.txt is located,
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
### Neural Network Structure
Each ship in the game can 'see' the five closest objects, whether they be the edges of the map or planets. For each obstacle, 
its distance and angle to the ship are calculated, as well as a boolean value indicating if it should be avoided or sought out.


<p float="left">
<img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/classifications.png" width=30%>
<img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/distances.png" width=30%>
<img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/angles.png" width=30%>
</p>

These 15 values are then normalized and used as inputs to the neural network. The ship is always moving forward, and at 
each timestep it can turn either left or right depending on the output neuron value.

<p align="center"><img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/neural_network.png" width=100%></p>

### Genetic Algorithm
#### Fitness

The fitness function is used to determine how well a ship performed during a round. The goal of the game is to avoid 
hitting red planets and fly into the white planet, so a ship that immediately crashes into a red planet would have a 
low fitness, whereas a ship that avoided crashing and navigated to the white planet would have a high fitness.
While seemingly simple, there are many ways to calculate fitness and slight alterations can cause large changes in behavior. 
Ships needed to be selected that maximized distance from the red planets and minimized distance to the white planet.

<p align="center"><img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/ship_goals.png" width=100%></p>

The basic fitness function that was developed can be seen in the image below. At each time step, the minimum bad distance 
(red distance) is added to the inverse of the good (white) distance. This function simultaneous measures how good the 
ships are at not crashing and seeking the white planet. A ship will get a good score if it keeps far away from the red 
planets but also gets close to the white planet.

<p align="center"><img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/fitness_calculation.png" width=100%></p>

#### Selection
Selection is the process of picking two ships to crossbreed, of which there are many viable methods. In this project 
_fitness proportionate selection_ was used, where a probability of selection is assigned to each ship based on their fitness. 
For example, if ship A had a fitness of 30, ship B had a fitness of 20, and ship C had a fitness of 10, ship A would have 
a 50% (30/60) probability of being chosen as a parent, ship B would have a 33% chance, and ship C would have a 16.6% 
chance of being selected. More than two ships can be selected to crossbreed, but in this project only two were used to 
keep it simple.

#### Crossover
Crossover is the process of combining characteristics from the selected parents. In the case of this project, weights 
and biases from the neural networks of the chosen ships were interchanged. Randomly selected weights and biases in 
ship A’s neural network would be replaced by the corresponding weights and biases from ship B. 

<p align="center"><img src="https://github.com/Niccolo-Bluhm/object-avoidance-neural-network/blob/master/media/crossover_example.png" width=100%></p>

#### Mutation
Mutation involves randomly changing characteristics of the populating in order to maintain genetic diversity and better 
explore the solution space. The number of weights and biases to be mutated was a random value, and could vary uniformly 
from 5% to 20%. The number of weights and biases to be mutated was kept relatively low, because if it was very high it 
introduced too much variation and the ships wouldn’t converge to a solution. The weights and biases to be mutated were 
picked at random from the neural net. Two possible mutation strategies could be picked, each with a probability of 50%. 
The first one involved taking each selected weight and replacing it with a new random weight between -1 and 1. 
The second strategy involved taking the selected weights and scaling them all by a random number between -1 and 3.
