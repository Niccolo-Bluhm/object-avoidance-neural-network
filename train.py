#!/usr/bin/env python

from copy import deepcopy
from sklearn.neural_network import MLPClassifier
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
from lib.genetic_algorithm import mutate, crossover
from lib.game_utilities import Colors
VEC = pygame.math.Vector2


game_settings = dict(
    gravity=0,
    land_angle=10,
    land_speed=0.25,
    delta_angle=2,
    thrust=0.01,
    dt=5,
    num_ships=10,
    starting_pos=(20, 20),
    starting_angle=45,
    speed_multiplier=1.35,
    time_limit=5,
    ship_file=None,
    default_level='training_levels\Train2.txt',
    normalize_fitness=False
)

TrainingLevels = ['training_levels\Train1.txt','training_levels\Train2.txt','training_levels\Train6.txt','training_levels\Train4.txt','training_levels\Train5.txt']

TestingLevels = ['levels\Train\Train3.txt','levels\Train\Train3.txt','levels\Train\Train1.txt','levels\Train\Train2.txt','levels\Train\Train4.txt']

TestingLevels = ['testing_levels\Test3.txt']

theLevels = TrainingLevels

#These are just used for initializing the scikitlearn Neural Networks
X_train = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
y_train = np.array([0,3,1,2])

#Neural Network Structure
n_inputs = 10
n_hidden = 7
n_output = 1

#These are used for initializing the scikitlearn Neural Networks
X = np.zeros(n_inputs)
X_train = np.array([X, X])
y_train = np.array(range(n_output + 1))


class PygView( object ):
    def __init__(self, level, width=1000, height=1000, fps=60 ):
        """Initialize pygame, window, background, font,...
        """

        pygame.init()
        pygame.display.set_caption("Press ESC to quit")

        self.level = level
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        self.planetFinished = False
        self.planets = []
        self.landing_points = None
        self.ship = space_ship( self.screen, self.landing_points, self.level )
        self.ships = []
        for i in range(game_settings['num_ships']):
            self.ships.append(space_ship(self.screen, self.landing_points, self.level))
        self.Newships = []
        self.game_over = False
        self.stop_printing = False
        self.generation = 0
        self.bestScore = 0
        self.prevShips = []
        self.prevFitness = []
        self.logLst = []
        self.nfitnesses = np.zeros(game_settings['num_ships'])

        self.maxes = np.zeros(len(theLevels))

        if game_settings['ship_file'] is not None:
            self.loadShips()

    def loadShips(self):
        with open(game_settings['ship_file'], 'rb') as f:
            lShipData = pickle.load(f)

        for i in range(game_settings['num_ships']):
            self.ships[i].mlp.intercepts_[0] = lShipData[-1]['intercepts1']
            self.ships[i].mlp.intercepts_[1] = lShipData[-1]['intercepts2']
            self.ships[i].mlp.coefs_[0] = lShipData[-1]['weights1']
            self.ships[i].mlp.coefs_[1] = lShipData[-1]['weights2']

    def reset(self):
        self.ship = space_ship( self.screen, self.landing_points )
        self.game_over = False

    def run(self):
        """The mainloop
        """
        running = True
        ai_key = "none"
        count = 0
        while running:
            da = 0
            thrust = 0.0
            for qqq in range(len(theLevels)):
                levelfile1 = open( theLevels[qqq] )
                level1 = json.load( levelfile1 )
                self.level = deepcopy(level1)
                
                for j in range(game_settings['num_ships']):
                    self.ships[j].level = deepcopy(self.level)
                    self.ships[j].physics(
                            delta_angle=da,
                            thrust=thrust,
                            stop=self.ships[j].crashed)
                
                self.resetShipLocs()        
                start_time = time.time()
                all_crashed = False
                shipsAlive = 10
                loopCount = 0
                while all_crashed == False:
                    self.draw_text("Generation:{}".format(self.generation))
                    self.draw_text_top(("Level: {} of {} Ships Alive: {}".format(qqq+1,len(theLevels),shipsAlive)))

                    # Render the planet
                    self.do_planet()

                    for j in range(game_settings['num_ships']):
                        if(time.time()-start_time > game_settings['time_limit']):
                            self.ships[j].crashed = True
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                all_crashed = True
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                                    all_crashed = True
                                if event.key == pygame.K_r:
                                    self.reset()
                                    #self.resetShipLocs()

                        keys = pygame.key.get_pressed()

                        da = 0
                        thrust = 0.0
                        ai_key = self.ships[j].predict(self.planets)
                        if ai_key == "left":
                            da = -game_settings["delta_angle"]
                        if ai_key == "right":
                            da = game_settings["delta_angle"]
                        thrust = game_settings["thrust"]

                        if(j==0):
                            theColor = (255, 0, 0)
                        elif(j==1):
                             theColor = (255, 114, 0)
                        elif(j==2):
                            theColor = (0, 76, 255)
                        elif(j==3):
                            theColor = (0, 114, 255)
                        elif(j==4):
                            theColor = (0, 144, 255)
                        elif(j==5):
                            theColor = (0, 174, 255)
                        elif(j==6):
                            theColor = (0, 200, 255)
                        elif(j==7):
                            theColor = (0, 230, 255)
                        elif(j==8):
                            theColor = (0, 255, 255)                                
                        else:
                            theColor = (0, 255, 255)

                        # Do the physics on the spaceship
                        self.ships[j].physics(
                            delta_angle=da,
                            thrust=thrust,
                            stop=self.ships[j].crashed,
                            color = theColor )
                        # Did we land?
                        if self.ships[j].check_on_planet() or self.ships[j].check_pos_screen()==False:
                            self.ships[j].crashed = True

                        self.ships[j].updateFitness(self.level['center_white'])

                        if self.ships[j].check_red_planets(self.planets) == False:
                            self.ships[j].crashed = True
                            # Give it a mean Penalty.
                            #self.ships[j].fitness = self.ships[j].fitness + 0.2

                        #Run this again to update fitness
                        #_ = self.ships[j].predict()
                        #Run this to update fitness


                    pygame.display.flip()
                    self.screen.blit( self.background, (0, 0) )

                    shipsAlive = 0
                    all_crashed = True
                    for j in range(game_settings['num_ships']):
                        if(self.ships[j].crashed == False):
                            all_crashed = False
                            shipsAlive = shipsAlive + 1 

                    loopCount = loopCount + 1

                #Normalize fitness by the number of loops 
                fitnesses = []
                for p in range(game_settings['num_ships']):
                    fitnesses.append(deepcopy(self.ships[p].fitness2))
                theMax = max(fitnesses)

                if(theMax > self.maxes[qqq]):
                    self.maxes[qqq] = theMax
                else:
                    theMax = self.maxes[qqq] 


                if(game_settings['normalize_fitness'] == True):
                    fitnesses = np.array(fitnesses) / loopCount
                
                self.nfitnesses = self.nfitnesses + np.array(fitnesses)


            self.updateWeights()

        pygame.quit()



    def updateWeights(self):
        newShips = []

        scores = np.zeros(game_settings['num_ships'])
        for i in range(game_settings['num_ships']):
            scores[i] = deepcopy(self.nfitnesses[i]) #deepcopy(self.ships[i].fitness2)

        scores_sort = np.sort(scores)[::-1]
        #Invert. Make highest scores to Lowest
        #scores_sort = 1/scores_sort

        reject = True
        if(scores_sort[0]>= self.bestScore):
            self.bestScore = scores_sort[0]
            reject = False
        scores_sort_ind = scores.argsort()[::-1] #Descending order (highest to lowest)

        ##### PRINT STUFF #####
        print("")
        print("Generation: " , self.generation)
        for i in range(game_settings['num_ships']):
            print("Ship Score:",scores[scores_sort_ind[i]],self.ships[scores_sort_ind[i]].fitnessDebug, "Weight:")

        #If we did worse than before, reject this generation

        
        reject = False
        if reject:
            scores = np.zeros(game_settings['num_ships'])
            for i in range(game_settings['num_ships']):
                scores[i] = deepcopy(self.prevFitness[i])
                self.ships[i].mlp = deepcopy(self.prevShips[i])


                #self.ships[i].fitness2 = deepcopy(self.prevFitness[i])

            scores_sort = np.sort(scores)[::-1]
            #Invert. Make highest scores to Lowest
            #scores_sort = 1/scores_sort
            print("Generation Rejected")
            
        
        self.generation = self.generation + 1
        for i in range(game_settings['num_ships']):
            #Get Weight value of best ship
            NN1= deepcopy(self.ships[scores_sort_ind[i]].mlp)
            intercepts= np.concatenate( (NN1.intercepts_[0],NN1.intercepts_[1]))
            weights1 = NN1.coefs_[0].flatten()
            weights2 = NN1.coefs_[1].flatten()
            allWeights = np.concatenate((intercepts,weights1,weights2))
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
                    'score':scores[scores_sort_ind[i]]
                }
                self.logLst.append(logdict)
                fname = "best.pkl"
                #fname = "{}_best.pkl".format(datetime.datetime.now().isoformat().replace(':','-'))

                with open(fname, 'wb') as pfd:
                    pickle.dump(  self.logLst, pfd )

            
        #print(self.bestScore)
        #########################


        # Sort the scores from low value to high values
        # Low values indicate a better score (Closer to landing zone)
        scores_sort_ind = scores.argsort()[::-1]
        sortedShips = []
        for i in range(game_settings['num_ships']):
            sortedShips.append( deepcopy(self.ships[scores_sort_ind[i]].mlp))

       #Normalize the fitness scores
        scores_sum = np.sum(scores_sort)
        scores_sort = scores_sort/scores_sum
        probabilities = scores_sort

        #Take best performing ships(Top 20%) and introduce directly to next round
        num_bestShips = int(np.floor(game_settings['num_ships'] * 0.2))
        for i in range(num_bestShips):
            newShips.append(deepcopy(self.ships[scores_sort_ind[i]].mlp))

        #Take two parents, mutate them, and introduce to next round (Skip crossover)
        for i in range(2):
            parents1 = np.random.choice(range(game_settings['num_ships']), size = 2, replace = False, p=probabilities)
            theNewMlp1 = mutate(sortedShips[parents1[0]])
            newShips.append(deepcopy(theNewMlp1))

        #Whatever ships we have left mutate + crossbreed
        for i in range(int(game_settings['num_ships'] - len(newShips))):
            #Select two parents
            parents = np.random.choice(range(game_settings['num_ships']), size = 2, replace = False, p=probabilities)

            NN = self.crossover(sortedShips[parents[0]],sortedShips[parents[1]])
            theNewMlp = mutate(NN)
            #theNewMlp = self.mutate(sortedShips[parents[0]])

            newShips.append(deepcopy(theNewMlp))



        #Save the previous ships incase all the new ships are worse
        #We don't currently need this because we're always carrying the
        #best ships to the next round
        if(reject == False):
            self.prevShips = []
            self.prevFitness = []
            for i in range(len(self.ships)):
                self.prevShips.append( deepcopy(self.ships[i].mlp))
                self.prevFitness = deepcopy(self.nfitnesses)


        for i in range(len(self.ships)):
            self.ships[i].mlp = deepcopy(newShips[i])
            self.ships[i].fitnessDebug = 0

        self.nfitnesses = np.zeros(game_settings['num_ships'])


    def draw_text( self, text ):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render( text, True, (0, 255, 0) )
        # // makes integer division in python3
        self.screen.blit(
            surface, ( ( self.width - fw ), ( self.height - fh )) )

    def draw_text_top( self, text ):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render( text, True, (0, 255, 0) )
        # // makes integer division in python3
        self.screen.blit(
            surface, ( ( self.width - fw ), ( 20 - fh )) )

    def do_planet( self ):
        """Draw the planet including the gaussian noise
        to simulate erruptions"""

        if self.level["radius_white"] != 0:
            # angle in radians between points defining the planet
            res = 0.01


            # numer of points defining the planet
            npoints = int( 2*math.pi//res + 1)
            thetas = np.arange(0, 2*math.pi, res)
            plist = np.zeros((npoints, 2))

            landform = np.random.normal( scale=2, size=( npoints, 2) )

            plist[:, 0] = self.level["center_white"][0] + self.level["radius_white"]*np.cos(thetas)
            plist[:, 1] = self.level["center_white"][1] + self.level["radius_white"]*np.sin(thetas)



            pygame.draw.circle(self.screen, Colors.white, self.level["center_white"], self.level["radius_white"])


        radii = self.level['radii_red']
        centers = self.level['centers_red']


        self.planets = []
        for i in range(len(centers)):
            np_center = VEC(centers[i])
            rp = red_planet( self.screen, np_center, radii[i]  )
            self.planets.append(rp)
            self.planets[i].render()



        return None

    def resetShipLocs(self):
        """Reset the ship locations, but not their neural net weights"""

        for i in range(game_settings['num_ships']):
            self.ships[i].pos = self.level['ship']['starting_pos']
            self.ships[i].angle = self.level["ship"]['starting_angle']
            self.ships[i].velocity = VEC(0, 0)
            self.ships[i].crashed = False
            self.ships[i].fitness2 = 0
            self.ships[i].sawTheGoodPlanet = False
            self.ships[i].donezo = False

class space_ship:
    """The space ship class"""
    def __init__(self, screen, landing_points, level  ):
        self.level = level
        self.pos = self.level['ship']['starting_pos']
        self.angle = self.level['ship']['starting_angle']
        self.screen = screen
        self.velocity = VEC(0, 0)
        self.landing_points = landing_points
        self.crashed = False
        self.fitness = 0
        self.inputs = np.zeros(n_inputs)
        self.fitness2 = 0
        self.fitnessDebug = 0
        # find mid point of landing
        #li = landing_points.shape[0]//2
        #self.mid_landing_point = VEC(list(self.landing_points[li]))
        self.sawTheGoodPlanet = False
        self.donezo = False

        '''
        # VEC can't be instantiated with array
        # so we convert to list
        lp0 = VEC(list(self.landing_points[0])) -  config["planet_center"]
        lpf = VEC(list(self.landing_points[-1])) - config["planet_center"]
        self.la0 = lp0.angle_to(VEC(1, 0))
        self.laf = lpf.angle_to(VEC(1, 0))
        '''


        self.mlp = MLPClassifier(hidden_layer_sizes=(n_hidden),max_iter=1, activation = "tanh")
        self.mlp.fit(X_train,y_train)
        #Initialize the MLP with random weights

        self.mlp.intercepts_[0] = np.random.rand(n_hidden)*2-1
        self.mlp.intercepts_[1] = np.random.rand(n_output)*2-1
        self.mlp.coefs_[0] = np.random.rand(n_inputs,n_hidden)*2-1
        self.mlp.coefs_[1] = np.random.rand(n_hidden,n_output)*2-1

        self.debug = False

    def validShipPos(self):
        if(self.tip[0] > 0 and self.tip[0] < 1000 and self.tip[1] > 0 and self.tip[1] < 800 and
           self.left[0] > 0 and self.left[0] < 1000 and self.left[1] > 0 and self.left[1] < 800 and
           self.right[0] > 0 and self.right[0] < 1000 and self.right[1] > 0 and self.right[1] < 800
            ):
            return True
        else:
            return False
        

    def updateFitness(self,planetCenter):
        ########Update the ships fitness value###############
        # Fitness is defined as the distance from the landing strip, dLandStrip
        maxD = VEC(1000,800).length()
        bad_distances = 0
        good_distances = 0
        badCount = 0
        goodCount = 0

        distances = self.inputs[range(5)]
        bad = self.inputs[range(5,10)]

        bad_inds = np.where(bad == 1)[0]
        bad_distances = distances[bad_inds]
        bad_distances = np.min(bad_distances)

        if self.validShipPos() == True:
            good_inds = np.where(bad == 0)[0]
            if(len(good_inds) !=0):
                good_distances = distances[good_inds]
                good_distances = np.min(good_distances)
                good_distances = 1/good_distances
                good_distances = good_distances * maxD

                if(good_distances > 50):
                    good_distances = 50

                self.fitnessDebug = self.fitnessDebug + good_distances
                self.fitness2 = self.fitness2 + bad_distances + good_distances
                self.sawTheGoodPlanet = True
            else:
                self.fitness2 = self.fitness2 + bad_distances
        else:
            self.fitness2 = self.fitness2 + 1


        #If we see the planet (once) double my current fitness score. 
        #Encourages ships to come into view of the good planet
        #if(self.sawTheGoodPlanet == True and self.donezo == False):
            #self.fitness2 = self.fitness2 * 2
            #self.donezo = True

        #########Calculate Inputs for fitness##########
        #ship_coors = self.pos
        #dPlanet = (ship_coors - planetCenter).length()

        #########Normalize inputs, want 0 to 1 range########
        #maxD = VEC(1000,800).length()
        #dPlanet = dPlanet/maxD

        #self.fitness = dPlanet

    def predict(self,red_planets):
        string_output = "none"
        X = self.calcInputs(red_planets)

        #########Normalize inputs, want 0 to 1 range########
        self.inputs = deepcopy(X)
        maxD = VEC(1000,800).length()
        X[range(5)] = X[range(5)]/maxD

        #########Make prediction based on inputs##########
        output = self.mlp.predict(X.reshape(1,-1))[0]
        if(output==0):
            string_output = "left"
        elif(output==1):
            string_output = "right"
        return string_output

    def calcInputs(self,red_planets):
        avoidObject = np.zeros(5)
        objectDistances = np.zeros(5)
        #For each direction
        for i in range(5):
            #For each planet (+1 is for the good planet)
            allObjDistances = []
            for j in range(len(red_planets)+1):
                distFromEdge = self.wallIntercept(i)
                #avoidObject[i] = 1
                if(j!=len(red_planets)):#If we're not equal to the last planet (that's the good one)
                    #red_planets[j][0] is the planet center
                    dist = self.circleIntercept(i,red_planets[j][0],red_planets[j][1])
                    if(dist == -1):
                        dist = distFromEdge
                    allObjDistances.append(dist)
                else:
                    #Make the last planet the good one
                    center = np.array(*[self.level['center_white']])
                    dist = self.circleIntercept(i,center, self.level['radius_white'])
                    if(dist == -1):
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
            if(ind != len(red_planets)):
                avoidObject[i] = 1

        return np.concatenate((objectDistances,avoidObject))


    def wallIntercept(self,direction):
            #m is the slope of the line. Used to describe line in direction of ship
            #direction = 4

            if(direction ==0):
                #straight
                m = self.tip - self.back
                x, y = self.tip[0],self.tip[1]
            if(direction == 1):
                #left
                m = self.left - self.right
                x, y = self.left[0],self.left[1]
            if(direction == 2):
                #right
                m = self.right - self.left
                x, y = self.right[0],self.right[1]
            if(direction == 3):
                #left-staight
                m = (self.left + self.tip)/2 - self.right
                x, y = ((self.left + self.tip)/2)[0],((self.left + self.tip)/2)[1]
            if(direction == 4):
                #right-straight
                m = (self.right + self.tip)/2 - self.left
                x, y = ((self.right + self.tip)/2)[0],((self.right + self.tip)/2)[1]
            #Don't want to divide by zero, so just give m a really high value if x in y/x is 0
            if(m[0]==0):
                m=999999
            else:
                m = m[1]/m[0]

            """ m_lw = the slope of the line that describes the left wall of the game world  """
            m_lw = 999999
            m_rw = 999999
            m_bw = 0
            m_tw = 0

            """rw = rightWall, lw = leftWall, bw = bottomWall, tw = topWall"""
            x_lw, y_lw = 0,0
            x_rw, y_rw = 1000,0
            x_bw, y_bw = 1000,800
            x_tw, y_tw = 1000,0

            m_walls = [m_lw,m_rw,m_bw,m_tw]
            x_w = [x_lw,x_rw,x_bw,x_tw]
            y_w = [y_lw,y_rw,y_bw,y_tw]

            lDistances = []
            for ii in range(4):
                if(m - m_walls[ii] == 0):
                    x_i = 999999
                else:
                    x_i = (m*x - y - m_walls[ii]*x_w[ii] + y_w[ii])/(m - m_walls[ii])

                y_i = m*(x_i - x) + y

                dist = (VEC(x_i,y_i) - VEC(x, y)).length()

                if(dist != -1):
                    if(direction == 0):
                        #straight
                        if (self.back - VEC(x_i,y_i)).length() < (self.tip - VEC(x_i,y_i)).length():
                            dist = -1
                    if(direction == 1):
                        #left
                        if (self.right - VEC(x_i,y_i)).length() < (self.left - VEC(x_i,y_i)).length():
                            dist = -1
                    if(direction == 2):
                        #right
                        if (self.left - VEC(x_i,y_i)).length() < (self.right - VEC(x_i,y_i)).length():
                            dist = -1
                    if(direction == 3):
                        #left-staight
                        if (self.right - VEC(x_i,y_i)).length() < ((self.left + self.tip)/2 - VEC(x_i,y_i)).length():
                            dist = -1
                    if(direction == 4):
                        #right-straight
                        if (self.left - VEC(x_i,y_i)).length() < ((self.right + self.tip)/2 - VEC(x_i,y_i)).length():
                            dist = -1
                if(dist != -1):
                    lDistances.append(dist)

            #For some reason, it didn't get any distances once. This will prevent the game from crashing if that happens
            if len(lDistances) == 0:
                lDistances.append(1)
                #print("Bug!")
            return np.min(lDistances)

    def circleIntercept(self,direction,planetCenter,planetRadius):
        """https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle"""

        #m is the slope of the line. c is the y intercept. used to describe line in direction of ship
        #direction = 4

        if(direction == 0):
            #straight
            m = self.tip - self.back
            lineStart = self.tip
        if(direction == 1):
            #left
            m = self.left - self.right
            lineStart = self.left
        if(direction == 2):
            #right
            m = self.right - self.left
            lineStart = self.right
        if(direction == 3):
            #left-staight
            m = (self.left + self.tip)/2 - self.right
            lineStart = (self.left + self.tip)/2
        if(direction == 4):
            #right-straight
            m = (self.right + self.tip)/2 - self.left
            lineStart = (self.right + self.tip)/2
        #Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if(m[0]==0):
            m=999999
        else:
            m = m[1]/m[0]

        #We want left and right 'seeing directions' to be at the back of the ship
        c = lineStart[1] - m * lineStart[0]


        p = planetCenter[0]  #config['planet_center'][0]
        q = planetCenter[1]  #config['planet_center'][1]
        r = planetRadius #config['planet_radius']

        A = m**2 + 1
        B = 2*(m*c - m*q - p)
        C = q**2-r**2+p**2-2*c*q+c**2

        #If B^2−4AC<0 then the line misses the circle
        #If B^2−4AC=0 then the line is tangent to the circle.
        #If B^2−4AC>0 then the line meets the circle in two distinct points.
        if(B**2 - 4*A*C < 0):
            x = -1
            y = -1
            dist = -1
        elif(B**2 - 4*A*C == 0 ):
            x = -B/(2*A)
            y = m*x + c
            dist = (VEC(x,y) - VEC(lineStart[0],lineStart[1])).length()
        else:
            x1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            x2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            y1 = m*x1 + c
            y2 = m*x2 + c

            l1 = (VEC(x1,y1) - VEC(lineStart[0],lineStart[1])).length()
            l2 = (VEC(x2,y2) - VEC(lineStart[0],lineStart[1])).length()

            #Pick the point on the circle that is closest to the ship
            if(l1 < l2):
                x = x1
                y = y1
                dist = l1
            else:
                x = x2
                y = y2
                dist = l2

        #Check to make sure the line intercepts the circle on the front side of the ship
        if(dist != -1):
            if(direction ==0):
                #straight
                if (self.back - VEC(x,y)).length() < (self.tip - VEC(x,y)).length():
                    dist = -1
            if(direction == 1):
                #left
                if (self.right - VEC(x,y)).length() < (self.left - VEC(x,y)).length():
                    dist = -1
            if(direction == 2):
                #right
                if (self.left - VEC(x,y)).length() < (self.right - VEC(x,y)).length():
                    dist = -1
            if(direction == 3):
                #left-staight
                if (self.right - VEC(x,y)).length() < ((self.left + self.tip)/2 - VEC(x,y)).length():
                    dist = -1
            if(direction == 4):
                #right-straight
                if (self.left - VEC(x,y)).length() < ((self.right + self.tip)/2 - VEC(x,y)).length():
                    dist = -1
        return dist

    def render(self, color ):

        tip = VEC( 10, 0)
        left = VEC(-5, -5)
        right = VEC(-5, 5)

        for pt in (tip, right, left):
            pt.rotate_ip( self.angle )
            pt += self.pos
        pygame.draw.polygon(self.screen, color, ( tip, left, right ) )

        self.back = (left + right)/2
        self.tip, self.left, self.right = tip, left, right

    def physics(self, thrust=0.0, delta_angle=0.0, stop=False, color = Colors.blue):
        ppos =  self.level['center_white']

        # gravity = config["gravity"]*(self.pos-ppos).normalize()
        dt = game_settings["dt"]
        if not stop:
            thrust_vector = VEC(1, 0).rotate(self.angle)*thrust
            # self.velocity = self.velocity + (gravity+thrust_vector)*dt
            self.velocity = game_settings['speed_multiplier'] * VEC(1, 0).rotate(self.angle)

            self.pos = self.pos + self.velocity*dt
            self.angle += delta_angle

        '''
        if thrust == 0:
            color = colors.green
        else:
            color = colors.blue
        '''

        self.render( color )

# Begin methods to check win conditions
    def check_orientation(self):
        pangle = ((self.left - self.right).angle_to(
            self.pos - game_settings["planet_center"]))

        if pangle > -90-game_settings["land_angle"] \
                and pangle < -90+game_settings["land_angle"]:

            return True
        else:
            return False

    def check_red_planets(self, rps):
        for (ppos, rad) in rps:
            if (self.tip - ppos).length() < rad:
                return False

        return True



    def check_speed(self):
        if self.velocity.length() < game_settings["land_speed"]:
            return True
        else:
            return False

    def check_pos_screen(self):
        if(self.pos[0] > 0 and self.pos[0] < 1000 and self.pos[1] > 0 and self.pos[1] < 800):
            return True
        else:
            return False
        #print(self.pos)

    def check_on_planet(self):
        # if any part of the ship is touching the planet
        # we have landed
        for pt in (self.tip, self.left, self.right):

            if (pt - VEC( self.level["center_white"] ) ).length()\
                    < self.level["radius_white"]:
                return True
        return False
    
    def NN_Inputs(self):
        ship_coors = self.tip
        land_coors = self.landing_points[0]
        
        ship_angle = self.angle%360
        dSurface = (ship_coors - game_settings["planet_center"]).length() - game_settings["planet_radius"]
        dLandStrip = (ship_coors - land_coors).length()
        
        #Normalize inputs, want -1 to 1 range
        ship_angle = ship_angle/360 * 2 - 1
        maxD = VEC(1000,800).length()
        dSurface = dSurface/maxD*2 - 1
        dLandStrip = dLandStrip/maxD*2 - 1
        
        
        return ship_angle,dSurface,dLandStrip
        




class red_planet:

    def __init__( self, screen, center, radius ):
        self.screen = screen
        self.radius = radius
        self.center = center
        self.idx = 0

    def render( self ):
        """
        # angle in radians between points defining the planet
        res = 0.01


        # numer of points defining the planet

        npoints = int( 2*math.pi//res + 1)
        thetas = np.arange(0, 2*math.pi, res)
        plist = np.zeros((npoints, 2))

        landform = np.random.normal( scale=5, size=( npoints, 2) )

        plist[:, 0] = self.center[0] + self.radius*np.cos(thetas)
        plist[:, 1] = self.center[1] + self.radius*np.sin(thetas)

        pygame.draw.polygon (self.screen, colors.red, plist+landform  )"""
        pygame.draw.circle(self.screen, Colors.red, np.int64(self.center), self.radius)

    def __getitem__( self, key ):
        if key == 0:
            return self.center

        elif key == 1:
            return self.radius

        raise KeyError("red planet index must be 0 or 1 not ", key )

    def __iter__( self ):
        self.idx = 0
        return self

    def __next__(self):
        try:
            item = (self.center, self.radius)[self.idx]
        except IndexError:
            raise StopIteration("Iter error")
        self.idx +=1
        return item


def open(path, *args, **kwargs):
    wpath = pathlib.PureWindowsPath(path)
    return builtins.open(str(pathlib.Path(wpath)), *args, **kwargs )

# End win condition methods.
if __name__ == '__main__':
    if len(sys.argv) == 2:
        levelfile = open( sys.argv[1] )
    else:
        levelfile = open(game_settings['default_level'])

    level = json.load( levelfile )
    # call with width of window and fps
    PygView(level, 1000, 800).run()

