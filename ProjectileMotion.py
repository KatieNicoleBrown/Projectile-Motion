#PHYSICS 3926 PROJECT 1 BY KATIE BROWN
from typing import List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

#BEGIN SIMULATION PARAMETERS --------------------------------------

part1 = False                    #True prompts horizontal range computation
part1Plot = False                #True prompts plotting of fig 2.3
part2 = False                    #True prompts computation of AB/HR-ratio
part3A = False                   #True prompts calculation of AB/HR for fence heights from 0.5-15.
part3B = True                    #True prompts calculating the fence height ratio required for AM/HR=10

initialSpeed = 50                #Initial ball speed (m/s) for part 1
theta0 = 45                      #Launch angle (degrees) for part 1
tau = 0.1                        #Time-step for numerical integration (s) in all parts
solveMethod = "Euler"            #Method of integration (Euler, Euler-Cromer or Midpoint) for part 1
airRes = False                   #True turns on air resistance for part 1

numSim = 500                     #Number of simulations to calculate AB/HR in pt.2 & 3

#END SIMULATION PARAMETERS ----------------------------------------

def simulation(initialSpeed, theta0, tau, solveMethod, airRes, ABHRcal):
    """ Simulates projectile motion through numerical integraion

    :param initialSpeed: initial ball speed (m/s)(int/float)
    :param theta0: launch angle (degree) (int/float)
    :param tau: time-step for numerical integration (s) (int/float)
    :param solveMethod: Method of integration (Euler, Euler-Cromer or Midpoint) (string)
    :param airRes: turns on/off air resistance (Boolean)
    :param ABHRcal: causes trajectory to end at fence (Boolean)
    :return: trajectory: list of every (x,y) position vector
    """
    m, d,  g, rho, cd, y0, maxsteps = 0.145, 0.074, 9.81, 1.2, 0.35, 1, 1000
    A = np.pi * (d / 2) ** 2
    r0 = np.array([0,y0])                                       #initial position vector
    v0 = np.array([initialSpeed*np.cos(theta0*np.pi/180), \
                   initialSpeed*np.cos(theta0*np.pi/180)])      #Calclates components of initial velocity vector
    r = np.copy(r0)
    v = np.copy(v0)
    a = np.array([0,0])                                         #Initializes acceleration vector
    trajectory: List[Union[Union[None, float, ndarray], Any]] = []
    if not airRes:                                              #Turns off air resistance
        rho = 0
    airResCoef = (cd*rho*A)/(2*m)

    for _ in range(maxsteps):
        a[0] = - airResCoef * v[0] * np.linalg.norm(v)           #Updates components of acceleration
        a[1] = -g - airResCoef * v[1] * np.linalg.norm(v)

        if solveMethod == 'Euler':
            r = r + v * tau
            v = v + a * tau

        elif solveMethod == 'Euler-Cromer':
            v = v + a * tau
            r = r + v * tau

        elif solveMethod == 'Midpoint':
            v_last = np.copy(v)
            v = v + a * tau
            r = r + tau * (v + v_last) / 2

        trajectory.append(r)
        if r[1] <= 0:                                           #End simulation when ball hits the ground
            break
        if ABHRcal:
            if r[0] >= 121.92:                                  #End simulation when ball reaches fence
                break

    return trajectory

def getXPos(initialSpeed, theta0, tau, solveMethod, airRes, ABHRcal):
    ''' Generates array of x-positions at each time-step of simulation '''
    xPositions = []
    trajectory = simulation(initialSpeed, theta0, tau, solveMethod, airRes, ABHRcal)
    for i in trajectory:
        xPositions.append(i[0])
    xPosArray = np.array(xPositions)
    return xPosArray

def getYPos(initialSpeed, theta0, tau, solveMethod, airRes, ABHRcal):
    ''' Generates array of y-positions at each time-step of simulation '''
    trajectory = simulation(initialSpeed, theta0, tau, solveMethod, airRes, ABHRcal)
    yPositions = []
    for i in trajectory:
        yPositions.append(i[1])
    yPosArray = np.array(yPositions)
    return yPosArray

if part1:                           #Calculate horizontal range as maximum x-position
    xPosArray = getXPos(initialSpeed, theta0, tau, solveMethod, airRes, False)
    xRange = np.amax(xPosArray)
    print(xRange)

if part1Plot:                       #Plots trajectory calculated with each integration method
    eulerTrajectoryX = getXPos(initialSpeed, theta0, tau, 'Euler', airRes, False)
    eulerTrajectoryY = getYPos(initialSpeed, theta0, tau, 'Euler', airRes, False)
    eulerCTrajectoryX = getXPos(initialSpeed, theta0, tau, 'Euler-Cromer', airRes, False)
    eulerCTrajectoryY = getYPos(initialSpeed, theta0, tau, 'Euler-Cromer', airRes, False)
    midpointTrajectoryX = getXPos(initialSpeed, theta0, tau, 'Midpoint', airRes, False)
    midpointTrajectoryY = getYPos(initialSpeed, theta0, tau, 'Midpoint', airRes, False)
    plt.plot(eulerTrajectoryX,eulerTrajectoryY,eulerCTrajectoryX,\
             eulerCTrajectoryY,midpointTrajectoryX,midpointTrajectoryY)
    plt.xlabel('Range (m)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Motion')
    plt.legend(['Euler', 'Euler-Cromer', 'Midpoint'])
    plt.show()

def ABHRfun(fenceHeight, tau, numSim):
    """Calculates the AB/HR ratio

    :param fenceHeight: height of fence (m) (int/float)
    :param tau: time-step for numerical integration (s) (int/float)
    :param numSim: number of iterations to calculate AB/HR
    :return: ABHR: AB/HR ratio (float)
    """
    airRes, solveMethod = True, 'Euler-Cromer'
    atBat = 0
    homeRun = 0
    for _ in range(numSim):
        initialSpeed = 44.7 + (6.7) * np.random.randn()  #Note: 100 mile/h = 44.7 m/s & 15 mph = 6.7 m/s
        theta0 = 45 + (10) * np.random.randn()
        xPosArray = getXPos(initialSpeed, theta0, tau, solveMethod, airRes, True)
        xRange = np.amax(xPosArray)
        yPosArray = getYPos(initialSpeed, theta0, tau, solveMethod, airRes, True)
        if xRange > 121.92:                     #Note: 400 ft = 122 m,
            if yPosArray[-1] > fenceHeight:
                homeRun += 1                    #Count home run if the ball reaches the fence AND clears it
        atBat += 1
    if homeRun > 0:                             #Eliminates divide by zero error
        ABHR = atBat / homeRun
        ABHR = round(ABHR, 2)
    else:
        ABHR = 0                                #ABHR = 0 corresponds to zero home runs
    return ABHR

if part2:                                       #Calculate AB/HR without a fence (fenceHeight = 0)
    ABHR = ABHRfun(0, tau, numSim)
    print(ABHR)

if part3A:                                      #Calculate (& plot) AB/HR for a range of fence heights
    ABHRvariation = []
    for height in range(0,41,1):
        ABHR = ABHRfun(height, tau, numSim)
        ABHRvariation.append(ABHR)
    print(ABHRvariation)
    heights = np.arange(0,41,1)
    plt.plot(heights, ABHRvariation)
    plt.xlabel('Fence Height')
    plt.ylabel('AB/HR')
    plt.show()

if part3B:                                      #Calculate fence height at which AB/HR = 10
    fenceHeight = 0
    ABHR = 0
    while ABHR <= 10:
        ABHR = ABHRfun(fenceHeight, tau, numSim)
        fenceHeight += 0.5
    print(fenceHeight)
