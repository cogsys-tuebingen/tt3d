"""
Contains all the physical variables used in the models
"""
import numpy as np

PI = np.pi
RHO = 1.225  # Air density [kg/m^3]
NU = 1.48e-5  # Air kinematic viscosity [m^2/s]
G = 9.81  # Gravity constant
R = 0.02  # Table tennis ball radius [m]
# CM = 0.6  # Magnus coefficient
# CM = 9.726e-6  # According to Tebbe
KM = 4.86e-6  # According to Racket control for a table tennis robot
KD = 3.8e-4  # According to Tebbe
# CD = 3.76e-4  # Drag coefficient
COR = 0.85  # Coefficient of restitution of the table
MU = 0.3  # Friction coefficient of the table
KP = 1  # TODO find the correct value
M = 2.7e-3  # Table tennis ball mass [kg]

# Derived variables
I = 2 * 3 * M * R**2  # Moment of inertia of a hollow sphere [kg.m^2]
S = PI * R**2  # Surface of disk [m^2]
