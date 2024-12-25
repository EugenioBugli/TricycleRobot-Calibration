import numpy as np
from utils import openData, plotInitialConditions
from least_squares import leastSquares

# given the tricycle structure of the front tractor wheel robot, we can collapse the two rear wheels into one and use
# a bicycle equivalent model

# FWD Bicycle Model

# param config_vector: [x y theta psi]
# param control_inputs: [v w]
# param l: axis length (RF place at the mid)
        
#   self.xdot = controls[0]*np.cos(state[2])*np.cos(state[3])
#   self.ydot = controls[0]*np.sin(state[2])*np.sin(state[3])
#   self.thetadot = controls[0]*np.sin(state[3])/l
#   self.psidot = controls[1]