import os
import numpy as np
from pathlib import Path
from yaml import safe_load
from TricycleRobotCalibration.Utils.utils import getRotationMatrix, v2T, T2v

this_dir = Path(__file__).resolve().parents[1]
with open(this_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

INITIAL_K_STEER = conf["INITIAL_K_STEER"]
INITIAL_K_TRACT = conf["INITIAL_K_TRACT"]
INITIAL_AXIS_LENGTH = conf["INITIAL_AXIS_LENGTH"]
INITIAL_STEER_OFFSET = conf["INITIAL_STEER_OFFSET"]

LASER_WRT_BASE_X = conf["LASER_WRT_BASE_X"]
LASER_WRT_BASE_ANGLE = conf["LASER_WRT_BASE_ANGLE"]

class Tricycle:
    def __init__(self, initial_pose):

        self.pose = initial_pose # x, y, theta
        self.sensor_pose = np.array([ # pose with respect to the robot
            LASER_WRT_BASE_X, 0, LASER_WRT_BASE_ANGLE
        ])
        self.kinematic_parameters = {
            "K_STEER": INITIAL_K_STEER,
            "K_TRACT": INITIAL_K_TRACT,
            "AXIS_LENGTH": INITIAL_AXIS_LENGTH,
            "STEER_OFFSET": INITIAL_STEER_OFFSET,
        }

    def getTransformation(self):
        return v2T(self.pose)
    
    def update(self, new_pose, new_kinematic_parameters):
        self.pose = new_pose
        self.kinematic_parameters = new_kinematic_parameters
    
    def Robot2Sensor(self):
        # transformation that brings you from the world frame to the sensor RF
        w_T_r = self.getTransformation() # robot frame wrt world
        r_T_s = v2T(self.sensor_pose) # sensor frame wrt robot
        return w_T_r @ r_T_s