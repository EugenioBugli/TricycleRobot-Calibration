import os
import numpy as np
from pathlib import Path
from yaml import safe_load
from TricycleRobotCalibration.Utils.utils import getRotationMatrix, v2T, T2v, get_steering_angle, get_traction_distance

this_dir = Path(__file__).resolve().parents[1]
with open(this_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

INITIAL_K_STEER = conf["INITIAL_K_STEER"]
INITIAL_K_TRACT = conf["INITIAL_K_TRACT"]
INITIAL_AXIS_LENGTH = conf["INITIAL_AXIS_LENGTH"]
INITIAL_STEER_OFFSET = conf["INITIAL_STEER_OFFSET"]

INITIAL_LASER_WRT_BASE_X = conf["INITIAL_LASER_WRT_BASE_X"]
INITIAL_LASER_WRT_BASE_Y = conf["INITIAL_LASER_WRT_BASE_Y"]
INITIAL_LASER_WRT_BASE_ANGLE = conf["INITIAL_LASER_WRT_BASE_ANGLE"]
class Tricycle:
    def __init__(self, initial_pose):

        self.pose = initial_pose # x, y, theta
        
        # the following ones are the parameters that will be calibrated aka my state 
        self.kinematic_parameters = {
            "K_STEER": INITIAL_K_STEER,
            "K_TRACT": INITIAL_K_TRACT,
            "AXIS_LENGTH": INITIAL_AXIS_LENGTH,
            "STEER_OFFSET": INITIAL_STEER_OFFSET,
        }
        self.sensor_pose = np.array([ # pose with respect to the robot
            INITIAL_LASER_WRT_BASE_X,
            INITIAL_LASER_WRT_BASE_Y,
            INITIAL_LASER_WRT_BASE_ANGLE
        ])

    def getTransformation(self):
        return v2T(self.pose.flatten())
    
    def updatePose(self, new_pose):
        r_T_m = v2T(new_pose.flatten()) # movement wrt the robot RF
        w_T_r = self.getTransformation() # robot wrt world RF
        self.pose = T2v(w_T_r @ r_T_m)
    
    def updateKinematicParam(self, K_steer, K_tract, axis_length, steer_offset):
        self.kinematic_parameters = {
            "K_STEER": K_steer,
            "K_TRACT": K_tract,
            "AXIS_LENGTH": axis_length,
            "STEER_OFFSET": steer_offset,
        }
    
    def Robot2Sensor(self):
        # transformation that brings you from the world frame to the sensor RF
        w_T_r = self.getTransformation() # robot frame wrt world
        r_T_s = v2T(self.sensor_pose) # sensor frame wrt robot
        return w_T_r @ r_T_s
    
    def ModelPrediction(self, steer_tick, current_tract_tick, next_tract_tick):

        _, _, theta_rob = self.pose

        K_steer = self.kinematic_parameters["K_STEER"]
        K_tract = self.kinematic_parameters["K_TRACT"] 
        axis_length = self.kinematic_parameters["AXIS_LENGTH"]

        steering_angle = get_steering_angle(steer_tick, K_steer)
        traction_distance = get_traction_distance(current_tract_tick, next_tract_tick, K_tract)

        # this is the kinematic model of the robot with the elimination of dt
        dx = np.cos(steering_angle)*np.cos(theta_rob) * traction_distance
        dy = np.cos(steering_angle)*np.sin(theta_rob) * traction_distance
        dtheta = (np.sin(steering_angle) / axis_length) * traction_distance
        dphi = steering_angle

        return dx.item(), dy.item(), dtheta.item(), dphi.item()