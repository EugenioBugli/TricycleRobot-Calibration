import os
import numpy as np
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Calibration.least_squares import leastSquares
from TricycleRobotCalibration.Utils.utils import openData, plotInitialConditions

this_dir = os.path.dirname(os.path.realpath(__file__))
with open(this_dir + '/config.yml', 'r') as file:
    conf = safe_load(file)

MAX_STEER_TICK = conf["MAX_STEER_TICKS"]
MAX_TRACT_TICK = conf["MAX_TRACT_TICKS"]

INITIAL_K_STEER = conf["INITIAL_K_STEER"]
INITIAL_K_TRACT = conf["INITIAL_K_TRACT"]
INITIAL_AXIS_LENGTH = conf["INITIAL_AXIS_LENGTH"]
INITIAL_STEER_OFFSET = conf["INITIAL_STEER_OFFSET"]

LASER_WRT_BASE_X = conf["LASER_WRT_BASE_X"]
LASER_WRT_BASE_ANGLE = conf["LASER_WRT_BASE_ANGLE"]

def get_steering_angle(tick, K_steer):
    # ABSOLUTE Encoder

    if tick > MAX_STEER_TICK/2:
        s = tick - MAX_STEER_TICK
    else:
        s = tick

    angle = s * K_steer

    return (2*np.pi/MAX_STEER_TICK) * angle

def get_traction_distance(tick, next_tick, K_tract):
    # INCREMENTAL Encoder
    # tick and next_tick are uint32 values
    # MAX_INT_32 = np.iinfo(np.int32).max
    # MAX_UINT_32 = np.iinfo(np.uint32).max

    t = next_tick - tick

    # fix possible overflow
    # if t > MAX_INT_32:
    #     t -= MAX_UINT_32
    # elif t < -MAX_INT_32:
    #     t += MAX_UINT_32

    return np.int32(t)*K_tract / MAX_TRACT_TICK

def model_prediction(robot_pose, steer_tick, current_tract_tick, next_tract_tick, kine_param):
    # with this function you use the model of the robot to predict the next state
    x_rob, y_rob, theta_rob = robot_pose
    K_steer, K_tract, baseline, x_sens, y_sens, theta_sens = kine_param

    steering_angle = get_steering_angle(steer_tick, K_steer)
    traction_distance = get_traction_distance(current_tract_tick, next_tract_tick, K_tract)

    linear_velocity = traction_distance
    angular_velocity = steering_angle

    # kinematic model of the robot with the elimination of dt

    dx = np.cos(steering_angle)*np.cos(theta_rob) * traction_distance
    dy = np.cos(steering_angle)*np.sin(theta_rob) * traction_distance
    dtheta = (np.sin(steering_angle) / baseline) * traction_distance
    dphi = steering_angle

    return dx, dy, dtheta, dphi # displacement dato dalla predizione del modello


def main():

    data = openData()
    time = data[:,0:1]
    steer_ticks = data[:,1:2]
    tract_ticks = data[:,2:3].astype(dtype=np.uint32)
    robot_pose = data[:,3:6]
    tracker_pose_robot_frame = data[:,6:9]
    tracker_pose_world_frame = data[:,9:]

    kinematic_parameters = 1

    # check if the model that you have defined is correct:

    for i in range(len(time[0])):

        # i-th measurement
        dx, dy, dtheta, dphi = model_prediction(robot_pose[i], steer_ticks[i], tract_ticks[i], tract_ticks[i+1])

        # get next poses of the robot and of the sensor

if __name__ == "__main__":
    main()