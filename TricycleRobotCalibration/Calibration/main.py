import os
import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.robot import Tricycle
from TricycleRobotCalibration.Utils.dataset import Dataset
from TricycleRobotCalibration.Utils.utils import v2T, T2v, getRotationMatrix
from TricycleRobotCalibration.Calibration.least_squares import leastSquares

this_dir = Path(__file__).resolve().parents[1]
print(this_dir)
with open(this_dir / "config.yml", 'r') as file:
    conf = safe_load(file)

parent_dir = this_dir.parent

PICS_PATH = parent_dir / "Pics"

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
    MAX_INT_32 = np.iinfo(np.int32).max
    MAX_UINT_32 = np.iinfo(np.uint32).max

    t = next_tick - tick

    # fix possible overflow
    if t > MAX_INT_32:
        t -= MAX_UINT_32
    elif t < -MAX_INT_32:
        t += MAX_UINT_32

    return t*K_tract / MAX_TRACT_TICK

def model_prediction(robot_pose, steer_tick, current_tract_tick, next_tract_tick, kine_param):
    # with this function you use the model of the robot to predict the next state
    x_rob, y_rob, theta_rob = robot_pose
    K_steer = kine_param["K_STEER"]
    K_tract = kine_param["K_TRACT"] 
    axis_length = kine_param["AXIS_LENGTH"] 
    steer_offset = kine_param["STEER_OFFSET"]
    # x_sens, y_sens, theta_sens = kine_param

    # print(f"kine param: {K_steer}, {K_tract}, {axis_length}, {steer_offset}")
    steering_angle = get_steering_angle(steer_tick, K_steer)
    traction_distance = get_traction_distance(current_tract_tick, next_tract_tick, K_tract)

    linear_velocity = traction_distance
    angular_velocity = steering_angle

    # print(f"s: {steering_angle}, t: {traction_distance}")

    # kinematic model of the robot with the elimination of dt

    dx = np.cos(steering_angle)*np.cos(theta_rob) * traction_distance
    dy = np.cos(steering_angle)*np.sin(theta_rob) * traction_distance
    dtheta = (np.sin(steering_angle) / axis_length) * traction_distance
    dphi = steering_angle

    return dx.item(), dy.item(), dtheta.item(), dphi.item() # displacement dato dalla predizione del modello


def main():

    data = Dataset()
    robot = Tricycle(np.array([0.0,0.0,0.0]))
    # check if the model that you have defined is correct:

    predicted_poses = np.array([[0,0,0]])

    fig, ax = plt.subplots()
    for i in range(data.length-1):
        # i-th measurement
        robot_pose = data.robot_poses[i]
        steer_tick = data.steer_ticks[i]
        tract_tick = data.tract_ticks[i]
        next_tract_tick = data.tract_ticks[i+1]

        # print(f" i-th measurement: {robot_pose}, {steer_tick}, {tract_tick}, {next_tract_tick}")
        
        dx, dy, dtheta, dphi = model_prediction(
            robot_pose, steer_tick, tract_tick, next_tract_tick, robot.kinematic_parameters)
        
        # print(f" model_prediction: {dx}, {dy}, {dtheta} \n")
        measurement = np.array([[dx, dy, dtheta]])

        # print(f"Movements: {predicted_poses.shape}, measurement: {measurement.shape}")
        # get next poses of the robot and of the sensor
        x, y, theta = measurement.T
        r_T = v2T(measurement.flatten())
        w_T_r = robot.getTransformation()


        robot.pose = T2v(w_T_r @ r_T)
        predicted_poses = np.vstack((predicted_poses, robot_pose))

        ax.scatter(robot_pose[0], robot_pose[1])

    print(predicted_poses.shape)
    ax.legend()
    plt.savefig(PICS_PATH / "model.png")
    # plt.show()
    plt.close()
if __name__ == "__main__":
    main()