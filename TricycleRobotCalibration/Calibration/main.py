import os
import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.robot import Tricycle
from TricycleRobotCalibration.Utils.dataset import Dataset
from TricycleRobotCalibration.Calibration.least_squares import leastSquares
from TricycleRobotCalibration.Utils.utils import v2T, T2v, getRotationMatrix, get_steering_angle, get_traction_distance

source_dir = Path(__file__).resolve().parents[1]
with open(source_dir / "config.yml", 'r') as file:
    conf = safe_load(file)

home_dir = source_dir.parent
PICS_PATH = home_dir / "Pics"

MAX_STEER_TICK = conf["MAX_STEER_TICKS"]
MAX_TRACT_TICK = conf["MAX_TRACT_TICKS"]

INITIAL_K_STEER = conf["INITIAL_K_STEER"]
INITIAL_K_TRACT = conf["INITIAL_K_TRACT"]
INITIAL_AXIS_LENGTH = conf["INITIAL_AXIS_LENGTH"]
INITIAL_STEER_OFFSET = conf["INITIAL_STEER_OFFSET"]

LASER_WRT_BASE_X = conf["LASER_WRT_BASE_X"]
LASER_WRT_BASE_ANGLE = conf["LASER_WRT_BASE_ANGLE"]

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

    predicted_poses = []

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
        
        r_T = v2T(measurement.flatten())
        w_T_r = robot.getTransformation()


        robot.pose = T2v(w_T_r @ r_T)
        predicted_poses.append(robot_pose.copy())

    predicted_poses = np.array(predicted_poses)
    fig, axs = plt.subplots(1,2)

    axs[0].scatter(predicted_poses[:, 0], predicted_poses[:, 1], color="mediumseagreen", label="Model Prediction")
    axs[1].scatter(data.robot_poses[:, 0], data.robot_poses[:, 1], color="orange", label="Ground Truth")
    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")
    axs[0].legend()
    axs[1].legend()
    fig.set_figheight(5)
    fig.set_figwidth(18)
    plt.savefig(PICS_PATH / "model_vs_gt.png")
    plt.close()

if __name__ == "__main__":
    main()