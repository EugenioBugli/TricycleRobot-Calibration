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

def checkModelValidity(predicted_poses, data):
    predicted_poses = np.array(predicted_poses)
    data.plotModelData(predicted_poses)

    if np.allclose(predicted_poses[:, :2], data.robot_poses[:-1, :2], atol=1e-6):
        print("The prediction given by the model is correct")

def main():

    data = Dataset()
    robot = Tricycle(np.array([0.0,0.0,0.0]))
    # check if the model that you have defined is correct:

    predicted_poses = []

    for i in range(data.length-1):
        # i-th measurement
        robot_pose, steer_tick, tract_tick, next_tract_tick, _ = data.getMeasurement(i)
        dx, dy, dtheta, _ = robot.ModelPrediction(steer_tick, tract_tick, next_tract_tick)
        
        prediction = np.array([[dx, dy, dtheta]]) # that's the prediction of the model aka how much you have moved from your previous reading
        robot.updatePose(prediction)
        predicted_poses.append(robot_pose.copy())

    checkModelValidity(predicted_poses, data)

if __name__ == "__main__":
    main()