import numpy as np
from pathlib import Path
from yaml import safe_load
from autograd import grad, jacobian
from TricycleRobotCalibration.Utils.robot import Tricycle
from TricycleRobotCalibration.Utils.dataset import Dataset
from TricycleRobotCalibration.Utils.utils import getRotationMatrix, v2T, T2v

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

STATE_DIM = conf["STATE_DIM"]
MEASUREMENT_DIM = conf["MEASUREMENT_DIM"]
NUM_ITERATIONS = conf["NUM_ITERATIONS"]
NUM_MEASUREMENTS = conf["NUM_MEASUREMENTS"]

def boxPlus(X, delta_x):
    # since part of the state it's already euclidean we need to divide the operation in two parts
    X_k, X_s = X[:4], X[4:]
    delta_x_k, delta_x_s = delta_x[:4], delta_x[4:]

    # kinematic parameters (already euclidean)
    X_prime_k = X_k + delta_x_k
    # sensor pose (manifold)
    X_prime_s = X_s @ v2T(delta_x_s)
    
    return np.concatenate((
        X_prime_k, X_prime_s
    ))

def boxMinus(h, Z):
    # h is the prediction given by the model aka the pose of the sensor wrt the world 
    # Z is the measurement of the pose of the sensor wrt the world
    return T2v(np.linalg.inv(v2T(Z.flatten())) @ v2T(h.flatten()))

def computeError(prediction, measurement):
    # prediction boxminus measurement
    return boxMinus(prediction, measurement)

def computeJacobian(error):
    # jacobian of the error wrt to the state calculated in the actual state
    return jacobian(error)

def leastSquares(data, robot):
    print("Init Least Squares Algo")

    H = np.zeros(shape=(STATE_DIM, STATE_DIM))
    b = np.zeros(shape=(STATE_DIM, 1))

    X_star = {
        "K_STEER": INITIAL_K_STEER,
        "K_TRACT": INITIAL_K_TRACT,
        "AXIS_LENGTH": INITIAL_AXIS_LENGTH,
        "STEER_OFFSET": INITIAL_STEER_OFFSET,
        "LASER_WRT_BASE_X": INITIAL_LASER_WRT_BASE_X,
        "LASER_WRT_BASE_Y": INITIAL_LASER_WRT_BASE_Y,
        "LASER_WRT_BASE_ANGLE": INITIAL_LASER_WRT_BASE_ANGLE
    }

    for i in range(NUM_ITERATIONS):
        print(f"Iteration number {i}")

        for j in range(NUM_MEASUREMENTS):
            # for each measurement you have to update H and b
            
            # this should be my prediction function 
            _, steer_tick, tract_tick, next_tract_tick, meas = data.getMeasurement(j)
            dx, dy, dtheta, _ = robot.ModelPrediction(steer_tick, tract_tick, next_tract_tick)
            pred_robot_pose = np.array([[dx, dy, dtheta]])
            robot.updatePose(pred_robot_pose)
            pred = robot.Robot2Sensor()
            ######

            # the measurement is given by the amount 

            omega = np.eye(MEASUREMENT_DIM) # (3,3)
            error = computeError(pred, meas) # (3,1)
            Jacobian = computeJacobian(error) # (3,7)

            H_value = Jacobian.T @ omega @ Jacobian
            b_value = Jacobian.T @ omega @ error

            H = H + H_value
            b = b + b_value

        # update your estimate with the perturbation

        # solve H @ delta_x = -b 
        # delta_x = - inv(H) @ b
        # X_star += delta_x  

    print("Finish Least Square Algo")

# DEFINITIONS :
"""
    STATE:
        X: {kinematic parameters | sensor pose relative to the robot} = 4 values that belongs to R and one that belongs to SE(2)
        delta_x: {K_steer K_tract axis_length steer_offset | r_x_s r_y_s r_theta_s} euclidean param needed only for the sensor
        box_plus:
            for the kinematic parameter: X_k <- X_k + delta_x_k
            for the sensor pose: X_s <- X_s @ v2T(delta_x_s)

    MEASUREMENT:
        Z: {sensor pose given by odometry of the sensor} = {x_s y_s theta_s} belongs to SE(2)
        delta_z: {x_s y_s theta_s} euclidean param needed
        box_minus:
            v2T(delta_z) <- Z'@ Z TODO finish this
"""
# > State (kinematic_param | sensor_pose) = [Ksteer Ktract axis_length steer_offset | x_sens y_sens theta_sens]
# > Measurements --> position of the sensor wrt the robot

if __name__ == "__main__":
    data = Dataset()
    robot = Tricycle(np.array([0.0,0.0,0.0]))
    leastSquares(data, robot)