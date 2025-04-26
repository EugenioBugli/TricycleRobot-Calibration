import numpy as np
from autograd import grad, jacobian
from TricycleRobotCalibration.Utils.robot import Tricycle
from TricycleRobotCalibration.Utils.dataset import Dataset
from TricycleRobotCalibration.Utils.utils import getRotationMatrix, v2T, T2v

STATE_DIM = 7
MEASUREMENT_DIM = 3
NUM_ITERATIONS = 1
NUM_MEASUREMENTS = 1

def boxminus():
    return 0

def boxplus():
    return 0

def computeError(prediction, measurement):
    # prediction boxminus measurement
    return boxminus(prediction, measurement)

def computeJacobian(error):
    # jacobian of the error wrt to the state calculated in the actual state
    return jacobian(error)

def leastSquares(data):
    print("Init Least Squares Algo")

    H = np.zeros(shape=(STATE_DIM, STATE_DIM))
    b = np.zeros(shape=(STATE_DIM, 1))

    for i in range(NUM_ITERATIONS):
        print(f"Iteration number {i}")

        for j in range(NUM_MEASUREMENTS):
            # for each measurement you have to update H and b
            omega = np.eye(MEASUREMENT_DIM) # (3,3)
            error = np.zeros(shape=(MEASUREMENT_DIM, 1)) # (3,1)
            Jacobian = np.zeros(shape=(MEASUREMENT_DIM, STATE_DIM)) # (3,7)

            H_value = Jacobian.T @ omega @ Jacobian
            b_value = Jacobian.T @ omega @ error

            H = H + H_value
            b = b + b_value

        # update your estimate with the perturbation

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
    leastSquares(data)