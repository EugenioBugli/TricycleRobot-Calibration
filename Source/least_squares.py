import numpy as np
from data_handler import rotationMatrix, openData

# TODO: implement the algorithm of least squares, function which plot the error behavior 
def leastSquares(measurements, predictions):
    H = np.zeros()
    b = np.zeros()
    e = [] # error container
    # iterate over all the measurements
    for j in range(len(measurements)):
        # e_jth = (jth error) current prediction wrt to the current xstar - current measurement
        # e.append(e_jth)
        # J = jacobian of the error wrt to the state and evaluated in the current xstar
        # H = H + J' * omega * J
        # b = b + J' * omega * e
        print(f"iter: {j}") 
    # now I need to extract the current perturbation
    # H * delta_x = -b 
    # x_star = x_star + delta_x
    return e

# DATA :
# timestamps
# absolute encoder ticks --> steering (max_value = 8192)
# incremental encoder ticks --> traction_wheel (max_value = 5000)
# model_pose (odometry) [x, y, theta]
# tracker_pose (sensor position wrt robot) [x_s, y_s, theta_s]

# Nominal values: Ksteer=0.1 Ktract=0.0106141 axis_length=1.4 steer_offset=0 

# DEFINITIONS :
# > State (robot_pose | odometry_param | sensor_pose) = [x_r y_r theta_r | Ksteer Ktract axis_length steer_offset | x_sens y_sens]
# steering wheel motion = r_steer = Ksteer * enc_steer
# traction wheel motion = r_tract = Ktract * enc_tract
# > Measurements --> position of the robot
# > Prediction Function --> gives me the position of the sensor
