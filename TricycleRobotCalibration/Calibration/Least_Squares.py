import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.Utils import Pose, Tricycle, Dataset

config_dir = Path(__file__).resolve().parents[1]
assets_dir = config_dir.resolve.parent

with open(config_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

STATE_DIM = conf["STATE_DIM"]
MEASUREMENT_DIM = conf["MEASUREMENT_DIM"]
NUM_ITERATIONS = conf["NUM_ITERATIONS"]
DATA_SIZE = conf["DATA_SIZE"]
EPSILON = conf["EPSILON"]

DATASET_PATH = assets_dir / "Data" / "dataset.txt"
PICS_PATH = assets_dir / "Pics"

def box_plus(X: np.array, delta_x: np.array):
    """
        > X: [kinematic_parameters | sensor pose]
        > delta_x: [kinematic_parameters | sensor pose]
    """

    X_euclidean, X_manifold = X[:4], Pose.from_vector(X[4:])
    delta_x_euclidean, delta_x_manifold = delta_x[:4], Pose.from_vector(delta_x[4:])

    euclidean_part = X_euclidean + delta_x_euclidean
    manifold_part = X_manifold @ delta_x_manifold
    
    return np.vstack((euclidean_part, manifold_part))

def box_minus(P: Pose, M: Pose):
    """
        > P: prediction
        > M: measurement
    """
    return Pose.from_transformation(np.linalg.inv(M.to_transformation()) @ P.to_transformation())

def get_observation(current_meas: Pose, next_meas: Pose):
    # our observation it's defined as "how much the robot has moved from the last measurement"
    return np.linalg.inv(current_meas.to_transformation()) @ next_meas.to_transformation()

def get_prediction(robot_sensor_pose: Pose, movement: Pose):
    return np.linalg.inv(robot_sensor_pose.to_transformation()) @ movement.to_transformation() @ robot_sensor_pose.to_transformation()

class LS:
    """
        STATE:
            X: {kinematic parameters | sensor pose relative to the robot} = 4 values that belongs to R and one that belongs to SE(2)
            delta_x: {K_steer K_tract axis_length steer_offset | r_x_s r_y_s r_theta_s} euclidean param needed only for the sensor
            box_plus:
                for the kinematic parameter: X_k <--- X_k + delta_x_k
                for the sensor pose: X_s <--- X_s @ v2T(delta_x_s)

        MEASUREMENT:
            Z: {difference btw sensor pose given by odometry of the sensor at instant i+1 and instant i} = {x_s y_s theta_s} belongs to SE(2)
            delta_z: {x_s y_s theta_s} euclidean param needed
            box_minus:
                v2T(delta_z) <- Z^{i+1}' @ Z^{i} TODO

        PREDICTION:
            h: {displacement of the sensor after one step of the model prediction}
    """
    def __init__(self, initial_pose: Pose, data_path: str):
        self.H = np.zeros((MEASUREMENT_DIM, STATE_DIM, DATA_SIZE))
        self.b = np.zeros((MEASUREMENT_DIM, 1, DATA_SIZE))
        self.omega = np.eye(MEASUREMENT_DIM)

        self.error = np.zeros((MEASUREMENT_DIM, 1))
        self.Jacobian = np.zeros((MEASUREMENT_DIM, STATE_DIM))

        self.dataset = Dataset(data_path)
        self.robot = Tricycle(initial_pose)

    def get_error(self, prediction: Pose, measurement: Pose):
        """
            Use this function to compute the error between the predicition and the measurement

            > prediction: how much the robot has moved from the prediction model
            > measurement: how much the model has moved from the last measurement

            error dimension is 3x1
        """
        return box_minus(prediction, measurement)
    
    def get_jacobian(self, param_plus: np.array, param_minus: np.array, measurement: np.array):
        """
            Use this function to compute the jacobian of the error with respect to the state

            > error: difference between how much the robot has moved from the prediction model 
                        and how much has moved from the last measurement
            > x: current state estimate
        
            The complete jacobian has a dimension (3,7) where the first 4 columns are related to
            the kinematic parameters, while the remaining ones to the sensor pose. The first block
            does not need the use of boxplus and boxminus since the quantities are already euclidean,
            while the second block uses them.

            error dimension is 3x1
        """
        # Compute the numerical jacobian
        euclidean_plus, manifold_plus = param_plus[:4], Pose.from_vector(param_plus[4:])
        euclidean_minus, manifold_minus = param_minus[:4], Pose.from_vector(param_minus[4:])

        # get the error 
        
        error_plus_euclidean, error_plus_manifold = self.get_error(param_plus, measurement)
        error_minus_euclidean, error_minus_manifold = self.get_error(param_minus, measurement) 
        # euclidean quantities: (euclidean_plus - euclidean_minus)/2*EPSILON
        J_euclidean = (error_plus_euclidean - error_minus_euclidean)/2*EPSILON
        J_manifold = box_minus(error_plus_manifold, error_minus_manifold)/2*EPSILON

        # manifold quantities: euclidean

        return J_euclidean, J_manifold

    def run(self):
        for i in range(NUM_ITERATIONS):
            for j in range(DATA_SIZE-1):
                _, steer_tick, current_tract_tick, next_tract_tick, sensor_pose = self.dataset.get_measurement(j)

                # this is the prediction of the model
                dx, dy, dtheta, dphi = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick)
                # this is the prediction function
                measurement = Pose(dx, dy, dtheta)
                prediction = get_prediction(sensor_pose, measurement)

                self.error = self.get_error(prediction, measurement)
                param_plus = np.array([steer_tick + EPSILON, 
                                       current_tract_tick + EPSILON, 
                                       next_tract_tick + EPSILON, 
                                       box_plus(sensor_pose, 
                                                Pose(EPSILON, EPSILON, EPSILON))])
                
                param_minus = np.array([steer_tick - EPSILON, 
                                        current_tract_tick - EPSILON,
                                        next_tract_tick - EPSILON, 
                                        box_minus(sensor_pose, 
                                                  Pose(EPSILON, EPSILON, EPSILON))])
                self.Jacobian = self.get_jacobian(param_plus, param_minus)

        return None