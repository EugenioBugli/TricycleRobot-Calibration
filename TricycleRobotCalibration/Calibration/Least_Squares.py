import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.Utils import Pose, Tricycle, Dataset

config_dir = Path(__file__).resolve().parents[1]
assets_dir = config_dir.resolve().parent

with open(config_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

STATE_DIM = conf["STATE_DIM"]
MEASUREMENT_DIM = conf["MEASUREMENT_DIM"]
NUM_ITERATIONS = conf["NUM_ITERATIONS"]
DATA_SIZE = conf["DATA_SIZE"]
EPSILON = conf["EPSILON"]

DATASET_PATH = assets_dir / "Data" / "dataset.txt"
PICS_PATH = assets_dir / "Pics"

class State:
    """
        X: {kinematic parameters | sensor pose relative to the robot} \in R^{4} \times SE(2)
        delta_x: {K_steer K_tract axis_length steer_offset | r_x_s r_y_s r_theta_s} euclidean param needed only for the sensor

        box_plus:
            for the kinematic parameter: X_k <--- X_k + delta_x_k
            for the sensor pose: X_s <--- X_s @ v2T(delta_x_s)
    """
    def __init__(self, kinem_param: np.array, sensor_pose: Pose):
        self.kinem_param = kinem_param
        self.sensor_pose = sensor_pose

    @classmethod
    def box_plus(cls, X, delta_x): 

        new_kinem_param = X.kinem_param + delta_x.kinem_param
        new_sensor_pose = Pose.from_transformation(
            X.sensor_pose.to_transformstion() @ delta_x.sensor_to_pose.to_transformation())
        
        return cls(new_kinem_param, new_sensor_pose)
    
    @classmethod
    def box_minus(cls, X, Y):

        X_inv = np.linalg.inv(X.sensor_pose.to_transformation())
        new_kinem_param = X.kinem_param - Y.kinem_param
        new_sensor_pose = Pose.from_transformation(
            X_inv @ Y.sensor_to_pose.to_transformation())
        
        return cls(new_kinem_param, new_sensor_pose)
    
class Measurement:
    """
        Z \in SE(2)

        The measurements represent the pose of the sensor with respect to the World frame.
        The box minus operation is needed since the quantities belong to SE(2).

        The observation is defined as the relative transformation between two measurements, which tells me
        how much the sensor moved from instant [i] to instant [i+1]
    """
    def __init__(self, sensor_pose: Pose):
        self.sensor_pose = sensor_pose

    @classmethod
    def box_minus(model_prediction: Pose, measurement: Pose):
        inv_measurement = np.linalg.inv(measurement.to_transformation())
        return Pose.from_transformation(inv_measurement @ model_prediction)
    
    @classmethod
    def get_observation(current_measurement: Pose, new_measurement: Pose):
        inv_current_measurement = np.linalg.inv(current_measurement.to_transformation())
        return Pose.from_transformation(inv_current_measurement @ new_measurement.to_transformation())

def prediction_function(sensor_pose: Pose, delta_sensor: Pose):
    """
        This is the definition of the prediction function.

        > sensor_pose: relative pose of the sensor wrt the robot in the current state
        > delta_sensor: actual motion of the robot (via model_prediction)
    """
    inv_current_sensor_pose = np.linalg.inv(sensor_pose.to_transformation())

    return Pose.from_transformation(inv_current_sensor_pose @ delta_sensor.to_transformation() @ sensor_pose.to_transformation())


class LS:

    def __init__(self, initial_pose: Pose, data_path: str):
        self.dataset = Dataset(data_path)
        self.robot = Tricycle(initial_pose)
        
        self.omega = np.eye(MEASUREMENT_DIM)

        self.error = np.zeros((MEASUREMENT_DIM, 1))
        self.Jacobian = np.zeros((MEASUREMENT_DIM, STATE_DIM))


    def get_error(self, prediction: Pose, observation: Pose):
        """
            Use this function to compute the error between the predicition and the observation

            > prediction: how much the sensor has moved from the prediction model
            > observation: how much the sensor has moved from the last measurement

            The dimension is [MEASUREMENT_DIM, 1] = [3, 1]
        """
        self.error = Measurement.box_minus(prediction, observation)


    def get_jacobian(self, X: State, delta_x: State, observation: Pose):
        """
            > X: current state
            > delta_x: how much the robot has moved (via model_prediction)
            > observation: how much the sensor has moved

            The complete jacobian has a dimension [MEASUREMENT_DIM, STATE_DIM] = [3, 7] where the first 4 columns are related to
            the kinematic parameters, while the remaining ones to the sensor pose.
        """

        X_plus = State.box_plus(X, EPSILON*np.ones(STATE_DIM))
        pred_plus = prediction_function(X_plus.sensor_pose, delta_x)
        error_plus = self.get_error(pred_plus, observation)

        X_minus = State.box_minus(X, EPSILON*np.ones(STATE_DIM))
        pred_minus = prediction_function(X_minus.sensor_pose, delta_x)
        error_minus = self.get_error(pred_minus, observation)

        self.Jacobian = (error_plus - error_minus)/(2*EPSILON)

    def run(self):
        for i in range(NUM_ITERATIONS):
            H = np.zeros((STATE_DIM, STATE_DIM))
            b = np.zeros((STATE_DIM, 1))
            for j in range(DATA_SIZE-1):

                X = State(
                    kinem_param=np.array(list(self.robot.kinematic_parameters.values)),
                    sensor_pose=self.robot.relative_sensor_pose
                )
                _, steer_tick, current_tract_tick, next_tract_tick, sensor_pose = self.dataset.get_measurement(j)

                # this is the movement of the robot
                dx, dy, dtheta, delta_phi = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick)
                delta_x = Pose(dx, dy, dtheta) # movement via model_prediction

                # my observation --> how much the sensor moved
                sensor_pose_measurement = Measurement(sensor_pose)
                # TODO get the previous measurement of the sensor pose
                sensor_movement_observation = Measurement.get_observation(sensor_pose_measurement, prev_measurement)
                sensor_movement_prediction = prediction_function(X.sensor_pose, delta_x)

                # compute the error and the jacobian
                self.get_error(sensor_movement_prediction, sensor_movement_observation)
                self.get_jacobian(X, delta_x, sensor_movement_observation)

                H += self.Jacobian.T @ self.omega @ self.Jacobian
                b += self.Jacobian.T @ self.omega @ self.error

            # solution
            # TODO check types
            delta = -np.linalg.solve(H, b)
            X_star = State.box_plus(X, delta)

            # update parameters
            self.robot.kinematic_parameters = X_star.kinem_param
            self.robot.relative_sensor_pose = X_star.sensor_pose