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
EPSILON = float(conf["EPSILON"])

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
            X.sensor_pose.to_transformation() @ delta_x.sensor_pose.to_transformation())
        
        return cls(new_kinem_param, new_sensor_pose)
    
    @classmethod
    def box_minus(cls, X, Y):

        X_inv = np.linalg.inv(X.sensor_pose.to_transformation())
        new_kinem_param = X.kinem_param - Y.kinem_param
        new_sensor_pose = Pose.from_transformation(
            X_inv @ Y.sensor_pose.to_transformation())
        
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

    @staticmethod
    def box_minus(model_prediction: Pose, measurement: Pose):
        inv_measurement = np.linalg.inv(measurement.to_transformation())
        return Pose.from_transformation(inv_measurement @ model_prediction.to_transformation())
    
    @staticmethod
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
            Use this function to compute the error between the predicition and the observation of the sensor pose

            > prediction: how much the sensor has moved from the prediction model
            > observation: how much the sensor has moved from the last measurement

            The dimension is [MEASUREMENT_DIM, 1] = [3, 1]
        """
        error = Measurement.box_minus(prediction, observation)

        return error.to_vector()


    def get_jacobian(self, X: State, delta_x: State, observation: Pose):
        """
            > X: current state
            > delta_x: how much the robot has moved (via model_prediction)
            > observation: how much the sensor has moved

            The complete jacobian has a dimension [MEASUREMENT_DIM, STATE_DIM] = [3, 7] where the first 4 columns are related to
            the kinematic parameters, while the remaining ones to the sensor pose.
        """
        Jacobian = np.zeros((MEASUREMENT_DIM, STATE_DIM))

        for i in range(STATE_DIM):
            # iter over the cols
            perturbation = np.zeros(STATE_DIM)
            perturbation[i] = EPSILON

            perturbed_state = State(perturbation[:4], Pose.from_vector(perturbation[4:]))

            X_plus = State.box_plus(X, perturbed_state)
            X_minus = State.box_minus(X, perturbed_state)

            pred_plus = prediction_function(X_plus.sensor_pose, delta_x)
            pred_minus = prediction_function(X_minus.sensor_pose, delta_x)

            error_plus = self.get_error(pred_plus, observation)
            error_minus = self.get_error(pred_minus, observation)

            Jacobian[:, i] = (error_plus - error_minus)/(2*EPSILON)

        return Jacobian

    def run(self):
        """
            Run Least Squares Algorithm:

                e = h(X*) [-] Z
                J = \partial( e(X* [+] \delta_x) )( \partial(\delta_x) )

                H += J.T @ \omega @ J
                b += J.T @ \omega @ e
            
            \delta_x = solve (H @ \delta_x = -b)
            X* = X* [+] \delta_x
        """
        chi_square = np.zeros((NUM_ITERATIONS, 1))
        for i in range(NUM_ITERATIONS):
            H = np.zeros((STATE_DIM, STATE_DIM))
            b = np.zeros((STATE_DIM, 1))
            chi_iteration = 0
            for j in range(DATA_SIZE-1):

                X = State(
                    kinem_param=np.array(list(self.robot.kinematic_parameters.values())),
                    sensor_pose=self.robot.relative_sensor_pose
                )
                _, steer_tick, current_tract_tick, next_tract_tick, current_sensor_pose, next_sensor_pose = self.dataset.get_measurement(j)

                # this is the movement of the robot
                dx, dy, dtheta, delta_phi = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick)
                delta_x = Pose(dx, dy, dtheta) # movement via model_prediction

                # my observation --> how much the sensor moved
                current_sensor_pose_measurement = Pose.from_vector(current_sensor_pose)
                next_sensor_pose_measurement = Pose.from_vector(next_sensor_pose)
                sensor_movement_observation = Measurement.get_observation(current_sensor_pose_measurement, next_sensor_pose_measurement)
                sensor_movement_prediction = prediction_function(X.sensor_pose, delta_x)

                # compute the error and the jacobian
                self.error = self.get_error(sensor_movement_prediction, sensor_movement_observation)
                self.Jacobian = self.get_jacobian(X, delta_x, sensor_movement_observation)

                H += self.Jacobian.T @ self.omega @ self.Jacobian
                b += self.Jacobian.T @ self.omega @ self.error.reshape(-1, 1)
                chi_iteration += self.error.T @ self.error

                self.robot.update_pose(delta_x)

            chi_square[i] = chi_iteration

            delta = -np.linalg.solve(H, b)
            X_star = State.box_plus(X, State(delta[:4], Pose.from_vector(delta[4:])))

            # update parameters
            self.robot.kinematic_parameters = X_star.kinem_param
            self.robot.relative_sensor_pose = X_star.sensor_pose

if __name__ == "__main__":
    print("Start Least Squares")

    algo = LS(initial_pose=Pose(0.0, 0.0, 0.0), data_path=DATASET_PATH)

    algo.run()