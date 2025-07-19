import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.Utils import *

config_dir = Path(__file__).resolve().parents[1]
assets_dir = config_dir.resolve().parent
with open(config_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

STATE_DIM = conf["STATE_DIM"]
MEASUREMENT_DIM = conf["MEASUREMENT_DIM"]
NUM_ITERATIONS = conf["NUM_ITERATIONS"]
DATA_SIZE = conf["DATA_SIZE"]
EPSILON = float(conf["EPSILON"])

INITIAL_K_STEER = conf["INITIAL_K_STEER"]
INITIAL_K_TRACT = conf["INITIAL_K_TRACT"]
INITIAL_AXIS_LENGTH = conf["INITIAL_AXIS_LENGTH"]
INITIAL_STEER_OFFSET = conf["INITIAL_STEER_OFFSET"]

INITIAL_LASER_WRT_BASE_X = conf["INITIAL_LASER_WRT_BASE_X"]
INITIAL_LASER_WRT_BASE_Y = conf["INITIAL_LASER_WRT_BASE_Y"]
INITIAL_LASER_WRT_BASE_ANGLE = conf["INITIAL_LASER_WRT_BASE_ANGLE"]

MAX_STEER_TICK = conf["MAX_STEER_TICKS"]
MAX_TRACT_TICK = conf["MAX_TRACT_TICKS"]

MAX_INT_32 = np.iinfo(np.int32).max
MAX_UINT_32 = np.iinfo(np.uint32).max

DATASET_PATH = assets_dir / "Data" / "dataset.txt"
PICS_PATH = assets_dir / "Pics"

class State:
    """
        X: {kinematic parameters | sensor pose relative to the robot} \in R^{4} \times SE(2)
        delta_x: {K_steer K_tract axis_length steer_offset | r_x_s r_y_s r_theta_s} euclidean param needed only for the sensor

        box_plus:
            for the kinematic parameter: X_k <--- X_k + delta_x_k
            for the sensor pose: X_s <--- v2T(delta_x_s) @ X_s
    """
    def __init__(self, kinem_param: np.array, sensor_pose: Pose):
        self.kinem_param = kinem_param
        self.sensor_pose = sensor_pose

    def __str__(self):
        return f"State: [Ks: {self.kinem_param[0]}, Kt: {self.kinem_param[1]}, a: {self.kinem_param[2]}, delta_s: {self.kinem_param[3]}], sensor: [{self.sensor_pose}]"

    @classmethod
    def box_plus(cls, X, delta_x): 

        new_kinem_param = X.kinem_param + delta_x.kinem_param

        new_sensor_pose = Pose.from_transformation(
            delta_x.sensor_pose.to_transformation() @ X.sensor_pose.to_transformation())
        
        return cls(new_kinem_param, new_sensor_pose)
    
    def to_vector(self):
        return np.vstack([
            self.kinem_param, self.sensor_pose])
    
class Measurement:
    """
        Z \in SE(2)

        The measurements represent how much the sensor moved from one instant to the next one. The box minus operation is needed since the quantities belong to SE(2).
    """

    def __init__(self, actual_measured_sensor_pose: Pose, next_measured_sensor_pose: Pose):
        self.delta_sensor = Pose.from_transformation(
            np.linalg.inv(actual_measured_sensor_pose.to_transformation()) @ next_measured_sensor_pose.to_transformation()
            )

    @staticmethod
    def box_minus(prediction: Pose, measurement: Pose):
        inv_measurement = np.linalg.inv(measurement.to_transformation())
        return Pose.from_transformation(
            inv_measurement @ prediction.to_transformation()
        )
    

# PREDICTION FUNCTION

def prediction_function(actual_sensor_pose: Pose, delta_robot: Pose):
    """
        With this function I would like to define how much the sensor moved from one instant to the other through the tricycle model 
    """

    inv_actual_sensor_pose = np.linalg.inv(actual_sensor_pose.to_transformation())
    return Pose.from_transformation(
        inv_actual_sensor_pose @ delta_robot.to_transformation() @ actual_sensor_pose.to_transformation())

def error_function(delta_sensor_predicted: Pose, delta_sensor_measured: Pose):
    error = Measurement.box_minus(delta_sensor_predicted, delta_sensor_measured)
    return error.to_vector()

def compute_jacobian(X: State, delta_robot: State, Z: Pose, 
                     steer_tick: float, current_tract_tick: float, next_tract_tick: float):
    """
        > X: current state
        > delta_robot: how much the robot has moved
        > Z: current measurement

        > steer_tick: steering tick
        > current_tract_tick: current tractor tick
        > next_tract_tick: next tractor tick

        The complete jacobian has a dimension [MEASUREMENT_DIM, STATE_DIM] = [3, 7] where the first 4 columns are related to the kinematic parameters, while the remaining ones to the sensor pose.
    """

    J = np.zeros((MEASUREMENT_DIM, STATE_DIM))

    for i in range(STATE_DIM):
        perturbation_vector = np.zeros(STATE_DIM)

        # PLUS part
        perturbation_vector[i] = EPSILON
        delta_plus_perturbed = State(
            kinem_param=perturbation_vector[:4],
            sensor_pose=Pose.from_vector(perturbation_vector[4:]))
        X_plus = State.box_plus(
            X=X,
            delta_x=delta_plus_perturbed)
        # print(f"Delta plus: {perturbation_vector},\n X_plus: {X_plus}")

        # MINUS part
        perturbation_vector[i] = -EPSILON
        delta_minus_perturbed = State(
            kinem_param=perturbation_vector[:4],
            sensor_pose=Pose.from_vector(perturbation_vector[4:]))
        X_minus = State.box_plus(
            X=X,
            delta_x=delta_minus_perturbed)
        
        if i < 4:
            # kinematic parameters -> recompute delta_robot without perturbing the sensor pose
            sensor_pose = X.sensor_pose
            
            # PLUS part
            Ks_plus, Kt_plus, a_plus, delta_s_plus = X_plus.kinem_param
            delta_robot_plus, _ = model_prediction(
                steer_tick, current_tract_tick, next_tract_tick,
                Ks_plus, Kt_plus, a_plus, delta_s_plus)
            
            h_plus = prediction_function(
                actual_sensor_pose=sensor_pose, 
                delta_robot=delta_robot_plus)
            
            # MINUS part
            Ks_minus, Kt_minus, a_minus, delta_s_minus = X_minus.kinem_param
            delta_robot_minus, _ = model_prediction(
                steer_tick, current_tract_tick, next_tract_tick,
                Ks_minus, Kt_minus, a_minus, delta_s_minus)
            
            h_minus = prediction_function(
                actual_sensor_pose=sensor_pose, 
                delta_robot=delta_robot_minus)
            
        else:
            # perturb only the sensor pose without recomputing delta_robot

            # PLUS part
            sensor_pose_plus = X_plus.sensor_pose
            h_plus = prediction_function(
                actual_sensor_pose=sensor_pose_plus, 
                delta_robot=delta_robot)
            
            # MINUS part
            sensor_pose_minus = X_minus.sensor_pose
            h_minus = prediction_function(
                actual_sensor_pose=sensor_pose_minus, 
                delta_robot=delta_robot)
            
        error_plus = error_function(
            delta_sensor_predicted=h_plus,
            delta_sensor_measured=Z
        )

        error_minus = error_function(
            delta_sensor_predicted=h_minus,
            delta_sensor_measured=Z
        )
            
        h_delta = error_plus - error_minus # h_plus.to_vector() - h_minus.to_vector()
        J[:, i] = h_delta/(2*EPSILON)
    
    return J

# PREDICTION MODEL 

def model_prediction(steer_tick, current_tract_tick, next_tract_tick, 
                           K_steer, K_tract, axis_length, steer_offset):

    steering_angle = get_steering_angle(steer_tick, K_steer, steer_offset)
    traction_distance = get_traction_distance(current_tract_tick, next_tract_tick, K_tract)

    # this is the kinematic model of the robot with the elimination of dt
    # apply dtheta rather than self.global_pose.theta to obtain a local movement
    dtheta = (np.sin(steering_angle) / axis_length) * traction_distance
    dx = np.cos(steering_angle)*np.cos(dtheta) * traction_distance
    dy = np.cos(steering_angle)*np.sin(dtheta) * traction_distance
    dphi = steering_angle

    return Pose(dx.item(), dy.item(), dtheta.item()), dphi.item()

class LS:
    def __init__(self, data_path: str):
        self.dataset = Dataset(data_path)

        self.kinematic_parameters = np.array([
            INITIAL_K_STEER,
            INITIAL_K_TRACT,
            INITIAL_AXIS_LENGTH,
            INITIAL_STEER_OFFSET
        ])

        self.relative_sensor_pose = Pose(INITIAL_LASER_WRT_BASE_X, 
                                         INITIAL_LASER_WRT_BASE_Y, 
                                         INITIAL_LASER_WRT_BASE_ANGLE)
        
        self.omega = np.eye(MEASUREMENT_DIM)

    def iteration(self, X: State, current_threshold: float, current_iter: int):
        H = np.zeros((STATE_DIM, STATE_DIM))
        b = np.zeros((STATE_DIM, 1))
        total_chi = 0.0
        num_outliers = 0

        Ks, Kt, a, delta_s = X.kinem_param
        sensor_pose = X.sensor_pose
        
        for j in range(DATA_SIZE-1):

            _, steer_tick, current_tract_tick, next_tract_tick, actual_sensor_pose, next_sensor_pose = self.dataset.get_measurement(j)

            delta_robot, _ = model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                              Ks, Kt, a, delta_s)
            
            # measured sensor movement
            Z = Measurement(
                actual_measured_sensor_pose=Pose.from_vector(actual_sensor_pose),
                next_measured_sensor_pose=Pose.from_vector(next_sensor_pose))

            # predicted sensor movement
            h = prediction_function(
                actual_sensor_pose=sensor_pose, 
                delta_robot=delta_robot)
            
            error = error_function(
                delta_sensor_predicted=h,
                delta_sensor_measured=Z.delta_sensor)
            
            J = compute_jacobian(
                X, delta_robot, Z.delta_sensor,
                steer_tick, current_tract_tick, next_tract_tick)
            
            chi = error.T @ error

            if chi > current_threshold and current_iter > 0:
                error *= np.sqrt(current_threshold/chi)
                chi = current_threshold
                num_outliers += 1
                continue

            H += J.T @ self.omega @ J
            b += J.T @ self.omega @ error.reshape(-1, 1)
            total_chi += chi

        H += 2 * np.eye(STATE_DIM)

        return H, b, total_chi, num_outliers
    
    def run(self):
        chi_square = np.zeros((NUM_ITERATIONS, 1))
        total_outliers = np.zeros((NUM_ITERATIONS, 1))
        current_threshold = 0
        for i in range(NUM_ITERATIONS):
            print(f"Iteration {i}")

            current_state = State(
                kinem_param=self.kinematic_parameters,
                sensor_pose=self.relative_sensor_pose)
            
            H, b, chi_iteration, num_outliers = self.iteration(current_state, current_threshold, i)

            chi_square[i] = chi_iteration
            total_outliers[i] = num_outliers

            current_threshold = 3.0*chi_iteration/DATA_SIZE # 3.0

            delta_x, _, _, _ = np.linalg.lstsq(H, -b, rcond=None)
            delta_x = delta_x.ravel()

            X_star = State.box_plus(
                X=current_state,
                delta_x=State(
                    kinem_param=delta_x[:4],
                    sensor_pose=Pose.from_vector(delta_x[4:])))
            
            # update the state
            self.kinematic_parameters = X_star.kinem_param
            self.relative_sensor_pose = X_star.sensor_pose

            print(f"{X_star}, \n \
                    Chi Square: {chi_iteration} \n \
                    Num_outliers: {num_outliers} \n")

        return X_star, chi_square, total_outliers