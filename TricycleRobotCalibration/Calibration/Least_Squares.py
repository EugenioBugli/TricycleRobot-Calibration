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

INITIAL_LASER_WRT_BASE_X = conf["INITIAL_LASER_WRT_BASE_X"]
INITIAL_LASER_WRT_BASE_Y = conf["INITIAL_LASER_WRT_BASE_Y"]
INITIAL_LASER_WRT_BASE_ANGLE = conf["INITIAL_LASER_WRT_BASE_ANGLE"]

DEBUG = False

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

    def __repr__(self):
        return f"State: [Ks: {self.kinem_param[0]}, Kt: {self.kinem_param[1]}, a: {self.kinem_param[2]}, delta_s: {self.kinem_param[3]}], [{self.sensor_pose}]"
    
    def __str__(self):
        return f"State: [Ks: {self.kinem_param[0]}, Kt: {self.kinem_param[1]}, a: {self.kinem_param[2]}, delta_s: {self.kinem_param[3]}], [{self.sensor_pose}]"

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


    def get_jacobian(self, X: State, delta_x: State, observation: Pose, steer_tick: float, current_tract_tick: float, next_tract_tick: float):
        """
            > X: current state            
            > delta_x: how much the robot has moved (via model_prediction)
            > observation: how much the sensor has moved

            > steer_tick: steering tick
            > current_tract_tick: current tractor tick
            > next_tract_tick: next tractor tick

            The complete jacobian has a dimension [MEASUREMENT_DIM, STATE_DIM] = [3, 7] where the first 4 columns are related to
            the kinematic parameters, while the remaining ones to the sensor pose.
        """
        Jacobian = np.zeros((MEASUREMENT_DIM, STATE_DIM))

        for i in range(STATE_DIM):

            # perturb only the i-th parameter
            perturbation = np.zeros(STATE_DIM)
            perturbation[i] = EPSILON
            perturbed_state = State(perturbation[:4], Pose.from_vector(perturbation[4:]))

            X_plus = State.box_plus(X, perturbed_state)
            X_minus = State.box_minus(X, perturbed_state)
            
            if DEBUG:
                print(f"Initial {X}")
                print(f"Perturbed {perturbed_state}")
                print (f"Plus {X_plus}")
                print (f"Minus {X_minus}")
                print("\n")

            if i < 4:
                # you are dealing with kinematic parameters -> need a robot perturbed motion
                # Plus perturbed motion
                Ks_plus, Kt_plus, a_plus, delta_s_plus = X_plus.kinem_param
                dx_plus, dy_plus, dtheta_plus, _ = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick, 
                                                                               Ks_plus, Kt_plus, a_plus, delta_s_plus)
                # Minus perturbed motion
                Ks_minus, Kt_minus, a_minus, delta_s_minus = X_minus.kinem_param
                dx_minus, dy_minus, dtheta_minus, _ = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                                                                  Ks_minus, Kt_minus, a_minus, delta_s_minus)

                delta_x_plus = Pose(dx_plus, dy_plus, dtheta_plus)
                delta_x_minus = Pose(dx_minus, dy_minus, dtheta_minus)

                # sensor pose remains the same since we are only perturbing the kinematic parameters here
                pred_plus = prediction_function(X.sensor_pose, delta_x_plus)
                pred_minus = prediction_function(X.sensor_pose, delta_x_minus)

            else:
                # no need to compute againg the perturbed motion since the kinematic parameters remain the same
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
            # self.robot.global_pose = Pose(0.0, 0.0, 0.0)

            H = np.zeros((STATE_DIM, STATE_DIM))
            b = np.zeros((STATE_DIM, 1))
            chi_iteration = 0
            print(f"Iteration number {i}")
            for j in range(DATA_SIZE-1):

                X = State(
                    kinem_param=np.array(list(self.robot.kinematic_parameters.values())),
                    sensor_pose=self.robot.relative_sensor_pose
                )
                _, steer_tick, current_tract_tick, next_tract_tick, current_sensor_pose, next_sensor_pose = self.dataset.get_measurement(j)

                # this is the movement of the robot
                dx, dy, dtheta, _ = self.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                                                self.robot.kinematic_parameters["K_STEER"], self.robot.kinematic_parameters["K_TRACT"], 
                                                                self.robot.kinematic_parameters["AXIS_LENGTH"], self.robot.kinematic_parameters["STEER_OFFSET"])
                delta_x = Pose(dx, dy, dtheta) # movement via model_prediction

                # my observation --> how much the sensor moved
                current_sensor_pose_measurement = Pose.from_vector(current_sensor_pose)
                next_sensor_pose_measurement = Pose.from_vector(next_sensor_pose)
                sensor_movement_observation = Measurement.get_observation(current_sensor_pose_measurement, next_sensor_pose_measurement)
                sensor_movement_prediction = prediction_function(X.sensor_pose, delta_x)

                # compute the error and the jacobian
                self.error = self.get_error(sensor_movement_prediction, sensor_movement_observation)
                self.Jacobian = self.get_jacobian(X, delta_x, sensor_movement_observation, steer_tick, current_tract_tick, next_tract_tick)
                # print(f"J[:,0]= {self.Jacobian[:,0]}, J[:,1]= {self.Jacobian[:,1]}, J[:,2]= {self.Jacobian[:,2]}, J[:,3]= {self.Jacobian[:,3]}, J[:,4]= {self.Jacobian[:,4]}, J[:,5]= {self.Jacobian[:,5]}, J[:,6]= {self.Jacobian[:,6]}")

                H += self.Jacobian.T @ self.omega @ self.Jacobian
                b += self.Jacobian.T @ self.omega @ self.error.reshape(-1, 1)
                chi_iteration += self.error.T @ self.error

                self.robot.update_pose(delta_x)

            chi_square[i] = chi_iteration

            H += np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # Regularization factor ???

            delta = -np.linalg.solve(H, b).ravel()
            X_star = State.box_plus(X, State(delta[:4], Pose.from_vector(delta[4:])))

            # update parameters
            K_s, K_t, a, delta_s = X_star.kinem_param
            self.robot.calibrate_parameters(K_s, K_t, a, delta_s, X_star.sensor_pose)

            new_state = State(
                    kinem_param=np.array(list(self.robot.kinematic_parameters.values())),
                    sensor_pose=self.robot.relative_sensor_pose
                )
            
            print(new_state)
            print(f"Chi iteration: {chi_iteration}")
    
        return new_state

if __name__ == "__main__":
    print("Start Least Squares")

    algo = LS(initial_pose=Pose(0.0, 0.0, 0.0), data_path=DATASET_PATH)

    result = algo.run()
    print(f"Result {result}")

    sensor_calibrated = np.zeros((DATA_SIZE, 2))
    sensor_uncalibrated = np.zeros((DATA_SIZE, 2))
    uncalibrated_robot = Tricycle(initial_pose=Pose(0.0, 0.0, 0.0))   
    for i in range(DATA_SIZE-1):

        _, steer_tick, current_tract_tick, next_tract_tick, _, _ = algo.dataset.get_measurement(i)

        # calibrated prediction
        dx, dy, dtheta, dphi = algo.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                                      algo.robot.kinematic_parameters["K_STEER"], algo.robot.kinematic_parameters["K_TRACT"], 
                                                      algo.robot.kinematic_parameters["AXIS_LENGTH"], algo.robot.kinematic_parameters["STEER_OFFSET"])
        delta = Pose(dx, dy, dtheta)
        algo.robot.update_pose(delta)
        world_sensor_calibrated = Pose.from_transformation(algo.robot.global_pose.to_transformation() @ algo.robot.relative_sensor_pose.to_transformation())

        # uncalibrated prediction
        dx, dy, dtheta, dphi = algo.robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                                      uncalibrated_robot.kinematic_parameters["K_STEER"], uncalibrated_robot.kinematic_parameters["K_TRACT"], 
                                                      uncalibrated_robot.kinematic_parameters["AXIS_LENGTH"], uncalibrated_robot.kinematic_parameters["STEER_OFFSET"])
        delta = Pose(dx, dy, dtheta)
        uncalibrated_robot.update_pose(delta)
        world_sensor_uncalibrated = Pose.from_transformation(uncalibrated_robot.global_pose.to_transformation() @ uncalibrated_robot.relative_sensor_pose.to_transformation())

        sensor_calibrated[i] = world_sensor_calibrated.to_vector()[:2]
        sensor_uncalibrated[i] = world_sensor_uncalibrated.to_vector()[:2]
        
    fig, axs = plt.subplots(1,3)

    axs[0].scatter(algo.dataset.sensor_poses[:, 0], algo.dataset.sensor_poses[:, 1], color="royalblue", label="Sensor Pose Measured (Dataset)")
    axs[1].scatter(sensor_uncalibrated[:, 0], sensor_uncalibrated[:, 1], color="firebrick", label="Sensor Pose Uncalibrated")
    axs[2].scatter(sensor_calibrated[:, 0], sensor_calibrated[:, 1], color="forestgreen", label="Sensor Pose Calibrated")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.set_figheight(5)
    fig.set_figwidth(18)

    plt.savefig(PICS_PATH / "sensor_calibration.png")
    plt.show()
    plt.close()