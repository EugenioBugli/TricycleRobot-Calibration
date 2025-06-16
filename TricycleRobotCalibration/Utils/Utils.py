import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt

config_dir = Path(__file__).resolve().parents[1]
assets_dir = config_dir.resolve().parent
with open(config_dir / 'config.yml', 'r') as file:
    conf = safe_load(file)

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

def openData(data_path):
    # TODO check if this can be done in a better way -> array of Poses ?
	time, ticks, model_poses, sensor_poses = [], [], [], []
    
	f = open(data_path)
	lines = f.read().splitlines()[8:] # skip the lines occupied by the infos

	for l in lines:
		# by doing this you will have something like: 
		# ['time', ' 1668091584.821040869 ticks', ' 290 4294859756 model_pose', ' 0 0 0 tracker_pose', ' 6.50242e-05 -0.00354605 0.000941697']
		tokens = l.split(":") 
		current_sensor_pose = list(map(float, tokens[-1].strip().split(" ")))
		current_timestamp = float(tokens[1].strip().strip().split(" ")[0])
		current_ticks = list(map(float, tokens[2].strip().split(" ")[:2])) # here I have first the absolute encoder tick then the incremental one
		
        # due to the structure of the dataset, the robot model string has a variable structure
		robot_pose_with_garbage = []
		model_with_garbage = tokens[3].strip().split(" ")
		for t in model_with_garbage:
			if t != '':
				robot_pose_with_garbage.append(t)
		current_robot_pose = list(map(float, robot_pose_with_garbage[:3]))
		
		model_poses.append(current_robot_pose)
		time.append(current_timestamp)
		ticks.append(current_ticks)
		sensor_poses.append(current_sensor_pose)

	return {
		"time": np.asarray(time).reshape(-1,1), 
		"ticks": np.asarray(ticks), 
		"model_poses": np.asarray(model_poses), 
		"sensor_poses": np.asarray(sensor_poses)}

def get_steering_angle(tick, K_steer):
    # ABSOLUTE Encoder

    if tick > MAX_STEER_TICK/2:
        s = tick - MAX_STEER_TICK
    else:
        s = tick

    angle = s * K_steer

    return (2*np.pi/MAX_STEER_TICK) * angle

def get_traction_distance(tick, next_tick, K_tract):
    # INCREMENTAL Encoder
    # tick and next_tick are uint32 values
    t = next_tick - tick

    # fix possible overflow
    if t > MAX_INT_32:
        t -= MAX_UINT_32
    elif t < -MAX_INT_32:
        t += MAX_UINT_32

    return t*K_tract / MAX_TRACT_TICK

class Pose:
    # SE(2) object
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta
        
    @classmethod
    def from_transformation(cls, T: np.ndarray):
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return cls(x, y, theta)

    def to_vector(self):
        return np.array([self.x , self.y, self.theta])
    
    def to_transformation(self):
        T = np.array([
            [np.cos(self.theta),-np.sin(self.theta), self.x],
            [np.sin(self.theta), np.cos(self.theta), self.y],
            [0, 0, 1]])
        return T


class Tricycle:
    """
        Use this class to model the behavior of the robot
    """
    def __init__(self, initial_pose: Pose):

        self.global_pose = initial_pose 

        self.kinematic_parameters = {
            "K_STEER": INITIAL_K_STEER,
            "K_TRACT": INITIAL_K_TRACT,
            "AXIS_LENGTH": INITIAL_AXIS_LENGTH,
            "STEER_OFFSET": INITIAL_STEER_OFFSET,
        }
        # relative pose, not a global one!
        self.relative_sensor_pose = Pose(INITIAL_LASER_WRT_BASE_X, 
                                         INITIAL_LASER_WRT_BASE_Y, 
                                         INITIAL_LASER_WRT_BASE_ANGLE)
    
    def updatePose(self, new_pose: Pose):
        # set the pose to the new value, which is a relative value
        r_T_m = new_pose.to_transformation() # movement wrt the robot RF
        w_T_r = self.global_pose.to_transformation() # robot wrt world RF
        self.global_pose = Pose.from_transformation(w_T_r @ r_T_m)
    
    def updateKinematicParam(self, K_steer, K_tract, axis_length, steer_offset):
        self.kinematic_parameters = {
            "K_STEER": K_steer,
            "K_TRACT": K_tract,
            "AXIS_LENGTH": axis_length,
            "STEER_OFFSET": steer_offset,
        }

    def model_prediction(self, steer_tick, current_tract_tick, next_tract_tick):

        K_steer = self.kinematic_parameters["K_STEER"]
        K_tract = self.kinematic_parameters["K_TRACT"] 
        axis_length = self.kinematic_parameters["AXIS_LENGTH"]

        steering_angle = get_steering_angle(steer_tick, K_steer)
        traction_distance = get_traction_distance(current_tract_tick, next_tract_tick, K_tract)

        # this is the kinematic model of the robot with the elimination of dt
        dx = np.cos(steering_angle)*np.cos(self.global_pose.theta) * traction_distance
        dy = np.cos(steering_angle)*np.sin(self.global_pose.theta) * traction_distance
        dtheta = (np.sin(steering_angle) / axis_length) * traction_distance
        dphi = steering_angle

        return dx.item(), dy.item(), dtheta.item(), dphi.item()
    
class Dataset:
     
     def __init__(self, data_path: str):
        raw_data = openData(data_path)

        # shapes will be (N, relative dimension)
        self.time = raw_data["time"]
        self.tract_ticks = raw_data["ticks"][:, 0:1]
        self.steer_ticks = raw_data["ticks"][:, 1:2]
        self.robot_poses = raw_data["model_poses"]
        self.sensor_poses = raw_data["sensor_poses"]
        self.length = self.time.shape[0]
    
if __name__ == "__main__":
    print("Check if everything works")
    dataset = Dataset(DATASET_PATH)
    robot = Tricycle(Pose(0.0, 0.0, 0.0))

    fig, axs = plt.subplots(1,2)

    axs[0].scatter(dataset.robot_poses[:, 0], dataset.robot_poses[:, 1], color="royalblue", label="Robot Pose")
    axs[1].scatter(dataset.sensor_poses[:, 0], dataset.sensor_poses[:, 1], color="darkorange", label="Sensor Pose without Calibration")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.set_figheight(5)
    fig.set_figwidth(18)

    plt.savefig(PICS_PATH / "initial_plot.png")
    # plt.show()
    plt.close()

    print("Testing the prediction given by the Model")
    prediction = np.zeros((dataset.length, 2))
    for i in range(dataset.length-1):

        steer_tick = dataset.steer_ticks[i]
        current_tract_tick = dataset.tract_ticks[i]
        next_tract_tick = dataset.tract_ticks[i+1]

        dx, dy, dtheta, dphi = robot.model_prediction(steer_tick, current_tract_tick, next_tract_tick)

        robot.updatePose(Pose(dx, dy, dtheta))
        print("New pose: ", robot.global_pose.to_vector())
        prediction[i] = robot.global_pose.to_vector()[:2]
        
    fig, axs = plt.subplots(1,2)

    axs[0].scatter(dataset.robot_poses[:, 0], dataset.robot_poses[:, 1], color="firebrick", label="Robot Pose (Dataset)")
    axs[1].scatter(prediction[:, 0], prediction[:, 1], color="forestgreen", label="Robot Pose (Model)")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.set_figheight(5)
    fig.set_figwidth(18)

    plt.savefig(PICS_PATH / "prediction_vs_gt.png")
    plt.show()
    plt.close()