import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt

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

def openData(data_path):
	time, ticks, model_poses, sensor_poses = [], [], [], []
    
	f = open(data_path)
	lines = f.read().splitlines()[8:] # skip the lines occupied by the infos

	for l in lines:
		# by doing this you will have something like: 
		# ['time', ' 1668091584.821040869 ticks', ' 290 4294859756 model_pose', ' 0 0 0 tracker_pose', ' 6.50242e-05 -0.00354605 0.000941697']
		tokens = l.split(":") 
		current_sensor_pose = list(map(float, tokens[-1].strip().split(" ")))
		current_timestamp = float(tokens[1].strip().strip().split(" ")[0])
        # here I have first the absolute encoder tick then the incremental one
		current_ticks = [np.uint32(int(x)) for x in tokens[2].strip().split(" ")[:2] ] 
		
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
		"ticks": np.asarray(ticks, dtype=np.uint32), 
		"model_poses": np.asarray(model_poses), 
		"sensor_poses": np.asarray(sensor_poses)}

def get_steering_angle(tick, K_steer, steer_offset):
    # ABSOLUTE Encoder

    abs_ticks = np.int64(tick)

    if abs_ticks > np.int64(MAX_STEER_TICK) // 2:
        s = abs_ticks - np.int64(MAX_STEER_TICK)
    else:
        s = abs_ticks

    angle = s * K_steer

    return ((2*np.pi/MAX_STEER_TICK) * angle) + steer_offset

def get_traction_distance(tick, next_tick, K_tract):
    # INCREMENTAL Encoder
    # tick and next_tick are uint32 values
    t = np.int64(next_tick) - np.int64(tick)

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
    
    def __str__(self):
        return f"Pose: {self.x}, {self.y}, {self.theta}"
            
    @classmethod
    def from_transformation(cls, T: np.ndarray):
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return cls(x, y, theta)
    
    @classmethod
    def from_vector(cls, v: np.ndarray):
        x = v[0]
        y = v[1]
        theta = v[2]
        return cls(x, y, theta)

    def to_vector(self):
        return np.array([self.x , self.y, self.theta])
    
    def to_transformation(self):
        T = np.array([
            [np.cos(self.theta),-np.sin(self.theta), self.x],
            [np.sin(self.theta), np.cos(self.theta), self.y],
            [0, 0, 1]])
        return T
    
class Dataset:
     
    def __init__(self, data_path: str):
        raw_data = openData(data_path)

        # shapes will be (N, relative dimension)
        self.time = raw_data["time"]
        self.steer_ticks = raw_data["ticks"][:, 0:1]
        self.tract_ticks = raw_data["ticks"][:, 1:2]
        self.robot_poses = raw_data["model_poses"]
        self.sensor_poses = raw_data["sensor_poses"]
        self.length = self.time.shape[0]
    
    def get_measurement(self, idx):
        return self.robot_poses[idx], self.steer_ticks[idx], self.tract_ticks[idx], self.tract_ticks[idx+1], self.sensor_poses[idx], self.sensor_poses[idx+1]