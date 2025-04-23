import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

utils_dir = Path(__file__).resolve().parents[2]

DATASET_PATH = utils_dir / "Data" / "dataset.txt"
PICS_PATH = utils_dir / "Pics"

def openData():
	time, ticks, model_poses, sensor_poses = [], [], [], []
	f = open(DATASET_PATH)
	lines = f.read().splitlines()[8:] # skip the lines occupied by the infos

	for l in lines:
		tokens = l.split(":") # by doing this you will have something like: ['time', ' 1668091584.821040869 ticks', ' 290 4294859756 model_pose', ' 0 0 0 tracker_pose', ' 6.50242e-05 -0.00354605 0.000941697']
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

class Dataset:
	
	def __init__(self):
		self.raw_data = openData() # just read the .txt file 
		
		def ticksPreProcessing(self):
			raw_ticks = self.raw_data["ticks"]
			
		@staticmethod
		def getSteeringAngle():
			return 0
		
		@staticmethod
		def getTractionDistance():
			return 0
		
if __name__ == "__main__":
	data = Dataset()
	print(data.raw_data)