import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Utils.utils import getRotationMatrix

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

		raw_data = openData() # just read the .txt file 
		# processed_ticks = ticksPreProcessing(raw_ticks)

		# shapes will be (N, relative dimension)
		self.time = raw_data["time"]
		self.tract_ticks = raw_data["ticks"][:, 0:1]
		self.steer_ticks = raw_data["ticks"][:, 1:2]
		self.robot_poses = raw_data["model_poses"]
		self.sensor_poses = raw_data["sensor_poses"]

		self.length = self.time.shape[0] 
	
	@staticmethod
	def ticksPreProcessing(raw_ticks):

		return {
			"steer": 0,
			"tract": 0}
			
	@staticmethod
	def getSteeringAngle():
		return 0
		
	@staticmethod
	def getTractionDistance():
		return 0
		
	def plotData(self):

		x_rob, y_rob, theta_rob = self.robot_poses.T
		x_sens_wrt_r, y_sens_wrt_r, _ = self.sensor_poses.T
		# given that the position of the sensor is given wrt to the robot frame, the position of the sensor wrt the world frame is R @ r_p_s + r_p_s

		fig, axs = plt.subplots(1,3)

		axs[0].scatter(x_rob, y_rob, color="yellowgreen", label="Robot Odometry")
		axs[1].scatter(x_sens_wrt_r, y_sens_wrt_r, color="darkorange", label="Sensor wrt Robot RF")
		#axs[2].scatter(x_sens_wrt_w, y_sens_wrt_w, color="cornflowerblue", label="Sensor wrt World RF")

		axs[0].axis("equal")
		axs[1].axis("equal")
		axs[2].axis("equal")
		axs[0].legend()
		axs[1].legend()
		#axs[2].legend()
		fig.set_figheight(5)
		fig.set_figwidth(18)
		plt.savefig(PICS_PATH / "initial_data.png")
		plt.show()
		plt.close()

		
if __name__ == "__main__":
	data = Dataset()
	# data.plotData()