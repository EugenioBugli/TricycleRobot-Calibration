import os
import numpy as np
import matplotlib.pyplot as plt

def getRotationMatrix(theta):
	return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def openData():
	this_dir = os.path.dirname(os.path.realpath(__file__)) # Source
	current_dir = os.path.dirname(this_dir) # ../Source 

	f = open(current_dir + "/Data/dataset.txt")
	lines = f.read().splitlines()
	c = 0
	time = []
	ticks = []
	model = []
	tracker = []
	world_tracker = []
	for l in lines:
		c += 1
		if(c < 9): # first 9 lines with infos
			continue
		tokens = l.split(":")
		# example of a token
		# ['time', ' 1668091584.821040869 ticks', ' 290 4294859756 model_pose', ' 0 0 0 tracker_pose', ' 6.50242e-05 -0.00354605 0.000941697']
		tracker_pose = tokens[-1].strip().split(" ") # x y theta
		timestamp = float(tokens[1].strip().strip().split(" ")[0])
		tick = tokens[2].strip().split(" ")[:2] # absolute, incremental
		
		# not sure if this is the best way to do this (needed since the number of strings is not fixed)
		model_with_garbage = tokens[3].strip().split(" ")
		clean_model = []
		for elem in model_with_garbage:
			if elem != '':
				clean_model.append(elem)
		model_pose = list(map(float, clean_model[:3])) # x y theta
		tracker_pose = list(map(float, tracker_pose))
		time.append(timestamp)
		ticks.append(list(map(float, tick)))
		model.append(model_pose)
		tracker.append(tracker_pose)
		world_tracker.append(rotationMatrix(model_pose[2]) @ np.asarray(tracker_pose)[0:2] + np.asarray(model_pose[0:2]))

	return np.hstack((np.asarray(time).reshape(-1,1), np.asarray(ticks), np.asarray(model), np.asarray(tracker), np.asarray(world_tracker)))

def plotInitialConditions(model_pose, tracker_pose_robot_frame, tracker_pose_world_frame):
	fig, axs = plt.subplots(1,3)
	axs[0].scatter(tracker_pose_robot_frame[:,0], tracker_pose_robot_frame[:,1], color="darkorange", label="sensor in robot frame")
	axs[1].scatter(model_pose[:,0], model_pose[:,1], color="yellowgreen", label="robot odometry")
	axs[2].scatter(tracker_pose_world_frame[:,0], tracker_pose_world_frame[:,1], color="cornflowerblue", label="sensor in world frame")
	axs[0].axis("equal")
	axs[1].axis("equal")
	axs[2].axis("equal")
	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	fig.set_figheight(5)
	fig.set_figwidth(18)
	plt.savefig("Pics/initial_data.png")
	plt.show()
	plt.close()

def v2T(v):
    # this function is used to transform a vector into an Homogeneous Transformation
	x, y, theta = v
	return np.array([
        [np.cos(theta),-np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def T2v(T):
    # this function is used to extract a vector from the related Homogeneous Transformation
    v = np.array([
        [T[0,-1]],
        [T[1,-1]],
        [np.atan2(T[1,-1], T[0,-1])]
    ])
    return v

def AnalyzeTime(timestamps):
	# timestamps: array of al the timestamps
	return np.concatenate((np.array([[0.0]]), np.diff(timestamps, axis=0)))

def AnalyzeTicks(ticks):
	# tractor_ticks --> incremental encoder
	init_ticks = np.array([[0.0]], dtype=np.int64)
	delta_ticks = np.diff(ticks, axis=0).astype(np.int64)
	for i in range(len(delta_ticks)):
		if delta_ticks[i] > MAX_TRACT // 2:
			delta_ticks[i] -= MAX_TRACT
		elif delta_ticks[i] < -MAX_TRACT // 2:
			delta_ticks[i] += MAX_TRACT
	data = np.cumsum(delta_ticks)
	norm_data = data % MAX_TRACT
	return np.concatenate((init_ticks, norm_data.reshape(-1,1)))

if __name__ == "__main__":
	data  = openData()
	print(data.shape)
	time = data[:,0:1]
	steer_ticks = data[:,1:2]
	tract_ticks = data[:,2:3].astype(dtype=np.uint32)
	model_pose = data[:,3:6]
	tracker_pose_robot_frame = data[:,6:9]
	tracker_pose_world_frame = data[:,9:]
	print("time: ", time.shape)
	print("steer_ticks: ", steer_ticks.shape)
	print("tract_ticks: ", tract_ticks.shape)
	print("model_pose: ", model_pose.shape)
	print("tracker_pose_robot_frame: ", tracker_pose_robot_frame.shape)
	print("tracker_pose_world_frame: ", tracker_pose_world_frame.shape)
	# plotInitialConditions(model_pose, tracker_pose_robot_frame, tracker_pose_world_frame)
	instants_of_time = AnalyzeTime(time)
	tractor_ticks = AnalyzeTicks(tract_ticks)

	print(np.unique(steer_ticks))