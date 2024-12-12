import numpy as np
import matplotlib.pyplot as plt
f = open("Data/dataset.txt")

def rotationMatrix(theta):
	return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def openData():
	f = open("Data/dataset.txt")
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

	return np.asarray(time), np.asarray(ticks), np.asarray(model), np.asarray(tracker), np.asarray(world_tracker)

time, ticks, model_pose, tracker_pose_robot_frame, tracker_pose_world_frame  = openData()
print("time: ", time.shape)
print("ticks: ", ticks.shape)
print("model_pose: ", model_pose.shape)
print("tracker_pose_robot_frame: ", tracker_pose_robot_frame.shape)
print("tracker_pose_world_frame: ", tracker_pose_world_frame.shape)

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
plt.show()
plt.close()