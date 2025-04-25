import numpy as np
import matplotlib.pyplot as plt

def getRotationMatrix(theta):
	return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

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
        [np.arctan2(T[1,-1], T[0,-1])]
    ])
    return v

def AnalyzeTime(timestamps):
	# timestamps: array of al the timestamps
	return np.concatenate((np.array([[0.0]]), np.diff(timestamps, axis=0)))