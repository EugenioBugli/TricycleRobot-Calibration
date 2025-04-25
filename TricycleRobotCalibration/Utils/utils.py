import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt

source_dir = Path(__file__).resolve().parents[1]
with open(source_dir / "config.yml", 'r') as file:
    conf = safe_load(file)

MAX_STEER_TICK = conf["MAX_STEER_TICKS"]
MAX_TRACT_TICK = conf["MAX_TRACT_TICKS"]

MAX_INT_32 = np.iinfo(np.int32).max
MAX_UINT_32 = np.iinfo(np.uint32).max

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