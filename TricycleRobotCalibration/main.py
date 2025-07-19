import numpy as np
from pathlib import Path
from yaml import safe_load
import matplotlib.pyplot as plt
from TricycleRobotCalibration.Calibration.Least_Squares import *

config_dir = Path(__file__).resolve().parent
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


print("Start Least Square")

algo = LS(DATASET_PATH)

X_final, chi_square, total_outliers = algo.run()

sensor_calibrated = np.zeros((DATA_SIZE-1, 2))

Ks, Kt, a, delta_s = X_final.kinem_param
sensor_pose = X_final.sensor_pose
inv_sensor_pose = Pose.from_transformation(np.linalg.inv(sensor_pose.to_transformation()))

_, _, _, _, init_sensor_meas, _ = algo.dataset.get_measurement(0)
curret_sensor_pose = Pose.from_vector(init_sensor_meas)
sensor_calibrated[0] = curret_sensor_pose.to_vector()[:2]

for i in range(DATA_SIZE-1):

    _, steer_tick, current_tract_tick, next_tract_tick, actual_sensor_pose, next_sensor_pose = algo.dataset.get_measurement(i)

    delta_robot, _ = model_prediction(steer_tick, current_tract_tick, next_tract_tick,
                                      Ks, Kt, a, delta_s)
    
    delta_sensor = Pose.from_transformation(
        inv_sensor_pose.to_transformation() @ delta_robot.to_transformation() @ sensor_pose.to_transformation()
    )

    curret_sensor_pose = Pose.from_transformation(
        curret_sensor_pose.to_transformation() @ delta_sensor.to_transformation()) 
        
    sensor_calibrated[i] = curret_sensor_pose.to_vector()[:2]

fig, ax = plt.subplots()

ax.scatter(algo.dataset.sensor_poses[:, 0], algo.dataset.sensor_poses[:, 1], color="royalblue", label="Sensor Pose Measured (Dataset)")
ax.scatter(sensor_calibrated[:, 0], sensor_calibrated[:, 1], color="forestgreen", label="Sensor Pose Calibrated (LS)")

ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.set_figheight(5)
fig.set_figwidth(8)

plt.savefig(PICS_PATH / "sensor_calibration.png")
plt.show()
plt.close()

fig, axs = plt.subplots(1,2)

axs[0].plot(chi_square, color="firebrick")
axs[0].scatter(np.arange(NUM_ITERATIONS), chi_square, color="darkorange", label="Error Norm")
axs[0].legend()
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Error")

axs[1].plot(total_outliers, color="darkblue")
axs[1].scatter(np.arange(NUM_ITERATIONS), total_outliers, color="royalblue", label="Outliers")
axs[1].legend()
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Outliers")


fig.set_figheight(6)
fig.set_figwidth(12)
plt.savefig(PICS_PATH / "chi_and_outliers.png")
plt.show()
plt.close()
