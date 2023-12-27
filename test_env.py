import numpy as np
from utils import get_collision_fn_PR2, load_env, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import pybullet
from PIL import Image
connect(use_gui=True)
# robots, obstacles = load_env('pr2doorway.json')
robots, obstacles = load_env('blocked.json')
# robots, obstacles = load_env('move_several_10.json')
# robots, obstacles = load_env('dinner.json')

base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
# goal_config = (2.6, -1.3, -np.pi)
goal_config = (1.8, -1.8, -np.pi)
# goal_config = (1.8, -1.8, -np.pi)
draw_sphere_marker((goal_config[0], goal_config[1], 0.5), 0.1, (0, 0, 0, 1))

# Adjust the screen coordinates based on your scene and desired view
left = -2.0
right = 2.0
bottom = -1.5
top = 1.5

# Adjust the near and far plane distances based on your scene depth
near = 5
far = 50.0

# Compute the projection matrix
projection_matrix = pybullet.computeProjectionMatrix(left, right, bottom, top, near, far)


view_matrix = pybullet.computeViewMatrix([2, 2, 20], [0, 0, 0.5], [0, 0, 1])

_, _, rgb_array, _, _ = pybullet.getCameraImage(
1024,
1024,
view_matrix,
projection_matrix,
renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
)

image = Image.fromarray(rgb_array)

image.save("tstimg.png")

wait_if_gui()
disconnect()