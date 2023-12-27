import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from queue import PriorityQueue
#########################
class Node:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = 99999999999999
        self.heuristic = 0
        self.parent = None

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __hash__(self):
        return hash((self.x, self.y, self.theta))

    def __str__(self):
        return f"Node(x={self.x}, y={self.y}, theta={self.theta})"

def traversalcost(n, m):
    xy = (n.x - m.x)**2 + (n.y - m.y)**2    
    t = (min(abs(n.theta - m.theta), ((2*np.pi)- abs(n.theta - m.theta))))**2
    return np.round(np.sqrt(xy + t),3)

def traversalcostf(n, m):
    xy = (n.x - m.x)**2 + (n.y - m.y)**2    
    t = (min(abs(n.theta - m.theta), ((2*np.pi)- abs(n.theta - m.theta))))**2
    return np.round(np.sqrt(xy),3)

def heuristiccost(n, g, h_fn_val):

    if h_fn_val=='octile':
        xdiff = np.abs(n.x - g.x)
        ydiff = np.abs(n.y - g.y)
        cheby_h = np.max([xdiff, ydiff])
        octile_h = cheby_h + ((np.sqrt(2) - 1)*np.min([xdiff, ydiff]))
        h_rot = (np.min(np.array(np.abs(g.theta-n.theta), 2*np.pi - np.abs(g.theta - n.theta))))**2
        h_tran = octile_h 
        dist = np.sqrt(h_rot + h_tran)

        return dist

    elif h_fn_val=='euclidean':
        h_trans = (n.x - g.x)**2 + (n.y - g.y)**2
        h_rot = (np.min(np.array(np.abs(n.theta-g.theta), 2*np.pi - np.abs(n.theta - g.theta))))**2
        dist = np.sqrt(h_trans + h_rot)

        return dist
    else: 
        assert False

def get_neighbors(node, inc):
    x, y, theta = node.x, node.y, node.theta
    del_dx = inc[0]
    del_dy = inc[1]
    det_theta = inc[2]

    neighbors = [
        
        Node(np.round(x-del_dx, 3), np.round(y-del_dy, 3), np.round(theta, 3)),
        Node(np.round(x-del_dx, 3), np.round(y, 3), np.round(theta, 3)),
        Node(np.round(x-del_dx, 3), np.round(y+del_dy, 3), np.round(theta, 3)),
        Node(np.round(x, 3), np.round(y-del_dy, 3), np.round(theta, 3)),
        Node(np.round(x, 3), np.round(y+del_dy, 3), np.round(theta, 3)),
        Node(np.round(x+del_dx, 3), np.round(y-del_dy, 3), np.round(theta, 3)),
        Node(np.round(x+del_dx, 3), np.round(y, 3), np.round(theta, 3)),
        Node(np.round(x+del_dx, 3), np.round(y+del_dy, 3), np.round(theta, 3)),

        
        Node(np.round(x-del_dx, 3), y, np.round((theta+det_theta), 3)),
        Node(np.round(x-del_dx, 3), y, np.round((theta-det_theta), 3)),
        Node(np.round(x, 3), y, np.round((theta+det_theta), 3)),
        Node(np.round(x, 3), y, np.round((theta-det_theta), 3)),
        Node(np.round(x+del_dx, 3), y, np.round((theta+det_theta), 3)),
        Node(np.round(x+del_dx, 3), y, np.round((theta-det_theta), 3)),

        Node(x, np.round(y-del_dy, 3), np.round((theta+det_theta), 3)),
        Node(x, np.round(y-del_dy, 3), np.round((theta-det_theta), 3)),
        Node(x, np.round(y, 3), np.round((theta+det_theta), 3)),
        Node(x, np.round(y, 3), np.round((theta-det_theta), 3)),
        Node(x, np.round(y+del_dy, 3), np.round((theta+det_theta), 3)),
        Node(x, np.round(y+del_dy, 3), np.round((theta-det_theta), 3)),

        Node(np.round(x-del_dx, 3), np.round(y-del_dy, 3), np.round((theta+det_theta), 3)),
        Node(np.round(x-del_dx, 3), np.round(y-del_dy, 3), np.round((theta-det_theta), 3)),
        Node(np.round(x+del_dx, 3), np.round(y+del_dy, 3), np.round((theta+det_theta), 3)),
        Node(np.round(x+del_dx, 3), np.round(y+del_dy, 3), np.round((theta-det_theta), 3)),
        Node(np.round(x+del_dx, 3), np.round(y-del_dy, 3), np.round((theta+det_theta), 3)),
        Node(np.round(x-del_dx, 3), np.round(y+del_dy, 3), np.round((theta-det_theta), 3))
    ]

    return neighbors

def goal_check(node1, node2, threshold=0.1):
    return (abs(node1.x - node2.x) < threshold and 
            abs(node1.y - node2.y) < threshold and 
            abs(node1.theta - node2.theta) < threshold)

def astar(env, gl, inc, hfnval):
    connect(use_gui=True)
    robots, obstacles = load_env(env)

    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))

    goal_config = gl
    path = []
    start_time = time.time()
    final_path_points = set() 
    collision_points = set() 
    non_collision_points = set() 
    
    
    start = Node(np.round(start_config[0],3), np.round(start_config[1],3), np.round(start_config[2],3))
    goal = Node(np.round(goal_config[0],3), np.round(goal_config[1],3), np.round(goal_config[2],3))

    open_list = PriorityQueue()
    open_dict = {start: np.round(start.cost,3)}  
    closed_list = set()

    start.cost = 0
    start.heuristic = heuristiccost(start, goal, hfnval)
    open_list.put(start)

    while not open_list.empty():

        current_node = open_list.get()
        if current_node in closed_list:
            continue

        closed_list.add(current_node)

        if goal_check(current_node, goal):
            while current_node is not None:
                final_path_points.add((current_node.x, current_node.y, 0.5))
                path.append((current_node.x, current_node.y, current_node.theta))
                current_node = current_node.parent
            break

        neighbors = get_neighbors(current_node, inc)

        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            if not collision_fn((neighbor.x, neighbor.y, neighbor.theta)):
                tentative_g = np.round(current_node.cost,3) + traversalcost(current_node, neighbor)
                if neighbor not in open_dict or tentative_g < open_dict[neighbor]:
                    non_collision_points.add((current_node.x, current_node.y, 0.5)) 
                    neighbor.parent = current_node
                    neighbor.cost = tentative_g
                    neighbor.heuristic = heuristiccost(neighbor, goal, hfnval)
                    open_list.put(neighbor)
                    open_dict[neighbor] = tentative_g
            else:
                collision_points.add((current_node.x, current_node.y, 0.5))
                

    path.reverse() 
    cost_final_path = 0
    count = 0

    for i in range(len(path)-1):
        n = Node(path[i][0], path[i][1], path[i][2])
        m = Node(path[i+1][0], path[i+1][1], path[i+1][2])
        cost_final_path+=traversalcostf(n, m)
        count += 1
    run_time = time.time() - start_time
    # print("final cost", cost_final_path)
    print("Planner run time: ", run_time)



    for point in final_path_points:
        draw_sphere_marker(point, 0.09, (0, 1, 0, 1))
    # for point in collision_points:
    #     if point not in final_path_points:
    #         draw_sphere_marker(point, 0.09, (1, 0, 0, 1))
    # for point in non_collision_points:
    #     if point not in final_path_points and point not in collision_points:
    #         draw_sphere_marker(point, 0.09, (0, 0, 1, 1))
    if path:
        print("Path Found")
    else:
        print("No solution found")

    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # wait_if_gui()
    
    # left = -2.0
    # right = 2.0
    # bottom = -1.5
    # top = 1.5

    # near = 5
    # far = 50.0

    # projection_matrix = pybullet.computeProjectionMatrix(left, right, bottom, top, near, far)
    # view_matrix = pybullet.computeViewMatrix([2, 2, 20], [0, 0, 0.5], [0, 0, 1])
    # _, _, rgb_array, _, _ = pybullet.getCameraImage(
    # 1024,
    # 1024,
    # view_matrix,
    # projection_matrix,
    # renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # image = Image.fromarray(rgb_array)

    # image.save(f"{env[:7]}_ASTAR1_{hfnval}.png")
    time.sleep(3)
    disconnect()

    return cost_final_path, run_time



# astar('blocked.json', (1.8, -1.8, -np.pi/2), (.4,.4,np.pi/4), 'octile')
# astar('blocked.json', (1.8, -1.8, -np.pi/2), (.4,.4,np.pi/4), 'euclidean')
# astar('pr2doorway.json', (2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4), 'octile')
# astar('pr2doorway.json', (2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4), 'euclidean')
# astar('move_several_10.json', (1.8, -1.8, -np.pi), (.3,.1,np.pi/4), 'octile')
# astar('move_several_10.json', (1.8, -1.8, -np.pi), (.2,.1,np.pi/4), 'euclidean')